"""Energy accounting: measured GPU energy and a hardware coupling-operation proxy.

Two quantities, reported separately because they answer different questions:

* **Measured GPU energy** (joules): board power sampled from NVML every 50 ms and
  integrated over time (trapezoid rule). The NVML cumulative energy counter is not used:
  on this machine (RTX 5060 Laptop, WSL2) it disagreed with sampled power by up to 5x
  (idle: 73.7 W vs 14.7 W). This is the cost of *simulating* the model on this GPU;
  masked couplings are still computed as dense matrix products, so it does not decrease
  with sparsity. Values are gross (include idle power).
* **Coupling operations per inference** = active edges x integration steps. On
  physical oscillator hardware each active coupling is a physical connection evaluated
  over the integration window; this proxy scales with that cost and is the metric for
  comparing masks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import torch

SAMPLE_INTERVAL_S = 0.05
MIN_INFER_WINDOW_S = 2.0


def coupling_ops_per_inference(edges: int, num_steps: int) -> int:
    """Hardware proxy: active couplings evaluated per integration step, summed over steps."""
    return int(edges) * int(num_steps)


def _nvml_handle():
    """Return an NVML handle for the current CUDA device, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        pynvml.nvmlDeviceGetPowerUsage(handle)  # probe support
        return handle
    except Exception:  # noqa: BLE001 - any NVML failure means "not measurable"
        return None


class GpuEnergyMeter:
    """Context manager integrating sampled GPU power over a block.

    After exit, ``joules`` and ``seconds`` hold the result (``joules`` is None if NVML
    is unavailable) and ``mean_power_w`` the average sampled power.
    """

    def __init__(self) -> None:
        """Initialise an idle meter."""
        self.joules: float | None = None
        self.seconds: float = 0.0
        self.mean_power_w: float | None = None
        self._handle = None
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        import pynvml

        while not self._stop.is_set():
            watts = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            self._samples.append((time.perf_counter(), watts))
            self._stop.wait(SAMPLE_INTERVAL_S)

    def __enter__(self) -> GpuEnergyMeter:
        """Synchronize and start sampling."""
        self._handle = _nvml_handle()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        if self._handle is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Synchronize, stop sampling and integrate."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.seconds = end - self._start
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        samples = self._samples
        if len(samples) < 2:  # block shorter than one interval: use the single reading
            watts = samples[0][1] if samples else 0.0
            self.mean_power_w, self.joules = watts, watts * self.seconds
            return
        joules = sum(
            0.5 * (w0 + w1) * (t1 - t0)
            for (t0, w0), (t1, w1) in zip(samples, samples[1:], strict=False)
        )
        covered = samples[-1][0] - samples[0][0]
        self.mean_power_w = joules / covered
        self.joules = self.mean_power_w * self.seconds  # extend to the full block


def measure_energy_per_call(fn: Callable[[], object]) -> float | None:
    """Average measured GPU energy (J) per ``fn()`` call over a window of >= 2 s."""
    fn()  # warm-up
    calls = 0
    with GpuEnergyMeter() as meter:
        start = time.perf_counter()
        while calls == 0 or time.perf_counter() - start < MIN_INFER_WINDOW_S:
            fn()
            calls += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    return None if meter.joules is None else meter.joules / calls


def gpu_info() -> dict:
    """GPU name, enforced power limit (W) and energy method, for provenance."""
    info = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    handle = _nvml_handle()
    if handle is not None:
        import pynvml

        info["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        info["energy_method"] = "nvml_power_integration_50ms"
    return info
