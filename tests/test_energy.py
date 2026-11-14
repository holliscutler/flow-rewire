import pytest
import torch

from flow_rewire.energy import GpuEnergyMeter, coupling_ops_per_inference, measure_energy_per_call


def test_coupling_ops_scales_with_edges_and_steps():
    assert coupling_ops_per_inference(1000, 10) == 10_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + NVML")
def test_gpu_energy_is_positive_for_real_work():
    a = torch.randn(2048, 2048, device="cuda")
    with GpuEnergyMeter() as meter:
        for _ in range(200):
            a = a @ a / 2048**0.5
    assert meter.joules is not None and meter.joules > 0
    per_call = measure_energy_per_call(lambda: a @ a)
    assert per_call is not None and per_call >= 0
