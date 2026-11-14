import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # skip torch.compile in unit tests

import math

import pytest
import torch
from torchdiffeq import odeint
from un0.model import build_cifar10_model

from flow_rewire.flow import abs_edge_flow, effective_coupling, measure_flow, unwrap
from flow_rewire.masks import apply_mask, topk_mask


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = build_cifar10_model(
        n_oscillators=64, num_steps=10, solver="euler", relativization="mean_relative"
    )
    return model.eval()


def test_abs_edge_flow_matches_bruteforce():
    torch.manual_seed(0)
    theta, K = torch.randn(5, 7), torch.randn(7, 7)
    expected = sum((K * torch.sin(theta[b][None, :] - theta[b][:, None])).abs() for b in range(5))
    torch.testing.assert_close(abs_edge_flow(theta, K, chunk=2), expected)


def test_locked_oscillators_carry_no_flow():
    K = torch.ones(3, 3)
    in_phase = torch.zeros(1, 3)
    anti_phase = torch.tensor([[0.0, math.pi, 0.0]])
    assert abs_edge_flow(in_phase, K).abs().max() < 1e-6
    assert abs_edge_flow(anti_phase, K).abs().max() < 1e-6


def test_measure_flow_reproduces_model_trajectory(tiny_model):
    class_id = torch.arange(10)
    stats = measure_flow(tiny_model, class_id, generator=torch.Generator().manual_seed(1))

    init = tiny_model._sample_initial_state(
        10, device="cpu", dtype=torch.float32, generator=torch.Generator().manual_seed(1)
    )
    drive = unwrap(tiny_model.dynamics).K_drive[class_id]
    ref = odeint(
        lambda t, s: tiny_model.dynamics(s, t, drive),
        init,
        tiny_model._time_grid(torch.device("cpu"), torch.float32),
        method="euler",
        options={"step_size": 0.1},
    )[-1]
    torch.testing.assert_close(stats.final_state, ref, rtol=1e-4, atol=1e-4)
    assert stats.flow.shape == (64, 64)
    assert stats.order_parameter.shape == (11,)


def test_topk_mask_density_and_diagonal():
    mask = topk_mask(torch.rand(20, 20), sparsity=0.75)
    assert not mask.diagonal().any()
    assert mask.sum().item() == round(0.25 * 20 * 19)


def test_attach_mask_blocks_forward_and_gradient(tiny_model):
    from flow_rewire.train_masked import attach_mask

    mask = topk_mask(torch.rand(64, 64), sparsity=0.9)
    attach_mask(tiny_model, mask)
    tiny_model.train()
    tiny_model(torch.arange(4)).sum().backward()
    dyn = unwrap(tiny_model.dynamics)
    assert (dyn.K[~mask] == 0).all()
    grad = dyn.parametrizations.K.original.grad
    assert (grad[~mask] == 0).all() and grad[mask].abs().sum() > 0


def test_apply_mask_zeroes_effective_coupling(tiny_model):
    mask = topk_mask(torch.rand(64, 64), sparsity=0.9)
    apply_mask(tiny_model, mask)
    assert (effective_coupling(tiny_model.dynamics)[~mask] == 0).all()
