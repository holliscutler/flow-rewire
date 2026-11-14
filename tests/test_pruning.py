import pytest
import torch

from flow_rewire.mnist1d_model import KuramotoClassifier
from flow_rewire.pruning import GradualPruner, PruneConfig, cubic_edge_schedule


def test_cubic_schedule_endpoints_and_monotone():
    counts = [
        cubic_edge_schedule(t, start_edges=1000, final_edges=100, t0=0, t1=6000)
        for t in range(0, 6001, 250)
    ]
    assert counts[0] == 1000 and counts[-1] == 100
    assert all(a >= b for a, b in zip(counts, counts[1:], strict=False))


@pytest.mark.parametrize("score", ["flow", "magnitude"])
def test_pruning_is_permanent_and_reaches_final_edges(score):
    torch.manual_seed(0)
    model = KuramotoClassifier(24)
    pruner = GradualPruner(
        model, final_edges=500, total_steps=1000, config=PruneConfig(score=score)
    )
    x = torch.randn(8, 40)
    previous = model.mask.clone()
    for step in range(250, 751, 250):
        assert pruner.due(step)
        pruner.prune(step, x)
        assert not (model.mask & ~previous).any()  # nothing regrows
        previous = model.mask.clone()
    assert model.num_edges == 500
    assert not pruner.due(1000)


def test_flow_prunes_edges_between_locked_oscillators_first():
    torch.manual_seed(0)
    model = KuramotoClassifier(0, num_steps=1)
    with torch.no_grad():
        model.K.fill_(1.0)
        model.omega.zero_()
    # Encoder pins inputs: oscillators 0 and 1 share a phase; all others differ.
    phases = torch.linspace(0.0, 3.0, 40)
    phases[1] = phases[0]
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(phases)
    pruner = GradualPruner(
        model, final_edges=model.num_edges - 2, total_steps=4, config=PruneConfig(interval=1)
    )
    pruner.prune(3, torch.zeros(4, 40))
    assert not model.mask[0, 1] and not model.mask[1, 0]


def test_default_trajectory_unchanged_by_shared_noise_option():
    torch.manual_seed(0)
    model = KuramotoClassifier(8)
    x = torch.randn(4, 40)
    a = model.trajectory(x, torch.Generator().manual_seed(3))[-1]
    b = model.trajectory(x, torch.Generator().manual_seed(3), shared_noise=False)[-1]
    torch.testing.assert_close(a, b)
    shared = model.initial_state(x, torch.Generator().manual_seed(3), shared_noise=True)
    assert (shared[:, 40:] == shared[:1, 40:]).all()  # non-input phases identical


def test_perturbation_score_prunes_and_reaches_final_edges():
    torch.manual_seed(0)
    model = KuramotoClassifier(24)
    pruner = GradualPruner(
        model, final_edges=500, total_steps=1000, config=PruneConfig(score="perturbation")
    )
    for step in (250, 500, 750):
        pruner.prune(step, torch.randn(8, 40))
    assert model.num_edges == 500
