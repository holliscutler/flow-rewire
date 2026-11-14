import torch

from flow_rewire.masks import random_mask
from flow_rewire.mnist1d_model import KuramotoClassifier
from flow_rewire.physarum import PhysarumConfig, PhysarumRewirer


def _model(seed=0, n_hidden=24, sparsity=0.8):
    torch.manual_seed(seed)
    n = 40 + n_hidden + 40
    mask = random_mask(n, sparsity, generator=torch.Generator().manual_seed(seed))
    return KuramotoClassifier(n_hidden, mask=mask)


def test_rewire_preserves_budget_and_diagonal():
    model = _model()
    rewirer = PhysarumRewirer(model, total_steps=1000, config=PhysarumConfig())
    budget = model.num_edges
    for _ in range(3):
        rewirer.rewire(torch.randn(16, 40))
    assert model.num_edges == budget
    assert not model.mask.diagonal().any()
    assert rewirer.stats.events == 3


def test_dormant_edge_with_strong_potential_flow_reactivates():
    model = _model()
    dormant = (~model.mask & ~torch.eye(model.n, dtype=torch.bool)).nonzero()[0]
    i, j = dormant.tolist()
    with torch.no_grad():
        model.K[i, j] = 1e3  # huge conductance -> dominant potential flow
    PhysarumRewirer(model, 1000, PhysarumConfig()).rewire(torch.randn(16, 40))
    assert model.mask[i, j]


def test_schedule_stops_at_stop_fraction():
    rewirer = PhysarumRewirer(_model(), total_steps=8000, config=PhysarumConfig())
    assert rewirer.due(250) and rewirer.due(6000)
    assert not rewirer.due(6250) and not rewirer.due(100)


def test_flow_uses_raw_coupling_for_dormant_edges():
    model = _model()
    flow = PhysarumRewirer(model, 1000, PhysarumConfig()).flow(torch.randn(8, 40))
    off_diag = ~torch.eye(model.n, dtype=torch.bool)
    assert (flow[~model.mask & off_diag] > 0).any()  # dormant edges get potential flow
    assert (flow.diagonal() == 0).all()
