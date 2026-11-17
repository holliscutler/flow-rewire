import pytest
import torch

from flow_rewire.mnist1d_model import N_IN, N_OUT
from flow_rewire.mnist1d_train import build_mask, matched_edge_count
from flow_rewire.web import hops_to_readout, rewire, web_layout, web_mask, wire_length

H = 128  # small web: N = 208


def test_layout_puts_inputs_on_rim_and_readout_at_hub():
    _, ring = web_layout(H)
    assert (ring[:N_IN] == ring.max()).all()
    assert ring[-N_OUT:].max() <= ring[N_IN:-N_OUT].min()


@pytest.mark.parametrize("direction", ["inward", "twoway"])
def test_exact_edge_count_no_diagonal(direction):
    m = web_mask(H, 3000, direction, generator=torch.Generator().manual_seed(0))
    assert int(m.sum()) == 3000 and not m.diagonal().any()


def test_inward_without_rewiring_has_no_outward_edges():
    _, ring = web_layout(H)
    m = web_mask(H, 3000, "inward", rewire_p=0.0)
    i, j = m.nonzero(as_tuple=True)
    assert (ring[j] >= ring[i]).all()  # source never closer to the hub than its receiver


def test_twoway_without_rewiring_is_symmetric_and_local():
    m = web_mask(H, 3000, "twoway", rewire_p=0.0)
    assert (m == m.T).all()
    n = m.shape[0]
    rand = torch.rand(n, n, generator=torch.Generator().manual_seed(0)) < 3000 / (n * (n - 1))
    rand &= ~torch.eye(n, dtype=torch.bool)
    assert wire_length(m, H) < 0.5 * wire_length(rand, H) * 3000 / int(rand.sum())


def test_rewire_preserves_count_and_sources():
    m = web_mask(H, 3000, "twoway", rewire_p=0.0)
    r = rewire(m, 0.5, torch.Generator().manual_seed(1))
    assert int(r.sum()) == 3000 and not r.diagonal().any()
    torch.testing.assert_close(r.sum(dim=0), m.sum(dim=0))  # out-degree per source unchanged


def test_hops_to_readout():
    m = torch.zeros(N_IN + 1 + N_OUT, N_IN + 1 + N_OUT, dtype=torch.bool)
    m[N_IN, 0] = True  # input 0 -> hidden
    assert hops_to_readout(m) is None
    m[N_IN + 1 :, N_IN] = True  # hidden -> every readout
    assert hops_to_readout(m) == 2


def test_build_mask_matches_random_edge_count():
    for kind in ("web_inward", "web_twoway"):
        m = build_mask(kind, H, 0.9375, seed=3)
        assert int(m.sum()) == matched_edge_count(H, 0.9375, seed=3)
