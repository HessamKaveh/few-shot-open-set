import numpy as np
import pytest
import torch

import evaluate as ev
from data import Episodes


def test_lower_quantile_finite_sample_rank():
    v = np.random.default_rng(0).permutation(np.arange(1, 20, dtype=float))
    assert ev.lower_quantile(v, 0.1) == 2.0          # floor(20 * 0.1) = 2nd smallest
    assert ev.lower_quantile(v[:8], 0.1) == -np.inf  # floor(9 * 0.1) = 0 -> accept everything


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_lower_quantile_acceptance_rate(alpha):
    rng = np.random.default_rng(1)
    acc = [(rng.normal(size=2000) >= ev.lower_quantile(rng.normal(size=300), alpha)).mean() for _ in range(200)]
    assert np.mean(acc) >= 1 - alpha - 0.01
    assert np.mean(acc) <= 1 - alpha + 1 / 301 + 0.01


def _embeddings(n_classes=12, per_class=40, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = 5 * torch.randn(n_classes, dim, generator=g)
    return {c: centres[c] + torch.randn(per_class, dim, generator=g) for c in range(n_classes)}


def test_episode_shapes_and_near_flags():
    E = _embeddings()
    coarse_of = {c: c // 2 for c in range(12)}       # pairs of classes share a superclass
    sup, qry, unq, near = ev.episode(E, list(range(12)), 3, np.random.default_rng(0), coarse_of)
    assert sup.shape == (ev.N_WAY, 3, 8)
    assert qry.shape == (ev.N_WAY * ev.Q, 8) and unq.shape == (ev.N_UNK * ev.Q, 8)
    assert near.shape == (ev.N_UNK * ev.Q,) and near.dtype == bool


def test_scores_nearest_prototype_and_ranges():
    E = _embeddings()
    sup = torch.stack([E[c][:5] for c in range(5)])
    qry = torch.cat([E[c][5:20] for c in range(5)])
    unk = torch.cat([E[c][:15] for c in range(5, 10)])
    sq, pred = ev.scores(sup, qry, T=1.0)
    su, _ = ev.scores(sup, unk, T=1.0)
    assert set(sq) == {"dist", "msp", "rel"}
    assert (pred == np.repeat(np.arange(5), 15)).mean() > 0.95      # well-separated clusters
    assert np.all((sq["msp"] > 0) & (sq["msp"] <= 1))
    assert np.median(sq["dist"]) > np.median(su["dist"])           # taught queries are closer than unknowns
    # with one shot there is no support spread, so the relative score is not defined
    assert "rel" not in ev.scores(sup[:, :1], qry, T=1.0)[0]


def test_loo_scores_one_per_support_image():
    E = _embeddings()
    sup = torch.stack([E[c][:4] for c in range(5)])
    lo = ev.loo_scores(sup, T=1.0)
    assert set(lo) == set(ev.SCORES)
    assert all(v.shape == (5 * 4,) for v in lo.values())
    assert np.all(lo["dist"] <= 0)


def test_backbone_embedding_dims():
    from model import Conv4, ResNet12
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert Conv4().eval()(x).shape == (2, Conv4.dim)
        assert ResNet12(num_classes=64).eval()(x).shape == (2, ResNet12.dim)


def test_training_episode_stream():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, (60, 32, 32, 3), dtype=np.uint8)
    Y = np.repeat(np.arange(6), 10)
    xs, xq = next(iter(Episodes(X, Y, range(6), n_way=3, k_shot=2, q=4)))
    assert xs.shape == (6, 3, 32, 32) and xq.shape == (12, 3, 32, 32)
