"""Unit tests for the perturbation-embedding baseline model and metrics.

Tests cover:
- BaselineModel forward-pass shape checks on dummy batches
- Unknown embedding index behaviour
- Pseudobulk delta computation against a hand-computed example
- Pert-to-index mapping
- Each metric function (MSE, MAE, Pearson, Spearman, top-k) against
  hand-computed toy examples
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

from perturbgpt.eval.prediction_metrics import (
    mae,
    mse,
    pearson_corr,
    spearman_corr,
    top_k_recovery,
)
from perturbgpt.models.baseline import (
    UNKNOWN_IDX,
    BaselineModel,
    build_pert_to_idx,
    compute_pseudobulk_deltas,
)
from perturbgpt.data.splitting import CONTROL_LABEL, PerturbationSplit

# ============================================================== model tests

N_HVGS = 20
EMBED_DIM = 8
HIDDEN_DIM = 16
NUM_PERTS = 4


@pytest.fixture
def model():
    return BaselineModel(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        n_hvgs=N_HVGS,
        num_perts=NUM_PERTS,
        dropout=0.0,
    )


def test_baseline_forward_shape(model):
    batch = 7
    pert_idx = torch.tensor([1, 2, 3, 0, 1, 2, 3])
    out = model(pert_idx)
    assert out.shape == (batch, N_HVGS)


def test_baseline_forward_single_batch(model):
    pert_idx = torch.tensor([2])
    out = model(pert_idx)
    assert out.shape == (1, N_HVGS)


def test_baseline_unknown_embedding_is_zero(model):
    """The unknown (index 0) embedding row must be all zeros."""
    emb = model.embedding.weight[UNKNOWN_IDX]
    assert torch.allclose(emb, torch.zeros_like(emb))


def test_baseline_unknown_produces_same_output(model):
    """Two unknown-pert inputs must produce identical outputs (since the
    unknown embedding is zero)."""
    model.eval()
    idx_known = torch.tensor([1, 1, 1])
    idx_unknown = torch.tensor([0, 0, 0])
    with torch.no_grad():
        out_known = model(idx_known)
        out_unknown = model(idx_unknown)
    assert not torch.allclose(out_known, out_unknown)
    with torch.no_grad():
        out_u2 = model(torch.tensor([0, 0, 0]))
    assert torch.allclose(out_unknown, out_u2)


def test_baseline_output_finite(model):
    pert_idx = torch.randint(0, NUM_PERTS + 1, (10,))
    out = model(pert_idx)
    assert torch.all(torch.isfinite(out))


def test_baseline_embedding_has_correct_size(model):
    assert model.embedding.num_embeddings == NUM_PERTS + 1
    assert model.embedding.embedding_dim == EMBED_DIM


# ------------------------------------------------------------- data tests


def _make_split_adata() -> AnnData:
    """3 control + 6 perturbed cells (3 genes 'A', 3 'B'), 5 genes."""
    labels = ["control"] * 3 + ["A"] * 3 + ["B"] * 3
    obs = pd.DataFrame({"perturbation": labels}, index=[f"cell{i}" for i in range(9)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(5)])
    X = np.array(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5],
            [3, 6, 3, 4, 5],
            [3, 6, 3, 4, 5],
            [3, 6, 3, 4, 5],
            [1, 2, 9, 4, 5],
            [1, 2, 9, 4, 5],
            [1, 2, 9, 4, 5],
        ],
        dtype=np.float32,
    )
    return AnnData(sparse.csr_matrix(X), obs=obs, var=var)


def _make_split() -> PerturbationSplit:
    return PerturbationSplit(
        train=["cell0", "cell1", "cell2", "cell3", "cell4", "cell5"],
        val=[],
        test=["cell6", "cell7", "cell8"],
        train_perts={"A"},
        val_perts=set(),
        test_perts={"B"},
        seed=42,
    )


def test_compute_pseudobulk_deltas_hand_computed():
    adata = _make_split_adata()
    split = _make_split()
    deltas = compute_pseudobulk_deltas(adata, split)
    # control cells: [1,2,3,4,5], [2,3,4,5,6], [1,2,3,4,5]
    ctrl_mean = np.array([4 / 3, 7 / 3, 10 / 3, 13 / 3, 16 / 3])
    delta_a = np.array([3, 6, 3, 4, 5]) - ctrl_mean
    np.testing.assert_allclose(deltas["A"], delta_a, atol=1e-6)
    delta_b = np.array([1, 2, 9, 4, 5]) - ctrl_mean
    np.testing.assert_allclose(deltas["B"], delta_b, atol=1e-6)


def test_build_pert_to_idx():
    mapping = build_pert_to_idx({"B", "A", "C"})
    assert mapping["A"] == 1
    assert mapping["B"] == 2
    assert mapping["C"] == 3
    assert mapping.get("Z", UNKNOWN_IDX) == UNKNOWN_IDX



# ============================================================ metric tests


def test_mse_hand_computed():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 4.0, 6.0])
    assert mse(pred, true) == pytest.approx(13.0 / 3.0)


def test_mse_2d():
    pred = np.array([[1, 2], [3, 4]])
    true = np.array([[1, 2], [3, 5]])
    assert mse(pred, true) == pytest.approx(0.25)


def test_mae_hand_computed():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 4.0, 6.0])
    assert mae(pred, true) == pytest.approx(5.0 / 3.0)


def test_mae_2d():
    pred = np.array([[1, 2], [3, 4]])
    true = np.array([[1, 2], [3, 5]])
    assert mae(pred, true) == pytest.approx(0.25)


def test_pearson_perfect_correlation():
    pred = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    true = np.array([[2, 4, 6, 8, 10]], dtype=np.float64)
    assert pearson_corr(pred, true) == pytest.approx(1.0)


def test_pearson_negative_correlation():
    pred = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    true = np.array([[5, 4, 3, 2, 1]], dtype=np.float64)
    assert pearson_corr(pred, true) == pytest.approx(-1.0)


def test_pearson_zero_variance():
    pred = np.array([[1, 1, 1, 1]], dtype=np.float64)
    true = np.array([[1, 2, 3, 4]], dtype=np.float64)
    assert pearson_corr(pred, true) == 0.0


def test_pearson_2d_averaged():
    pred = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float64)
    true = np.array([[2, 4, 6, 8], [8, 6, 4, 2]], dtype=np.float64)
    assert pearson_corr(pred, true) == pytest.approx(1.0)


def test_spearman_perfect_monotonic():
    pred = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    true = np.array([[10, 20, 30, 40, 50]], dtype=np.float64)
    assert spearman_corr(pred, true) == pytest.approx(1.0)


def test_spearman_negative_monotonic():
    pred = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    true = np.array([[50, 40, 30, 20, 10]], dtype=np.float64)
    assert spearman_corr(pred, true) == pytest.approx(-1.0)


def test_spearman_nonlinear_but_monotonic():
    pred = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    true = np.array([[1, 4, 9, 16, 25]], dtype=np.float64)
    assert spearman_corr(pred, true) == pytest.approx(1.0)
    assert pearson_corr(pred, true) < 1.0


def test_top_k_recovery_perfect():
    pred = np.array([[5, 4, 3, 2, 1]], dtype=np.float64)
    true = np.array([[5, 4, 3, 2, 1]], dtype=np.float64)
    result = top_k_recovery(pred, true, k=3)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_top_k_recovery_partial():
    pred = np.array([[5, 3, 1, 0, -1]], dtype=np.float64)
    true = np.array([[10, 1, -1, 8, 6]], dtype=np.float64)
    result = top_k_recovery(pred, true, k=3)
    assert result["precision"] == pytest.approx(1.0 / 3.0)
    assert result["recall"] == pytest.approx(1.0 / 3.0)


def test_top_k_recovery_no_overlap():
    pred = np.array([[5, 4, 3, 2, 1]], dtype=np.float64)
    true = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    result = top_k_recovery(pred, true, k=2)
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)


def test_top_k_recovery_2d_averaged():
    pred = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], dtype=np.float64)
    true = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], dtype=np.float64)
    result = top_k_recovery(pred, true, k=3)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_top_k_k_exceeds_n_genes():
    pred = np.array([[5, 4, 3]], dtype=np.float64)
    true = np.array([[5, 4, 3]], dtype=np.float64)
    result = top_k_recovery(pred, true, k=10)
    assert result["precision"] == pytest.approx(1.0)

