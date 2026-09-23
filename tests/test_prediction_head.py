"""Shape and forward-pass sanity tests for the gene-embedding MLP head.

Tests cover:
- GeneMLP forward-pass shape checks on dummy batches
- Conditioning regression test: changing the perturbation (gene) embedding
  *must* change the output (guards against the embedding being ignored)
- GeneMLP attribute and num_layers clamping checks
- compute_perturbation_embedding for single, combo, and unknown genes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from perturbgpt.models.prediction_head import (
    GeneMLP,
    compute_perturbation_embedding,
)

# ============================================================== constants

PERT_DIM = 16
HIDDEN = 32
N_HVGS = 20
NUM_LAYERS = 3
BATCH = 7


# ============================================================ GeneMLP tests


@pytest.fixture
def model():
    return GeneMLP(
        pert_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN,
        num_layers=NUM_LAYERS,
        n_hvgs=N_HVGS,
        dropout=0.0,
    )


def test_forward_shape(model):
    pert = torch.randn(BATCH, PERT_DIM)
    out = model(pert)
    assert out.shape == (BATCH, N_HVGS)


def test_forward_single_batch(model):
    pert = torch.randn(1, PERT_DIM)
    out = model(pert)
    assert out.shape == (1, N_HVGS)


def test_conditioning_changes_output(model):
    """Changing the perturbation embedding MUST change the output.

    Primary regression test against the gene embedding silently being
    ignored by the MLP.
    """
    model.eval()
    pert1 = torch.randn(1, PERT_DIM)
    pert2 = torch.randn(1, PERT_DIM)
    with torch.no_grad():
        out1 = model(pert1)
        out2 = model(pert2)
    assert not torch.allclose(out1, out2, atol=1e-8), (
        "gene-embedding MLP is a no-op: different perturbation embeddings "
        "produced identical predicted deltas."
    )


def test_num_layers_minimum():
    """num_layers=0 should be clamped to 1."""
    m = GeneMLP(
        pert_emb_dim=PERT_DIM, hidden_dim=HIDDEN, num_layers=0,
        n_hvgs=N_HVGS, dropout=0.0,
    )
    assert m.num_layers == 1
    out = m(torch.randn(1, PERT_DIM))
    assert out.shape == (1, N_HVGS)


def test_output_finite(model):
    out = model(torch.randn(BATCH, PERT_DIM))
    assert torch.all(torch.isfinite(out))


def test_model_attributes(model):
    assert model.pert_emb_dim == PERT_DIM
    assert model.hidden_dim == HIDDEN
    assert model.num_layers == NUM_LAYERS
    assert model.n_hvgs == N_HVGS


# ========================================= compute_perturbation_embedding


def test_pert_emb_single_known_gene():
    gene_map = {"KLF1": np.ones(8, dtype=np.float32)}
    emb = compute_perturbation_embedding("KLF1", gene_map, emb_dim=8)
    np.testing.assert_array_equal(emb, np.ones(8, dtype=np.float32))


def test_pert_emb_combo_sum():
    gene_map = {
        "KLF1": np.ones(8, dtype=np.float32),
        "AHR": 2.0 * np.ones(8, dtype=np.float32),
    }
    emb = compute_perturbation_embedding("AHR_KLF1", gene_map, emb_dim=8)
    np.testing.assert_array_equal(emb, 3.0 * np.ones(8, dtype=np.float32))


def test_pert_emb_unknown_gene_fallback():
    gene_map = {"KLF1": np.ones(8, dtype=np.float32)}
    emb = compute_perturbation_embedding("UNKNOWN_GENE", gene_map, emb_dim=8)
    np.testing.assert_array_equal(emb, np.zeros(8, dtype=np.float32))


def test_pert_emb_control_returns_zero():
    gene_map = {"KLF1": np.ones(8, dtype=np.float32)}
    emb = compute_perturbation_embedding("control", gene_map, emb_dim=8)
    np.testing.assert_array_equal(emb, np.zeros(8, dtype=np.float32))


def test_pert_emb_partial_combo():
    """One known + one unknown gene: only the known gene contributes."""
    gene_map = {"KLF1": np.ones(8, dtype=np.float32)}
    emb = compute_perturbation_embedding("KLF1_MISSING", gene_map, emb_dim=8)
    np.testing.assert_array_equal(emb, np.ones(8, dtype=np.float32))
