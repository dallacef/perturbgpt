"""Shape and forward-pass sanity tests for the FiLM-MLP prediction head.

Tests cover:
- FiLMLayer shape checks on dummy batches
- FiLM conditioning regression test: changing the perturbation embedding
  *must* change the output for a fixed cell embedding (guards against the
  FiLM conditioning silently becoming a no-op)
- FiLMLayer identity-at-initialisation behaviour
- FiLMMLP forward-pass shape and conditioning tests
- compute_perturbation_embedding for single, combo, and unknown genes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from perturbgpt.models.prediction_head import (
    FiLMLayer,
    FiLMMLP,
    compute_perturbation_embedding,
)

# ============================================================== constants

CELL_DIM = 16
PERT_DIM = 16
HIDDEN = 32
N_HVGS = 20
NUM_LAYERS = 3
BATCH = 7


# ============================================================ FiLMLayer tests


@pytest.fixture
def film_layer():
    return FiLMLayer(hidden_dim=HIDDEN, cond_dim=PERT_DIM)


def test_film_forward_shape(film_layer):
    x = torch.randn(BATCH, HIDDEN)
    c = torch.randn(BATCH, PERT_DIM)
    out = film_layer(x, c)
    assert out.shape == (BATCH, HIDDEN)


def test_film_single_batch():
    layer = FiLMLayer(hidden_dim=HIDDEN, cond_dim=PERT_DIM)
    x = torch.randn(1, HIDDEN)
    c = torch.randn(1, PERT_DIM)
    out = layer(x, c)
    assert out.shape == (1, HIDDEN)


def test_film_identity_at_init():
    """At initialisation scale=1 and shift=0, so output equals input."""
    layer = FiLMLayer(hidden_dim=HIDDEN, cond_dim=PERT_DIM)
    x = torch.randn(BATCH, HIDDEN)
    c = torch.randn(BATCH, PERT_DIM)
    out = layer(x, c)
    assert torch.allclose(out, x, atol=1e-6)


def test_film_conditioning_changes_output(film_layer):
    """Changing the perturbation embedding MUST change the FiLM output for
    a fixed hidden state.  This is the primary regression test against the
    FiLM conditioning silently becoming a no-op."""
    film_layer.eval()
    x = torch.ones(1, HIDDEN)
    c1 = torch.randn(1, PERT_DIM)
    c2 = torch.randn(1, PERT_DIM)

    # Force non-zero weights so modulation is active.
    with torch.no_grad():
        film_layer.scale_proj.weight.normal_(std=0.1)
        film_layer.shift_proj.weight.normal_(std=0.1)

    with torch.no_grad():
        out1 = film_layer(x, c1)
        out2 = film_layer(x, c2)

    assert not torch.allclose(out1, out2, atol=1e-8), (
        "FiLM conditioning is a no-op: changing the perturbation embedding "
        "did not change the output for a fixed cell embedding."
    )


def test_film_different_conditions_different_outputs_batch():
    """Each row in a batch with a different condition should produce a
    different output from the same hidden state."""
    layer = FiLMLayer(hidden_dim=HIDDEN, cond_dim=PERT_DIM)
    with torch.no_grad():
        layer.scale_proj.weight.normal_(std=0.1)
        layer.shift_proj.weight.normal_(std=0.1)

    x = torch.ones(BATCH, HIDDEN)
    c = torch.randn(BATCH, PERT_DIM)

    with torch.no_grad():
        out = layer(x, c)

    # All rows should differ from each other (probabilistically near-certain).
    for i in range(BATCH):
        for j in range(i + 1, BATCH):
            assert not torch.allclose(out[i], out[j], atol=1e-8), (
                f"rows {i} and {j} have identical FiLM output despite "
                f"different conditions"
            )


def test_film_output_finite(film_layer):
    x = torch.randn(BATCH, HIDDEN)
    c = torch.randn(BATCH, PERT_DIM)
    out = film_layer(x, c)
    assert torch.all(torch.isfinite(out))


# ============================================================ FiLMMLP tests


@pytest.fixture
def model():
    return FiLMMLP(
        cell_emb_dim=CELL_DIM,
        pert_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN,
        num_layers=NUM_LAYERS,
        n_hvgs=N_HVGS,
        dropout=0.0,
    )


def test_model_forward_shape(model):
    cell = torch.randn(BATCH, CELL_DIM)
    pert = torch.randn(BATCH, PERT_DIM)
    out = model(cell, pert)
    assert out.shape == (BATCH, N_HVGS)


def test_model_single_batch():
    m = FiLMMLP(
        cell_emb_dim=CELL_DIM, pert_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN, num_layers=2, n_hvgs=N_HVGS, dropout=0.0,
    )
    cell = torch.randn(1, CELL_DIM)
    pert = torch.randn(1, PERT_DIM)
    out = m(cell, pert)
    assert out.shape == (1, N_HVGS)


def test_model_output_finite(model):
    cell = torch.randn(BATCH, CELL_DIM)
    pert = torch.randn(BATCH, PERT_DIM)
    out = model(cell, pert)
    assert torch.all(torch.isfinite(out))


def test_model_conditioning_changes_output():
    """Regression test: for a fixed cell embedding, different perturbation
    embeddings must produce different predicted deltas.  Guards against
    the FiLM conditioning becoming a no-op anywhere in the full model.

    At initialisation the FiLM layers are identity (gamma=1, beta=0), so we
    randomise the FiLM projection weights to confirm the conditioning path
    is actually wired through the network.
    """
    m = FiLMMLP(
        cell_emb_dim=CELL_DIM, pert_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN, num_layers=NUM_LAYERS,
        n_hvgs=N_HVGS, dropout=0.0,
    )
    m.eval()

    # Randomise FiLM projections so modulation is active.
    with torch.no_grad():
        for film in m.film_layers:
            film.scale_proj.weight.normal_(std=0.1)
            film.shift_proj.weight.normal_(std=0.1)

    cell = torch.randn(1, CELL_DIM)
    pert1 = torch.randn(1, PERT_DIM)
    pert2 = torch.randn(1, PERT_DIM)

    with torch.no_grad():
        out1 = m(cell, pert1)
        out2 = m(cell, pert2)

    assert not torch.allclose(out1, out2, atol=1e-8), (
        "FiLM-MLP conditioning is a no-op: different perturbation embeddings "
        "produced identical predicted deltas for the same cell embedding."
    )


def test_model_different_cells_different_outputs():
    """Different cell embeddings with the same perturbation should produce
    different outputs (the cell embedding is not ignored)."""
    m = FiLMMLP(
        cell_emb_dim=CELL_DIM, pert_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN, num_layers=NUM_LAYERS,
        n_hvgs=N_HVGS, dropout=0.0,
    )
    m.eval()
    cell1 = torch.randn(1, CELL_DIM)
    cell2 = torch.randn(1, CELL_DIM)
    pert = torch.randn(1, PERT_DIM)

    with torch.no_grad():
        out1 = m(cell1, pert)
        out2 = m(cell2, pert)

    assert not torch.allclose(out1, out2, atol=1e-8)


def test_model_num_layers_minimum():
    """num_layers=0 should be clamped to 1."""
    m = FiLMMLP(
        cell_emb_dim=CELL_DIM, pert_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN, num_layers=0, n_hvgs=N_HVGS, dropout=0.0,
    )
    assert m.num_layers == 1
    cell = torch.randn(1, CELL_DIM)
    pert = torch.randn(1, PERT_DIM)
    out = m(cell, pert)
    assert out.shape == (1, N_HVGS)


def test_model_attributes(model):
    assert model.cell_emb_dim == CELL_DIM
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

