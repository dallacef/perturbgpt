"""Shape and forward-pass sanity tests for the gene-embedding MLP head.

Tests cover:
- GeneMLP forward-pass shape checks on dummy batches
- Conditioning regression test: changing the perturbation (gene) embedding
  *must* change the output (guards against the embedding being ignored)
- GeneMLP attribute and num_layers clamping checks
- compute_perturbation_embedding for single, combo, and unknown genes
- gaussian_nll_loss: reference value, log-variance clamping, finite gradients
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from perturbgpt.models.prediction_head import (
    LOG_VAR_MAX,
    LOG_VAR_MIN,
    FiLMPredictionHead,
    GeneMLP,
    compute_perturbation_embedding,
    gaussian_nll_loss,
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


# =========================================================== gaussian NLL


def test_gaussian_nll_reference_value():
    """With log_var == 0 the variance is 1, so the loss is 0.5 * MSE."""
    mu = torch.zeros(3)
    target = torch.tensor([1.0, -1.0, 0.5])
    expected = 0.5 * (target**2).mean()
    assert torch.isclose(gaussian_nll_loss(mu, torch.zeros(3), target), expected)


def test_gaussian_nll_clamps_low_log_var():
    """A collapsing variance must not blow the loss up to inf (the bug)."""
    loss = gaussian_nll_loss(
        torch.zeros(2), torch.full((2,), -50.0), torch.ones(2)
    )
    expected = 0.5 * (LOG_VAR_MIN + 1.0 / math.exp(LOG_VAR_MIN))
    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-6)


def test_gaussian_nll_clamps_high_log_var():
    loss = gaussian_nll_loss(
        torch.zeros(2), torch.full((2,), 50.0), torch.ones(2)
    )
    expected = 0.5 * (LOG_VAR_MAX + 1.0 / math.exp(LOG_VAR_MAX))
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-6)


def test_gaussian_nll_gradients_finite_under_extreme_log_var():
    """Gradients must stay usable when the head predicts an absurd variance."""
    mu = torch.zeros(2, requires_grad=True)
    log_var = torch.full((2,), -50.0, requires_grad=True)
    gaussian_nll_loss(mu, log_var, torch.ones(2)).backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(log_var.grad).all()
    # Outside the clamp window the log-variance receives no gradient at all,
    # which is what stops the head from running away.
    assert torch.allclose(log_var.grad, torch.zeros(2))


def test_gaussian_nll_log_var_gradient_inside_clamp_window():
    """Inside the window d(loss)/d(log_var) == 0.5 (plus the residual term)."""
    mu = torch.zeros(1)
    log_var = torch.zeros(1, requires_grad=True)
    gaussian_nll_loss(mu, log_var, torch.ones(1)).backward()
    # d/dlog_var of 0.5*(log_var + r^2*exp(-log_var)) at log_var=0, r=1 is
    # 0.5*(1 - 1) == 0.
    assert torch.allclose(log_var.grad, torch.zeros(1))
    log_var2 = torch.full((1,), 2.0, requires_grad=True)
    gaussian_nll_loss(mu, log_var2, torch.ones(1)).backward()
    expected = 0.5 * (1.0 - 1.0 / math.exp(2.0))
    assert torch.isclose(log_var2.grad, torch.tensor([expected]))


# ==================================================== uncertainty head plumbing


def test_log_variance_head_does_not_backprop_into_trunk():
    """The auxiliary variance head must not perturb the mu/trunk features."""
    torch.manual_seed(0)
    model = FiLMPredictionHead(
        pert_emb_dim=PERT_DIM,
        control_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN,
        num_layers=NUM_LAYERS,
        n_hvgs=N_HVGS,
        dropout=0.0,
    )
    ctrl = torch.randn(BATCH, PERT_DIM)
    pert = torch.randn(BATCH, PERT_DIM)
    _mu, log_var = model(ctrl, pert)
    log_var.sum().backward()

    assert model.logvar_head.weight.grad is not None
    assert model.mu_head.weight.grad is None
    assert all(p.grad is None for p in model.film_generator.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


def test_mu_gradient_reaches_trunk_through_mu_head():
    """Sanity: the mu path still trains the trunk."""
    torch.manual_seed(0)
    model = FiLMPredictionHead(
        pert_emb_dim=PERT_DIM,
        control_emb_dim=PERT_DIM,
        hidden_dim=HIDDEN,
        num_layers=NUM_LAYERS,
        n_hvgs=N_HVGS,
        dropout=0.0,
    )
    ctrl = torch.randn(BATCH, PERT_DIM)
    pert = torch.randn(BATCH, PERT_DIM)
    mu, _log_var = model(ctrl, pert)
    mu.sum().backward()
    assert model.mu_head.weight.grad is not None
    assert model.logvar_head.weight.grad is None
    assert any(p.grad is not None for p in model.film_generator.parameters())
