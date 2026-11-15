"""Tests for the prediction-head checkpoint contract.

Covers the train -> inference artefact boundary: what
``save_prediction_checkpoint`` writes must be loadable by
``load_prediction_checkpoint`` (both architectures) and by the consumers in
:mod:`perturbgpt.agent.tools` and ``scripts/build_retrieval_index.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from perturbgpt.agent import tools
from perturbgpt.models.prediction_head import (
    ARCH_FILM,
    ARCH_GENE_MLP,
    FiLMPredictionHead,
    GeneMLP,
    load_prediction_checkpoint,
    read_checkpoint_arch,
    save_prediction_checkpoint,
)

D_MODEL = 8
N_HVGS = 5
GENE_NAMES = [f"GENE{i}" for i in range(N_HVGS)]


def _gene_emb_map() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "KLF1": rng.standard_normal(D_MODEL).astype(np.float32),
        "AHR": rng.standard_normal(D_MODEL).astype(np.float32),
    }


def _gene_mlp() -> GeneMLP:
    torch.manual_seed(0)
    return GeneMLP(
        pert_emb_dim=D_MODEL, hidden_dim=16, num_layers=2, n_hvgs=N_HVGS, dropout=0.0
    )


def _film_head(control_dim: int = D_MODEL) -> FiLMPredictionHead:
    torch.manual_seed(0)
    return FiLMPredictionHead(
        pert_emb_dim=D_MODEL,
        control_emb_dim=control_dim,
        hidden_dim=16,
        num_layers=2,
        n_hvgs=N_HVGS,
        dropout=0.0,
        film_hidden_dim=8,
    )


# ----------------------------------------------------------- save / load


def test_gene_mlp_roundtrip_matches_direct_forward(tmp_path):
    model = _gene_mlp()
    path = save_prediction_checkpoint(
        tmp_path / "mlp.pt",
        model,
        arch=ARCH_GENE_MLP,
        model_config={"pert_emb_dim": D_MODEL, "hidden_dim": 16, "num_layers": 2},
        gene_names=GENE_NAMES,
    )
    bundle = load_prediction_checkpoint(path)

    assert bundle.arch == ARCH_GENE_MLP
    assert bundle.gene_names == GENE_NAMES
    assert bundle.n_hvgs == N_HVGS

    got = bundle.predict_delta(["KLF1", "AHR", "KLF1"], _gene_emb_map())

    # The bundle builds features from gene embeddings, so compare against the
    # same features fed to the original model.
    from perturbgpt.models.prediction_head import compute_perturbation_embedding

    feats = np.stack(
        [compute_perturbation_embedding(p, _gene_emb_map(), D_MODEL)
         for p in ("KLF1", "AHR", "KLF1")]
    )
    with torch.no_grad():
        expected = model(torch.tensor(feats)).numpy()

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_film_roundtrip_uses_stored_control_mean(tmp_path):
    model = _film_head()
    control_mean = np.linspace(-0.05, 0.05, D_MODEL).astype(np.float32)
    path = save_prediction_checkpoint(
        tmp_path / "film.pt",
        model,
        arch=ARCH_FILM,
        model_config={
            "pert_emb_dim": D_MODEL,
            "control_emb_dim": D_MODEL,
            "hidden_dim": 16,
            "num_layers": 2,
            "film_hidden_dim": 8,
        },
        gene_names=GENE_NAMES,
        control_emb_mean=control_mean,
    )
    bundle = load_prediction_checkpoint(path)

    assert bundle.arch == ARCH_FILM
    np.testing.assert_allclose(bundle.control_emb_mean, control_mean, rtol=1e-6)

    from perturbgpt.models.prediction_head import compute_perturbation_embedding

    labels = ["KLF1", "AHR"]
    feats = np.stack(
        [compute_perturbation_embedding(p, _gene_emb_map(), D_MODEL) for p in labels]
    )
    control = np.tile(control_mean, (len(labels), 1))
    with torch.no_grad():
        expected, _ = model(
            torch.tensor(control, dtype=torch.float32),
            torch.tensor(feats, dtype=torch.float32),
        )
    got = bundle.predict_delta(labels, _gene_emb_map())
    np.testing.assert_allclose(got, expected.numpy(), rtol=1e-6, atol=1e-6)


def test_save_fills_n_hvgs_from_gene_names(tmp_path):
    path = save_prediction_checkpoint(
        tmp_path / "mlp.pt",
        _gene_mlp(),
        arch=ARCH_GENE_MLP,
        model_config={"pert_emb_dim": D_MODEL, "hidden_dim": 16, "num_layers": 2},
        gene_names=GENE_NAMES,
    )
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["model_config"]["n_hvgs"] == N_HVGS
    assert ckpt["format_version"] == 1
    assert ckpt["gene_names"] == GENE_NAMES
    assert "created" in ckpt


def test_read_checkpoint_arch(tmp_path):
    path = save_prediction_checkpoint(
        tmp_path / "film.pt",
        _film_head(),
        arch=ARCH_FILM,
        model_config={"pert_emb_dim": D_MODEL, "control_emb_dim": D_MODEL,
                      "hidden_dim": 16, "num_layers": 2, "film_hidden_dim": 8},
        gene_names=GENE_NAMES,
        control_emb_mean=np.zeros(D_MODEL, dtype=np.float32),
    )
    assert read_checkpoint_arch(path) == ARCH_FILM


# ----------------------------------------------------------- validation


def test_film_checkpoint_requires_control_mean(tmp_path):
    with pytest.raises(ValueError, match="control_emb_mean"):
        save_prediction_checkpoint(
            tmp_path / "film.pt",
            _film_head(),
            arch=ARCH_FILM,
            model_config={"pert_emb_dim": D_MODEL, "control_emb_dim": D_MODEL,
                          "hidden_dim": 16, "num_layers": 2},
            gene_names=GENE_NAMES,
        )


def test_unknown_architecture_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown architecture"):
        save_prediction_checkpoint(
            tmp_path / "x.pt",
            _gene_mlp(),
            arch="transformer",
            model_config={},
            gene_names=GENE_NAMES,
        )


def test_n_hvgs_gene_name_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not match len\\(gene_names\\)"):
        save_prediction_checkpoint(
            tmp_path / "x.pt",
            _gene_mlp(),
            arch=ARCH_GENE_MLP,
            model_config={"pert_emb_dim": D_MODEL, "n_hvgs": 99},
            gene_names=GENE_NAMES,
        )


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prediction_checkpoint(tmp_path / "nope.pt")


def test_non_checkpoint_file_raises(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"encoder.weight": torch.zeros(2, 2)}, path)
    with pytest.raises(ValueError, match="not a prediction-head checkpoint"):
        load_prediction_checkpoint(path)


def test_legacy_checkpoint_defaults_to_gene_mlp(tmp_path):
    """Pre-`arch` checkpoints (model_config/model_state_dict/gene_names) load."""
    model = _gene_mlp()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_config": {
                "pert_emb_dim": D_MODEL,
                "hidden_dim": 16,
                "num_layers": 2,
                "n_hvgs": N_HVGS,
            },
            "model_state_dict": model.state_dict(),
            "gene_names": GENE_NAMES,
        },
        path,
    )
    bundle = load_prediction_checkpoint(path)
    assert bundle.arch == ARCH_GENE_MLP
    assert read_checkpoint_arch(path) == ARCH_GENE_MLP


# ----------------------------------------------------------- tool integration


@pytest.fixture()
def patched_tools(tmp_path, monkeypatch):
    """Point the agent tools at a temporary checkpoint + gene embeddings."""
    ckpt_path = save_prediction_checkpoint(
        tmp_path / "film_head.pt",
        _film_head(),
        arch=ARCH_FILM,
        model_config={"pert_emb_dim": D_MODEL, "control_emb_dim": D_MODEL,
                      "hidden_dim": 16, "num_layers": 2, "film_hidden_dim": 8},
        gene_names=GENE_NAMES,
        control_emb_mean=np.zeros(D_MODEL, dtype=np.float32),
    )
    emb_map = _gene_emb_map()
    emb_path = tmp_path / "gene_embeddings.npz"
    np.savez(
        emb_path,
        embeddings=np.stack(list(emb_map.values())),
        gene_symbols=np.array(list(emb_map.keys())),
    )
    monkeypatch.setattr(tools, "DEFAULT_CHECKPOINT", ckpt_path)
    monkeypatch.setattr(tools, "DEFAULT_GENE_EMBEDDINGS", emb_path)
    tools.reset_caches()
    yield tools
    tools.reset_caches()


def test_get_predicted_genes_uses_film_checkpoint(patched_tools):
    out = patched_tools.get_predicted_genes("KLF1", top_k=3)
    assert len(out) == 3
    assert {r["gene"] for r in out} <= set(GENE_NAMES)
    deltas = [abs(r["delta"]) for r in out]
    assert deltas == sorted(deltas, reverse=True)
    assert all(np.isfinite(r["delta"]) for r in out)


def test_get_predicted_genes_without_checkpoint_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DEFAULT_CHECKPOINT", tmp_path / "missing.pt")
    tools.reset_caches()
    try:
        assert tools.get_predicted_genes("KLF1") == []
    finally:
        tools.reset_caches()


def test_get_predicted_genes_unknown_perturbation_is_zero_fallback(patched_tools):
    out = patched_tools.get_predicted_genes("NOT_A_GENE", top_k=2)
    # compute_perturbation_embedding returns a zero vector for unknown genes,
    # so the model still runs and returns finite (near-constant) deltas.
    assert len(out) == 2
    assert all(np.isfinite(r["delta"]) for r in out)
