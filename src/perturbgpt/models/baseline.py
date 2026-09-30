"""Perturbation-embedding baseline model for perturbation-response prediction.

The MLP takes a learned perturbation embedding and predicts the pseudobulk
expression delta Δx = mean(perturbed) - mean(control) for that perturbation,
restricted to the highly-variable genes selected during preprocessing.

A learned ``nn.Embedding`` maps perturbation identities to dense vectors;
index 0 is reserved as the "unknown" fallback for perturbations not seen
during training (val/test in the perturbation-level split).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import torch
import torch.nn as nn

from perturbgpt.data.splitting import CONTROL_LABEL, PerturbationSplit

UNKNOWN_IDX = 0  # reserved embedding index for unseen perturbations


class BaselineModel(nn.Module):
    """Perturbation embedding → MLP → predicted Δx.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the learned perturbation embedding.
    hidden_dim : int
        Width of MLP hidden layers.
    num_layers : int
        Number of hidden layers (minimum 1).
    n_hvgs : int
        Number of output genes (highly-variable genes).
    num_perts : int
        Number of known perturbation identities (excluding unknown).
    dropout : float
        Dropout probability between hidden layers.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_hvgs: int,
        num_perts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # +1 for the unknown index (0); known perts occupy 1..num_perts
        self.embedding = nn.Embedding(num_perts + 1, embed_dim, padding_idx=None)
        nn.init.normal_(self.embedding.weight, std=0.02)
        # Zero-init the unknown row so unseen perts produce a neutral embedding
        with torch.no_grad():
            self.embedding.weight[UNKNOWN_IDX].zero_()

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_hvgs))
        self.mlp = nn.Sequential(*layers)
        self.n_hvgs = n_hvgs

    def forward(self, pert_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pert_idx : Tensor [batch] (long)
            Perturbation embedding indices; 0 = unknown.

        Returns
        -------
        Tensor [batch, n_hvgs]
            Predicted pseudobulk expression delta Δx.
        """
        emb = self.embedding(pert_idx)
        return self.mlp(emb)


# ------------------------------------------------------------- data utilities


def compute_pseudobulk_deltas(
    adata: ad.AnnData,
    split: PerturbationSplit,
    perturbation_col: str = "perturbation",
) -> dict[str, np.ndarray]:
    """Compute Δx = mean(perturbed) - mean(control) for each perturbation.

    Only perturbations that appear in the given split's cell list are
    included. Control mean is computed from train-split control cells.

    Returns
    -------
    dict
        Mapping perturbation label → Δx vector of shape ``[n_hvgs]``.
    """
    perts = adata.obs[perturbation_col].astype(str)
    train_cells = set(split.train)

    # Control mean from train-split control cells
    ctrl_mask = (perts == CONTROL_LABEL) & adata.obs_names.isin(train_cells)
    X_ctrl = adata[ctrl_mask].X
    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
    ctrl_mean = np.asarray(X_ctrl, dtype=np.float64).mean(axis=0)

    deltas = {}
    split_cells_by_part = {
        "train": set(split.train),
        "val": set(split.val),
        "test": set(split.test),
    }
    all_perts = set()
    for cells in split_cells_by_part.values():
        mask = adata.obs_names.isin(cells)
        all_perts |= set(perts[mask].unique()) - {CONTROL_LABEL}

    for pert in sorted(all_perts):
        # Find which split this perturbation belongs to
        for part_name, cells in split_cells_by_part.items():
            mask = (perts == pert) & adata.obs_names.isin(cells)
            if mask.any():
                X_p = adata[mask].X
                if hasattr(X_p, "toarray"):
                    X_p = X_p.toarray()
                pert_mean = np.asarray(X_p, dtype=np.float64).mean(axis=0)
                deltas[pert] = pert_mean - ctrl_mean
                break
    return deltas


def build_pert_to_idx(train_perts: set[str]) -> dict[str, int]:
    """Map perturbation labels to embedding indices.

    Index 0 is reserved for ``UNKNOWN_IDX`` (unseen perturbations).
    Known perturbations get indices 1..N.

    Returns
    -------
    dict
        Mapping perturbation label → integer index. Unseen perturbations
        should be looked up with ``.get(label, UNKNOWN_IDX)``.
    """
    return {pert: i + 1 for i, pert in enumerate(sorted(train_perts))}


def compute_control_statistics(
    adata: ad.AnnData,
    split: PerturbationSplit,
    perturbation_col: str = "perturbation",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-gene mean and std from train-split control cells.

    The control distribution defines the "unperturbed" reference, so its
    per-gene standard deviation is a natural unit for rescaling perturbation
    effects. Genes with zero control variance are clamped to a std of 1.0 so
    that standardization never divides by zero.

    Parameters
    ----------
    adata : AnnData
        Processed dataset.
    split : PerturbationSplit
        Perturbation-level split; control cells are taken from ``split.train``.
    perturbation_col : str
        Column in ``adata.obs`` holding perturbation labels.

    Returns
    -------
    (mu, sigma)
        Per-gene mean and std, both of shape ``[n_genes]``.
    """
    perts = adata.obs[perturbation_col].astype(str)
    control_mask = (perts == CONTROL_LABEL) & adata.obs_names.isin(set(split.train))
    X = adata[control_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1.0, 1.0, sigma)  # avoid division by zero
    return mu, sigma


def standardize_deltas(
    deltas: dict[str, np.ndarray],
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    """Divide each perturbation's delta by the per-gene control std.

    Parameters
    ----------
    deltas : dict
        Mapping perturbation label → Δx vector ``[n_genes]``.
    sigma : np.ndarray
        Per-gene control std ``[n_genes]`` from
        :func:`compute_control_statistics`.

    Returns
    -------
    dict
        Standardized deltas (element-wise ``delta / sigma``).
    """
    return {pert: delta / sigma for pert, delta in deltas.items()}


def unstandardize_predictions(
    preds: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Multiply standardized predictions back by the per-gene control std.

    Parameters
    ----------
    preds : np.ndarray
        Standardized predictions, shape ``[n_perts, n_genes]`` or ``[n_genes]``.
    sigma : np.ndarray
        Per-gene control std ``[n_genes]``.

    Returns
    -------
    np.ndarray
        Predictions in the original delta scale (``preds * sigma``).
    """
    return preds * sigma

