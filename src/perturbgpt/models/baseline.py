"""PCA + MLP baseline model for perturbation-response prediction.

PCA is fit only on train-split control cells. The MLP takes
``[PCA(control_mean); perturbation_embedding]`` and predicts the pseudobulk
expression delta Δx = mean(perturbed) - mean(control) for that perturbation,
restricted to the highly-variable genes selected during preprocessing.

A learned ``nn.Embedding`` maps perturbation identities to dense vectors;
index 0 is reserved as the "unknown" fallback for perturbations not seen
during training (val/test in the perturbation-level split).
"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from perturbgpt.data.splitting import CONTROL_LABEL, PerturbationSplit

UNKNOWN_IDX = 0  # reserved embedding index for unseen perturbations


class BaselineModel(nn.Module):
    """PCA features + perturbation embedding → MLP → predicted Δx.

    Parameters
    ----------
    pca_dim : int
        Dimensionality of the PCA-reduced control expression.
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
        pca_dim: int,
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
        in_dim = pca_dim + embed_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_hvgs))
        self.mlp = nn.Sequential(*layers)
        self.pca_dim = pca_dim
        self.n_hvgs = n_hvgs

    def forward(self, pca_features: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pca_features : Tensor [batch, pca_dim]
            PCA-reduced control expression (same vector for every perturbation,
            or per-cell if using augmentation).
        pert_idx : Tensor [batch] (long)
            Perturbation embedding indices; 0 = unknown.

        Returns
        -------
        Tensor [batch, n_hvgs]
            Predicted pseudobulk expression delta Δx.
        """
        emb = self.embedding(pert_idx)
        x = torch.cat([pca_features, emb], dim=-1)
        return self.mlp(x)


# ------------------------------------------------------------- data utilities


def fit_pca_on_control(
    adata: ad.AnnData,
    split: PerturbationSplit,
    variance: Optional[float] = 0.95,
    n_components: Optional[int] = None,
) -> PCA:
    """Fit PCA on train-split control cells only.

    Parameters
    ----------
    adata : AnnData
        Processed dataset (log-normalized, HVG-subsetted).
    split : PerturbationSplit
        The perturbation-level split (control cells are in ``split.train``).
    variance : float, optional
        Fraction of variance to explain (e.g. 0.95). If provided, the number
        of components is automatically determined. Default 0.95.
    n_components : int, optional
        Fixed number of components. Used only when ``variance`` is None.

    Returns
    -------
    PCA
        Fitted sklearn PCA object.
    """
    perts = adata.obs["perturbation"].astype(str)
    control_mask = (perts == CONTROL_LABEL) & adata.obs_names.isin(set(split.train))
    X = adata[control_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    if variance is not None:
        pca = PCA(n_components=variance, svd_solver="full")
    else:
        pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca



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


def transform_pca(pca: PCA, adata: ad.AnnData, cell_names: list[str]) -> np.ndarray:
    """Apply a fitted PCA to specific cells, returning the PCA features.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object from :func:`fit_pca_on_control`.
    adata : AnnData
        Processed dataset.
    cell_names : list[str]
        Cell barcodes to transform.

    Returns
    -------
    np.ndarray [n_cells, pca_dim]
    """
    mask = adata.obs_names.isin(set(cell_names))
    X = adata[mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pca.transform(np.asarray(X, dtype=np.float64))

