"""Perturbation (gene-embedding) MLP prediction head.

Maps a perturbation's gene embedding (summed scGPT gene embeddings, for
single-gene or combinatorial perturbations) through an MLP to predict the
pseudobulk expression delta Δx over the same HVG gene set used by the
baseline. There is no cell-state input — the model predicts perturbation
effects from the perturbation identity alone, so it can generalise to
unseen perturbations via gene-level semantics.

Architecture
------------
::

    pert_emb -> Linear -> ReLU -> Dropout -> ...  (num_layers blocks)
             -> Linear -> delta_x_hat
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from perturbgpt.data.splitting import parse_perturbation


# --------------------------------------------------------- prediction head


class GeneMLP(nn.Module):
    """MLP prediction head over perturbation gene embeddings.

    Parameters
    ----------
    pert_emb_dim : int
        Dimensionality of the perturbation (gene) embedding.
    hidden_dim : int
        Width of the MLP hidden layers.
    num_layers : int
        Number of hidden layers (minimum 1).
    n_hvgs : int
        Number of output genes (highly-variable genes).
    dropout : float
        Dropout probability between hidden layers.
    """

    def __init__(
        self,
        pert_emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_hvgs: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = pert_emb_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_hvgs))
        self.mlp = nn.Sequential(*layers)
        self.pert_emb_dim = pert_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = max(num_layers, 1)
        self.n_hvgs = n_hvgs

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pert_emb : Tensor [batch, pert_emb_dim]
            Perturbation gene embedding (frozen, cached).

        Returns
        -------
        Tensor [batch, n_hvgs]
            Predicted expression delta.
        """
        return self.mlp(pert_emb)


# --------------------------------------------------------- helpers


def compute_perturbation_embedding(
    pert_label: str,
    gene_emb_map: dict[str, np.ndarray],
    emb_dim: int,
) -> np.ndarray:
    """Compute a perturbation embedding by summing constituent gene embeddings.

    For single-gene perturbations the gene embedding is returned directly.
    For combinatorial perturbations the constituent gene embeddings are
    summed, matching ``ScGPTWrapper.get_combination_embedding``
    with ``method="sum"``.

    Genes not found in *gene_emb_map* contribute a zero vector.  If **no**
    constituent gene is found, the all-zero fallback is returned (analogous
    to the baseline UNKNOWN_IDX zero-initialised embedding row).

    Parameters
    ----------
    pert_label : str
        Perturbation label, e.g. "KLF1" or "AHR_KLF1".
    gene_emb_map : dict
        Mapping gene symbol to embedding vector (np.ndarray [emb_dim]).
    emb_dim : int
        Expected embedding dimensionality.

    Returns
    -------
    np.ndarray [emb_dim]
    """
    genes = parse_perturbation(pert_label)
    if not genes:
        return np.zeros(emb_dim, dtype=np.float32)

    emb = np.zeros(emb_dim, dtype=np.float32)
    for g in genes:
        if g in gene_emb_map:
            emb = emb + gene_emb_map[g]
    return emb
