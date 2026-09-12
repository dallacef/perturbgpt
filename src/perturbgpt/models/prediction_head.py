"""FiLM-conditioned MLP prediction head.

Takes a cached scGPT control-cell embedding (the *carrier* signal) and a
cached scGPT perturbation (gene) embedding (the *condition*).  The
perturbation embedding is projected into per-layer scale (gamma) and
shift (beta) parameters that modulate an MLP processing the cell
embedding via Feature-wise Linear Modulation (FiLM; Perez et al., 2018).

The output is a predicted expression delta over the same gene set
(highly-variable genes) used by the baseline.

Architecture
------------
::

    cell_emb -> Linear(cell_emb_dim, hidden) -> h0
    pert_emb -> FiLM(gamma_1, beta_1) -> ReLU(h0 * gamma_1 + beta_1) -> Dropout -> h1
             -> FiLM(gamma_2, beta_2) -> ReLU(h1 * gamma_2 + beta_2) -> Dropout -> h2
             ...  (num_layers FiLM-conditioned blocks)
             -> Linear(hidden, n_hvgs) -> delta_x_hat

Each FiLM layer initialises its scale projection to zero so that
gamma = 1 + 0 = 1 at the start of training, making the modulation an
identity at initialisation and ensuring stable early gradients.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from perturbgpt.data.splitting import parse_perturbation


# --------------------------------------------------------- FiLM primitives


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Given a hidden state *x* of dimension *hidden_dim* and a condition
    vector *c* of dimension *cond_dim*, computes::

        gamma = 1 + W_gamma @ c   (scale, initialised to identity)
        beta  = W_beta  @ c      (shift,  initialised to zero)
        out   = gamma * x + beta

    Weight-zero initialisation of the scale projection ensures the layer
    is the identity at the start of training, preventing early FiLM
    modulation from destabilising the carrier signal.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        # Zero-init weights so gamma=1 and beta=0 at initialisation.
        self.scale_proj = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

        self.shift_proj = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        x : Tensor [batch, hidden_dim]
            Hidden state to modulate.
        condition : Tensor [batch, cond_dim]
            Conditioning vector (perturbation embedding).

        Returns
        -------
        Tensor [batch, hidden_dim]
        """
        gamma = 1.0 + self.scale_proj(condition)
        beta = self.shift_proj(condition)
        return gamma * x + beta


# --------------------------------------------------------- Full prediction head


class FiLMMLP(nn.Module):
    """FiLM-conditioned MLP prediction head.

    Parameters
    ----------
    cell_emb_dim : int
        Dimensionality of the cached scGPT cell embedding (d_model).
    pert_emb_dim : int
        Dimensionality of the cached scGPT perturbation (gene) embedding.
    hidden_dim : int
        Width of the MLP hidden layers.
    num_layers : int
        Number of FiLM-conditioned hidden layers (minimum 1).
    n_hvgs : int
        Number of output genes (highly-variable genes).
    dropout : float
        Dropout probability between FiLM-conditioned layers.
    """

    def __init__(
        self,
        cell_emb_dim: int,
        pert_emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_hvgs: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_layers = max(num_layers, 1)

        self.cell_proj = nn.Linear(cell_emb_dim, hidden_dim)

        self.film_layers = nn.ModuleList(
            [FiLMLayer(hidden_dim, pert_emb_dim) for _ in range(num_layers)]
        )
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, n_hvgs)

        self.cell_emb_dim = cell_emb_dim
        self.pert_emb_dim = pert_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_hvgs = n_hvgs

    def forward(
        self, cell_emb: torch.Tensor, pert_emb: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        cell_emb : Tensor [batch, cell_emb_dim]
            scGPT cell embedding (frozen, cached).
        pert_emb : Tensor [batch, pert_emb_dim]
            scGPT perturbation gene embedding (frozen, cached).

        Returns
        -------
        Tensor [batch, n_hvgs]
            Predicted expression delta.
        """
        h = self.cell_proj(cell_emb)
        for linear, film in zip(self.linears, self.film_layers):
            h = linear(h)
            h = film(h, pert_emb)
            h = torch.relu(h)
            h = self.dropout(h)
        return self.output_proj(h)


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
