"""Prediction heads for perturbation-response modelling.

Two prediction heads are provided:

* ``GeneMLP`` — a simple MLP that maps a perturbation's gene embedding
  (summed scGPT gene embeddings, for single-gene or combinatorial
  perturbations) to a predicted pseudobulk expression delta Δx over the HVG
  gene set. There is no cell-state input, so it generalises to unseen
  perturbations via gene-level semantics.

* ``FiLMPredictionHead`` — a FiLM (Feature-wise Linear Modulation) head that
  takes a *control* (baseline) embedding and a *perturbation* embedding. The
  perturbation embedding conditions a backbone network driven by the control
  embedding, and the head outputs both a mean ``mu`` and a ``log_variance``.
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
        


# --------------------------------------------------------- FiLM prediction head


class FiLMPredictionHead(nn.Module):
    """FiLM-conditioned prediction head.

    Takes a control (baseline) embedding and a perturbation embedding. The
    perturbation embedding conditions the model via Feature-wise Linear
    Modulation (FiLM): a small generator network maps the perturbation
    embedding to per-layer affine parameters ``(gamma, beta)``, which modulate
    the activations of a backbone network driven by the control embedding.

    The backbone's final hidden representation is decoded by two linear heads
    into:

    * ``mu``           — predicted pseudobulk expression delta (mean), and
    * ``log_variance`` — predicted log-variance of that delta (for
      uncertainty-aware losses).

    Architecture
    ------------
    ::

        pert_emb -> film_generator -> (gamma_i, beta_i) for each layer
        control_emb -> [Linear -> LayerNorm -> FiLM -> ReLU -> Dropout] x L
                    -> mu_head     -> mu            [batch, n_hvgs]
                    -> logvar_head -> log_variance  [batch, n_hvgs]

    Parameters
    ----------
    pert_emb_dim : int
        Dimensionality of the perturbation (gene) embedding used for
        conditioning.
    control_emb_dim : int
        Dimensionality of the control (baseline) embedding.
    hidden_dim : int
        Width of the backbone hidden layers.
    num_layers : int
        Number of FiLM-conditioned backbone layers (minimum 1).
    n_hvgs : int
        Number of output genes (highly-variable genes).
    dropout : float
        Dropout probability applied after each backbone activation.
    film_hidden_dim : int, optional
        Hidden width of the FiLM generator MLP. Defaults to ``hidden_dim``.
    """

    def __init__(
        self,
        pert_emb_dim: int,
        control_emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_hvgs: int,
        dropout: float = 0.1,
        film_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_layers = max(num_layers, 1)
        film_hidden_dim = film_hidden_dim or hidden_dim

        # Backbone: per-layer Linear + LayerNorm. ReLU, FiLM modulation and
        # dropout are applied in ``forward`` so the modulation parameters can
        # be injected between the norm and the activation.
        self.backbone = nn.ModuleList()
        in_dim = control_emb_dim
        for _ in range(self.num_layers):
            self.backbone.append(
                nn.ModuleDict(
                    {
                        "linear": nn.Linear(in_dim, hidden_dim),
                        # "norm": nn.LayerNorm(hidden_dim),
                    }
                )
            )
            in_dim = hidden_dim

        # FiLM generator: pert_emb -> (gamma, beta) for every backbone layer.
        # The output is ``num_layers`` chunks of ``2 * hidden_dim``; each chunk
        # splits into gamma (first half) and beta (second half).
        self.film_generator = nn.Sequential(
            nn.Linear(pert_emb_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, film_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(film_hidden_dim//2, 2 * hidden_dim * self.num_layers),
        )

        # Output heads.
        self.mu_head = nn.Linear(hidden_dim, n_hvgs)
        self.logvar_head = nn.Linear(hidden_dim, n_hvgs)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.pert_emb_dim = pert_emb_dim
        self.control_emb_dim = control_emb_dim
        self.hidden_dim = hidden_dim
        self.n_hvgs = n_hvgs

        self._init_film_generator()

    def _init_film_generator(self) -> None:
        """Initialise the FiLM generator to start as identity modulation.

        Zero the final generator layer's weights and set its bias so that every
        ``gamma`` starts at 1 and every ``beta`` starts at 0. The backbone
        therefore begins as a plain (unmodulated) network, which stabilises
        early training before the perturbation signal is learned.
        """
        last: nn.Linear = self.film_generator[-1]
        nn.init.zeros_(last.weight)
        # Bias layout: (num_layers, 2, hidden_dim) -> [gamma, beta] per layer.
        bias = torch.zeros(self.num_layers, 2, self.hidden_dim)
        bias[:, 0, :] = 1.0  # gamma -> identity scale (beta stays 0)
        with torch.no_grad():
            last.bias.copy_(bias.reshape(-1))

    def forward(
        self,
        control_emb: torch.Tensor,
        pert_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        control_emb : Tensor [batch, control_emb_dim]
            Control (baseline) cell-state embedding.
        pert_emb : Tensor [batch, pert_emb_dim]
            Perturbation gene embedding used to condition the backbone.

        Returns
        -------
        (mu, log_variance)
            mu : Tensor [batch, n_hvgs]
                Predicted mean expression delta.
            log_variance : Tensor [batch, n_hvgs]
                Predicted log-variance of the expression delta.
        """
        film_params = self.film_generator(pert_emb)  # [B, 2 * hidden_dim * L]
        chunks = torch.chunk(film_params, self.num_layers, dim=-1)

        h = control_emb
        for block, chunk in zip(self.backbone, chunks):
            gamma, beta = torch.chunk(chunk, 2, dim=-1)
            h = block["linear"](h)
            # h = block["norm"](h)
            h = gamma * h + beta
            h = torch.relu(h)
            h = self.dropout(h)

        mu = self.mu_head(h)
        log_variance = self.logvar_head(h)
        return mu, log_variance





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
