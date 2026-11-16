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

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

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
        # The uncertainty head is auxiliary and unused at inference time
        # (``predict_delta`` returns ``mu``).  It is trained on *detached* trunk
        # features so its likelihood objective can never perturb the features
        # ``mu`` depends on — otherwise the variance term destabilises the mean
        # head (val MSE rose from 6.9 to 8.2 while train loss kept falling).
        log_variance = self.logvar_head(h.detach())
        return mu, log_variance





# --------------------------------------------------------- helpers


# Bounds on the predicted log-variance used by :func:`gaussian_nll_loss`.
# Without a floor, the NLL chases zero variance (log_var -> -inf) as soon as a
# single gene is badly under-predicted; ``exp(log_var)`` in the denominator then
# makes the loss explode and training stalls.  Observed on the Norman split:
# val NLL went -0.58 -> 0.12 -> 2.95 -> 196 -> ... -> 210,379 over 20 epochs.
LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood of ``target`` under N(mu, exp(log_var)).

    The constant ``0.5 * log(2 * pi)`` term is dropped because it does not
    affect the gradient.  ``log_var`` is clamped to
    ``[LOG_VAR_MIN, LOG_VAR_MAX]`` to keep the loss finite; the returned value
    is the mean over all elements, so it is comparable across batch sizes.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean, shape ``[..., n_genes]``.
    log_var : torch.Tensor
        Predicted log-variance, same shape as ``mu``.
    target : torch.Tensor
        Observed value, broadcastable to ``mu``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
    var = log_var.exp()
    return 0.5 * (log_var + (target - mu).pow(2) / var).mean()


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

# --------------------------------------------------------- checkpoint I/O

#: Current on-disk checkpoint schema version (see :func:`save_prediction_checkpoint`).
CHECKPOINT_FORMAT_VERSION = 1

#: Architecture tags understood by :func:`load_prediction_checkpoint`.
ARCH_FILM = "film"
ARCH_GENE_MLP = "gene_mlp"


@dataclass
class PredictionBundle:
    """A loaded prediction head plus everything inference needs.

    Attributes
    ----------
    arch : str
        ``"film"`` (control-cell conditioned) or ``"gene_mlp"``
        (perturbation-embedding only).
    model : torch.nn.Module
        The model, already in ``eval()`` mode on *device*.
    model_config : dict
        Architecture hyper-parameters, including ``n_hvgs``.
    gene_names : list[str]
        Output gene symbols aligned with the model's ``n_hvgs`` outputs.
    control_emb_mean : np.ndarray or None
        Mean control-cell embedding ``[control_emb_dim]``.  Required for
        ``"film"`` models, ``None`` for ``"gene_mlp"`` models.
    device : torch.device
        Device the model lives on.
    """

    arch: str
    model: nn.Module
    model_config: dict
    gene_names: list[str]
    control_emb_mean: np.ndarray | None
    device: "torch.device"

    @property
    def n_hvgs(self) -> int:
        return int(self.model_config.get("n_hvgs", len(self.gene_names)))

    def predict_delta(
        self,
        pert_labels: Sequence[str],
        gene_emb_map: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict pseudobulk expression deltas for a batch of perturbations.

        Parameters
        ----------
        pert_labels : sequence of str
            Perturbation labels, e.g. ``["KLF1", "AHR_KLF1"]``.
        gene_emb_map : dict[str, np.ndarray]
            Gene symbol -> frozen scGPT gene embedding.

        Returns
        -------
        np.ndarray
            Shape ``[len(pert_labels), n_hvgs]``, dtype ``float32``.
            Row *i* is the predicted delta for ``pert_labels[i]``.
        """
        labels = [str(p) for p in pert_labels]
        if not labels:
            return np.zeros((0, self.n_hvgs), dtype=np.float32)

        emb_dim = int(self.model_config.get("pert_emb_dim", 512))
        feats = np.stack(
            [compute_perturbation_embedding(p, gene_emb_map, emb_dim) for p in labels]
        )
        pert = torch.tensor(feats, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            if self.arch == ARCH_FILM:
                if self.control_emb_mean is None:
                    raise ValueError(
                        "FiLM prediction bundle has no control_emb_mean; "
                        "cannot run inference without a control-cell embedding."
                    )
                control = torch.tensor(
                    np.tile(
                        np.asarray(self.control_emb_mean, dtype=np.float32),
                        (len(labels), 1),
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                mu, _log_var = self.model(control, pert)
                preds = mu
            else:
                preds = self.model(pert)

        return preds.detach().cpu().numpy().astype(np.float32)


def save_prediction_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    arch: str,
    model_config: dict,
    gene_names: Sequence[str],
    control_emb_mean: np.ndarray | None = None,
) -> Path:
    """Serialize a trained prediction head to an inference-ready checkpoint.

    The checkpoint is a plain dict so that ``torch.load(...)["model_config"]``
    / ``["model_state_dict"]`` / ``["gene_names"]`` work without importing
    this module first::

        {
          "format_version": 1,
          "arch": "film" | "gene_mlp",
          "model_config": {...},
          "model_state_dict": {...},
          "gene_names": [...],
          "control_emb_mean": np.ndarray,   # FiLM only
          "created": "<ISO-8601 UTC timestamp>",
        }

    Parameters
    ----------
    path : str or Path
        Destination file, e.g. ``runs/film_head.pt``.  Parent directories are
        created as needed.
    model : torch.nn.Module
        Trained model.
    arch : str
        ``"film"`` or ``"gene_mlp"``.
    model_config : dict
        Architecture hyper-parameters.  ``n_hvgs`` is filled in from
        *gene_names* when absent.
    gene_names : sequence of str
        Output gene symbols, in model output order.
    control_emb_mean : np.ndarray, optional
        Required for ``arch="film"``: the mean control-cell embedding used
        as the FiLM backbone input at inference time.

    Returns
    -------
    Path
        The written checkpoint path.

    Raises
    ------
    ValueError
        If *arch* is unknown, if a FiLM checkpoint lacks *control_emb_mean*,
        or if ``n_hvgs`` and ``len(gene_names)`` disagree.
    """
    if arch not in (ARCH_FILM, ARCH_GENE_MLP):
        raise ValueError(
            f"Unknown architecture {arch!r}; expected one of "
            f"{ARCH_FILM!r} or {ARCH_GENE_MLP!r}."
        )

    gene_names = [str(g) for g in gene_names]
    config = dict(model_config)
    config.setdefault("n_hvgs", len(gene_names))
    config["n_hvgs"] = int(config["n_hvgs"])
    if gene_names and config["n_hvgs"] != len(gene_names):
        raise ValueError(
            f"n_hvgs ({config['n_hvgs']}) does not match len(gene_names) "
            f"({len(gene_names)}); the gene names must align with the outputs."
        )
    if arch == ARCH_FILM and control_emb_mean is None:
        raise ValueError(
            "FiLM checkpoints require control_emb_mean so that inference can "
            "run without re-loading per-cell embeddings."
        )

    payload: dict = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "arch": arch,
        "model_config": config,
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "gene_names": gene_names,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if control_emb_mean is not None:
        payload["control_emb_mean"] = np.asarray(
            control_emb_mean, dtype=np.float32
        ).ravel()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def read_checkpoint_arch(path: Union[str, Path]) -> str:
    """Return the architecture tag of a checkpoint without building the model.

    Checkpoints written before this helper existed have no ``arch`` key and
    are treated as ``"gene_mlp"`` for backwards compatibility.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} is not a prediction-head checkpoint.")
    cfg = ckpt.get("model_config") or {}
    return str(ckpt.get("arch") or cfg.get("arch") or ARCH_GENE_MLP)


def load_prediction_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
) -> PredictionBundle:
    """Load a checkpoint written by :func:`save_prediction_checkpoint`.

    Both architectures are supported and dispatched on the ``arch`` tag, so
    callers (agent tools, retrieval-index builder, API) share one code path.

    Parameters
    ----------
    path : str or Path
        Checkpoint file.
    device : str
        Torch device string, e.g. ``"cpu"``.

    Returns
    -------
    PredictionBundle
        Model in ``eval()`` mode plus its config, gene names and (for FiLM)
        the mean control embedding.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not a prediction-head checkpoint, the architecture is
        unknown, or required fields are missing/inconsistent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not a prediction-head checkpoint "
            "(missing 'model_state_dict')."
        )

    cfg = dict(ckpt.get("model_config") or {})
    arch = str(ckpt.get("arch") or cfg.get("arch") or ARCH_GENE_MLP)
    gene_names = [str(g) for g in (ckpt.get("gene_names") or cfg.get("gene_names") or [])]
    n_hvgs = int(cfg.get("n_hvgs") or len(gene_names))
    if n_hvgs <= 0:
        raise ValueError(f"{path}: cannot determine n_hvgs (no config, no gene_names).")
    if gene_names and len(gene_names) != n_hvgs:
        raise ValueError(
            f"{path}: len(gene_names)={len(gene_names)} != n_hvgs={n_hvgs}."
        )
    cfg["n_hvgs"] = n_hvgs

    control_emb_mean: np.ndarray | None = None
    if arch == ARCH_FILM:
        raw_mean = ckpt.get("control_emb_mean")
        if raw_mean is None:
            raise ValueError(
                f"{path}: FiLM checkpoint is missing 'control_emb_mean'."
            )
        control_emb_mean = np.asarray(raw_mean, dtype=np.float32).ravel()
        model: nn.Module = FiLMPredictionHead(
            pert_emb_dim=int(cfg.get("pert_emb_dim", 512)),
            control_emb_dim=int(cfg.get("control_emb_dim", control_emb_mean.shape[0])),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_layers=int(cfg.get("num_layers", 2)),
            n_hvgs=n_hvgs,
            dropout=0.0,  # inference: no stochasticity
            film_hidden_dim=cfg.get("film_hidden_dim"),
        )
    elif arch == ARCH_GENE_MLP:
        model = GeneMLP(
            pert_emb_dim=int(cfg.get("pert_emb_dim", 512)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_layers=int(cfg.get("num_layers", 2)),
            n_hvgs=n_hvgs,
            dropout=0.0,
        )
    else:
        raise ValueError(
            f"{path}: unknown architecture {arch!r}; expected "
            f"{ARCH_FILM!r} or {ARCH_GENE_MLP!r}."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return PredictionBundle(
        arch=arch,
        model=model,
        model_config=cfg,
        gene_names=gene_names,
        control_emb_mean=control_emb_mean,
        device=torch.device(device),
    )
