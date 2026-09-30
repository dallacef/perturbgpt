#!/usr/bin/env python3
"""Plot predicted vs. true pseudobulk deltas for the baseline and FiLM models.

Trains both models in-memory on the same perturbation-level split, then
scatters predicted vs. true Δx per gene for a selection of test-split
perturbations spanning strong, medium, and weak effect sizes. Each panel is
annotated with Pearson r, MAE, and the OLS slope of ``pred`` on ``true`` —
a slope far from 1.0 flags a scale mismatch.

* **baseline** — a learned ``nn.Embedding`` over perturbation identities.
* **film_head** — a FiLM-conditioned head over summed frozen scGPT gene
  embeddings and individual control-cell embeddings (predictions averaged
  over all control cells).

Usage
-----
    python scripts/plot_pred_vs_true.py

    python scripts/plot_pred_vs_true.py \\
        --n-perts 6 \\
        --epochs 100 \\
        --output results/pred_vs_true.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — headless backend before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad  # noqa: E402

from perturbgpt.data.splitting import CONTROL_LABEL, perturbation_split  # noqa: E402
from perturbgpt.eval.prediction_metrics import mae, pearson_corr  # noqa: E402
from perturbgpt.models.baseline import (  # noqa: E402
    UNKNOWN_IDX,
    BaselineModel,
    build_pert_to_idx,
    compute_control_statistics,
    compute_pseudobulk_deltas,
    standardize_deltas,
    unstandardize_predictions,
)
from perturbgpt.models.prediction_head import (  # noqa: E402
    FiLMPredictionHead,
    compute_perturbation_embedding,
)


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_gene_embeddings(embeddings_dir: Path) -> tuple[dict, int]:
    """Load cached scGPT gene embeddings from the NPZ file."""
    gene_path = embeddings_dir / "gene_embeddings.npz"
    if not gene_path.exists():
        raise FileNotFoundError(f"Cached gene embeddings not found at {gene_path}.")
    npz = np.load(gene_path, allow_pickle=True)
    emb = npz["embeddings"]
    syms = npz["gene_symbols"].astype(str)
    gene_emb_map = {s: emb[i] for i, s in enumerate(syms)}
    return gene_emb_map, int(emb.shape[1])


def load_cell_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load cached scGPT cell embeddings from the NPZ file."""
    cell_path = embeddings_dir / "cell_embeddings.npz"
    if not cell_path.exists():
        raise FileNotFoundError(f"Cached cell embeddings not found at {cell_path}.")
    npz = np.load(cell_path, allow_pickle=True)
    cell_emb = npz["embeddings"]
    cell_ids = npz["cell_ids"].astype(str)
    return cell_emb, list(cell_ids)


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL of ``target`` under N(mu, exp(log_var)) (constant dropped)."""
    var = log_var.exp()
    return 0.5 * (log_var + (target - mu).pow(2) / var).mean()


class ControlPertDataset(Dataset):
    """Cartesian product of control-cell embeddings and perturbations."""

    def __init__(self, ctrl_embs, pert_embs, targets):
        self.ctrl_embs = ctrl_embs
        self.pert_embs = pert_embs
        self.targets = targets
        self.n_ctrl = int(ctrl_embs.shape[0])
        self.n_pert = int(pert_embs.shape[0])

    def __len__(self) -> int:
        return self.n_ctrl * self.n_pert

    def __getitem__(self, idx: int):
        i, j = divmod(idx, self.n_pert)
        return self.ctrl_embs[i], self.pert_embs[j], self.targets[j]


def ols_slope(true: np.ndarray, pred: np.ndarray) -> float:
    """Ordinary least-squares slope of ``pred`` on ``true`` (NaN if degenerate)."""
    if np.std(true) < 1e-12:
        return float("nan")
    return float(np.polyfit(true, pred, 1)[0])


# ------------------------------------------------------------- baseline


def train_baseline_model(adata, split, raw_deltas, ctrl_sigma, model_cfg, device, epochs):
    """Train the perturbation-embedding baseline.

    Targets are standardized for training; ``test_deltas`` stay raw.
    Returns ``(model, test_deltas, pert_to_idx)``.
    """
    train_deltas = standardize_deltas(
        {p: raw_deltas[p] for p in split.train_perts if p in raw_deltas}, ctrl_sigma,
    )
    test_deltas = {p: raw_deltas[p] for p in split.test_perts if p in raw_deltas}

    pert_to_idx = build_pert_to_idx(split.train_perts)
    n_hvgs = adata.n_vars
    model = BaselineModel(
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        n_hvgs=n_hvgs,
        num_perts=len(pert_to_idx),
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"], weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_perts = sorted(train_deltas.keys())
    train_targets = torch.tensor(
        np.stack([train_deltas[p] for p in train_perts]), dtype=torch.float32, device=device,
    )
    train_indices = torch.tensor(
        [pert_to_idx.get(p, UNKNOWN_IDX) for p in train_perts], dtype=torch.long, device=device,
    )

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(train_indices)
        loss = loss_fn(preds, train_targets)
        loss.backward()
        optimizer.step()

    return model, test_deltas, pert_to_idx


def baseline_predictions(model, test_deltas, pert_to_idx, ctrl_sigma, device):
    """Return ``{pert: (true, pred)}`` in original delta scale."""
    model.eval()
    perts = sorted(test_deltas.keys())
    indices = torch.tensor(
        [pert_to_idx.get(p, UNKNOWN_IDX) for p in perts], dtype=torch.long, device=device,
    )
    with torch.no_grad():
        preds = model(indices).cpu().numpy()
    preds = unstandardize_predictions(preds, ctrl_sigma)
    return {p: (test_deltas[p], preds[i]) for i, p in enumerate(perts)}


# ------------------------------------------------------------- gene MLP


def train_film_model(adata, split, raw_deltas, ctrl_sigma, gene_emb_map, d_model, ctrl_embs, model_cfg, device, epochs):
    """Train the FiLM-conditioned head.

    Targets are standardized for training; ``test_deltas`` stay raw.
    Returns ``(model, test_deltas)``.
    """
    train_deltas = standardize_deltas(
        {p: raw_deltas[p] for p in split.train_perts if p in raw_deltas}, ctrl_sigma,
    )
    test_deltas = {p: raw_deltas[p] for p in split.test_perts if p in raw_deltas}

    n_hvgs = adata.n_vars
    model = FiLMPredictionHead(
        pert_emb_dim=d_model,
        control_emb_dim=d_model,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        n_hvgs=n_hvgs,
        dropout=model_cfg["dropout"],
        film_hidden_dim=model_cfg.get("film_hidden_dim"),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg.get("weight_decay", 1e-5),
    )

    train_perts = sorted(train_deltas.keys())
    pert_emb = torch.tensor(
        np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model) for p in train_perts]),
        dtype=torch.float32, device=device,
    )
    targets = torch.tensor(
        np.stack([train_deltas[p] for p in train_perts]), dtype=torch.float32, device=device,
    )

    batch_size = model_cfg.get("batch_size", 256)
    loader = DataLoader(
        ControlPertDataset(ctrl_embs, pert_emb, targets),
        batch_size=batch_size,
        shuffle=True,
    )

    for _ in range(epochs):
        model.train()
        for ctrl_batch, pert_batch, target_batch in loader:
            optimizer.zero_grad()
            mu, log_var = model(ctrl_batch, pert_batch)
            loss = gaussian_nll_loss(mu, log_var, target_batch)
            loss.backward()
            optimizer.step()

    return model, test_deltas


def film_predictions(model, ctrl_embs, test_deltas, gene_emb_map, d_model, ctrl_sigma, device):
    """Return ``{pert: (true, pred)}`` in original delta scale, averaging mu over cells."""
    model.eval()
    perts = sorted(test_deltas.keys())
    n_ctrl = int(ctrl_embs.shape[0])
    pert_feat = torch.tensor(
        np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model) for p in perts]),
        dtype=torch.float32, device=device,
    )
    preds_list = []
    with torch.no_grad():
        for j in range(int(pert_feat.shape[0])):
            ctrl_batch = ctrl_embs.to(device)
            pert_batch = pert_feat[j].unsqueeze(0).expand(n_ctrl, -1)
            mu, _log_var = model(ctrl_batch, pert_batch)
            preds_list.append(mu.mean(dim=0).cpu().numpy())
    preds = unstandardize_predictions(np.stack(preds_list), ctrl_sigma)
    return {p: (test_deltas[p], preds[i]) for i, p in enumerate(perts)}


# ------------------------------------------------------------- plotting


def select_perturbations(deltas: dict, n_perts: int) -> list[str]:
    """Select ``n_perts`` perturbations spanning strong → weak effect sizes."""
    names = sorted(deltas.keys(), key=lambda p: -np.linalg.norm(deltas[p]))
    n = len(names)
    if n_perts >= n:
        return names
    idxs = np.linspace(0, n - 1, n_perts).astype(int)
    return [names[i] for i in idxs]


def effect_size_label(pert: str, deltas: dict) -> str:
    """Return a coarse effect-size label for a perturbation."""
    names = sorted(deltas.keys(), key=lambda p: -np.linalg.norm(deltas[p]))
    n = len(names)
    rank = names.index(pert)
    if rank < n // 3:
        return "strong"
    if rank >= 2 * n // 3:
        return "weak"
    return "medium"


def plot_grid(
    selected: list[str],
    baseline_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    film_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    raw_deltas: dict,
    out_path: Path,
) -> None:
    """Plot a ``n_perts × 2`` grid of pred-vs-true scatter panels."""
    models = [
        ("baseline (learned embedding)", baseline_preds),
        ("film_head (FiLM, control-conditioned)", film_preds),
    ]
    n_perts = len(selected)
    fig, axes = plt.subplots(
        n_perts, 2, figsize=(10, 2.6 * n_perts), squeeze=False,
    )

    for row, pert in enumerate(selected):
        label = effect_size_label(pert, raw_deltas)
        for col, (model_name, preds) in enumerate(models):
            true, pred = preds[pert]
            ax = axes[row, col]
            ax.scatter(true, pred, s=8, alpha=0.5, c=np.abs(true), cmap="viridis")

            m = max(np.max(np.abs(true)), np.max(np.abs(pred))) * 1.1
            if not np.isfinite(m) or m <= 0:
                m = 1.0
            ax.set_xlim(-m, m)
            ax.set_ylim(-m, m)
            ax.axline((0, 0), slope=1.0, color="r", ls="--", lw=1.0)

            r = pearson_corr(pred, true)
            e = mae(pred, true)
            slope = ols_slope(true, pred)
            ax.set_title(
                f"{pert} ({label}) — {model_name}\n"
                f"r={r:.3f}  MAE={e:.3f}  slope={slope:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("true Δx")
            ax.set_ylabel("predicted Δx")
            ax.grid(alpha=0.2)

    fig.suptitle(
        "Predicted vs. true pseudobulk delta (per gene) — "
        "slope ≉ 1 indicates a scale mismatch",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter grid to {out_path}")


# ----------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs" / "model.yaml"),
    )
    parser.add_argument(
        "--training-config",
        default=str(PROJECT_ROOT / "configs" / "training.yaml"),
    )
    parser.add_argument(
        "--data-config",
        default=str(PROJECT_ROOT / "configs" / "data.yaml"),
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(PROJECT_ROOT / "data" / "embeddings"),
    )
    parser.add_argument(
        "--n-perts", type=int, default=4,
        help="Number of perturbations to plot (strong → weak).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs for both models.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "pred_vs_true.png"),
    )
    args = parser.parse_args(argv)

    model_cfg = load_config(Path(args.model_config))
    train_cfg = load_config(Path(args.training_config))
    data_cfg = load_config(Path(args.data_config))

    baseline_cfg = model_cfg["baseline"]
    film_cfg = model_cfg["prediction_head"]

    seed = train_cfg["splitting"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load processed data and build the split (same as training scripts).
    processed_path = PROJECT_ROOT / data_cfg["dataset"]["processed_file"]
    adata = ad.read_h5ad(processed_path)
    print(f"Loaded processed data: {adata.n_obs} cells x {adata.n_vars} genes")

    ps_cfg = train_cfg["splitting"]["perturbation_split"]
    split = perturbation_split(
        adata, seed=seed,
        train_frac=ps_cfg["train_frac"], val_frac=ps_cfg["val_frac"],
        test_frac=ps_cfg["test_frac"], control_split=ps_cfg["control_split"],
    )
    split.assert_no_overlap()
    print(f"Split: train={len(split.train_perts)} val={len(split.val_perts)} "
          f"test={len(split.test_perts)} perturbations")

    # Raw deltas are shared by both models.
    raw_deltas = compute_pseudobulk_deltas(adata, split)
    test_raw = {p: raw_deltas[p] for p in split.test_perts}
    selected = select_perturbations(test_raw, args.n_perts)
    print(f"Selected perturbations: {selected}")

    # Per-gene control statistics for target standardization.
    ctrl_mu, ctrl_sigma = compute_control_statistics(adata, split)

    # 2. Baseline (learned perturbation embedding).
    print("\nTraining baseline (perturbation embedding) ...")
    b_epochs = args.epochs or baseline_cfg["epochs"]
    b_model, b_test, b_pert_to_idx = train_baseline_model(
        adata, split, raw_deltas, ctrl_sigma, baseline_cfg, device, b_epochs,
    )
    baseline_preds = baseline_predictions(b_model, b_test, b_pert_to_idx, ctrl_sigma, device)

    # 3. FiLM head (scGPT gene embeddings + control-cell conditioning).
    print("Training film_head (FiLM-conditioned) ...")
    gene_emb_map, d_model = load_gene_embeddings(Path(args.embeddings_dir))
    print(f"  {len(gene_emb_map)} gene embeddings, d_model={d_model}")

    print(f"Loading cached cell embeddings from {args.embeddings_dir} ...")
    cell_emb, cell_ids = load_cell_embeddings(Path(args.embeddings_dir))
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    ctrl_cells = sorted(
        set(adata.obs_names[adata.obs["perturbation"].astype(str) == CONTROL_LABEL])
    )
    ctrl_idx = np.array([cell_id_to_idx[c] for c in ctrl_cells], dtype=np.int64)
    ctrl_embs = torch.tensor(cell_emb[ctrl_idx], dtype=torch.float32, device=device)
    print(f"  {ctrl_embs.shape[0]} control cells")

    f_epochs = args.epochs or film_cfg["epochs"]
    f_model, f_test = train_film_model(
        adata, split, raw_deltas, ctrl_sigma, gene_emb_map, d_model, ctrl_embs,
        film_cfg, device, f_epochs,
    )
    film_preds = film_predictions(f_model, ctrl_embs, f_test, gene_emb_map, d_model, ctrl_sigma, device)

    # 4. Plot and save.
    plot_grid(selected, baseline_preds, film_preds, raw_deltas, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
