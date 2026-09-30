#!/usr/bin/env python3
"""Train and evaluate the FiLM prediction head.

Loads cached scGPT gene and cell embeddings (from
``scripts/build_embeddings.py``), pairs every control cell embedding with each
perturbation's summed gene embedding, and trains a FiLM-conditioned head to
predict the pseudobulk expression delta Δx over the HVG gene set. The model
outputs a mean ``mu`` and a ``log_variance`` per HVG, and is trained by
maximizing the likelihood of the true Δx under the predicted normal
distribution (a Gaussian negative log-likelihood loss).

The split is **identical** to the one used by ``scripts/train_baseline.py``
(same seed, same perturbation-level partition), and evaluation uses the **same**
``eval/prediction_metrics.py`` functions (MSE, MAE, Pearson/Spearman, top-k),
with predictions averaged over all control cells for each perturbation.
Results are appended to the same ``results/metrics.csv`` so the two models are
directly comparable. A markdown comparison table (baseline vs. FiLM head) is
printed at the end.

Usage
-----
    python scripts/train_prediction_head.py \
        --embeddings-dir data/embeddings \
        --model-config configs/model.yaml \
        --training-config configs/training.yaml \
        --data-config configs/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from perturbgpt.data.splitting import (  # noqa: E402
    CONTROL_LABEL,
    perturbation_split,
    persist_split,
)
from perturbgpt.eval.prediction_metrics import (  # noqa: E402
    mae,
    mse,
    pearson_corr,
    spearman_corr,
    top_k_recovery,
)
from perturbgpt.models.baseline import (  # noqa: E402
    compute_control_statistics,
    compute_pseudobulk_deltas,
    standardize_deltas,
    unstandardize_predictions,
)
from perturbgpt.models.prediction_head import (  # noqa: E402
    FiLMPredictionHead,
    compute_perturbation_embedding,
)

MODEL_NAME = "film_head"

def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_gene_embeddings(embeddings_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load cached scGPT gene embeddings from the NPZ file.

    Returns
    -------
    gene_emb_map : dict[str, np.ndarray]
        Maps gene symbol → embedding vector ``[d_model]``.
    d_model : int
        Embedding dimensionality.
    """
    gene_path = embeddings_dir / "gene_embeddings.npz"
    if not gene_path.exists():
        raise FileNotFoundError(
            f"Cached gene embeddings not found at {gene_path}. "
            "Run scripts/build_embeddings.py first."
        )

    gene_npz = np.load(gene_path, allow_pickle=True)
    gene_emb = gene_npz["embeddings"]
    gene_syms = gene_npz["gene_symbols"].astype(str)
    gene_emb_map = {gs: gene_emb[i] for i, gs in enumerate(gene_syms)}

    return gene_emb_map, int(gene_emb.shape[1])


def load_cell_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load cached scGPT cell embeddings from the NPZ file.

    Returns
    -------
    cell_emb : np.ndarray
        Float array of shape ``[n_cells, d_model]``.
    cell_ids : list[str]
        Cell barcode strings aligned to the rows of ``cell_emb``.
    """
    cell_path = embeddings_dir / "cell_embeddings.npz"
    if not cell_path.exists():
        raise FileNotFoundError(
            f"Cached cell embeddings not found at {cell_path}. "
            "Run scripts/build_embeddings.py first."
        )

    cell_npz = np.load(cell_path, allow_pickle=True)
    cell_emb = cell_npz["embeddings"]
    cell_ids = cell_npz["cell_ids"].astype(str)
    return cell_emb, list(cell_ids)


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood of ``target`` under N(mu, exp(log_var)).

    The constant ``0.5 * log(2 * pi)`` term is dropped because it does not
    affect the gradient. ``log_var`` is the per-gene log-variance predicted by
    the model.
    """
    var = log_var.exp()
    return 0.5 * (log_var + (target - mu).pow(2) / var).mean()


class ControlPertDataset(Dataset):
    """Cartesian product of control-cell embeddings and perturbations.

    Each ``(control_cell_i, perturbation_j)`` pair is one training sample whose
    target is perturbation ``j``'s standardized pseudobulk delta (broadcast over
    all control cells). ``__len__`` is ``n_ctrl * n_pert``.
    """

    def __init__(
        self,
        ctrl_embs: torch.Tensor,
        pert_embs: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.ctrl_embs = ctrl_embs
        self.pert_embs = pert_embs
        self.targets = targets
        self.n_ctrl = int(ctrl_embs.shape[0])
        self.n_pert = int(pert_embs.shape[0])

    def __len__(self) -> int:
        return self.n_ctrl * self.n_pert

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j = divmod(idx, self.n_pert)
        return self.ctrl_embs[i], self.pert_embs[j], self.targets[j]


def compute_split_loss(
    model: FiLMPredictionHead,
    ctrl_embs: torch.Tensor,
    pert_embs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Mean Gaussian NLL over all (control cell, perturbation) pairs."""
    model.eval()
    n_ctrl = int(ctrl_embs.shape[0])
    total = 0.0
    n_pairs = 0
    with torch.no_grad():
        for j in range(int(pert_embs.shape[0])):
            ctrl_batch = ctrl_embs
            pert_batch = pert_embs[j].unsqueeze(0).expand(n_ctrl, -1)
            target_batch = targets[j].unsqueeze(0).expand(n_ctrl, -1)
            mu, log_var = model(ctrl_batch, pert_batch)
            total += float(gaussian_nll_loss(mu, log_var, target_batch)) * n_ctrl
            n_pairs += n_ctrl
    return total / n_pairs if n_pairs else float("inf")


def append_results(
    path: Path,
    model_name: str,
    split_name: str,
    metrics: dict[str, float],
) -> None:
    """Append a row to the results CSV, creating the file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "split", "pearson", "spearman", "mse", "mae",
        "top50_precision", "top50_recall", "timestamp",
    ]
    row = {
        "model": model_name,
        "split": split_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row.update({k: v for k, v in metrics.items()})
    for fn in fieldnames:
        row.setdefault(fn, "")
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate(
    model: FiLMPredictionHead,
    ctrl_embs: torch.Tensor,
    deltas: dict[str, np.ndarray],
    gene_emb_map: dict[str, np.ndarray],
    d_model: int,
    sigma: np.ndarray,
    device: torch.device,
    top_k: int,
) -> dict[str, float]:
    """Evaluate model predictions against true pseudobulk deltas.

    For each perturbation, the model is run on every control-cell embedding
    paired with that perturbation's gene embedding, and the ``mu`` predictions
    are averaged over control cells to give a single perturbation-level
    prediction. Predictions and targets are un-standardized (multiplied by
    ``sigma``) before metrics are computed so every metric is reported in the
    original delta scale.
    """
    model.eval()
    perts = sorted(deltas.keys())
    if not perts:
        return {
            "pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mae": 0.0,
            "top50_precision": 0.0, "top50_recall": 0.0,
        }

    n_ctrl = int(ctrl_embs.shape[0])
    pert_feat = torch.tensor(
        np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model) for p in perts]),
        dtype=torch.float32, device=device,
    )

    mu_preds: list[np.ndarray] = []
    with torch.no_grad():
        for j in range(int(pert_feat.shape[0])):
            ctrl_batch = ctrl_embs.to(device)
            pert_batch = pert_feat[j].unsqueeze(0).expand(n_ctrl, -1)
            mu, _log_var = model(ctrl_batch, pert_batch)
            mu_preds.append(mu.mean(dim=0).cpu().numpy())

    preds = unstandardize_predictions(np.stack(mu_preds), sigma)
    trues = unstandardize_predictions(np.stack([deltas[p] for p in perts]), sigma)
    tk = top_k_recovery(preds, trues, k=top_k)
    return {
        "pearson": pearson_corr(preds, trues),
        "spearman": spearman_corr(preds, trues),
        "mse": mse(preds, trues),
        "mae": mae(preds, trues),
        f"top{top_k}_precision": tk["precision"],
        f"top{top_k}_recall": tk["recall"],
    }


def print_comparison_table(results_path: Path, film_metrics: dict[str, dict[str, float]]) -> None:
    """Print a markdown comparison table: baseline vs. FiLM head.

    Reads the latest baseline rows from ``results_path`` and compares
    against the freshly computed ``film_metrics`` dict keyed by split name.
    """
    baseline_metrics: dict[str, dict[str, float]] = {}
    if results_path.exists():
        with results_path.open() as fh:
            reader = csv.DictReader(fh)
            # Keep the *last* row per (model, split) so we compare against
            # the most recent baseline run.
            for row in reader:
                if row["model"] == "baseline":
                    split = row["split"]
                    baseline_metrics[split] = {
                        "pearson": float(row["pearson"]),
                        "spearman": float(row["spearman"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "top50_precision": float(row["top50_precision"]),
                        "top50_recall": float(row["top50_recall"]),
                    }

    metric_keys = ["pearson", "spearman", "mse", "mae", "top50_precision", "top50_recall"]
    splits = sorted(set(baseline_metrics.keys()) | set(film_metrics.keys()))

    print("\n" + "=" * 72)
    print("  COMPARISON: baseline vs. film_head")
    print("=" * 72)

    for split in splits:
        if split not in baseline_metrics:
            print(f"\n  [{split}]  (no baseline results found)\n")
            b = {}
        else:
            b = baseline_metrics[split]
        f = film_metrics.get(split, {})

        if not f:
            print(f"\n  [{split}]  (no film_head results)\n")
            continue

        print(f"\n  ── {split} ──")
        print(f"  {'metric':<22s} {'baseline':>14s} {'film_head':>14s} {'Δ':>14s}")
        print(f"  {'─' * 22} {'─' * 14} {'─' * 14} {'─' * 14}")
        for key in metric_keys:
            bv = b.get(key)
            fv = f.get(key)
            if bv is not None and fv is not None:
                delta = fv - bv
                print(f"  {key:<22s} {bv:>14.6f} {fv:>14.6f} {delta:>+14.6f}")
            elif fv is not None:
                print(f"  {key:<22s} {'—':>14s} {fv:>14.6f} {'—':>14s}")
            elif bv is not None:
                print(f"  {key:<22s} {bv:>14.6f} {'—':>14s} {'—':>14s}")

    print("\n" + "=" * 72 + "\n")


def parse_hparam_overrides(unknown_args: list[str], model_cfg: dict) -> dict:
    """Parse ``--key=value`` / ``--key value`` flags into a dict of overrides.

    Types are coerced to match the existing value in ``model_cfg`` when the
    key is known; otherwise we try int, then float, then keep the string.
    """
    overrides: dict[str, object] = {}
    i = 0
    while i < len(unknown_args):
        tok = unknown_args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
        else:
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                key, raw = body, unknown_args[i + 1]
                i += 1
            else:
                key, raw = body, "true"
        key = key.strip()
        raw = raw.strip()

        def coerce(val: str):
            existing = model_cfg.get(key)
            if isinstance(existing, bool):
                return val.lower() in ("1", "true", "yes", "on")
            if isinstance(existing, int):
                return int(val)
            if isinstance(existing, float):
                return float(val)
            try:
                return int(val)
            except ValueError:
                pass
            try:
                return float(val)
            except ValueError:
                return val

        overrides[key] = coerce(raw)
        i += 1
    return overrides


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs" / "training.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument(
        "--embeddings-dir",
        default=str(PROJECT_ROOT / "data" / "embeddings"),
        help="Directory containing cached cell_embeddings.npz and gene_embeddings.npz.",
    )
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results" / "metrics.csv"))
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--use-wandb", action="store_true", help="enable W&B tracking")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    args, unknown = parser.parse_known_args(argv)

    model_cfg = load_config(Path(args.model_config))["prediction_head"]
    train_cfg = load_config(Path(args.training_config))
    data_cfg = load_config(Path(args.data_config))

    # Apply sweep/CLI hyperparameter overrides (highest precedence).
    overrides = parse_hparam_overrides(unknown, model_cfg)
    if overrides:
        print(f"Hyperparameter overrides: {overrides}")
    model_cfg.update(overrides)

    seed = args.seed if args.seed is not None else train_cfg["splitting"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- W&B initialization ---
    import os
    wandb_cfg = train_cfg.get("wandb", {})
    _under_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
    use_wandb = args.use_wandb or wandb_cfg.get("enabled", False) or _under_sweep
    if use_wandb and not _WANDB_AVAILABLE:
        print("WARNING: --use-wandb requested but wandb not installed; skipping.")
        use_wandb = False
    if use_wandb:
        run_config = {
            "seed": seed,
            "pert_emb_dim": model_cfg["pert_emb_dim"],
            "control_emb_dim": model_cfg.get("control_emb_dim", "auto"),
            "hidden_dim": model_cfg["hidden_dim"],
            "num_layers": model_cfg["num_layers"],
            "film_hidden_dim": model_cfg.get("film_hidden_dim"),
            "batch_size": model_cfg.get("batch_size", 256),
            "dropout": model_cfg["dropout"],
            "lr": model_cfg["lr"],
            "epochs": args.epochs or model_cfg["epochs"],
            "patience": model_cfg.get("patience", 10),
            "top_k": model_cfg.get("top_k", 50),
            "split_seed": seed,
            "train_frac": train_cfg["splitting"]["perturbation_split"]["train_frac"],
            "val_frac": train_cfg["splitting"]["perturbation_split"]["val_frac"],
            "test_frac": train_cfg["splitting"]["perturbation_split"]["test_frac"],
        }
        wandb.init(
            project=args.wandb_project or wandb_cfg.get("project", "perturbgpt"),
            entity=wandb_cfg.get("entity"),
            name=args.wandb_run_name or wandb_cfg.get("run_name"),
            tags=(wandb_cfg.get("tags", []) + ["film_head"]),
            config=run_config,
        )


    # 1. Load processed data
    processed_path = PROJECT_ROOT / data_cfg["dataset"]["processed_file"]
    adata = ad.read_h5ad(processed_path)
    print(f"Loaded processed data: {adata.n_obs} cells x {adata.n_vars} genes")

    # 2. Build perturbation-level split (identical to baseline)
    ps_cfg = train_cfg["splitting"]["perturbation_split"]
    split = perturbation_split(
        adata, seed=seed,
        train_frac=ps_cfg["train_frac"], val_frac=ps_cfg["val_frac"],
        test_frac=ps_cfg["test_frac"], control_split=ps_cfg["control_split"],
    )
    split.assert_no_overlap()
    print(f"Split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"  perts: train={len(split.train_perts)} val={len(split.val_perts)} test={len(split.test_perts)}")

    persist_dir = PROJECT_ROOT / train_cfg["splitting"]["persist_dir"]
    persist_split(split, persist_dir / "perturbation_split.json")

    # 3. Load cached scGPT gene embeddings
    embeddings_dir = Path(args.embeddings_dir)
    print(f"Loading cached gene embeddings from {embeddings_dir} ...")
    gene_emb_map, d_model = load_gene_embeddings(embeddings_dir)
    print(f"  {len(gene_emb_map)} gene embeddings, d_model={d_model}")

    # Use cached d_model if the config left the embedding dims unset/mismatched
    pert_emb_dim = model_cfg.get("pert_emb_dim", d_model)
    if pert_emb_dim != d_model:
        print(
            f"  WARNING: config pert_emb_dim ({pert_emb_dim}) does not match "
            f"cached d_model={d_model}; using cached d_model."
        )
        pert_emb_dim = d_model

    control_emb_dim = model_cfg.get("control_emb_dim", d_model)
    if control_emb_dim != d_model:
        print(
            f"  WARNING: config control_emb_dim ({control_emb_dim}) does not "
            f"match cached d_model={d_model}; using cached d_model."
        )
        control_emb_dim = d_model

    # Load cached cell embeddings and select the control cells used to
    # condition the FiLM head. All control cells live in split.train.
    print(f"Loading cached cell embeddings from {embeddings_dir} ...")
    cell_emb, cell_ids = load_cell_embeddings(embeddings_dir)
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    ctrl_cells = sorted(
        set(adata.obs_names[adata.obs["perturbation"].astype(str) == CONTROL_LABEL])
    )
    ctrl_idx = np.array([cell_id_to_idx[c] for c in ctrl_cells], dtype=np.int64)
    ctrl_embs = torch.tensor(cell_emb[ctrl_idx], dtype=torch.float32, device=device)
    print(f"  {ctrl_embs.shape[0]} control cells, control_emb_dim={control_emb_dim}")

    # 4. Compute pseudobulk deltas (identical to baseline)
    all_deltas = compute_pseudobulk_deltas(adata, split)
    train_deltas = {p: d for p, d in all_deltas.items() if p in split.train_perts}
    val_deltas = {p: d for p, d in all_deltas.items() if p in split.val_perts}
    test_deltas = {p: d for p, d in all_deltas.items() if p in split.test_perts}
    print(f"Deltas: train={len(train_deltas)} val={len(val_deltas)} test={len(test_deltas)}")

    # Per-gene control statistics for target standardization.
    ctrl_mu, ctrl_sigma = compute_control_statistics(adata, split)
    train_deltas = standardize_deltas(train_deltas, ctrl_sigma)
    val_deltas = standardize_deltas(val_deltas, ctrl_sigma)
    test_deltas = standardize_deltas(test_deltas, ctrl_sigma)

    # 5. Build model
    n_hvgs = adata.n_vars
    model = FiLMPredictionHead(
        pert_emb_dim=pert_emb_dim,
        control_emb_dim=control_emb_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        n_hvgs=n_hvgs,
        dropout=model_cfg["dropout"],
        film_hidden_dim=model_cfg.get("film_hidden_dim"),
    ).to(device)
    print(f"Model: FiLMPredictionHead(pert_emb_dim={pert_emb_dim}, "
          f"control_emb_dim={control_emb_dim}, hidden={model_cfg['hidden_dim']}, "
          f"layers={model_cfg['num_layers']}, n_hvgs={n_hvgs})")

    # 6. Build training tensors: control-cell x perturbation cartesian pairs.
    def _make_split_tensors(
        deltas: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        perts = sorted(deltas.keys())
        pert_emb = torch.tensor(
            np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model)
                      for p in perts]),
            dtype=torch.float32, device=device,
        )
        targets = torch.tensor(
            np.stack([deltas[p] for p in perts]),
            dtype=torch.float32, device=device,
        )
        return pert_emb, targets

    train_pert_emb, train_targets = _make_split_tensors(train_deltas)
    # for i,j in zip(train_pert_emb, train_targets):
    #     print(f"train_pert_emb={i}, train_targets={j}")
    val_pert_emb, val_targets = _make_split_tensors(val_deltas)

    batch_size = model_cfg.get("batch_size", 256)
    train_dataset = ControlPertDataset(ctrl_embs, train_pert_emb, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training samples: {len(train_dataset)} "
          f"(control cells x perturbations), batch_size={batch_size}")

    # 7. Training loop
    lr = model_cfg["lr"]
    epochs = args.epochs if args.epochs is not None else model_cfg["epochs"]
    weight_decay = model_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    patience = model_cfg.get("patience", 10)
    top_k = model_cfg.get("top_k", 50)
    log_interval = wandb_cfg.get("log_interval", 1)
    full_val_interval = wandb_cfg.get("full_val_interval", 5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for ctrl_batch, pert_batch, target_batch in train_loader:
            optimizer.zero_grad()
            mu, log_var = model(ctrl_batch, pert_batch)
            loss = gaussian_nll_loss(mu, log_var, target_batch)
            # print(f"Epoch {epoch+1}")
            # print(f"pert_batch={pert_batch[0]}")
            # print(f"mu={mu[0].mean().item():.6f}, log_var={log_var[0].mean().item():.6f}, target={target_batch[0].mean().item():.6f}")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / n_batches if n_batches else 0.0

        # Evaluate val loss every epoch (mean NLL over all control-cell pairs)
        val_loss = float("inf")
        if val_pert_emb.shape[0]:
            val_loss = compute_split_loss(model, ctrl_embs, val_pert_emb, val_targets)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
            if use_wandb:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Full val metrics every full_val_interval
        if val_pert_emb.shape[0] and ((epoch + 1) % full_val_interval == 0 or epoch == 0):
            val_metrics = evaluate(
                model, ctrl_embs, val_deltas, gene_emb_map, d_model, ctrl_sigma, device, top_k,
            )
            if use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
                wandb.log({"epoch": epoch + 1})

        # Early stopping
        if model_cfg.get("early_stopping", True) and val_loss < float("inf"):
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # 8. Evaluate on val and test
    results_path = Path(args.results)
    final_metrics: dict[str, dict[str, float]] = {}
    for split_name, deltas in [("val", val_deltas), ("test", test_deltas)]:
        if not deltas:
            print(f"No perturbations in {split_name} split, skipping.")
            continue
        metrics = evaluate(model, ctrl_embs, deltas, gene_emb_map, d_model, ctrl_sigma, device, top_k)
        print(f"\n=== {split_name} metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        append_results(results_path, MODEL_NAME, split_name, metrics)
        final_metrics[split_name] = metrics
        if use_wandb:
            for k, v in metrics.items():
                wandb.run.summary[f"{split_name}_{k}"] = v
            wandb.log({f"final/{split_name}/{k}": v for k, v in metrics.items()})

    print(f"\nResults appended to {results_path}")

    # 9. Print comparison table
    print_comparison_table(results_path, final_metrics)

    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())


