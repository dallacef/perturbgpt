#!/usr/bin/env python3
"""Train and evaluate the FiLM-conditioned MLP prediction head.

Loads cached scGPT cell and gene embeddings (from
``scripts/build_embeddings.py``), trains a FiLM-MLP that modulates a
cell-embedding network with per-layer scale/shift parameters derived
from the perturbation gene embedding, and predicts the pseudobulk
expression delta Δx over the same HVG gene set used by the baseline.

The split is **identical** to the one used by
``scripts/train_baseline.py`` (same seed, same perturbation-level
partition), and evaluation uses the **same** ``eval/prediction_metrics.py``
functions (MSE, MAE, Pearson/Spearman, top-k).  Results are appended to
the same ``results/metrics.csv`` so the two models are directly
comparable.  A markdown comparison table (baseline vs. FiLM head) is
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
import torch.nn as nn
import yaml

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
from perturbgpt.models.baseline import compute_pseudobulk_deltas  # noqa: E402
from perturbgpt.models.prediction_head import (  # noqa: E402
    FiLMMLP,
    compute_perturbation_embedding,
)

MODEL_NAME = "film_head"

def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_cached_embeddings(embeddings_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Load cached scGPT cell and gene embeddings from NPZ files.

    Returns
    -------
    cell_emb_map : dict[str, np.ndarray]
        Maps cell barcode → embedding vector ``[d_model]``.
    gene_emb_map : dict[str, np.ndarray]
        Maps gene symbol → embedding vector ``[d_model]``.
    d_model : int
        Embedding dimensionality.
    """
    cell_path = embeddings_dir / "cell_embeddings.npz"
    gene_path = embeddings_dir / "gene_embeddings.npz"
    if not cell_path.exists():
        raise FileNotFoundError(
            f"Cached cell embeddings not found at {cell_path}. "
            "Run scripts/build_embeddings.py first."
        )
    if not gene_path.exists():
        raise FileNotFoundError(
            f"Cached gene embeddings not found at {gene_path}. "
            "Run scripts/build_embeddings.py first."
        )

    cell_npz = np.load(cell_path, allow_pickle=True)
    cell_emb = cell_npz["embeddings"]
    cell_ids = cell_npz["cell_ids"].astype(str)
    cell_emb_map = {cid: cell_emb[i] for i, cid in enumerate(cell_ids)}

    gene_npz = np.load(gene_path, allow_pickle=True)
    gene_emb = gene_npz["embeddings"]
    gene_syms = gene_npz["gene_symbols"].astype(str)
    gene_emb_map = {gs: gene_emb[i] for i, gs in enumerate(gene_syms)}

    d_model = int(cell_emb.shape[1])
    return cell_emb_map, gene_emb_map, d_model


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


def build_training_tensors(
    adata: ad.AnnData,
    split,
    cell_emb_map: dict[str, np.ndarray],
    gene_emb_map: dict[str, np.ndarray],
    train_deltas: dict[str, np.ndarray],
    d_model: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-cell training tensors.

    Each train-split cell contributes one sample:
        cell_emb[cell]            -- the scGPT embedding of that cell
        pert_emb[cell's pert]     -- the scGPT gene embedding for the perturbation
        target = delta[that pert] -- the pseudobulk delta for that perturbation

    Returns
    -------
    cell_embs : Tensor [n_train_cells, d_model]
    pert_embs : Tensor [n_train_cells, d_model]
    targets   : Tensor [n_train_cells, n_hvgs]
    """
    perts_col = adata.obs["perturbation"].astype(str)
    train_cell_set = set(split.train)

    cell_list, pert_list, target_list = [], [], []
    for cell_id in adata.obs_names:
        if cell_id not in train_cell_set:
            continue
        if cell_id not in cell_emb_map:
            continue
        pert = str(perts_col.loc[cell_id])
        if pert == CONTROL_LABEL:
            continue
        if pert not in train_deltas:
            continue
        cell_list.append(cell_emb_map[cell_id])
        pert_list.append(compute_perturbation_embedding(pert, gene_emb_map, d_model))
        target_list.append(train_deltas[pert])

    if not cell_list:
        raise RuntimeError(
            "No training cells with matching cached embeddings found. "
            "Ensure build_embeddings.py was run on the same dataset."
        )

    cell_embs = torch.tensor(np.stack(cell_list), dtype=torch.float32, device=device)
    pert_embs = torch.tensor(np.stack(pert_list), dtype=torch.float32, device=device)
    targets = torch.tensor(np.stack(target_list), dtype=torch.float32, device=device)
    return cell_embs, pert_embs, targets


def evaluate(
    model: FiLMMLP,
    deltas: dict[str, np.ndarray],
    gene_emb_map: dict[str, np.ndarray],
    ctrl_cell_emb: np.ndarray,
    d_model: int,
    device: torch.device,
    top_k: int,
) -> dict[str, float]:
    """Evaluate model predictions against true pseudobulk deltas.

    Uses the mean control-cell embedding as the carrier signal (analogous
    to the baseline's ``pca_ctrl_mean``) and the perturbation gene embedding
    as the FiLM condition.
    """
    model.eval()
    perts = sorted(deltas.keys())
    if not perts:
        return {
            "pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mae": 0.0,
            "top50_precision": 0.0, "top50_recall": 0.0,
        }

    cell_feat = torch.tensor(
        np.tile(ctrl_cell_emb[np.newaxis, :], (len(perts), 1)),
        dtype=torch.float32, device=device,
    )
    pert_feat = torch.tensor(
        np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model) for p in perts]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        preds = model(cell_feat, pert_feat).cpu().numpy()

    trues = np.stack([deltas[p] for p in perts])
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
    parser.add_argument("--batch-size", type=int, default=None, help="override config batch_size")
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
            "cell_emb_dim": model_cfg["cell_emb_dim"],
            "pert_emb_dim": model_cfg["pert_emb_dim"],
            "hidden_dim": model_cfg["hidden_dim"],
            "num_layers": model_cfg["num_layers"],
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

    # 3. Load cached scGPT embeddings
    embeddings_dir = Path(args.embeddings_dir)
    print(f"Loading cached embeddings from {embeddings_dir} ...")
    cell_emb_map, gene_emb_map, d_model = load_cached_embeddings(embeddings_dir)
    print(f"  {len(cell_emb_map)} cell embeddings, {len(gene_emb_map)} gene embeddings, d_model={d_model}")

    # Use cached d_model if the config left emb dims unset or mismatched
    cell_emb_dim = model_cfg.get("cell_emb_dim", d_model)
    pert_emb_dim = model_cfg.get("pert_emb_dim", d_model)
    if cell_emb_dim != d_model or pert_emb_dim != d_model:
        print(
            f"  WARNING: config emb dims (cell={cell_emb_dim}, pert={pert_emb_dim}) "
            f"do not match cached d_model={d_model}; using cached d_model."
        )
        cell_emb_dim = d_model
        pert_emb_dim = d_model

    # 4. Compute pseudobulk deltas (identical to baseline)
    all_deltas = compute_pseudobulk_deltas(adata, split)
    train_deltas = {p: d for p, d in all_deltas.items() if p in split.train_perts}
    val_deltas = {p: d for p, d in all_deltas.items() if p in split.val_perts}
    test_deltas = {p: d for p, d in all_deltas.items() if p in split.test_perts}
    print(f"Deltas: train={len(train_deltas)} val={len(val_deltas)} test={len(test_deltas)}")

    # 5. Compute mean control-cell scGPT embedding (for evaluation)
    perts_col = adata.obs["perturbation"].astype(str)
    ctrl_mask = (perts_col == CONTROL_LABEL) & adata.obs_names.isin(set(split.train))
    ctrl_cell_ids = adata.obs_names[ctrl_mask].tolist()
    ctrl_embs = [cell_emb_map[c] for c in ctrl_cell_ids if c in cell_emb_map]
    if not ctrl_embs:
        raise RuntimeError("No control cells with matching cached embeddings found.")
    ctrl_cell_emb = np.mean(ctrl_embs, axis=0).astype(np.float32)
    print(f"Mean control cell embedding from {len(ctrl_embs)} cells")


    # 6. Build model
    n_hvgs = adata.n_vars
    model = FiLMMLP(
        cell_emb_dim=cell_emb_dim,
        pert_emb_dim=pert_emb_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        n_hvgs=n_hvgs,
        dropout=model_cfg["dropout"],
    ).to(device)
    print(f"Model: FiLMMLP(cell_emb_dim={cell_emb_dim}, pert_emb_dim={pert_emb_dim}, "
          f"hidden={model_cfg['hidden_dim']}, layers={model_cfg['num_layers']}, n_hvgs={n_hvgs})")

    # 7. Build training tensors (per-cell)
    train_cell, train_pert, train_targets = build_training_tensors(
        adata, split, cell_emb_map, gene_emb_map, train_deltas, d_model, device,
    )
    print(f"Training samples: {train_cell.shape[0]} cells")

    # 8. Training loop
    lr = model_cfg["lr"]
    epochs = args.epochs if args.epochs is not None else model_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else model_cfg.get("batch_size", 64)
    weight_decay = model_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    patience = model_cfg.get("patience", 10)
    top_k = model_cfg.get("top_k", 50)
    log_interval = wandb_cfg.get("log_interval", 1)
    full_val_interval = wandb_cfg.get("full_val_interval", 5)

    n_samples = train_cell.shape[0]
    indices = np.arange(n_samples)

    # Precompute val evaluation tensors
    val_perts_sorted = sorted(val_deltas.keys()) if val_deltas else []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            optimizer.zero_grad()
            preds = model(train_cell[batch_idx], train_pert[batch_idx])
            loss = loss_fn(preds, train_targets[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate val loss every epoch
        model.eval()
        val_loss = float("inf")
        if val_perts_sorted:
            with torch.no_grad():
                vc = torch.tensor(
                    np.tile(ctrl_cell_emb[np.newaxis, :], (len(val_perts_sorted), 1)),
                    dtype=torch.float32, device=device,
                )
                vp = torch.tensor(
                    np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model)
                              for p in val_perts_sorted]),
                    dtype=torch.float32, device=device,
                )
                vt = torch.tensor(
                    np.stack([val_deltas[p] for p in val_perts_sorted]),
                    dtype=torch.float32, device=device,
                )
                val_loss = float(loss_fn(model(vc, vp), vt))

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  train_loss={avg_train_loss:.6f}  val_loss={val_loss:.6f}")
            if use_wandb:
                wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})

        # Full val metrics every full_val_interval
        if val_perts_sorted and ((epoch + 1) % full_val_interval == 0 or epoch == 0):
            val_metrics = evaluate(
                model, val_deltas, gene_emb_map, ctrl_cell_emb, d_model, device, top_k,
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

    # 9. Evaluate on val and test
    results_path = Path(args.results)
    final_metrics: dict[str, dict[str, float]] = {}
    for split_name, deltas in [("val", val_deltas), ("test", test_deltas)]:
        if not deltas:
            print(f"No perturbations in {split_name} split, skipping.")
            continue
        metrics = evaluate(model, deltas, gene_emb_map, ctrl_cell_emb, d_model, device, top_k)
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

    # 10. Print comparison table
    print_comparison_table(results_path, final_metrics)

    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())


