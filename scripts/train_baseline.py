#!/usr/bin/env python3
"""Train and evaluate the perturbation-embedding baseline.

Trains on train-split perturbations (using only a learned perturbation
embedding — no cell-state or expression features), evaluates on val/test
using MSE, MAE, Pearson/Spearman correlation, and top-k gene recovery, then
appends results to ``results/metrics.csv`` (one row per model/split/run).

Usage
-----
    python scripts/train_baseline.py
    python scripts/train_baseline.py \
        --model-config configs/model.yaml \
        --training-config configs/training.yaml \
        --data-config configs/data.yaml \
        --use-wandb
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
    UNKNOWN_IDX,
    BaselineModel,
    build_pert_to_idx,
    compute_pseudobulk_deltas,
)


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def evaluate(
    model: BaselineModel,
    deltas: dict[str, np.ndarray],
    pert_to_idx: dict[str, int],
    device: torch.device,
    top_k: int,
) -> dict[str, float]:
    """Evaluate model predictions against true pseudobulk deltas."""
    model.eval()
    perts = sorted(deltas.keys())
    if not perts:
        return {"pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mae": 0.0,
                "top50_precision": 0.0, "top50_recall": 0.0}

    indices = torch.tensor(
        [pert_to_idx.get(p, UNKNOWN_IDX) for p in perts],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        preds = model(indices).cpu().numpy()

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
    row = {"model": model_name, "split": split_name, "timestamp": datetime.now(timezone.utc).isoformat()}
    row.update({k: v for k, v in metrics.items()})
    # Ensure all fieldnames present
    for fn in fieldnames:
        row.setdefault(fn, "")
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_hparam_overrides(unknown_args: list[str], model_cfg: dict) -> dict:
    """Parse ``--key=value`` / ``--key value`` flags injected by the wandb
    agent (or typed manually) into a dict of hyperparameter overrides.

    Types are coerced to match the existing value in ``model_cfg`` when the
    key is known; otherwise we try int, then float, then keep the string.
    Unknown keys (not present in ``model_cfg``) are still returned so the
    caller can decide what to do with them.
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
            # value is the next token (unless it's another flag)
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                key, raw = body, unknown_args[i + 1]
                i += 1
            else:
                # bare boolean flag
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
            # not in config → infer
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
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results" / "metrics.csv"))
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--use-wandb", action="store_true", help="enable W&B tracking")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    # Use parse_known_args: the wandb agent injects sampled hyperparameters as
    # extra --key=value flags, which argparse would otherwise reject.
    args, unknown = parser.parse_known_args(argv)

    model_cfg = load_config(Path(args.model_config))["baseline"]
    train_cfg = load_config(Path(args.training_config))
    data_cfg = load_config(Path(args.data_config))

    # Apply sweep/CLI hyperparameter overrides (highest precedence).
    hparam_overrides = parse_hparam_overrides(unknown, model_cfg)
    for key, value in hparam_overrides.items():
        old = model_cfg.get(key)
        model_cfg[key] = value
        print(f"  override: {key} = {value}  (was {old!r})")

    seed = args.seed if args.seed is not None else train_cfg["splitting"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- W&B initialization ---
    wandb_cfg = train_cfg.get("wandb", {})
    # Auto-enable wandb when running under a sweep agent (env var is set by wandb)
    import os
    _under_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
    use_wandb = args.use_wandb or wandb_cfg.get("enabled", False) or _under_sweep
    wandb_run = None
    if use_wandb:
        if not _WANDB_AVAILABLE:
            print("WARNING: --use-wandb requested but wandb not installed; skipping.")
            use_wandb = False
        else:
            run_config = {
                "seed": seed,
                "embed_dim": model_cfg["embed_dim"],
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
            wandb_run = wandb.init(
                project=args.wandb_project or wandb_cfg.get("project", "perturbgpt-baseline"),
                entity=wandb_cfg.get("entity"),
                name=args.wandb_run_name or wandb_cfg.get("run_name"),
                tags=wandb_cfg.get("tags", ["baseline"]),
                config=run_config,
            )
            # Sweep agent overrides: wandb.config may contain hyperparameters
            # injected by the sweep controller. These take precedence.
            for key in ("lr", "embed_dim", "hidden_dim", "num_layers", "dropout"):
                if key in wandb.config:
                    model_cfg[key] = wandb.config[key]
                    print(f"  sweep override: {key} = {wandb.config[key]}")
            if "epochs" in wandb.config:
                args.epochs = wandb.config["epochs"]

    # 1. Load processed data
    processed_path = PROJECT_ROOT / data_cfg["dataset"]["processed_file"]
    adata = ad.read_h5ad(processed_path)
    print(f"Loaded processed data: {adata.n_obs} cells x {adata.n_vars} genes")

    # 2. Build perturbation-level split
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

    # 3. Compute pseudobulk deltas on RAW expression (same space as the
    #    film head targets).
    all_deltas = compute_pseudobulk_deltas(adata, split)
    train_deltas = {p: d for p, d in all_deltas.items() if p in split.train_perts}
    val_deltas = {p: d for p, d in all_deltas.items() if p in split.val_perts}
    test_deltas = {p: d for p, d in all_deltas.items() if p in split.test_perts}
    print(f"Deltas: train={len(train_deltas)} val={len(val_deltas)} test={len(test_deltas)}")

    # 4. Build pert-to-idx and model
    pert_to_idx = build_pert_to_idx(split.train_perts)
    n_hvgs = adata.n_vars
    model = BaselineModel(
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"],
        n_hvgs=n_hvgs, num_perts=len(pert_to_idx), dropout=model_cfg["dropout"],
    ).to(device)



    # 5. Train
    lr = model_cfg["lr"]
    epochs = args.epochs if args.epochs is not None else model_cfg["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_perts = sorted(train_deltas.keys())
    train_targets = torch.tensor(
        np.stack([train_deltas[p] for p in train_perts]),
        dtype=torch.float32, device=device,
    )
    train_indices = torch.tensor(
        [pert_to_idx.get(p, UNKNOWN_IDX) for p in train_perts],
        dtype=torch.long, device=device,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = model_cfg.get("patience", 10)
    top_k = model_cfg.get("top_k", 50)
    log_interval = wandb_cfg.get("log_interval", 1)
    full_val_interval = wandb_cfg.get("full_val_interval", 5)

    # Precompute val tensors for efficient per-epoch evaluation
    val_perts_sorted = sorted(val_deltas.keys()) if val_deltas else []
    val_indices = torch.tensor(
        [pert_to_idx.get(p, UNKNOWN_IDX) for p in val_perts_sorted],
        dtype=torch.long, device=device,
    ) if val_perts_sorted else None
    val_targets = torch.tensor(
        np.stack([val_deltas[p] for p in val_perts_sorted]),
        dtype=torch.float32, device=device,
    ) if val_perts_sorted else None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(train_indices)
        loss = loss_fn(preds, train_targets)
        loss.backward()
        optimizer.step()

        # Evaluate val loss every epoch (cheap — ~15 perturbations)
        model.eval()
        val_loss = float("inf")
        if val_perts_sorted:
            with torch.no_grad():
                vp = model(val_indices)
                val_loss = float(loss_fn(vp, val_targets))

        # Print + log to wandb at log_interval or first epoch
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  train_loss={loss.item():.6f}  val_loss={val_loss:.6f}")
            if use_wandb:
                wandb.log({"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": val_loss})

        # Full val metrics (pearson, spearman, mse, mae, top-k) every full_val_interval
        if val_perts_sorted and ((epoch + 1) % full_val_interval == 0 or epoch == 0):
            with torch.no_grad():
                val_preds_np = model(val_indices).cpu().numpy()
            val_trues_np = np.stack([val_deltas[p] for p in val_perts_sorted])
            val_metrics = {
                "val/pearson": pearson_corr(val_preds_np, val_trues_np),
                "val/spearman": spearman_corr(val_preds_np, val_trues_np),
                "val/mse": mse(val_preds_np, val_trues_np),
                "val/mae": mae(val_preds_np, val_trues_np),
            }
            tk = top_k_recovery(val_preds_np, val_trues_np, k=top_k)
            val_metrics[f"val/top{top_k}_precision"] = tk["precision"]
            val_metrics[f"val/top{top_k}_recall"] = tk["recall"]
            val_metrics["epoch"] = epoch + 1
            if use_wandb:
                wandb.log(val_metrics)

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

    # 6. Evaluate on val and test
    results_path = Path(args.results)
    for split_name, deltas in [("val", val_deltas), ("test", test_deltas)]:
        if not deltas:
            print(f"No perturbations in {split_name} split, skipping.")
            continue
        metrics = evaluate(model, deltas, pert_to_idx, device, top_k)
        print(f"\n=== {split_name} metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        append_results(results_path, "baseline", split_name, metrics)
        # Log final summary metrics to wandb (no epoch step — these are final)
        if use_wandb:
            summary_metrics = {f"{split_name}_{k}": v for k, v in metrics.items()}
            for k, v in summary_metrics.items():
                wandb.run.summary[k] = v
            # Also log as a final step for the charts
            wandb.log({f"final/{split_name}/{k}": v for k, v in metrics.items()})

    print(f"\nResults appended to {results_path}")
    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())

