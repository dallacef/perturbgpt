#!/usr/bin/env python3
"""Train and evaluate the PCA+MLP baseline on the perturbation-level split.

Trains on train-split perturbations, evaluates on val/test using MSE, MAE,
Pearson/Spearman correlation, and top-k gene recovery, then appends results
to ``results/metrics.csv`` (one row per model/split/run).
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
    UNKNOWN_IDX,
    BaselineModel,
    build_pert_to_idx,
    compute_pseudobulk_deltas,
    fit_pca_on_control,
)


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def evaluate(
    model: BaselineModel,
    deltas: dict[str, np.ndarray],
    pert_to_idx: dict[str, int],
    pca_ctrl_mean: np.ndarray,
    device: torch.device,
    top_k: int,
) -> dict[str, float]:
    """Evaluate model predictions against true pseudobulk deltas."""
    model.eval()
    perts = sorted(deltas.keys())
    if not perts:
        return {"pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mae": 0.0,
                "top50_precision": 0.0, "top50_recall": 0.0}

    n_hvgs = model.n_hvgs
    pca_dim = model.pca_dim
    pca_feat = torch.tensor(
        np.tile(pca_ctrl_mean[np.newaxis, :], (len(perts), 1)),
        dtype=torch.float32,
        device=device,
    )
    indices = torch.tensor(
        [pert_to_idx.get(p, UNKNOWN_IDX) for p in perts],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        preds = model(pca_feat, indices).cpu().numpy()

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



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs" / "training.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results" / "metrics.csv"))
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    args = parser.parse_args(argv)

    model_cfg = load_config(Path(args.model_config))["baseline"]
    train_cfg = load_config(Path(args.training_config))
    data_cfg = load_config(Path(args.data_config))

    seed = args.seed if args.seed is not None else train_cfg["splitting"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 3. Fit PCA on train control cells (95% variance by default)
    variance = model_cfg.get("pca_variance", 0.95)
    n_comp = model_cfg.get("pca_components")
    pca = fit_pca_on_control(adata, split, variance=variance, n_components=n_comp)
    pca_dim = pca.n_components_
    print(f"PCA: {pca_dim} components (variance threshold={variance})")

    # PCA-project the control mean expression
    perts_col = adata.obs["perturbation"].astype(str)
    ctrl_mask = (perts_col == CONTROL_LABEL) & adata.obs_names.isin(set(split.train))
    X_ctrl = adata[ctrl_mask].X
    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
    ctrl_mean_expr = np.asarray(X_ctrl, dtype=np.float64).mean(axis=0)
    pca_ctrl_mean = pca.transform(ctrl_mean_expr[np.newaxis, :])[0]

    # 4. Compute pseudobulk deltas for each split
    all_deltas = compute_pseudobulk_deltas(adata, split)
    train_deltas = {p: d for p, d in all_deltas.items() if p in split.train_perts}
    val_deltas = {p: d for p, d in all_deltas.items() if p in split.val_perts}
    test_deltas = {p: d for p, d in all_deltas.items() if p in split.test_perts}
    print(f"Deltas: train={len(train_deltas)} val={len(val_deltas)} test={len(test_deltas)}")

    # 5. Build pert-to-idx and model
    pert_to_idx = build_pert_to_idx(split.train_perts)
    n_hvgs = adata.n_vars
    model = BaselineModel(
        pca_dim=pca_dim, embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"],
        n_hvgs=n_hvgs, num_perts=len(pert_to_idx), dropout=model_cfg["dropout"],
    ).to(device)



    # 6. Train
    lr = model_cfg["lr"]
    epochs = args.epochs if args.epochs is not None else model_cfg["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_perts = sorted(train_deltas.keys())
    train_targets = torch.tensor(
        np.stack([train_deltas[p] for p in train_perts]),
        dtype=torch.float32, device=device,
    )
    train_pca_feat = torch.tensor(
        np.tile(pca_ctrl_mean[np.newaxis, :], (len(train_perts), 1)),
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

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(train_pca_feat, train_indices)
        loss = loss_fn(preds, train_targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_loss = float("inf")
            if val_deltas:
                val_perts_sorted = sorted(val_deltas.keys())
                with torch.no_grad():
                    vp = model(
                        torch.tensor(
                            np.tile(pca_ctrl_mean[np.newaxis, :], (len(val_perts_sorted), 1)),
                            dtype=torch.float32, device=device,
                        ),
                        torch.tensor(
                            [pert_to_idx.get(p, UNKNOWN_IDX) for p in val_perts_sorted],
                            dtype=torch.long, device=device,
                        ),
                    )
                    val_loss = float(loss_fn(vp, torch.tensor(
                        np.stack([val_deltas[p] for p in val_perts_sorted]),
                        dtype=torch.float32, device=device,
                    )))
            print(f"Epoch {epoch+1:3d}/{epochs}  train_loss={loss.item():.6f}  val_loss={val_loss:.6f}")
            if model_cfg.get("early_stopping", True) and val_loss < float("inf"):
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

    # 7. Evaluate on val and test
    results_path = Path(args.results)
    for split_name, deltas in [("val", val_deltas), ("test", test_deltas)]:
        if not deltas:
            print(f"No perturbations in {split_name} split, skipping.")
            continue
        metrics = evaluate(model, deltas, pert_to_idx, pca_ctrl_mean, device, top_k)
        print(f"\n=== {split_name} metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        append_results(results_path, "baseline", split_name, metrics)

    print(f"\nResults appended to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

