#!/usr/bin/env python3
"""QC -> embedding filter -> normalization -> HVG -> split for Perturb-seq.

Loads the raw Norman et al. 2019 dataset (schema-validated), applies QC
filters (guide-assignment coverage, min genes/cell, mitochondrial fraction,
doublet/multiplet removal), drops perturbations whose constituent genes lack
cached scGPT embeddings, then does library-size normalization + log1p and
highly-variable-gene selection. Finally it builds the perturbation-level
train/val/test split and persists it as JSON. Saves the processed AnnData to
``data/processed/perturbseq.h5ad`` and prints a QC summary report with cell
counts before/after each filter, the dropped-perturbation report, the split
breakdown, and the perturbation label distribution.

Usage
-----
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from perturbgpt.data import preprocessing as pp  # noqa: E402
from perturbgpt.data.loading import load_config, load_dataset  # noqa: E402
from perturbgpt.data.splitting import (  # noqa: E402
    perturbation_split,
    persist_split,
)


def load_gene_symbols_with_embeddings(path: Path) -> set[str]:
    """Load the set of gene symbols that have cached scGPT embeddings."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cached gene embeddings not found at {path}. "
            "Run scripts/build_embeddings.py first."
        )
    gene_npz = np.load(path, allow_pickle=True)
    return set(gene_npz["gene_symbols"].astype(str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "data.yaml")
    )
    parser.add_argument(
        "--training-config",
        default=str(PROJECT_ROOT / "configs" / "training.yaml"),
        help="YAML config holding the splitting section.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    training_config = load_config(args.training_config)
    cfg = config["preprocessing"]
    schema = config["schema"]
    split_cfg = training_config["splitting"]
    pert_split_cfg = split_cfg["perturbation_split"]

    adata = load_dataset(args.config)
    print(f"Loaded raw dataset: {adata.n_obs} cells x {adata.n_vars} genes\n")

    # Track cell counts after every step for the QC report.
    counts = [("raw (schema validated)", adata.n_obs)]

    # Optional guide-assignment QC: drop cells without good guide coverage.
    cov_col = cfg.get("good_coverage_column", "good_coverage")
    if cfg.get("require_good_coverage") and cov_col in adata.obs.columns:
        keep = adata.obs[cov_col].astype(bool).to_numpy()
        adata = adata[keep].copy()
        counts.append((f"{cov_col} == True", adata.n_obs))

    adata = pp.filter_min_counts(
        adata, cfg["min_counts_per_cell"], n_counts_col=schema.get("n_counts_col", "ncounts")
    )
    counts.append((f"total counts >= {cfg['min_counts_per_cell']}", adata.n_obs))

    adata = pp.filter_min_genes(
        adata, cfg["min_genes_per_cell"], n_genes_col=schema.get("n_genes_col", "n_genes")
    )
    counts.append((f"n_genes >= {cfg['min_genes_per_cell']}", adata.n_obs))

    adata = pp.filter_mito_fraction(
        adata, cfg["max_mito_pct"], mito_pct_col=schema.get("mito_pct_col", "pct_mito")
    )
    counts.append((f"mito pct <= {cfg['max_mito_pct']}", adata.n_obs))

    adata = pp.filter_doublets(adata, column=cfg.get("doublet_column"))
    counts.append((f"doublet removal ({cfg.get('doublet_column') or 'auto'})", adata.n_obs))

    # Drop perturbations whose genes have no cached scGPT embedding.
    gene_emb_file = Path(config["dataset"].get("gene_embeddings_file", ""))
    if not gene_emb_file.is_absolute():
        gene_emb_file = PROJECT_ROOT / gene_emb_file
    gene_symbols_with_emb = load_gene_symbols_with_embeddings(gene_emb_file)
    adata, emb_stats = pp.filter_perturbations_by_embedding_coverage(
        adata,
        gene_symbols_with_emb,
        perturbation_col=schema["perturbation_col"],
    )
    counts.append(
        (f"perturbations with gene embeddings (dropped {emb_stats['n_perts_dropped']})",
         adata.n_obs)
    )

    log1p = bool(cfg.get("log1p", True))
    adata = pp.normalize_total_log1p(
        adata, target_sum=float(cfg["target_sum"]), log1p=log1p
    )
    counts.append(("normalize + log1p" if log1p else "normalize (no log1p)", adata.n_obs))

    adata = pp.select_highly_variable_genes(adata, n_top_genes=int(cfg["n_top_hvgs"]))

    # Perturbation-level train/val/test split (same seed/fracs as training).
    split = perturbation_split(
        adata,
        train_frac=float(pert_split_cfg["train_frac"]),
        val_frac=float(pert_split_cfg["val_frac"]),
        test_frac=float(pert_split_cfg["test_frac"]),
        seed=int(split_cfg["seed"]),
        perturbation_col=schema["perturbation_col"],
        control_split=pert_split_cfg.get("control_split", "train"),
    )
    split.assert_no_overlap()
    persist_dir = Path(split_cfg["persist_dir"])
    if not persist_dir.is_absolute():
        persist_dir = PROJECT_ROOT / persist_dir
    split_path = persist_split(split, persist_dir / "perturbation_split.json")

    # Provenance + QC metadata travel with the processed object.
    adata.uns["dataset"] = {
        k: config["dataset"][k]
        for k in ("name", "accession", "source_url", "license", "download_date")
        if k in config["dataset"]
    }
    adata.uns["preprocessing"] = dict(cfg)
    adata.uns["embedding_filter"] = emb_stats
    # parallel arrays: anndata cannot serialize lists of mixed-type dicts
    adata.uns["qc_summary"] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "step": [name for name, _ in counts],
        "n_cells": np.asarray([n for _, n in counts], dtype=np.int64),
    }

    out_path = Path(config["dataset"]["processed_file"])
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")

    # ---- QC summary report ----
    print("=== QC summary ===")
    print(f"{'step':<34}{'cells':>10}{'removed':>10}")
    prev = None
    for name, n in counts:
        removed = "" if prev is None else str(prev - n)
        print(f"{name:<34}{n:>10}{removed:>10}")
        prev = n
    print(f"\nFinal object: {adata.n_obs} cells x {adata.n_vars} genes (HVGs)")
    print(f"Saved to: {out_path}\n")

    # ---- embedding-coverage drop report ----
    print("=== Perturbations dropped (missing gene embeddings) ===")
    print(
        f"kept:   {emb_stats['n_perts_after']} perturbations "
        f"({adata.n_obs} cells)"
    )
    print(
        f"dropped: {emb_stats['n_perts_dropped']} perturbations "
        f"({emb_stats['n_cells_dropped']} cells)"
    )
    if emb_stats["dropped_perturbations"]:
        dropped_list = emb_stats["dropped_perturbations"]
        print(f"dropped labels: {', '.join(dropped_list)}")
    print()

    # ---- split report ----
    print("=== Perturbation split ===")
    print(
        f"train: {len(split.train):>6} cells "
        f"({len(split.train_perts)} perturbations)"
    )
    print(
        f"val:   {len(split.val):>6} cells "
        f"({len(split.val_perts)} perturbations)"
    )
    print(
        f"test:  {len(split.test):>6} cells "
        f"({len(split.test_perts)} perturbations)"
    )
    print(f"Saved split to: {split_path}\n")

    pert_col = schema["perturbation_col"]
    dist = adata.obs[pert_col].value_counts()
    print(f"=== Perturbation distribution ({pert_col}) ===")
    print(f"unique perturbations: {dist.size}")
    n_ctrl = int(dist.get("control", 0))
    print(f"control cells: {n_ctrl} ({100.0 * n_ctrl / adata.n_obs:.1f}%)")
    print("\ntop 15 perturbations:")
    for label, n in dist.head(15).items():
        print(f"  {label:<28}{n:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
