#!/usr/bin/env python3
"""QC -> normalization -> HVG selection for the raw Perturb-seq dataset.

Loads the raw Norman et al. 2019 dataset (schema-validated), applies QC
filters (guide-assignment coverage, min genes/cell, mitochondrial fraction,
doublet/multiplet removal), library-size normalization + log1p, and
highly-variable-gene selection. Saves the processed AnnData to
``data/processed/perturbseq.h5ad`` and prints a QC summary report with cell
counts before/after each filter and the perturbation label distribution.

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "data.yaml")
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    cfg = config["preprocessing"]
    schema = config["schema"]

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

    log1p = bool(cfg.get("log1p", True))
    adata = pp.normalize_total_log1p(
        adata, target_sum=float(cfg["target_sum"]), log1p=log1p
    )
    counts.append(("normalize + log1p" if log1p else "normalize (no log1p)", adata.n_obs))

    adata = pp.select_highly_variable_genes(adata, n_top_genes=int(cfg["n_top_hvgs"]))

    # Provenance + QC metadata travel with the processed object.
    adata.uns["dataset"] = {
        k: config["dataset"][k]
        for k in ("name", "accession", "source_url", "license", "download_date")
        if k in config["dataset"]
    }
    adata.uns["preprocessing"] = dict(cfg)
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
