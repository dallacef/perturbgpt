#!/usr/bin/env python3
"""Extract and cache scGPT embeddings for the full processed dataset.

Runs the frozen scGPT model over every cell and every gene in the processed
Perturb-seq dataset, then saves the results to disk so that no downstream
step needs to reload scGPT.

Outputs (in ``--output-dir``):
    cell_embeddings.npz   -- compressed NPZ with keys:
        * ``embeddings``  : float32 array, shape (n_cells, d_model)
        * ``cell_ids``    : array of cell barcode strings
    gene_embeddings.npz   -- compressed NPZ with keys:
        * ``embeddings``  : float32 array, shape (n_genes, d_model)
        * ``gene_symbols`` : array of gene symbol strings
    metadata.json         -- run metadata (timestamp, model_dir, shapes)

Usage
-----
    python scripts/build_embeddings.py \
        --model-dir /path/to/scGPT_checkpoint \
        --data data/processed/perturbseq.h5ad \
        --output-dir data/embeddings \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad  # noqa: E402

from perturbgpt.foundation.scgpt_wrapper import ScGPTWrapper  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the scGPT checkpoint directory (contains args.json, "
        "vocab.json, best_model.pt).",
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "processed" / "perturbseq.h5ad"),
        help="Path to the processed AnnData file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "embeddings"),
        help="Directory to write cached embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for the cell-embedding forward pass.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (cpu or cuda).",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Processed data not found at {data_path}")
        print("Run scripts/preprocess_data.py first.")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load processed dataset ---------------------------------------------
    print(f"Loading processed data from {data_path} ...")
    adata = ad.read_h5ad(data_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # ---- initialise wrapper --------------------------------------------------
    print(f"Loading scGPT checkpoint from {args.model_dir} ...")
    wrapper = ScGPTWrapper(
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"  d_model={wrapper.d_model}, vocab={len(wrapper.vocab)} tokens")

    # ---- cell embeddings ------------------------------------------------------
    print("Extracting cell embeddings ...")
    cell_emb = wrapper.extract_cell_embeddings(adata)
    cell_ids = np.array(adata.obs_names.to_numpy(), dtype=str)
    print(f"  cell embeddings: shape={cell_emb.shape}, dtype={cell_emb.dtype}")

    cell_path = out_dir / "cell_embeddings.npz"
    np.savez_compressed(
        cell_path,
        embeddings=cell_emb,
        cell_ids=cell_ids,
    )
    print(f"  saved to {cell_path}")

    # ---- gene embeddings ------------------------------------------------------
    print("Extracting gene embeddings ...")
    gene_symbols = sorted(adata.var_names.tolist())
    gene_emb_list = []
    matched_genes = []
    for gs in gene_symbols:
        try:
            gene_emb_list.append(wrapper.get_gene_embedding(gs))
            matched_genes.append(gs)
        except KeyError:
            pass  # gene not in scGPT vocabulary -- skip
    gene_emb = np.stack(gene_emb_list)
    gene_sym = np.array(matched_genes, dtype=str)
    print(
        f"  gene embeddings: shape={gene_emb.shape}, dtype={gene_emb.dtype} "
        f"({len(matched_genes)}/{len(gene_symbols)} genes matched)"
    )

    gene_path = out_dir / "gene_embeddings.npz"
    np.savez_compressed(
        gene_path,
        embeddings=gene_emb,
        gene_symbols=gene_sym,
    )
    print(f"  saved to {gene_path}")

    # ---- metadata ---------------------------------------------------------------
    meta = {
        "model_dir": str(args.model_dir),
        "data_file": str(data_path),
        "created": datetime.now(timezone.utc).isoformat(),
        "n_cells": int(adata.n_obs),
        "n_genes_matched": len(matched_genes),
        "n_genes_total": len(gene_symbols),
        "d_model": wrapper.d_model,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  metadata saved to {meta_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
