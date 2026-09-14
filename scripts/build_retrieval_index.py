#!/usr/bin/env python3
"""Build FAISS perturbation and response-signature retrieval indices.

Loads the Prompt 4 cached scGPT gene embeddings and the processed
Perturb-seq AnnData, builds two :class:`~faiss.IndexFlatIP` indices
(cosine similarity via inner product on L2-normalized vectors):

1. **Perturbation index** — over summed scGPT gene embeddings per
   perturbation.
2. **Response index** — over predicted response signatures (Δx̂ from a
   trained FiLM-MLP checkpoint, or pseudobulk deltas as fallback).

Then evaluates both against a pathway-co-membership ground truth using
Recall@K, Precision@K, and MRR, comparing against:

* **Random baseline** — analytical expected values.
* **Raw-expression-correlation baseline** — Pearson ρ between pseudobulk
  delta vectors, ranked descending.

Usage
-----
    python scripts/build_retrieval_index.py \
        --embeddings-dir data/embeddings \
        --data data/processed/perturbseq.h5ad \
        --output-dir data/indices

    # With a trained FiLM-MLP checkpoint for predicted response signatures:
    python scripts/build_retrieval_index.py \
        --embeddings-dir data/embeddings \
        --checkpoint runs/film_head.pt \
        --output-dir data/indices
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anndata as ad  # noqa: E402

from perturbgpt.data.splitting import CONTROL_LABEL, parse_perturbation  # noqa: E402
from perturbgpt.models.baseline import compute_pseudobulk_deltas  # noqa: E402
from perturbgpt.retrieval.index import (  # noqa: E402
    _FAISS_AVAILABLE,
    build_perturbation_index,
    build_response_index,
)
from perturbgpt.retrieval.metrics import (  # noqa: E402
    build_pathway_co_membership,
    evaluate_retrieval,
    random_baseline_metrics,
)
from perturbgpt.eval.prediction_metrics import pearson_corr  # noqa: E402


def load_gene_embeddings(embeddings_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load cached scGPT gene embeddings from NPZ."""
    gene_path = embeddings_dir / "gene_embeddings.npz"
    if not gene_path.exists():
        raise FileNotFoundError(
            f"Cached gene embeddings not found at {gene_path}. "
            "Run scripts/build_embeddings.py first."
        )
    npz = np.load(gene_path, allow_pickle=True)
    emb = npz["embeddings"]
    syms = npz["gene_symbols"].astype(str)
    gene_emb_map = {s: emb[i] for i, s in enumerate(syms)}
    return gene_emb_map, int(emb.shape[1])


def get_unique_perturbations(adata: ad.AnnData) -> tuple[list[str], dict[str, int]]:
    """Extract unique non-control perturbation labels and their cell counts."""
    perts = adata.obs["perturbation"].astype(str)
    labels = [p for p in perts.unique() if p != CONTROL_LABEL]
    labels = sorted(labels)
    counts = {p: int((perts == p).sum()) for p in labels}
    return labels, counts


def load_predicted_response_vectors(
    checkpoint_path: Path,
    gene_emb_map: dict[str, np.ndarray],
    pert_labels: list[str],
    ctrl_cell_emb_path: Path,
    device: str = "cpu",
) -> np.ndarray:
    """Load a trained FiLM-MLP checkpoint and predict Δx̂ per perturbation.

    Falls back to None if the checkpoint or cell embeddings are not
    available; the caller should then use pseudobulk deltas.
    """
    import torch
    from perturbgpt.models.prediction_head import FiLMMLP, compute_perturbation_embedding

    if not checkpoint_path.exists():
        return None
    if not ctrl_cell_emb_path.exists():
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    cell_emb_dim = cfg.get("cell_emb_dim", 512)
    pert_emb_dim = cfg.get("pert_emb_dim", 512)
    hidden_dim = cfg.get("hidden_dim", 256)
    num_layers = cfg.get("num_layers", 3)
    n_hvgs = cfg.get("n_hvgs")
    dropout = cfg.get("dropout", 0.0)

    model = FiLMMLP(
        cell_emb_dim=cell_emb_dim,
        pert_emb_dim=pert_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_hvgs=n_hvgs,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cell_npz = np.load(ctrl_cell_emb_path, allow_pickle=True)
    ctrl_emb = cell_npz["embeddings"].mean(axis=0).astype(np.float32)

    d_model = pert_emb_dim
    cell_feat = torch.tensor(
        np.tile(ctrl_emb[np.newaxis, :], (len(pert_labels), 1)),
        dtype=torch.float32, device=device,
    )
    pert_feat = torch.tensor(
        np.stack([compute_perturbation_embedding(p, gene_emb_map, d_model) for p in pert_labels]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        preds = model(cell_feat, pert_feat).cpu().numpy()
    return preds


def expression_correlation_ranking(
    pert_ids: list[str],
    delta_matrix: np.ndarray,
    k: int,
    exclude_self: bool = True,
) -> dict[str, list[str]]:
    """Rank perturbations by Pearson correlation of pseudobulk delta vectors.

    For each perturbation, computes the Pearson correlation between its
    delta vector and every other perturbation's delta vector, then ranks
    by descending correlation.

    Parameters
    ----------
    pert_ids : list[str]
        Perturbation labels, position-aligned with *delta_matrix*.
    delta_matrix : np.ndarray ``[n_perts, n_hvgs]``
        Pseudobulk delta vectors.
    k : int
        Number of neighbours to return per query.
    exclude_self : bool
        Exclude the query from its own results.

    Returns
    -------
    dict[str, list[str]]
        pert_id → list of top-k pert_ids by descending correlation.
    """
    from scipy import stats

    n = len(pert_ids)
    results: dict[str, list[str]] = {}

    for i, pid in enumerate(pert_ids):
        corrs = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if exclude_self and j == i:
                corrs[j] = -np.inf
                continue
            if np.std(delta_matrix[i]) < 1e-12 or np.std(delta_matrix[j]) < 1e-12:
                corrs[j] = 0.0
            else:
                r, _ = stats.pearsonr(delta_matrix[i], delta_matrix[j])
                corrs[j] = r if not np.isnan(r) else 0.0
        ranked = np.argsort(-corrs, kind="stable")[:k]
        results[pid] = [pert_ids[j] for j in ranked]
    return results


def evaluate_ranking(
    rankings: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: Sequence,
) -> dict[int, dict[str, float]]:
    """Evaluate a pre-computed ranking against ground truth.

    Uses the same metric functions as :func:`evaluate_retrieval` but
    operates on a dict of pre-ranked lists instead of a FAISS index.
    """
    from perturbgpt.retrieval.metrics import (
        mean_reciprocal_rank,
        precision_at_k,
        recall_at_k,
    )

    pids = list(rankings.keys())
    results: dict[int, dict[str, float]] = {}
    for k in k_values:
        recalls, precisions, mrrs = [], [], []
        for pid in pids:
            relevant = ground_truth.get(pid, set())
            retrieved = rankings[pid]
            recalls.append(recall_at_k(retrieved, relevant, k))
            precisions.append(precision_at_k(retrieved, relevant, k))
            mrrs.append(mean_reciprocal_rank(retrieved, relevant))
        n = len(pids) if pids else 1
        results[k] = {
            "recall": sum(recalls) / n,
            "precision": sum(precisions) / n,
            "mrr": sum(mrrs) / n,
        }
    return results


def print_comparison_table(
    system_results: dict[str, dict[int, dict[str, float]]],
    k_values: Sequence,
) -> None:
    """Print a markdown-style comparison table across all retrieval systems."""
    metric_names = ["recall", "precision", "mrr"]
    systems = list(system_results.keys())

    print("\n" + "=" * 80)
    print("  RETRIEVAL EVALUATION: Recall@K, Precision@K, MRR")
    print("=" * 80)

    for k in k_values:
        print(f"\n  ── K={k} ──")
        header = f"  {'system':<30s}"
        for m in metric_names:
            header += f" {m:>14s}"
        print(header)
        print(f"  {'─' * 30}" + f" {'─' * 14}" * len(metric_names))
        for sys_name in systems:
            row = f"  {sys_name:<30s}"
            for m in metric_names:
                val = system_results[sys_name].get(k, {}).get(m, 0.0)
                row += f" {val:>14.6f}"
            print(row)

    print("\n" + "=" * 80 + "\n")



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        default=str(PROJECT_ROOT / "data" / "embeddings"),
        help="Directory containing cached gene_embeddings.npz and cell_embeddings.npz.",
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "processed" / "perturbseq.h5ad"),
        help="Path to the processed AnnData file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "indices"),
        help="Directory to save the built FAISS indices.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained FiLM-MLP checkpoint (.pt) for predicted response "
             "signatures. If not provided, pseudobulk deltas are used.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for Recall@K / Precision@K evaluation.",
    )
    args = parser.parse_args(argv)

    if not _FAISS_AVAILABLE:
        print("ERROR: faiss is not installed. Install with: pip install faiss-cpu")
        return 1

    embeddings_dir = Path(args.embeddings_dir)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    if not data_path.exists():
        print(f"ERROR: Processed data not found at {data_path}")
        print("Run scripts/preprocess_data.py first.")
        return 1

    # 1. Load cached gene embeddings
    print(f"Loading cached gene embeddings from {embeddings_dir} ...")
    gene_emb_map, d_model = load_gene_embeddings(embeddings_dir)
    print(f"  {len(gene_emb_map)} gene embeddings, d_model={d_model}")

    # 2. Load processed AnnData
    print(f"Loading processed data from {data_path} ...")
    adata = ad.read_h5ad(data_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # 3. Extract unique perturbations and cell counts
    pert_labels, cell_counts = get_unique_perturbations(adata)
    print(f"  {len(pert_labels)} unique perturbations (excluding control)")

    # 4. Build perturbation index
    print("\nBuilding perturbation index ...")
    pert_index = build_perturbation_index(
        gene_emb_map=gene_emb_map,
        pert_labels=pert_labels,
        cell_counts=cell_counts,
        emb_dim=d_model,
    )
    print(f"  {pert_index.n_entries} entries, dim={pert_index.dim}")

    # 5. Build response index
    print("\nBuilding response index ...")
    response_vectors = None
    response_source = "pseudobulk_deltas"

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        ctrl_cell_emb_path = embeddings_dir / "cell_embeddings.npz"
        print(f"  Loading predicted response vectors from checkpoint {checkpoint_path} ...")
        response_vectors = load_predicted_response_vectors(
            checkpoint_path, gene_emb_map, pert_labels, ctrl_cell_emb_path,
        )
        if response_vectors is not None:
            response_source = "predicted_film_mlp"
            print(f"  Predicted {response_vectors.shape[0]} response vectors "
                  f"of dim {response_vectors.shape[1]}")
        else:
            print("  WARNING: Could not load checkpoint; falling back to pseudobulk deltas.")

    if response_vectors is None:
        print("  Computing pseudobulk deltas as response signatures ...")
        from perturbgpt.data.splitting import PerturbationSplit
        all_cells = adata.obs_names.tolist()
        all_perts = set(pert_labels)
        dummy_split = PerturbationSplit(
            train=all_cells, val=[], test=[],
            train_perts=all_perts, val_perts=set(), test_perts=set(),
            seed=42, params={}, created="",
        )
        all_deltas = compute_pseudobulk_deltas(adata, dummy_split)
        response_vectors = np.stack([all_deltas[p] for p in pert_labels])
        print(f"  {response_vectors.shape[0]} pseudobulk delta vectors "
              f"of dim {response_vectors.shape[1]}")

    gene_syms_map = {p: parse_perturbation(p) for p in pert_labels}
    response_index = build_response_index(
        pert_ids=pert_labels,
        response_vectors=response_vectors,
        cell_counts=cell_counts,
        gene_symbols_map=gene_syms_map,
    )
    print(f"  {response_index.n_entries} entries, dim={response_index.dim}")

    # 6. Build ground truth (pathway co-membership — stub returns empty)
    print("\nBuilding pathway-co-membership ground truth ...")
    ground_truth = build_pathway_co_membership(pert_labels)
    n_with_gt = sum(1 for v in ground_truth.values() if v)
    print(f"  {n_with_gt}/{len(ground_truth)} perturbations have non-empty ground truth")
    if n_with_gt == 0:
        print("  (Stub pathway lookup returns empty — all metrics will be 0.0. "
              "Prompt 7 will implement real pathway memberships.)")

    # 7. Evaluate all systems
    k_values = args.k_values
    print(f"\nEvaluating retrieval at K={k_values} ...")

    system_results: dict[str, dict[int, dict[str, float]]] = {}

    print("  - FAISS perturbation index ...")
    system_results["faiss_perturbation"] = evaluate_retrieval(
        pert_index, ground_truth, k_values=k_values,
    )

    print("  - FAISS response index ...")
    system_results["faiss_response"] = evaluate_retrieval(
        response_index, ground_truth, k_values=k_values,
    )

    print("  - Random baseline (analytical) ...")
    system_results["random_baseline"] = random_baseline_metrics(
        n_entries=pert_index.n_entries, k_values=k_values,
    )

    print("  - Raw-expression-correlation baseline ...")
    max_k = max(k_values)
    corr_rankings = expression_correlation_ranking(
        pert_ids=pert_labels,
        delta_matrix=response_vectors,
        k=max_k,
        exclude_self=True,
    )
    system_results["expr_correlation"] = evaluate_ranking(
        corr_rankings, ground_truth, k_values=k_values,
    )

    # 8. Print comparison table
    print_comparison_table(system_results, k_values)

    # 9. Save indices
    output_dir.mkdir(parents=True, exist_ok=True)
    pert_path = output_dir / "perturbation_index"
    resp_path = output_dir / "response_index"
    print(f"Saving perturbation index to {pert_path}.{{faiss,json}} ...")
    pert_index.save(pert_path)
    print(f"Saving response index to {resp_path}.{{faiss,json}} ...")
    response_index.save(resp_path)

    # 10. Save evaluation results
    results_meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "n_perturbations": pert_index.n_entries,
        "d_model": d_model,
        "response_source": response_source,
        "k_values": list(k_values),
        "n_with_ground_truth": n_with_gt,
        "systems": {
            sys_name: {str(k): v for k, v in k_results.items()}
            for sys_name, k_results in system_results.items()
        },
    }
    meta_path = output_dir / "retrieval_results.json"
    with meta_path.open("w") as fh:
        json.dump(results_meta, fh, indent=2)
    print(f"  evaluation results saved to {meta_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


