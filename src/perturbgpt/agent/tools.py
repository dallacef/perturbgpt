"""Implements the agent's callable tools.

Tools
-----
* ``get_predicted_genes`` — runs the Prompt 5 FiLM-MLP model for a
  perturbation and returns top-k predicted DE genes.
* ``find_similar_perturbations`` — queries the Prompt 6 retrieval index
  for perturbations with similar gene embeddings or response signatures.
* ``get_pathways`` — runs hypergeometric pathway enrichment (Prompt 7)
  on a list of gene symbols.
* ``retrieve_literature`` — searches the Prompt 8 PubMed FAISS index
  for relevant abstracts with PMID citations.

Each tool returns a plain ``list[dict]`` (or empty list) so the
orchestrator and agent-eval can inspect results without model-specific
types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from perturbgpt.eval.pathway_metrics import run_enrichment


# --------------------------------------------------------- module-level state
# These are lazily initialised so importing this module doesn't require
# all heavy dependencies (torch, faiss, sentence-transformers).

_pert_index = None      # retrieval/index.PerturbationIndex
_response_index = None  # retrieval/index.ResponseIndex
_lit_index = None       # rag/literature_index.LiteratureIndex
_model_checkpoint = None  # dict from torch.load
_gene_emb_map = None    # dict[str, np.ndarray]
_gene_names = None      # list[str] — HVG gene names aligned to model output
_ctrl_cell_emb = None   # np.ndarray [d_model]


def _ensure_pert_index():
    """Lazily load the perturbation retrieval index."""
    global _pert_index
    if _pert_index is not None:
        return _pert_index
    from perturbgpt.retrieval.index import PerturbationIndex
    path = Path("data/indices/perturbation_index")
    if not path.with_suffix(".faiss").exists():
        return None
    _pert_index = PerturbationIndex.load(path)
    return _pert_index


def _ensure_lit_index():
    """Lazily load the literature retrieval index."""
    global _lit_index
    if _lit_index is not None:
        return _lit_index
    try:
        from perturbgpt.rag.literature_index import build_literature_index
        path = Path("data/literature/abstracts.json")
        if path.exists():
            _lit_index = build_literature_index(path)
    except (ImportError, FileNotFoundError):
        pass
    return _lit_index


# --------------------------------------------------------- get_predicted_genes


def get_predicted_genes(
    perturbation_id: str,
    top_k: int = 20,
) -> list[dict]:
    """Run the FiLM-MLP model for a perturbation and return top-k DE genes.

    Parameters
    ----------
    perturbation_id : str
        Perturbation label (e.g. ``"KLF1"``).
    top_k : int
        Number of top genes by absolute predicted delta to return.

    Returns
    -------
    list[dict]
        Each: ``{"gene": str, "delta": float}`` sorted by ``|delta|`` desc.
        Empty list if the model or embeddings are not available.
    """
    global _model_checkpoint, _gene_emb_map, _gene_names, _ctrl_cell_emb

    try:
        import torch
        from perturbgpt.models.prediction_head import FiLMMLP, compute_perturbation_embedding
    except ImportError:
        return []

    ckpt_path = Path("runs/film_head.pt")
    if not ckpt_path.exists():
        return []

    if _model_checkpoint is None:
        _model_checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt = _model_checkpoint
    cfg = ckpt.get("model_config", {})

    if _gene_emb_map is None:
        gene_npz = np.load("data/embeddings/gene_embeddings.npz", allow_pickle=True)
        _gene_emb_map = {s: gene_npz["embeddings"][i] for i, s in enumerate(gene_npz["gene_symbols"].astype(str))}
    if _gene_names is None:
        _gene_names = ckpt.get("gene_names", [])
    if _ctrl_cell_emb is None:
        cell_npz = np.load("data/embeddings/cell_embeddings.npz", allow_pickle=True)
        _ctrl_cell_emb = cell_npz["embeddings"].mean(axis=0).astype(np.float32)

    model = FiLMMLP(
        cell_emb_dim=cfg.get("cell_emb_dim", 512),
        pert_emb_dim=cfg.get("pert_emb_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 3),
        n_hvgs=cfg.get("n_hvgs", len(_gene_names)),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    d_model = cfg.get("pert_emb_dim", 512)
    cell_feat = torch.tensor(_ctrl_cell_emb[np.newaxis, :], dtype=torch.float32)
    pert_feat = torch.tensor(
        compute_perturbation_embedding(perturbation_id, _gene_emb_map, d_model)[np.newaxis, :],
        dtype=torch.float32,
    )
    with torch.no_grad():
        delta = model(cell_feat, pert_feat).cpu().numpy()[0]

    order = np.argsort(-np.abs(delta))[:top_k]
    return [
        {"gene": _gene_names[i] if i < len(_gene_names) else f"gene_{i}", "delta": float(delta[i])}
        for i in order
    ]


# --------------------------------------------------------- find_similar_perturbations


def find_similar_perturbations(
    perturbation_id: str,
    top_k: int = 10,
    index_type: str = "perturbation",
) -> list[dict]:
    """Query the retrieval index for perturbations similar to the query.

    Parameters
    ----------
    perturbation_id : str
        Query perturbation label (must be in the index).
    top_k : int
    index_type : str
        ``"perturbation"`` (gene-embedding index) or ``"response"``
        (response-signature index).

    Returns
    -------
    list[dict]
        Each: ``{"pert_id": str, "score": float, "gene_symbols": list[str],
        "cell_count": int}``.  Empty list if the index is not available.
    """
    index = _ensure_pert_index()
    if index is None:
        return []
    if perturbation_id not in index.ids:
        return []
    results = index.query(perturbation_id, k=top_k, exclude_self=True)
    return [
        {
            "pert_id": pid,
            "score": score,
            "gene_symbols": meta.gene_symbols,
            "cell_count": meta.cell_count,
        }
        for pid, score, meta in results
    ]


# --------------------------------------------------------- get_pathways


def get_pathways(gene_symbols: list[str]) -> list[dict]:
    """Run pathway enrichment on a list of gene symbols.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene symbols (e.g. from :func:`get_predicted_genes` top-k output).

    Returns
    -------
    list[dict]
        Each: ``{"pathway": str, "p_adj": float, "genes": list[str]}``.
        Sorted by ascending adjusted p-value.  Empty list if no pathway
        is enriched or if gene sets are unavailable.
    """
    results = run_enrichment(gene_symbols)
    return [
        {"pathway": r.pathway, "p_adj": r.p_adj, "genes": r.overlap_genes}
        for r in results
    ]


# --------------------------------------------------------- retrieve_literature


def retrieve_literature(query: str, top_k: int = 5) -> list[dict]:
    """Search the literature index for relevant PubMed abstracts.

    Parameters
    ----------
    query : str
        Free-text search query (e.g. gene name + biological context).
    top_k : int
        Maximum number of results.

    Returns
    -------
    list[dict]
        Each: ``{"pmid": str, "title": str, "abstract": str,
        "score": float, "journal": str, "year": str}``.
        **Empty list** if no relevant results are found (score below
        relevance threshold) or if the index is not available.
        The agent must treat an empty list as "no supporting evidence"
        and must not fabricate citations.
    """
    index = _ensure_lit_index()
    if index is None:
        return []
    return index.retrieve_literature(query, top_k=top_k)

