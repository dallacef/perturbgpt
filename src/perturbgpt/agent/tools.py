"""Implements the agent's callable tools.

Tools
-----
* ``get_predicted_genes`` — runs the trained prediction head (FiLM or MLP)
  for a perturbation and returns top-k predicted DE genes.
* ``find_similar_perturbations`` — queries the FAISS retrieval index for
  perturbations with similar gene embeddings or response signatures.
* ``get_pathways`` — runs hypergeometric pathway enrichment on a list of
  gene symbols.
* ``retrieve_literature`` — searches the PubMed FAISS index for relevant
  abstracts with PMID citations.

Each tool returns a plain ``list[dict]`` (or empty list) so the
orchestrator and agent-eval can inspect results without model-specific
types.  Artefact paths are resolved relative to the repository root rather
than the current working directory, so the tools behave identically under
pytest, the CLI scripts, and uvicorn.  A missing artefact always degrades
to an empty result — never an exception — because the agent must be able to
say "no supporting evidence" instead of crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from perturbgpt.eval.pathway_metrics import run_enrichment

logger = logging.getLogger(__name__)

#: Repository root: ``<root>/src/perturbgpt/agent/tools.py`` -> ``parents[3]``.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "film_head.pt"
DEFAULT_GENE_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings" / "gene_embeddings.npz"
DEFAULT_INDICES_DIR = PROJECT_ROOT / "data" / "indices"
DEFAULT_ABSTRACTS = PROJECT_ROOT / "data" / "literature" / "abstracts.json"
DEFAULT_LITERATURE_INDEX = PROJECT_ROOT / "data" / "literature" / "index"

# --------------------------------------------------------- module-level state
# These are lazily initialised so importing this module doesn't require
# all heavy dependencies (torch, faiss, sentence-transformers).

_pert_index = None       # retrieval.index.PerturbationIndex
_response_index = None   # retrieval.index.ResponseIndex
_lit_index = None        # rag.literature_index.LiteratureIndex
_model_bundle = None     # models.prediction_head.PredictionBundle
_gene_emb_map = None     # dict[str, np.ndarray]


# --------------------------------------------------------- lazy artefact loaders


def _ensure_gene_embeddings() -> dict[str, np.ndarray] | None:
    """Load and cache the frozen scGPT gene embeddings."""
    global _gene_emb_map
    if _gene_emb_map is not None:
        return _gene_emb_map
    if not DEFAULT_GENE_EMBEDDINGS.exists():
        return None
    # NOTE: NpzFile decompresses on *every* ``npz[key]`` access, so the
    # array must be read once before building the per-gene views.
    with np.load(DEFAULT_GENE_EMBEDDINGS, allow_pickle=True) as npz:
        embeddings = npz["embeddings"]
        symbols = npz["gene_symbols"].astype(str)
    _gene_emb_map = {str(s): embeddings[i] for i, s in enumerate(symbols)}
    return _gene_emb_map


def _ensure_model_bundle():
    """Load and cache the trained prediction-head checkpoint."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle
    if not DEFAULT_CHECKPOINT.exists():
        return None
    from perturbgpt.models.prediction_head import load_prediction_checkpoint

    _model_bundle = load_prediction_checkpoint(DEFAULT_CHECKPOINT, device="cpu")
    return _model_bundle


def _load_index(index_type: str):
    """Load a saved FAISS index of the requested kind, or ``None``."""
    if index_type == "perturbation":
        stem = DEFAULT_INDICES_DIR / "perturbation_index"
        cls_name = "PerturbationIndex"
    else:
        stem = DEFAULT_INDICES_DIR / "response_index"
        cls_name = "ResponseIndex"
    if not stem.with_suffix(".faiss").exists():
        return None
    from perturbgpt.retrieval.index import PerturbationIndex, ResponseIndex

    cls = {"PerturbationIndex": PerturbationIndex, "ResponseIndex": ResponseIndex}[cls_name]
    return cls.load(stem)


def _ensure_index(index_type: str = "perturbation"):
    """Lazily load the perturbation or response retrieval index.

    Parameters
    ----------
    index_type : str
        ``"perturbation"`` (scGPT gene-embedding index) or ``"response"``
        (response-signature index).

    Returns
    -------
    PerturbationIndex or ResponseIndex or None
        ``None`` when the index has not been built yet.
    """
    global _pert_index, _response_index
    key = str(index_type).lower()
    if key in ("perturbation", "pert"):
        if _pert_index is None:
            _pert_index = _load_index("perturbation")
        return _pert_index
    if key in ("response", "signature"):
        if _response_index is None:
            _response_index = _load_index("response")
        return _response_index
    raise ValueError(
        f"Unknown index_type {index_type!r}; expected 'perturbation' or 'response'."
    )


def load_index(index_type: str = "perturbation"):
    """Public accessor for a lazily-loaded retrieval index (``None`` if absent)."""
    return _ensure_index(index_type)


def _ensure_lit_index():
    """Lazily load (or build) the literature retrieval index."""
    global _lit_index
    if _lit_index is not None:
        return _lit_index
    try:
        from perturbgpt.rag.literature_index import LiteratureIndex, build_literature_index
    except ImportError as exc:
        logger.warning("Literature index unavailable: %s", exc)
        return None

    # Prefer a persisted index (no re-encoding cost); fall back to encoding
    # the stored abstracts on first use.
    try:
        if (
            (DEFAULT_LITERATURE_INDEX / "literature.faiss").exists()
            and hasattr(LiteratureIndex, "load")
        ):
            _lit_index = LiteratureIndex.load(DEFAULT_LITERATURE_INDEX)
            return _lit_index
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("Could not load saved literature index: %s", exc)

    if not DEFAULT_ABSTRACTS.exists():
        return None
    try:
        _lit_index = build_literature_index(DEFAULT_ABSTRACTS)
    except (ImportError, ValueError, OSError) as exc:
        logger.warning("Could not build literature index: %s", exc)
        return None
    return _lit_index


def reset_caches() -> None:
    """Drop every cached artefact so the next call reloads from disk."""
    global _pert_index, _response_index, _lit_index, _model_bundle, _gene_emb_map
    _pert_index = None
    _response_index = None
    _lit_index = None
    _model_bundle = None
    _gene_emb_map = None


# --------------------------------------------------------- get_predicted_genes


def get_predicted_genes(
    perturbation_id: str,
    top_k: int = 20,
) -> list[dict]:
    """Run the trained prediction head for a perturbation, return top-k genes.

    Works with both checkpoint architectures (FiLM and gene-MLP) because the
    checkpoint tags its architecture and :class:`PredictionBundle` dispatches
    on it.

    Parameters
    ----------
    perturbation_id : str
        Perturbation label (e.g. ``"KLF1"`` or ``"AHR_KLF1"``).
    top_k : int
        Number of top genes by absolute predicted delta to return.

    Returns
    -------
    list[dict]
        Each: ``{"gene": str, "delta": float}`` sorted by ``|delta|`` desc.
        Empty list if the checkpoint or gene embeddings are not available.
    """
    try:
        bundle = _ensure_model_bundle()
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        logger.warning("Prediction head unavailable: %s", exc)
        return []
    if bundle is None:
        return []

    gene_emb_map = _ensure_gene_embeddings()
    if gene_emb_map is None:
        return []

    try:
        delta = bundle.predict_delta([perturbation_id], gene_emb_map)[0]
    except (KeyError, ValueError, RuntimeError) as exc:
        logger.warning("Prediction failed for %s: %s", perturbation_id, exc)
        return []

    k = max(int(top_k), 0)
    order = np.argsort(-np.abs(delta))[:k]
    gene_names = bundle.gene_names
    return [
        {
            "gene": gene_names[i] if i < len(gene_names) else f"gene_{i}",
            "delta": float(delta[i]),
        }
        for i in order
    ]


# --------------------------------------------------------- find_similar_perturbations


def find_similar_perturbations(
    perturbation_id: str,
    top_k: int = 10,
    index_type: str = "perturbation",
) -> list[dict]:
    """Query a retrieval index for perturbations similar to the query.

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
    try:
        index = _ensure_index(index_type)
    except (ImportError, ValueError, OSError) as exc:
        logger.warning("Retrieval index unavailable: %s", exc)
        return []
    if index is None:
        return []
    if perturbation_id not in index.ids:
        return []

    results = index.query(perturbation_id, k=top_k, exclude_self=True)
    return [
        {
            "pert_id": pid,
            "score": float(score),
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
    try:
        results = run_enrichment(gene_symbols)
    except (ImportError, FileNotFoundError, OSError) as exc:
        logger.warning("Pathway enrichment unavailable: %s", exc)
        return []
    return [
        {"pathway": r.pathway, "p_adj": float(r.p_adj), "genes": r.overlap_genes}
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
