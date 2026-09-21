"""Pathway enrichment (hypergeometric test) and enriched-pathway-set overlap.

Database choice: **MSigDB Hallmark (H)** — 50 well-curated, non-redundant
pathways covering major biological processes.  Small enough to load
quickly and broad enough for meaningful enrichment on HVG gene lists.

Obtaining gene sets: ``gseapy.get_library("MSigDB_Hallmark_2020")``
returns ``{pathway_name: [gene_symbols]}``.  A local JSON cache is used
as a fallback so tests can run without network access.  The gseapy
package (https://gseapy.readthedocs.io) is the maintained Python
interface to MSigDB and downloads gene-set files from the official
GSEA/MSigDB site on first use, then caches locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "pathways" / "msigdb_hallmark.json"
DEFAULT_LIBRARY = "MSigDB_Hallmark_2020"
DEFAULT_BACKGROUND = 19_000  # approx protein-coding genes


# --------------------------------------------------------- gene-set loading


def load_gene_sets(
    library: str = DEFAULT_LIBRARY,
    cache_path: Optional[Path] = None,
) -> dict[str, set[str]]:
    """Load a gene-set library as ``{pathway_name: set(gene_symbols)}``.

    Tries, in order:
      1. Local JSON cache at *cache_path* (or :data:`DEFAULT_CACHE`).
      2. ``gseapy.get_library(library)`` (downloads from MSigDB on first
         use, then caches to *cache_path* for offline runs).
    """
    if cache_path is None:
        cache_path = DEFAULT_CACHE

    if cache_path.exists():
        with cache_path.open() as fh:
            raw = json.load(fh)
        return {k: set(v) for k, v in raw.items()}

    try:
        import gseapy
        raw = gseapy.get_library(name=library, organism="Human")
        result = {k: set(v) for k, v in raw.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as fh:
            json.dump({k: sorted(v) for k, v in result.items()}, fh, indent=2)
        return result
    except ImportError:
        raise ImportError(
            "No cached gene-set file found and gseapy is not installed. "
            "Install with: pip install gseapy  (or place a JSON file at "
            f"{cache_path})"
        )


# --------------------------------------------------------- enrichment result


@dataclass
class EnrichmentResult:
    """Single pathway enrichment result."""

    pathway: str
    p_value: float
    p_adj: float
    overlap_genes: list[str]
    overlap_size: int
    pathway_size: int


# --------------------------------------------------------- enrichment test


def _benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjustment."""
    n = len(pvalues)
    if n == 0:
        return []
    order = np.argsort(pvalues)
    ranked = np.array(pvalues, dtype=np.float64)[order]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[-1] = ranked[-1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], ranked[i] * n / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(n, dtype=np.float64)
    result[order] = adjusted
    return result.tolist()


def run_enrichment(
    gene_list: list[str],
    background_size: int = DEFAULT_BACKGROUND,
    gene_sets: Optional[dict[str, set[str]]] = None,
    p_adj_threshold: float = 0.05,
) -> list[EnrichmentResult]:
    """Hypergeometric enrichment test for each gene set.

    For a gene set *S* with ``K = |S|`` genes, a query list of ``n``
    genes, and ``k = |query ∩ S|`` overlap::

        p = 1 - hypergeom.cdf(k-1, N, K, n)

    where ``N = background_size``.  P-values are Benjamini-Hochberg
    adjusted; only results with ``p_adj <= p_adj_threshold`` are returned.

    Parameters
    ----------
    gene_list : list[str]
        Query gene symbols (e.g. top-k predicted DE genes).
    background_size : int
        Total number of genes in the background universe.
    gene_sets : dict[str, set[str]], optional
        Pre-loaded gene sets.  Loaded via :func:`load_gene_sets` if omitted.
    p_adj_threshold : float

    Returns
    -------
    list[EnrichmentResult]
        Sorted by ascending p_adj.
    """
    if not gene_list:
        return []
    if gene_sets is None:
        gene_sets = load_gene_sets()

    query_set = set(gene_list)
    n = len(query_set)
    N = background_size

    raw: list[tuple[str, float, list[str], int, int]] = []
    pvalues: list[float] = []

    for pathway, pathway_genes in gene_sets.items():
        overlap = query_set & pathway_genes
        k = len(overlap)
        K = len(pathway_genes)
        if k == 0:
            continue
        p = float(stats.hypergeom.sf(k - 1, N, K, n))
        raw.append((pathway, p, sorted(overlap), k, K))
        pvalues.append(p)

    if not raw:
        return []

    adjusted = _benjamini_hochberg(pvalues)
    results = []
    for (pathway, pval, overlap, k, K), padj in zip(raw, adjusted):
        if padj <= p_adj_threshold:
            results.append(EnrichmentResult(
                pathway=pathway, p_value=pval, p_adj=padj,
                overlap_genes=overlap, overlap_size=k, pathway_size=K,
            ))
    results.sort(key=lambda r: r.p_adj)
    return results


# --------------------------------------------------------- pathway-set overlap


def enriched_pathway_names(
    gene_list: list[str],
    background_size: int = DEFAULT_BACKGROUND,
    gene_sets: Optional[dict[str, set[str]]] = None,
    p_adj_threshold: float = 0.05,
) -> set[str]:
    """Return the set of enriched pathway names for a gene list."""
    return {
        r.pathway
        for r in run_enrichment(gene_list, background_size, gene_sets, p_adj_threshold)
    }


def pathway_jaccard_overlap(
    genes_a: list[str],
    genes_b: list[str],
    background_size: int = DEFAULT_BACKGROUND,
    gene_sets: Optional[dict[str, set[str]]] = None,
    p_adj_threshold: float = 0.05,
) -> float:
    """Jaccard index between enriched-pathway sets of two gene lists.

    ``J = |E_a ∩ E_b| / |E_a ∪ E_b|``

    Returns 0.0 if neither list has any enriched pathways.

    Parameters
    ----------
    genes_a, genes_b : list[str]
        Two gene lists (e.g. predicted DE genes vs observed DE genes).

    Returns
    -------
    float
        Jaccard index in [0, 1].
    """
    if gene_sets is None:
        gene_sets = load_gene_sets()
    set_a = enriched_pathway_names(genes_a, background_size, gene_sets, p_adj_threshold)
    set_b = enriched_pathway_names(genes_b, background_size, gene_sets, p_adj_threshold)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)

