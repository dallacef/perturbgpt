"""Unit tests for pathway enrichment and pathway-set overlap.

Uses a small hand-crafted gene-set file so tests run offline without
gseapy or network access.  The test fixture writes a JSON cache to a
tmp directory and patches ``DEFAULT_CACHE`` to point at it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perturbgpt.eval import pathway_metrics as pm


# ── synthetic gene sets for offline testing ──────────────────────────

SYNTHETIC_GENE_SETS = {
    "PATHWAY_ERYTHROID": [
        "KLF1", "GATA1", "HBB", "HBA1", "ALAS2", "EPOR", "SLC4A1",
    ],
    "PATHWAY_HYPOXIA": [
        "HIF1A", "EPAS1", "VEGFA", "LDHA", "SLC2A1", "EGLN1",
    ],
    "PATHWAY_WNT": [
        "CTNNB1", "WNT1", "WNT3A", "AXIN2", "TCF7L2", "LGR5",
    ],
}


@pytest.fixture
def patched_gene_sets(tmp_path, monkeypatch):
    """Patch pathway_metrics to use a synthetic gene-set JSON cache."""
    cache = tmp_path / "test_gene_sets.json"
    with cache.open("w") as fh:
        json.dump(SYNTHETIC_GENE_SETS, fh)
    monkeypatch.setattr(pm, "DEFAULT_CACHE", cache)
    return pm.load_gene_sets(cache_path=cache)


# ── gene-set loading ─────────────────────────────────────────────────


def test_load_gene_sets_from_cache(tmp_path, monkeypatch):
    cache = tmp_path / "gs.json"
    with cache.open("w") as fh:
        json.dump({"P1": ["A", "B"]}, fh)
    monkeypatch.setattr(pm, "DEFAULT_CACHE", cache)
    gs = pm.load_gene_sets()
    assert gs == {"P1": {"A", "B"}}


def test_load_gene_sets_missing_no_gseapy(tmp_path, monkeypatch):
    cache = tmp_path / "nonexistent.json"
    monkeypatch.setattr(pm, "DEFAULT_CACHE", cache)
    monkeypatch.setitem(__import__("sys").modules, "gseapy", None)
    # Should raise ImportError when neither cache nor gseapy available
    with pytest.raises((ImportError, Exception)):
        pm.load_gene_sets()


# ── enrichment ───────────────────────────────────────────────────────


def test_enrichment_known_pathway(patched_gene_sets):
    """Genes from PATHWAY_ERYTHROID should enrich for that pathway."""
    query = ["KLF1", "GATA1", "HBB", "HBA1", "ALAS2"]
    results = pm.run_enrichment(
        query, background_size=100, gene_sets=patched_gene_sets, p_adj_threshold=0.5,
    )
    names = {r.pathway for r in results}
    assert "PATHWAY_ERYTHROID" in names
    erythroid = next(r for r in results if r.pathway == "PATHWAY_ERYTHROID")
    assert set(erythroid.overlap_genes) == {"KLF1", "GATA1", "HBB", "HBA1", "ALAS2"}
    assert erythroid.overlap_size == 5
    assert erythroid.pathway_size == 7


def test_enrichment_empty_list(patched_gene_sets):
    results = pm.run_enrichment([], gene_sets=patched_gene_sets)
    assert results == []


def test_enrichment_no_overlap(patched_gene_sets):
    query = ["XYZ1", "XYZ2", "XYZ3"]
    results = pm.run_enrichment(query, gene_sets=patched_gene_sets)
    assert results == []


def test_enrichment_p_value_decreasing(patched_gene_sets):
    """Results should be sorted by ascending p_adj."""
    query = ["KLF1", "GATA1", "HBB", "HIF1A", "VEGFA", "CTNNB1"]
    results = pm.run_enrichment(
        query, background_size=200, gene_sets=patched_gene_sets, p_adj_threshold=1.0,
    )
    padjs = [r.p_adj for r in results]
    assert padjs == sorted(padjs)


def test_enrichment_result_fields(patched_gene_sets):
    query = ["KLF1", "GATA1"]
    results = pm.run_enrichment(
        query, background_size=100, gene_sets=patched_gene_sets, p_adj_threshold=1.0,
    )
    assert len(results) > 0
    r = results[0]
    assert hasattr(r, "pathway")
    assert hasattr(r, "p_value")
    assert hasattr(r, "p_adj")
    assert hasattr(r, "overlap_genes")
    assert hasattr(r, "overlap_size")
    assert hasattr(r, "pathway_size")


# ── BH adjustment ─────────────────────────────────────────────────────


def test_benjamini_hochberg_single():
    assert pm._benjamini_hochberg([0.01]) == [0.01]


def test_benjamini_hochberg_empty():
    assert pm._benjamini_hochberg([]) == []


def test_benjamini_hochberg_monotonic():
    pvals = [0.01, 0.02, 0.03, 0.10]
    adj = pm._benjamini_hochberg(pvals)
    assert all(0 <= a <= 1 for a in adj)
    # Adjusted should be >= raw
    for raw, a in zip(sorted(pvals), [adj[i] for i in __import__("numpy").argsort(pvals)]):
        assert a >= raw - 1e-12


# ── pathway jaccard overlap ───────────────────────────────────────────


def test_pathway_jaccard_identical(patched_gene_sets):
    genes = ["KLF1", "GATA1", "HBB"]
    j = pm.pathway_jaccard_overlap(genes, genes, gene_sets=patched_gene_sets, p_adj_threshold=1.0)
    assert j == pytest.approx(1.0)


def test_pathway_jaccard_disjoint(patched_gene_sets):
    genes_a = ["KLF1", "GATA1", "HBB"]
    genes_b = ["CTNNB1", "WNT1", "AXIN2"]
    j = pm.pathway_jaccard_overlap(
        genes_a, genes_b, background_size=100,
        gene_sets=patched_gene_sets, p_adj_threshold=1.0,
    )
    assert j == pytest.approx(0.0)


def test_pathway_jaccard_partial_overlap(patched_gene_sets):
    genes_a = ["KLF1", "GATA1", "HBB"]
    genes_b = ["KLF1", "GATA1", "HIF1A"]
    # A enriches erythroid; B enriches erythroid + hypoxia
    j = pm.pathway_jaccard_overlap(
        genes_a, genes_b, background_size=100,
        gene_sets=patched_gene_sets, p_adj_threshold=1.0,
    )
    assert 0 < j <= 1


def test_enriched_pathway_names(patched_gene_sets):
    genes = ["KLF1", "GATA1", "HBB"]
    names = pm.enriched_pathway_names(
        genes, background_size=100, gene_sets=patched_gene_sets, p_adj_threshold=1.0,
    )
    assert "PATHWAY_ERYTHROID" in names


# ── get_pathways tool ─────────────────────────────────────────────────


def test_get_pathways_tool(patched_gene_sets, monkeypatch):
    """The agent tool get_pathways should return list[dict]."""
    from perturbgpt.agent import tools
    monkeypatch.setattr(pm, "DEFAULT_CACHE", pm.DEFAULT_CACHE)
    # Patch load_gene_sets to return our synthetic sets
    monkeypatch.setattr(tools, "run_enrichment", lambda g, **kw: pm.run_enrichment(
        g, gene_sets=patched_gene_sets, p_adj_threshold=1.0, **{k: v for k, v in kw.items() if k != "gene_sets"}))
    result = tools.get_pathways(["KLF1", "GATA1", "HBB"])
    assert isinstance(result, list)
    assert len(result) > 0
    assert "pathway" in result[0]
    assert "p_adj" in result[0]
    assert "genes" in result[0]

