#!/usr/bin/env python3
"""Build the MSigDB Hallmark gene-set JSON used for pathway enrichment.

Writes ``{pathway: [gene_symbol, ...]}`` to
``data/pathways/msigdb_hallmark.json`` — exactly the schema
:func:`perturbgpt.eval.pathway_metrics.load_gene_sets` reads.  That file
powers both pathway enrichment in the agent and the pathway-co-membership
ground truth used to score retrieval (Recall@K / Precision@K / MRR).

Sources
-------
``gseapy`` (preferred)
    ``gseapy.get_library("MSigDB_Hallmark_2020")`` downloads the curated
    Hallmark collection on first use and caches it locally.  Requires
    ``pip install gseapy`` and network access on the first run.
``builtin`` (offline fallback)
    A small curated set shipped in
    :mod:`perturbgpt.data.builtin_pathways`, used when gseapy is missing or
    the download fails.  Clearly tagged as approximate in the metadata
    sidecar so downstream results can be interpreted correctly.

Usage
-----
    python scripts/build_pathway_sets.py
    python scripts/build_pathway_sets.py --source gseapy --output data/pathways/msigdb_hallmark.json
    python scripts/build_pathway_sets.py --source builtin   # fully offline

Outputs
-------
``<output>``          gene-set JSON (pathway -> sorted gene symbols)
``<output>.meta.json`` provenance: source, library, counts, filters, timestamp
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "pathways" / "msigdb_hallmark.json"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "perturbseq.h5ad"
DEFAULT_LIBRARY = "MSigDB_Hallmark_2020"


def fetch_gseapy(library: str) -> dict[str, set[str]]:
    """Fetch a gene-set library through gseapy (downloads/caches on first use)."""
    import gseapy

    raw = gseapy.get_library(name=library, organism="Human")
    return {str(name): {str(g) for g in genes} for name, genes in raw.items()}


def load_builtin() -> dict[str, set[str]]:
    """Load the curated offline fallback gene sets."""
    from perturbgpt.data.builtin_pathways import HALLMARK_FALLBACK

    return {str(name): {str(g) for g in genes} for name, genes in HALLMARK_FALLBACK.items()}


def filter_gene_sets(
    gene_sets: dict[str, set[str]],
    min_genes: int,
    max_genes: int,
) -> dict[str, set[str]]:
    """Drop pathways outside the ``[min_genes, max_genes]`` size range."""
    filtered: dict[str, set[str]] = {}
    for name, genes in gene_sets.items():
        clean = {g.strip() for g in genes if g and g.strip()}
        if min_genes <= len(clean) <= max_genes:
            filtered[name] = clean
    return filtered


def coverage_report(
    gene_sets: dict[str, set[str]],
    data_path: Path,
) -> dict[str, object]:
    """Report how many perturbation genes / HVGs the gene sets cover.

    Reads the AnnData in backed mode so the 60 MB expression matrix is not
    loaded into memory just to inspect ``obs``/``var``.
    """
    if not data_path.exists():
        return {"available": False}

    import anndata as ad

    from perturbgpt.data.splitting import CONTROL_LABEL, parse_perturbation

    adata = ad.read_h5ad(data_path, backed="r")
    try:
        labels = adata.obs["perturbation"].astype(str).unique().tolist()
        pert_genes: set[str] = set()
        for label in labels:
            if label == CONTROL_LABEL:
                continue
            pert_genes.update(parse_perturbation(label))
        hvg_genes = {str(g) for g in adata.var_names}
    finally:
        if adata.isbacked:
            adata.file.close()

    covered = set().union(*gene_sets.values()) if gene_sets else set()
    return {
        "available": True,
        "n_perturbation_genes": len(pert_genes),
        "n_perturbation_genes_covered": len(pert_genes & covered),
        "n_hvgs": len(hvg_genes),
        "n_hvgs_covered": len(hvg_genes & covered),
        "uncovered_perturbation_genes": sorted(pert_genes - covered),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Destination gene-set JSON (default: data/pathways/msigdb_hallmark.json).",
    )
    parser.add_argument(
        "--library", default=DEFAULT_LIBRARY,
        help=f"gseapy library name (default: {DEFAULT_LIBRARY}).",
    )
    parser.add_argument(
        "--source", choices=("auto", "gseapy", "builtin"), default="auto",
        help="'auto' tries gseapy then falls back to the builtin sets.",
    )
    parser.add_argument(
        "--min-genes", type=int, default=10,
        help="Drop pathways with fewer genes than this (default: 10).",
    )
    parser.add_argument(
        "--max-genes", type=int, default=500,
        help="Drop pathways with more genes than this (default: 500).",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Processed AnnData used only for the coverage report.",
    )
    args = parser.parse_args(argv)

    source = args.source
    raw_sets: dict[str, set[str]] = {}

    if source in ("auto", "gseapy"):
        try:
            print(f"Fetching {args.library!r} via gseapy ...")
            raw_sets = fetch_gseapy(args.library)
            source = f"gseapy:{args.library}"
            print(f"  {len(raw_sets)} pathways downloaded")
        except ImportError:
            print("  gseapy is not installed.")
            if source == "gseapy":
                print("ERROR: --source gseapy requested but gseapy is missing.")
                print("       Install with: pip install gseapy")
                return 1
        except Exception as exc:  # network / MSigDB errors -> fallback
            print(f"  gseapy fetch failed: {exc}")
            if source == "gseapy":
                return 1

    if not raw_sets or source == "builtin":
        print("Using builtin curated fallback gene sets (approximate).")
        raw_sets = load_builtin()
        source = "builtin"

    gene_sets = filter_gene_sets(raw_sets, args.min_genes, args.max_genes)
    if not gene_sets:
        print("ERROR: no gene sets survived the size filter.")
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        json.dump({name: sorted(genes) for name, genes in sorted(gene_sets.items())}, fh, indent=2)

    coverage = coverage_report(gene_sets, Path(args.data))
    meta = {
        "source": source,
        "library": args.library if source.startswith("gseapy") else None,
        "created": datetime.now(timezone.utc).isoformat(),
        "n_pathways": len(gene_sets),
        "n_unique_genes": len(set().union(*gene_sets.values())),
        "min_genes": args.min_genes,
        "max_genes": args.max_genes,
        "coverage": {k: v for k, v in coverage.items() if k != "uncovered_perturbation_genes"},
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nWrote {len(gene_sets)} pathways "
          f"({meta['n_unique_genes']} unique genes) to {output}")
    print(f"Metadata: {meta_path}")
    if coverage.get("available"):
        print(f"Coverage: {coverage['n_perturbation_genes_covered']}"
              f"/{coverage['n_perturbation_genes']} perturbation genes, "
              f"{coverage['n_hvgs_covered']}/{coverage['n_hvgs']} HVGs")
        if coverage["uncovered_perturbation_genes"]:
            missing = coverage["uncovered_perturbation_genes"]
            print(f"  not covered ({len(missing)}): {', '.join(missing[:12])}"
                  + (" ..." if len(missing) > 12 else ""))
    if source == "builtin":
        print("\nNOTE: builtin sets are approximate. Re-run with gseapy installed "
              "for publication-quality MSigDB Hallmark results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
