#!/usr/bin/env python3
"""Build the PubMed literature retrieval index used by the agent.

Pipeline
--------
1. Select gene queries (test-split perturbation genes by default).
2. Fetch abstracts from PubMed E-utilities and store them in
   ``data/literature/abstracts.json`` (skipped when the file already exists,
   unless ``--refresh`` is given).
3. Encode the abstracts with ``sentence-transformers/all-MiniLM-L6-v2`` and
   build a FAISS IndexFlatIP over L2-normalised vectors.
4. Persist the index to ``data/literature/index/`` so the agent can load it
   without re-encoding.

Rate limits: NCBI allows ~3 requests/second without an API key and ~10/s
with one.  Set ``NCBI_API_KEY`` to use the higher rate.  ``--max-per-query``
defaults to 3 to keep a 30-gene run comfortably within the limits.

Usage
-----
    python scripts/build_literature_index.py
    python scripts/build_literature_index.py --genes-from all --max-per-query 2
    python scripts/build_literature_index.py --gene KLF1 --gene GATA1 --refresh

Outputs
-------
``data/literature/abstracts.json``  raw PubMed records
``data/literature/index/``          ``literature.faiss`` + ``literature.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from perturbgpt.data.splitting import CONTROL_LABEL, parse_perturbation  # noqa: E402
from perturbgpt.rag.literature_index import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_RELEVANCE_THRESHOLD,
    build_literature_index,
)
from perturbgpt.rag.pubmed_ingest import ingest_gene_queries, load_stored_abstracts  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "literature"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "processed" / "splits" / "perturbation_split.json"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "perturbseq.h5ad"
DEFAULT_QUERY_TEMPLATE = (
    "{gene}[Title/Abstract] AND "
    "(CRISPR OR overexpression OR knockdown OR perturbation)"
)


def _genes_from_split(split_path: Path, which: str) -> set[str]:
    """Perturbation genes belonging to the requested split(s)."""
    payload = json.loads(split_path.read_text())
    if which == "all":
        perts = set()
        for key in ("train_perts", "val_perts", "test_perts"):
            perts.update(payload.get(key, []))
    else:
        perts = set(payload.get(f"{which}_perts", []))
    genes: set[str] = set()
    for pert in perts:
        if str(pert) == CONTROL_LABEL:
            continue
        genes.update(parse_perturbation(pert))
    return genes


def _genes_from_data(data_path: Path) -> set[str]:
    """Fallback: every perturbation gene in the processed AnnData."""
    import anndata as ad

    adata = ad.read_h5ad(data_path, backed="r")
    try:
        labels = adata.obs["perturbation"].astype(str).unique().tolist()
    finally:
        if adata.isbacked:
            adata.file.close()
    genes: set[str] = set()
    for label in labels:
        if label == CONTROL_LABEL:
            continue
        genes.update(parse_perturbation(label))
    return genes


def select_genes(args) -> list[str]:
    """Resolve the gene query list from --gene / --genes-from."""
    genes: set[str] = {g.strip() for g in args.gene if g and g.strip()}
    if genes:
        return sorted(genes)

    split_path = Path(args.split)
    if split_path.exists():
        genes = _genes_from_split(split_path, args.genes_from)
        print(f"Genes from {split_path.name} ({args.genes_from} split): {len(genes)}")
    else:
        genes = _genes_from_data(Path(args.data))
        print(f"Split file not found; genes from {args.data}: {len(genes)}")

    if not genes:
        raise SystemExit("No genes selected; pass --gene or check --genes-from.")
    return sorted(genes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for abstracts.json and the saved FAISS index.",
    )
    parser.add_argument(
        "--genes-from", choices=("train", "val", "test", "all"), default="test",
        help="Which split's perturbation genes to query (default: test).",
    )
    parser.add_argument(
        "--gene", action="append", default=[],
        help="Explicit gene symbol (repeatable). Overrides --genes-from.",
    )
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--max-per-query", type=int, default=5,
        help="Max PubMed records per gene query (default: 5).",
    )
    parser.add_argument(
        "--query-template", default=DEFAULT_QUERY_TEMPLATE,
        help="PubMed query template; must contain '{gene}'.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch abstracts even if abstracts.json already exists.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--relevance-threshold", type=float, default=DEFAULT_RELEVANCE_THRESHOLD,
    )
    parser.add_argument(
        "--skip-save", action="store_true",
        help="Build and smoke-test the index without persisting it.",
    )
    args = parser.parse_args(argv)

    if "{gene}" not in args.query_template:
        print("ERROR: --query-template must contain '{gene}'.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    abstracts_path = output_dir / "abstracts.json"

    genes = select_genes(args)
    queries = [args.query_template.format(gene=g) for g in genes]
    print(f"{len(queries)} PubMed queries, max {args.max_per_query} records each")

    if abstracts_path.exists() and not args.refresh:
        records = load_stored_abstracts(abstracts_path)
        print(f"Reusing {len(records)} stored abstracts from {abstracts_path}")
        print("  (pass --refresh to re-fetch)")
    else:
        print(f"Fetching abstracts from PubMed into {abstracts_path} ...")
        records = ingest_gene_queries(
            queries, output_dir=output_dir, max_per_query=args.max_per_query,
        )
        print(f"  stored {len(records)} unique abstracts")

    if not records:
        print("ERROR: no abstracts retrieved; PubMed may be unreachable. "
              "The literature tool will return an empty list until this succeeds.")
        return 1

    print(f"\nEncoding {len(records)} abstracts with {args.model_name} ...")
    index = build_literature_index(
        abstracts_path,
        model_name=args.model_name,
        relevance_threshold=args.relevance_threshold,
    )

    # Smoke tests: the index must find a record by its own title, and a
    # nonsense query must return nothing (the no-evidence contract).
    smoke_query = records[0].title
    hits = index.retrieve_literature(smoke_query, top_k=3)
    print(f"\nSmoke query (first record title): {len(hits)} hit(s)")
    for hit in hits:
        print(f"  PMID {hit['pmid']} (score={hit['score']:.3f}) {hit['title'][:90]}")
    misses = index.retrieve_literature(
        "zxqv nonexistent synthetic construct qwerty", top_k=3
    )
    print(f"No-evidence control query: {len(misses)} hit(s) "
          f"({'OK' if not misses else 'threshold may be too low'})")

    if not args.skip_save:
        index_dir = output_dir / "index"
        index.save(index_dir)
        print(f"\nSaved literature index to {index_dir}/")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
