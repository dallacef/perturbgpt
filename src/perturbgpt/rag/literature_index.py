"""FAISS index over sentence-embedded PubMed abstracts.

Uses a general-purpose sentence-embedding model
(``sentence-transformers/all-MiniLM-L6-v2``, 384-d) kept in a **separate**
embedding space from scGPT's biological embeddings.  This ensures the
literature retrieval captures semantic text similarity, not gene-biology
similarity.

The ``retrieve_literature`` function implements an explicit **relevance
threshold**: queries with no sufficiently-relevant results return an
empty list rather than weak matches.  This is critical for the agent's
grounding behavior (Prompt 9) — the agent must distinguish "no evidence
found" from "weak evidence".
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np

from perturbgpt.rag.pubmed_ingest import PubMedRecord, load_stored_abstracts
from perturbgpt.retrieval.index import _FAISS_AVAILABLE, _require_faiss

DEFAULT_MODEL = "all-MiniLM-L6-v2"
#: Cosine-similarity cutoff below which a hit is treated as "no evidence".
#: Calibrated on the bundled PubMed corpus: off-topic / gibberish queries
#: peak at ~0.14-0.15, single gene symbols at ~0.19, and contextual
#: biology queries (e.g. "KLF1 erythroid differentiation") at ~0.30-0.45.
#: 0.20 keeps contextual matches while suppressing unrelated abstracts,
#: which is what makes the agent's no-evidence rule trustworthy.
DEFAULT_RELEVANCE_THRESHOLD = 0.20


def _require_sentence_transformers():
    """Import SentenceTransformer, raising a helpful error if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'sentence-transformers' package is required for literature "
            "retrieval. Install with: pip install sentence-transformers"
        ) from exc


class LiteratureIndex:
    """FAISS IndexFlatIP over L2-normalized sentence embeddings of abstracts.

    Parameters
    ----------
    records : list[PubMedRecord]
        Abstracts to index.
    model_name : str
        sentence-transformers model name.
    relevance_threshold : float
        Minimum cosine similarity for a result to be returned.
    """

    def __init__(
        self,
        records: list[PubMedRecord],
        model_name: str = DEFAULT_MODEL,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        if not _FAISS_AVAILABLE:
            _require_faiss()
        faiss = _require_faiss()
        SentenceTransformer = _require_sentence_transformers()

        self.records = records
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        self.model = SentenceTransformer(model_name)

        texts = [f"{r.title} {r.abstract}" for r in records]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self._normed = np.asarray(embeddings, dtype=np.float32)
        self._dim = self._normed.shape[1]

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(self._normed)


    def retrieve_literature(
        self, query: str, top_k: int = 5,
    ) -> list[dict]:
        """Retrieve top-k most relevant abstracts for a text query.

        Returns
        -------
        list[dict]
            Each: ``{"pmid": str, "title": str, "abstract": str,
            "score": float, "journal": str, "year": str}``.

            **Empty list** if all scores are below ``relevance_threshold``
            (no relevant results found).  The caller must treat an empty
            list as "no supporting evidence" and must not fabricate
            citations.
        """
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        search_k = min(top_k, len(self.records))
        scores, indices = self._index.search(q_emb, search_k)
        scores = scores[0]
        indices = indices[0]

        results: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            if score < self.relevance_threshold:
                continue
            rec = self.records[idx]
            results.append({
                "pmid": rec.pmid,
                "title": rec.title,
                "abstract": rec.abstract,
                "score": float(score),
                "journal": rec.journal,
                "year": rec.year,
            })
        return results

    # ----- save / load

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the FAISS index and abstract metadata to a directory.

        Writes::

            {path}/literature.faiss   serialised FAISS IndexFlatIP
            {path}/literature.json    records + model name + threshold

        Returns the FAISS file path.
        """
        faiss = _require_faiss()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss_path = path / "literature.faiss"
        faiss.write_index(self._index, str(faiss_path))
        payload = {
            "model_name": self.model_name,
            "relevance_threshold": float(self.relevance_threshold),
            "dim": int(self._dim),
            "records": [asdict(r) for r in self.records],
        }
        with (path / "literature.json").open("w") as fh:
            json.dump(payload, fh, indent=2)
        return faiss_path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model_name: str | None = None,
    ) -> "LiteratureIndex":
        """Load an index previously written by :meth:`save`.

        The sentence-embedding model is re-instantiated (it is needed to
        encode queries) but the abstracts are **not** re-encoded.
        """
        faiss = _require_faiss()
        SentenceTransformer = _require_sentence_transformers()
        path = Path(path)

        with (path / "literature.json").open() as fh:
            payload = json.load(fh)

        obj = cls.__new__(cls)
        obj.records = [PubMedRecord(**r) for r in payload["records"]]
        obj.model_name = model_name or payload["model_name"]
        obj.relevance_threshold = float(payload["relevance_threshold"])
        obj.model = SentenceTransformer(obj.model_name)
        obj._dim = int(payload["dim"])
        obj._index = faiss.read_index(str(path / "literature.faiss"))
        obj._normed = faiss.rev_swig_ptr(
            obj._index.get_xb(), obj._index.ntotal * obj._index.d
        ).reshape(obj._index.ntotal, obj._index.d).astype(np.float32)
        return obj


def build_literature_index(
    abstracts_path: Path = Path("data/literature/abstracts.json"),
    model_name: str = DEFAULT_MODEL,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> LiteratureIndex:
    """Load stored abstracts and build a :class:`LiteratureIndex`.

    Parameters
    ----------
    abstracts_path : Path
        Path to ``abstracts.json`` from :func:`ingest_gene_queries`.
    model_name : str
    relevance_threshold : float

    Returns
    -------
    LiteratureIndex
    """
    records = load_stored_abstracts(abstracts_path)
    if not records:
        raise ValueError(f"No abstracts found at {abstracts_path}")
    return LiteratureIndex(records, model_name=model_name, relevance_threshold=relevance_threshold)

