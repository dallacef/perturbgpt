"""FAISS-based retrieval indices over perturbation and response-signature vectors.

Two indices are built:

* **PerturbationIndex** — over L2-normalized scGPT perturbation (gene)
  embeddings.  Each perturbation's vector is the sum of its constituent
  gene embeddings (single-gene: the gene embedding itself; combo:
  element-wise sum), L2-normalized for cosine-similarity search via
  inner product on ``IndexFlatIP``.

* **ResponseIndex** — over L2-normalized predicted/observed response
  signature vectors (the model's predicted delta-x-hat, or pseudobulk
  deltas).

Both share a common ``_FAISSIndex`` base that handles normalisation,
top-k query (with optional self-exclusion), and save/load.

All ``faiss`` imports are deferred so this module is importable (and
unit-testable for the non-FAISS helper functions) even when faiss is not
installed.  Instantiating an index without faiss raises a clear
ImportError.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from perturbgpt.data.splitting import parse_perturbation


# --------------------------------------------------------- FAISS availability


def _require_faiss():
    """Import faiss, raising a helpful error if unavailable."""
    try:
        import faiss  # noqa: F401
        return faiss
    except ImportError as exc:
        raise ImportError(
            "The 'faiss' package is required for building/querying retrieval "
            "indices. Install it with: pip install faiss-cpu "
            "(or faiss-gpu for CUDA support)."
        ) from exc


try:
    import faiss as _faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# --------------------------------------------------------- helpers


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize an array along the last axis.

    Zero vectors map to zero vectors (no division-by-zero / NaN).

    Parameters
    ----------
    v : np.ndarray
        1-D ``[d]`` or 2-D ``[n, d]`` array.
    eps : float
        Numerical stability constant.

    Returns
    -------
    np.ndarray
        Same shape as *v*, float32.
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < eps:
            return np.zeros_like(v)
        return (v / norm).astype(np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (v / norms).astype(np.float32)


# --------------------------------------------------------- base FAISS index


@dataclass
class IndexMetadata:
    """Metadata associated with a single entry in a retrieval index.

    Attributes
    ----------
    pert_id : str
        Perturbation label, e.g. ``"KLF1"`` or ``"AHR_KLF1"``.
    gene_symbols : list[str]
        Constituent gene symbols of the perturbation.
    cell_count : int
        Number of cells in the dataset with this perturbation.
    extra : dict
        Additional metadata (e.g. split assignment, observed vs predicted).
    """

    pert_id: str
    gene_symbols: list[str] = field(default_factory=list)
    cell_count: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pert_id": self.pert_id,
            "gene_symbols": self.gene_symbols,
            "cell_count": self.cell_count,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndexMetadata":
        return cls(
            pert_id=d["pert_id"],
            gene_symbols=d.get("gene_symbols", []),
            cell_count=d.get("cell_count", 0),
            extra=d.get("extra", {}),
        )


class _FAISSIndex:
    """Base class for FAISS IndexFlatIP over L2-normalized vectors.

    Subclasses set ``_index_kind`` for serialization bookkeeping.
    """

    _index_kind: str = "base"

    def __init__(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[IndexMetadata]] = None,
    ):
        """Build the index.

        Parameters
        ----------
        ids : sequence of str
            Identifier for each row, position-aligned with *embeddings*.
        embeddings : np.ndarray ``[n, d]``
            Vectors to index.  Will be L2-normalized (a copy is made;
            the input is not modified).
        metadata : sequence of IndexMetadata, optional
            Per-entry metadata.  If None, empty IndexMetadata is created.
        """
        faiss = _require_faiss()
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D [n, d], got shape {embeddings.shape}"
            )
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"len(ids)={len(ids)} does not match embeddings rows={embeddings.shape[0]}"
            )

        self._ids: list[str] = list(ids)
        self._id_to_pos: dict[str, int] = {pid: i for i, pid in enumerate(self._ids)}
        self._dim: int = int(embeddings.shape[1])
        self._normed = l2_normalize(embeddings).astype(np.float32)

        if metadata is None:
            self._metadata: list[IndexMetadata] = [
                IndexMetadata(pert_id=pid) for pid in self._ids
            ]
        else:
            if len(metadata) != len(self._ids):
                raise ValueError(
                    f"len(metadata)={len(metadata)} does not match len(ids)={len(self._ids)}"
                )
            self._metadata = list(metadata)

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(self._normed)

    # ----- properties

    @property
    def n_entries(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def get_metadata(self, pert_id: str) -> IndexMetadata:
        """Return the metadata for a given perturbation ID."""
        pos = self._id_to_pos[pert_id]
        return self._metadata[pos]

    # ----- query

    def query(
        self,
        query: Union[str, np.ndarray],
        k: int = 10,
        exclude_self: bool = True,
    ) -> list[tuple[str, float, IndexMetadata]]:
        """Top-k nearest-neighbour query (cosine similarity).

        Parameters
        ----------
        query : str or np.ndarray
            If a string, it is treated as a perturbation ID already in the
            index and its stored (normalised) vector is used.  If an
            ndarray ``[d]``, it is L2-normalised on the fly.
        k : int
            Number of neighbours to return.
        exclude_self : bool
            If *query* is a perturbation ID in the index, exclude it from
            the results.  Has no effect for raw-vector queries.

        Returns
        -------
        list of (pert_id, similarity_score, metadata) tuples
            Sorted by descending similarity.  Length is ``min(k, n_entries)``
            (or ``min(k, n_entries - 1)`` when self is excluded).
        """
        if isinstance(query, str):
            if query not in self._id_to_pos:
                raise KeyError(f"perturbation ID {query!r} not in index")
            qvec = self._normed[self._id_to_pos[query]]
            self_pos = self._id_to_pos[query] if exclude_self else None
        else:
            qvec = l2_normalize(np.asarray(query, dtype=np.float32))
            self_pos = None

        qvec = np.ascontiguousarray(qvec[np.newaxis, :], dtype=np.float32)
        # Search k+1 so we can filter the self-match if needed.
        search_k = min(k + 1, self.n_entries)
        scores, indices = self._index.search(qvec, search_k)
        scores = scores[0]
        indices = indices[0]

        results: list[tuple[str, float, IndexMetadata]] = []
        for score, pos in zip(scores, indices):
            if pos < 0:
                continue
            if self_pos is not None and int(pos) == self_pos:
                continue
            results.append((self._ids[pos], float(score), self._metadata[pos]))
            if len(results) >= k:
                break
        return results


    # ----- save / load

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the index + metadata to *path* (an ``.npz`` + ``.json`` pair).

        Writes two files:
            ``{path}.faiss``  — serialised FAISS index
            ``{path}.json``   — ids, metadata, dim, kind
        """
        faiss = _require_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_path = path.with_suffix(".faiss")
        json_path = path.with_suffix(".json")

        faiss.write_index(self._index, str(faiss_path))

        payload = {
            "kind": self._index_kind,
            "dim": self._dim,
            "ids": self._ids,
            "metadata": [m.to_dict() for m in self._metadata],
        }
        with json_path.open("w") as fh:
            json.dump(payload, fh, indent=2)

        return faiss_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "_FAISSIndex":
        """Load an index previously saved with :meth:`save`.

        *path* should be the stem (without ``.faiss`` / ``.json`` suffix).
        """
        faiss = _require_faiss()
        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        json_path = path.with_suffix(".json")

        index = faiss.read_index(str(faiss_path))

        with json_path.open() as fh:
            payload = json.load(fh)

        obj = cls.__new__(cls)
        obj._index = index
        obj._dim = payload["dim"]
        obj._ids = payload["ids"]
        obj._id_to_pos = {pid: i for i, pid in enumerate(obj._ids)}
        obj._metadata = [IndexMetadata.from_dict(m) for m in payload["metadata"]]
        # Reconstruct the normalised matrix from the FAISS index for
        # query-by-ID lookups.
        obj._normed = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
        obj._normed = obj._normed.reshape(index.ntotal, index.d).astype(np.float32)
        return obj


# --------------------------------------------------------- concrete indices


class PerturbationIndex(_FAISSIndex):
    """FAISS IndexFlatIP over L2-normalized scGPT perturbation embeddings."""

    _index_kind = "perturbation"


class ResponseIndex(_FAISSIndex):
    """FAISS IndexFlatIP over L2-normalized response-signature vectors."""

    _index_kind = "response"


# --------------------------------------------------------- factory functions


def build_perturbation_index(
    gene_emb_map: dict[str, np.ndarray],
    pert_labels: Sequence[str],
    cell_counts: Optional[dict[str, int]] = None,
    emb_dim: Optional[int] = None,
) -> PerturbationIndex:
    """Build a :class:`PerturbationIndex` from cached gene embeddings.

    For each perturbation label, the constituent gene embeddings are
    summed (single-gene → the gene embedding; combo → element-wise sum).

    Parameters
    ----------
    gene_emb_map : dict[str, np.ndarray]
        Mapping gene symbol → embedding ``[d_model]``.
    pert_labels : sequence of str
        Unique perturbation labels to index (e.g. ``["KLF1", "AHR_KLF1"]``).
        ``"control"`` is skipped automatically.
    cell_counts : dict[str, int], optional
        Mapping perturbation label → number of cells.  If None, ``0`` is
        stored for every entry.
    emb_dim : int, optional
        Embedding dimensionality.  Inferred from *gene_emb_map* if omitted.

    Returns
    -------
    PerturbationIndex
    """
    if emb_dim is None:
        if gene_emb_map:
            emb_dim = next(iter(gene_emb_map.values())).shape[0]
        else:
            raise ValueError("Cannot infer emb_dim from empty gene_emb_map")

    if cell_counts is None:
        cell_counts = {}

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    metadata: list[IndexMetadata] = []

    for label in sorted(set(str(p) for p in pert_labels)):
        genes = parse_perturbation(label)
        if not genes:  # skip "control"
            continue
        emb = np.zeros(emb_dim, dtype=np.float32)
        matched: list[str] = []
        for g in genes:
            if g in gene_emb_map:
                emb = emb + gene_emb_map[g]
                matched.append(g)
        ids.append(label)
        vectors.append(emb)
        metadata.append(IndexMetadata(
            pert_id=label,
            gene_symbols=genes,
            cell_count=cell_counts.get(label, 0),
            extra={"matched_genes": matched},
        ))

    if not vectors:
        raise ValueError(
            "No valid perturbations found to index. Check pert_labels and "
            "gene_emb_map coverage."
        )

    embeddings = np.stack(vectors).astype(np.float32)
    return PerturbationIndex(ids, embeddings, metadata)


def build_response_index(
    pert_ids: Sequence[str],
    response_vectors: np.ndarray,
    cell_counts: Optional[dict[str, int]] = None,
    gene_symbols_map: Optional[dict[str, list[str]]] = None,
) -> ResponseIndex:
    """Build a :class:`ResponseIndex` from predicted/observed response signatures.

    Parameters
    ----------
    pert_ids : sequence of str
        Perturbation labels, position-aligned with *response_vectors*.
    response_vectors : np.ndarray ``[n, n_hvgs]``
        Predicted Δx̂ or pseudobulk delta vectors.
    cell_counts : dict[str, int], optional
        Perturbation label → cell count.
    gene_symbols_map : dict[str, list[str]], optional
        Perturbation label → constituent gene symbols.

    Returns
    -------
    ResponseIndex
    """
    if cell_counts is None:
        cell_counts = {}
    if gene_symbols_map is None:
        gene_symbols_map = {}

    metadata = [
        IndexMetadata(
            pert_id=str(pid),
            gene_symbols=gene_symbols_map.get(str(pid), parse_perturbation(str(pid))),
            cell_count=cell_counts.get(str(pid), 0),
            extra={"source": "response"},
        )
        for pid in pert_ids
    ]

    return ResponseIndex(
        ids=[str(p) for p in pert_ids],
        embeddings=np.asarray(response_vectors, dtype=np.float32),
        metadata=metadata,
    )


