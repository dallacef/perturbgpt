"""Sanity tests for FAISS index build/query correctness and retrieval metrics.

Tests use a small synthetic embedding set with known, constructed nearest
neighbors to verify the index returns the correct top-k.  Metric functions
(Recall@K, Precision@K, MRR) and pathway co-membership ground-truth
construction are tested with hand-computed toy examples.

FAISS-dependent tests are skipped when faiss is not installed; pure-Python
helpers (l2_normalize, metrics, co-membership) always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbgpt.retrieval.index import (
    _FAISS_AVAILABLE,
    IndexMetadata,
    PerturbationIndex,
    ResponseIndex,
    l2_normalize,
)
from perturbgpt.retrieval.metrics import (
    build_pathway_co_membership,
    get_pathway_memberships,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

# ============================================================== constants

DIM = 8

# Cluster A: pert_0, pert_1, pert_2 — vectors near e0 = [1,0,0,...]
# Cluster B: pert_3, pert_4, pert_5 — vectors near e7 = [0,...,0,1]
PERT_IDS = [f"pert_{i}" for i in range(6)]


def _make_synthetic_embeddings() -> np.ndarray:
    """6 vectors in 2 clusters of 3."""
    e0 = np.zeros(DIM, dtype=np.float32)
    e0[0] = 1.0
    e7 = np.zeros(DIM, dtype=np.float32)
    e7[7] = 1.0

    rng = np.random.RandomState(42)
    vecs = []
    for i in range(3):
        v = e0 + 0.01 * rng.randn(DIM).astype(np.float32)
        vecs.append(v)
    for i in range(3):
        v = e7 + 0.01 * rng.randn(DIM).astype(np.float32)
        vecs.append(v)
    return np.stack(vecs)


def _make_synthetic_metadata() -> list[IndexMetadata]:
    return [
        IndexMetadata(
            pert_id=PERT_IDS[i],
            gene_symbols=[f"GENE_{i}"],
            cell_count=100 + i,
            extra={"cluster": "A" if i < 3 else "B"},
        )
        for i in range(6)
    ]


# ====================================================== l2_normalize tests


def test_l2_normalize_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(v)
    assert normed.shape == (2,)
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-6)


def test_l2_normalize_2d():
    v = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normed = l2_normalize(v)
    assert normed.shape == (2, 2)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_l2_normalize_zero_vector():
    v = np.zeros(DIM, dtype=np.float32)
    normed = l2_normalize(v)
    assert normed.shape == (DIM,)
    assert np.all(normed == 0.0)
    assert not np.any(np.isnan(normed))


def test_l2_normalize_zero_row_2d():
    v = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(v)
    assert np.all(normed[0] == 0.0)
    assert np.isclose(np.linalg.norm(normed[1]), 1.0, atol=1e-6)


# ====================================================== metric tests


def test_recall_at_k_hand_computed():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C", "E"}
    # top-3: [A, B, C], hits={A,C}=2, min(3,3)=3 → 2/3
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_all_relevant_found():
    retrieved = ["A", "C", "B"]
    relevant = {"A", "C"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_precision_at_k_hand_computed():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C", "E"}
    # top-3: [A, B, C], hits=2, k=3 → 2/3
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_no_hits():
    retrieved = ["X", "Y", "Z"]
    relevant = {"A"}
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_mrr_hand_computed():
    retrieved = ["B", "D", "A", "C"]
    relevant = {"A"}
    # first hit at rank 3 → 1/3
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_mrr_first_position():
    retrieved = ["A", "B", "C"]
    relevant = {"A"}
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)


def test_mrr_no_relevant_found():
    retrieved = ["B", "D", "C"]
    relevant = {"A"}
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.0)


def test_recall_empty_relevant():
    assert recall_at_k(["A", "B"], set(), k=2) == 0.0


def test_precision_empty_relevant():
    assert precision_at_k(["A", "B"], set(), k=2) == 0.0


def test_mrr_empty_relevant():
    assert mean_reciprocal_rank(["A", "B"], set()) == 0.0


# ================================================== pathway / co-membership


def test_pathway_stub_returns_empty():
    assert get_pathway_memberships() == {}


def test_co_membership_stub_all_empty():
    perts = ["KLF1", "AHR", "CEBPA"]
    gt = build_pathway_co_membership(perts)
    for p in perts:
        assert gt[p] == set()


def test_co_membership_with_injected_pathways():
    perts = ["KLF1", "AHR", "CEBPA", "FOXA1"]
    # KLF1 and AHR share "pathway_X"; CEBPA and FOXA1 share "pathway_Y"
    memberships = {
        "KLF1": {"pathway_X"},
        "AHR": {"pathway_X", "pathway_Z"},
        "CEBPA": {"pathway_Y"},
        "FOXA1": {"pathway_Y"},
    }
    gt = build_pathway_co_membership(perts, pathway_memberships=memberships)
    assert gt["KLF1"] == {"AHR"}
    assert gt["AHR"] == {"KLF1"}
    assert gt["CEBPA"] == {"FOXA1"}
    assert gt["FOXA1"] == {"CEBPA"}


def test_co_membership_combo_perturbation():
    perts = ["KLF1", "AHR_KLF1", "CEBPA"]
    memberships = {
        "KLF1": {"pathway_X"},
        "AHR": {"pathway_X"},
        "CEBPA": {"pathway_Y"},
    }
    gt = build_pathway_co_membership(perts, pathway_memberships=memberships)
    # KLF1 shares pathway_X with AHR_KLF1 (via AHR)
    assert "AHR_KLF1" in gt["KLF1"]
    assert "KLF1" in gt["AHR_KLF1"]
    # CEBPA shares nothing
    assert gt["CEBPA"] == set()


def test_co_membership_no_pathway_for_gene():
    perts = ["KLF1", "AHR"]
    memberships = {"KLF1": {"pathway_X"}}  # AHR has no pathway
    gt = build_pathway_co_membership(perts, pathway_memberships=memberships)
    assert gt["KLF1"] == set()  # AHR has no pathways, so no overlap
    assert gt["AHR"] == set()


# ============================================ FAISS index tests (skip if no faiss)

skip_no_faiss = pytest.mark.skipif(
    not _FAISS_AVAILABLE, reason="faiss not installed"
)


@pytest.fixture
def pert_index():
    """PerturbationIndex over the synthetic 6-vector dataset."""
    if not _FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    embs = _make_synthetic_embeddings()
    meta = _make_synthetic_metadata()
    return PerturbationIndex(PERT_IDS, embs, meta)


@pytest.fixture
def response_index():
    """ResponseIndex with different-dim synthetic vectors."""
    if not _FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    # Use 20-dim response vectors mimicking n_hvgs
    rng = np.random.RandomState(123)
    e_a = np.zeros(20, dtype=np.float32)
    e_a[:5] = 1.0
    e_b = np.zeros(20, dtype=np.float32)
    e_b[15:] = 1.0
    vecs = []
    for i in range(3):
        vecs.append(e_a + 0.01 * rng.randn(20).astype(np.float32))
    for i in range(3):
        vecs.append(e_b + 0.01 * rng.randn(20).astype(np.float32))
    embs = np.stack(vecs)
    meta = _make_synthetic_metadata()
    return ResponseIndex(PERT_IDS, embs, meta)


@skip_no_faiss
def test_pert_index_query_top_k(pert_index):
    """Querying pert_0 (cluster A) should return the other 2 cluster-A members."""
    results = pert_index.query("pert_0", k=2, exclude_self=True)
    assert len(results) == 2
    returned_ids = {r[0] for r in results}
    assert returned_ids == {"pert_1", "pert_2"}


@skip_no_faiss
def test_pert_index_query_cluster_b(pert_index):
    """Querying pert_3 (cluster B) should return pert_4 and pert_5."""
    results = pert_index.query("pert_3", k=2, exclude_self=True)
    assert len(results) == 2
    returned_ids = {r[0] for r in results}
    assert returned_ids == {"pert_4", "pert_5"}


@skip_no_faiss
def test_response_index_query_top_k(response_index):
    """Response index should also return same-cluster neighbours."""
    results = response_index.query("pert_0", k=2, exclude_self=True)
    assert len(results) == 2
    returned_ids = {r[0] for r in results}
    assert returned_ids == {"pert_1", "pert_2"}


@skip_no_faiss
def test_index_excludes_self(pert_index):
    """query(pert_id) must not include pert_id in results."""
    for pid in PERT_IDS:
        results = pert_index.query(pid, k=5, exclude_self=True)
        returned_ids = [r[0] for r in results]
        assert pid not in returned_ids, f"self-match {pid!r} appeared in results"


@skip_no_faiss
def test_index_metadata_keys(pert_index):
    results = pert_index.query("pert_0", k=2, exclude_self=True)
    for pid, score, meta in results:
        assert isinstance(pid, str)
        assert isinstance(score, float)
        assert isinstance(meta, IndexMetadata)
        assert hasattr(meta, "gene_symbols")
        assert hasattr(meta, "cell_count")
        assert meta.cell_count > 0


@skip_no_faiss
def test_index_scores_descending(pert_index):
    results = pert_index.query("pert_0", k=5, exclude_self=True)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


@skip_no_faiss
def test_index_n_entries_and_dim(pert_index):
    assert pert_index.n_entries == 6
    assert pert_index.dim == DIM


@skip_no_faiss
def test_index_query_by_vector(pert_index):
    """Query with a raw vector should also work (not just by ID)."""
    qvec = _make_synthetic_embeddings()[0]
    results = pert_index.query(qvec, k=2, exclude_self=False)
    # With exclude_self=False for a raw vector, the top hit should be pert_0
    assert results[0][0] == "pert_0"


@skip_no_faiss
def test_index_query_unknown_id_raises(pert_index):
    with pytest.raises(KeyError):
        pert_index.query("nonexistent", k=2)


@skip_no_faiss
def test_index_save_load_roundtrip(tmp_path, pert_index):
    """save() then load() should produce an index with identical query results."""
    path = tmp_path / "test_index"
    pert_index.save(path)

    loaded = PerturbationIndex.load(path)
    assert loaded.n_entries == pert_index.n_entries
    assert loaded.dim == pert_index.dim
    assert loaded.ids == pert_index.ids

    original_results = pert_index.query("pert_0", k=5, exclude_self=True)
    loaded_results = loaded.query("pert_0", k=5, exclude_self=True)
    assert [r[0] for r in original_results] == [r[0] for r in loaded_results]
    for (o_id, o_score, _), (l_id, l_score, _) in zip(original_results, loaded_results):
        assert o_id == l_id
        assert np.isclose(o_score, l_score, atol=1e-5)


@skip_no_faiss
def test_index_metadata_roundtrip(tmp_path, pert_index):
    path = tmp_path / "test_index"
    pert_index.save(path)
    loaded = PerturbationIndex.load(path)
    for pid in PERT_IDS:
        orig_meta = pert_index.get_metadata(pid)
        load_meta = loaded.get_metadata(pid)
        assert orig_meta.pert_id == load_meta.pert_id
        assert orig_meta.gene_symbols == load_meta.gene_symbols
        assert orig_meta.cell_count == load_meta.cell_count


@skip_no_faiss
def test_index_construction_mismatched_lengths():
    with pytest.raises(ValueError, match="does not match"):
        PerturbationIndex(["a", "b"], np.zeros((3, DIM), dtype=np.float32))


@skip_no_faiss
def test_index_construction_wrong_ndim():
    with pytest.raises(ValueError, match="must be 2-D"):
        PerturbationIndex(["a"], np.zeros(DIM, dtype=np.float32))


# ============================================ evaluate_retrieval & random baseline


@skip_no_faiss
def test_evaluate_retrieval_structure(pert_index):
    from perturbgpt.retrieval.metrics import evaluate_retrieval

    gt = {pid: set() for pid in PERT_IDS}
    results = evaluate_retrieval(pert_index, gt, k_values=[2, 5])
    assert set(results.keys()) == {2, 5}
    for k, kres in results.items():
        assert set(kres.keys()) == {"recall", "precision", "mrr"}
        # Empty ground truth → all metrics 0.0
        assert kres["recall"] == 0.0
        assert kres["precision"] == 0.0
        assert kres["mrr"] == 0.0


@skip_no_faiss
def test_evaluate_retrieval_with_ground_truth(pert_index):
    from perturbgpt.retrieval.metrics import evaluate_retrieval

    # Ground truth: cluster A members are neighbours of each other
    gt = {
        "pert_0": {"pert_1", "pert_2"},
        "pert_1": {"pert_0", "pert_2"},
        "pert_2": {"pert_0", "pert_1"},
        "pert_3": {"pert_4", "pert_5"},
        "pert_4": {"pert_3", "pert_5"},
        "pert_5": {"pert_3", "pert_4"},
    }
    results = evaluate_retrieval(pert_index, gt, k_values=[2])
    # At K=2 with perfect clustering, recall should be 1.0
    assert results[2]["recall"] == pytest.approx(1.0)
    assert results[2]["precision"] == pytest.approx(1.0)


def test_random_baseline_metrics():
    from perturbgpt.retrieval.metrics import random_baseline_metrics

    results = random_baseline_metrics(n_entries=11, k_values=[5, 10])
    # pool = 10, K=5 → 5/10 = 0.5
    assert results[5]["recall"] == pytest.approx(0.5)
    assert results[5]["precision"] == pytest.approx(0.5)
    # K=10 → 10/10 = 1.0
    assert results[10]["recall"] == pytest.approx(1.0)
    # MRR ≈ 1/11
    assert results[5]["mrr"] == pytest.approx(1 / 11)


def test_random_baseline_metrics_small_n():
    from perturbgpt.retrieval.metrics import random_baseline_metrics

    results = random_baseline_metrics(n_entries=2, k_values=[5])
    # pool=1, K clamped to 1 → 1/1 = 1.0
    assert results[5]["recall"] == pytest.approx(1.0)

