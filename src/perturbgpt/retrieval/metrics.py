"""Recall@K, Precision@K, and mean reciprocal rank for retrieval evaluation.

Ground truth is defined by **pathway co-membership**: a perturbation's
"true positive" neighbors are other perturbations whose target gene(s)
share at least one curated gene-set / pathway membership with the query
perturbation's target gene(s).

The pathway-membership lookup (``get_pathway_memberships``) is currently a
**stub** that returns an empty mapping.  Prompt 7 will replace it with a
real implementation backed by MSigDB / Reactome / KEGG.
"""

from __future__ import annotations

from typing import Optional, Sequence

from perturbgpt.data.splitting import parse_perturbation


# --------------------------------------------------------- ground-truth stub


def get_pathway_memberships() -> dict[str, set[str]]:
    """Return a mapping gene_symbol → set of pathway/gene-set names.

    .. note::

       **Stub implementation.**  Returns an empty dict so that all
       co-membership ground-truth sets are empty and all metrics evaluate
       to 0.0.  Prompt 7 will replace this with a real curated pathway
       lookup (MSigDB C2/Reactome/KEGG).

    Returns
    -------
    dict[str, set[str]]
        Gene symbol → pathway names.  Empty in the stub.
    """
    return {}


# --------------------------------------------------------- ground-truth build


def build_pathway_co_membership(
    pert_ids: Sequence[str],
    pathway_memberships: Optional[dict[str, set[str]]] = None,
) -> dict[str, set[str]]:
    """Build the ground-truth neighbour sets from pathway co-membership.

    For each perturbation *p*, the relevant (true-positive) set is every
    other perturbation *q* (``q != p``) such that at least one constituent
    gene of *p* and at least one constituent gene of *q* share a pathway.

    Parameters
    ----------
    pert_ids : sequence of str
        Perturbation labels in the index.
    pathway_memberships : dict[str, set[str]], optional
        Gene symbol → pathway names.  Defaults to
        :func:`get_pathway_memberships` (the stub).

    Returns
    -------
    dict[str, set[str]]
        pert_id → set of pert_ids that are true-positive neighbours.
    """
    if pathway_memberships is None:
        pathway_memberships = get_pathway_memberships()

    pert_ids = [str(p) for p in pert_ids]

    # Pre-compute the union of pathways for each perturbation.
    pert_pathways: dict[str, set[str]] = {}
    for pid in pert_ids:
        genes = parse_perturbation(pid)
        paths: set[str] = set()
        for g in genes:
            paths |= pathway_memberships.get(g, set())
        pert_pathways[pid] = paths

    ground_truth: dict[str, set[str]] = {}
    for pid in pert_ids:
        gt: set[str] = set()
        p_paths = pert_pathways[pid]
        if p_paths:  # if this pert has no pathway memberships, GT is empty
            for other in pert_ids:
                if other == pid:
                    continue
                if p_paths & pert_pathways[other]:
                    gt.add(other)
        ground_truth[pid] = gt

    return ground_truth


# --------------------------------------------------------- ranking metrics


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant items in the top-k retrieved.

    .. math::
        \\text{Recall@K} = \\frac{|\\text{retrieved[:K]} \\cap \\text{relevant}|}{\\min(K, |\\text{relevant}|)}

    Returns 0.0 when *relevant_ids* is empty.
    """
    if not relevant_ids:
        return 0.0
    retrieved_top_k = [str(r) for r in retrieved_ids[:k]]
    hits = sum(1 for r in retrieved_top_k if r in relevant_ids)
    return hits / min(k, len(relevant_ids))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Precision@K: fraction of the top-k retrieved that are relevant.

    .. math::
        \\text{Precision@K} = \\frac{|\\text{retrieved[:K]} \\cap \\text{relevant}|}{K}

    Returns 0.0 when *relevant_ids* is empty.
    """
    if not relevant_ids:
        return 0.0
    retrieved_top_k = [str(r) for r in retrieved_ids[:k]]
    hits = sum(1 for r in retrieved_top_k if r in relevant_ids)
    return hits / k


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant item in the ranked list.

    .. math::
        \\text{MRR} = \\frac{1}{\\text{rank of first hit}}

    Returns 0.0 when *relevant_ids* is empty or no relevant item is found.
    """
    if not relevant_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if str(rid) in relevant_ids:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------- batch evaluation


def evaluate_retrieval(
    index,
    ground_truth: dict[str, set[str]],
    k_values: Sequence[int] = (5, 10, 20),
    exclude_self: bool = True,
) -> dict[int, dict[str, float]]:
    """Evaluate a retrieval index against ground-truth neighbour sets.

    Queries every perturbation ID in *index*, computes Recall@K,
    Precision@K, and MRR for each *k*, and averages over all queries.

    Parameters
    ----------
    index : _FAISSIndex
        A :class:`~perturbgpt.retrieval.index.PerturbationIndex` or
        :class:`~perturbgpt.retrieval.index.ResponseIndex`.
    ground_truth : dict[str, set[str]]
        pert_id → set of true-positive pert_ids (from
        :func:`build_pathway_co_membership`).
    k_values : sequence of int
        K cutoffs at which to compute Recall and Precision.
    exclude_self : bool
        Whether to exclude the query perturbation from its own results.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping k → {"recall": float, "precision": float, "mrr": float}.
    """
    pert_ids = index.ids
    results: dict[int, dict[str, float]] = {}

    for k in k_values:
        recalls: list[float] = []
        precisions: list[float] = []
        mrrs: list[float] = []

        for pid in pert_ids:
            relevant = ground_truth.get(pid, set())
            retrieved = [r[0] for r in index.query(pid, k=k, exclude_self=exclude_self)]
            recalls.append(recall_at_k(retrieved, relevant, k))
            precisions.append(precision_at_k(retrieved, relevant, k))
            mrrs.append(mean_reciprocal_rank(retrieved, relevant))

        n = len(pert_ids) if pert_ids else 1
        results[k] = {
            "recall": sum(recalls) / n,
            "precision": sum(precisions) / n,
            "mrr": sum(mrrs) / n,
        }

    return results


def random_baseline_metrics(
    n_entries: int,
    k_values: Sequence[int] = (5, 10, 20),
) -> dict[int, dict[str, float]]:
    """Analytical expected metrics for uniform random retrieval.

    With *N* items and *N - 1* candidates per query (self excluded),
    the expected fraction of relevant items in a random top-K is
    ``K / (N - 1)`` — this holds for both Recall@K (when the relevant
    set is large enough) and Precision@K.  The expected MRR for a
    random permutation with at least one relevant item is
    ``1 / N`` (harmonic mean approximation).

    Parameters
    ----------
    n_entries : int
        Number of items in the index.
    k_values : sequence of int
        K cutoffs.

    Returns
    -------
    dict[int, dict[str, float]]
    """
    n = max(n_entries, 2)
    pool = n - 1  # self excluded
    results: dict[int, dict[str, float]] = {}
    for k in k_values:
        k_eff = min(k, pool)
        rate = k_eff / pool
        results[k] = {
            "recall": rate,
            "precision": rate,
            "mrr": 1.0 / n,
        }
    return results

