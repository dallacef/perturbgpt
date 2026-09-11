"""MSE, MAE, Pearson/Spearman correlation, and top-k gene recovery metrics.

Each function is independently testable and accepts either 1-D arrays
(single perturbation: ``[n_genes]``) or 2-D arrays (batch of perturbations:
``[n_perts, n_genes]``). For 2-D inputs, correlation metrics are computed
per-row and then averaged.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy import stats

ArrayLike = Union[np.ndarray, list]


def _to_2d(x: ArrayLike) -> np.ndarray:
    """Coerce input to a 2-D float64 array of shape ``[n_rows, n_genes]``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def mse(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Mean squared error averaged over all elements."""
    yp, yt = _to_2d(y_pred), _to_2d(y_true)
    return float(np.mean((yp - yt) ** 2))


def mae(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Mean absolute error averaged over all elements."""
    yp, yt = _to_2d(y_pred), _to_2d(y_true)
    return float(np.mean(np.abs(yp - yt)))


def pearson_corr(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Mean per-row Pearson correlation between predicted and true deltas.

    Rows with zero variance in either array contribute 0.0 to the mean
    (Pearson is undefined for constant inputs).
    """
    yp, yt = _to_2d(y_pred), _to_2d(y_true)
    corrs = []
    for i in range(yp.shape[0]):
        if np.std(yp[i]) < 1e-12 or np.std(yt[i]) < 1e-12:
            corrs.append(0.0)
            continue
        r, _ = stats.pearsonr(yp[i], yt[i])
        corrs.append(r if not np.isnan(r) else 0.0)
    return float(np.mean(corrs))


def spearman_corr(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Mean per-row Spearman rank correlation between predicted and true deltas.

    Rows with zero variance contribute 0.0.
    """
    yp, yt = _to_2d(y_pred), _to_2d(y_true)
    corrs = []
    for i in range(yp.shape[0]):
        if np.std(yp[i]) < 1e-12 or np.std(yt[i]) < 1e-12:
            corrs.append(0.0)
            continue
        r, _ = stats.spearmanr(yp[i], yt[i])
        corrs.append(r if not np.isnan(r) else 0.0)
    return float(np.mean(corrs))


def top_k_recovery(
    y_pred: ArrayLike, y_true: ArrayLike, k: int = 50
) -> dict[str, float]:
    """Precision/recall of top-k genes by |Δx̂| vs top-k by |Δx|.

    For each row, the k genes with the largest absolute predicted delta are
    compared to the k genes with the largest absolute true delta. Precision
    and recall are both |pred_top ∩ true_top| / k (since both sets have
    exactly k elements unless k > n_genes). The returned dict has keys
    ``"precision"`` and ``"recall"``.
    """
    yp, yt = _to_2d(y_pred), _to_2d(y_true)
    n_genes = yp.shape[1]
    k_eff = min(k, n_genes)
    precisions, recalls = [], []
    for i in range(yp.shape[0]):
        pred_top = set(np.argsort(-np.abs(yp[i]))[:k_eff].tolist())
        true_top = set(np.argsort(-np.abs(yt[i]))[:k_eff].tolist())
        overlap = len(pred_top & true_top)
        precisions.append(overlap / k_eff)
        recalls.append(overlap / k_eff)
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
    }

