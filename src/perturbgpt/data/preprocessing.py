"""QC filtering, library-size normalization + log1p, and HVG selection.

Each step is a separate, individually testable function that takes an
:class:`anndata.AnnData` and returns a *new* AnnData (the input is never
modified), so steps compose freely and cell counts can be compared
before/after every filter. All functions work on dense or sparse ``.X``.
"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

#: obs columns auto-detected as doublet flags when no column is given.
STANDARD_DOUBLET_COLUMNS = (
    "predicted_doublet",
    "is_doublet",
    "doublet",
    "doublet_flag",
    "scrublet_doublet",
)


def _as_csr_float32(X) -> sparse.csr_matrix:
    if sparse.issparse(X):
        return X.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(X, dtype=np.float32))


def _n_genes_per_cell(adata: ad.AnnData) -> np.ndarray:
    """Number of detected (non-zero) genes per cell."""
    X = adata.X
    if sparse.issparse(X):
        return np.asarray(X.getnnz(axis=1)).ravel()
    return (np.asarray(X) > 0).sum(axis=1)


def _mito_pct_per_cell(adata: ad.AnnData, mito_prefix: str = "MT-") -> np.ndarray:
    """Percentage of each cell's counts mapping to mitochondrial genes."""
    X = _as_csr_float32(adata.X)
    total = np.asarray(X.sum(axis=1)).ravel()
    mito_mask = np.array([str(g).startswith(mito_prefix) for g in adata.var_names])
    if mito_mask.any():
        mito = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
    else:
        mito = np.zeros(adata.n_obs, dtype=np.float32)
    return np.where(total > 0, 100.0 * mito / np.maximum(total, 1e-12), 0.0)


def compute_qc_metrics(adata: ad.AnnData, mito_prefix: str = "MT-") -> ad.AnnData:
    """Return a copy of ``adata`` with per-cell QC metrics added to ``.obs``:

    ``total_counts`` (library size), ``n_genes`` (detected genes) and
    ``pct_mito`` (percent of counts from genes starting with ``mito_prefix``).
    """
    out = adata.copy()
    X = _as_csr_float32(out.X)
    out.obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    out.obs["n_genes"] = _n_genes_per_cell(out)
    out.obs["pct_mito"] = _mito_pct_per_cell(out, mito_prefix)
    return out


def filter_min_counts(
    adata: ad.AnnData, min_counts: int, n_counts_col: str = "total_counts"
) -> ad.AnnData:
    """Remove cells with fewer than ``min_counts`` total UMI counts.

    Uses ``adata.obs[n_counts_col]`` if present, otherwise computes the
    per-cell library size from ``.X``.
    """
    if n_counts_col in adata.obs.columns:
        n_counts = adata.obs[n_counts_col].to_numpy()
    else:
        X = adata.X
        n_counts = np.asarray(X.sum(axis=1)).ravel()
    return adata[n_counts >= min_counts].copy()


def filter_min_genes(
    adata: ad.AnnData, min_genes: int, n_genes_col: str = "n_genes"
) -> ad.AnnData:
    """Remove cells with fewer than ``min_genes`` detected genes.

    Uses ``adata.obs[n_genes_col]`` if present, otherwise computes the counts.
    """
    if n_genes_col in adata.obs.columns:
        n_genes = adata.obs[n_genes_col].to_numpy()
    else:
        n_genes = _n_genes_per_cell(adata)
    return adata[n_genes >= min_genes].copy()


def filter_mito_fraction(
    adata: ad.AnnData,
    max_mito_pct: float,
    mito_pct_col: str = "pct_mito",
    mito_prefix: str = "MT-",
) -> ad.AnnData:
    """Remove cells whose mitochondrial fraction exceeds ``max_mito_pct``.

    ``max_mito_pct`` is in percent units (e.g. 15.0 = 15%). Uses
    ``adata.obs[mito_pct_col]`` if present, otherwise computes the fraction
    from genes whose names start with ``mito_prefix``.
    """
    if mito_pct_col in adata.obs.columns:
        pct = adata.obs[mito_pct_col].to_numpy()
    else:
        pct = _mito_pct_per_cell(adata, mito_prefix)
    return adata[pct <= max_mito_pct].copy()


def filter_doublets(adata: ad.AnnData, column: Optional[str] = None) -> ad.AnnData:
    """Remove cells flagged as doublets/multiplets.

    ``column`` may hold booleans (``True`` = doublet), strings
    ("true"/"doublet"/...), or numeric per-barcode cell counts (values > 1
    are removed as multiplets). If ``column`` is None, common doublet-flag
    columns are auto-detected; if none exists the data is returned unchanged.
    """
    col = column or next(
        (c for c in STANDARD_DOUBLET_COLUMNS if c in adata.obs.columns), None
    )
    if col is None or col not in adata.obs.columns:
        return adata.copy()
    values = adata.obs[col]
    if pd.api.types.is_bool_dtype(values):
        is_doublet = values.to_numpy()
    elif pd.api.types.is_numeric_dtype(values):
        is_doublet = values.to_numpy() > 1
    else:
        is_doublet = (
            values.astype(str)
            .str.lower()
            .isin(["true", "1", "doublet", "yes"])
            .to_numpy()
        )
    return adata[~is_doublet].copy()


def filter_perturbations_by_embedding_coverage(
    adata: ad.AnnData,
    gene_symbols_with_embeddings,
    perturbation_col: str = "perturbation",
    combo_separator: str = "_",
    control_label: str = "control",
) -> tuple[ad.AnnData, dict]:
    """Remove cells whose perturbation references genes without embeddings.

    A perturbation is kept only when **every** constituent gene (split on
    ``combo_separator``) is present in ``gene_symbols_with_embeddings``.
    Control cells carry no gene perturbation, so they are always kept. Cells
    are dropped wholesale by perturbation label: if any constituent gene lacks
    an embedding, every cell for that perturbation is removed.

    Parameters
    ----------
    adata : AnnData
        Dataset to filter. Must contain a perturbation label column.
    gene_symbols_with_embeddings : iterable of str
        Gene symbols that have cached embeddings.
    perturbation_col : str
        ``obs`` column holding the perturbation label.
    combo_separator : str
        Separator used to join constituent genes in combinatorial labels.
    control_label : str
        Label for control (non-perturbed) cells.

    Returns
    -------
    (adata, stats)
        adata : AnnData
            New AnnData with unsupported perturbations removed.
        stats : dict
            ``n_perts_before`` / ``n_perts_after`` / ``n_perts_dropped`` /
            ``n_cells_dropped`` counts plus ``dropped_perturbations`` (a
            sorted list of dropped perturbation labels).
    """
    genes_with_emb = set(gene_symbols_with_embeddings)
    perts = adata.obs[perturbation_col].astype(str)
    unique_before = set(perts.unique())

    def _has_full_coverage(label: str) -> bool:
        if label == control_label:
            return True
        return all(g in genes_with_emb for g in label.split(combo_separator))

    keep = np.array([_has_full_coverage(p) for p in perts])
    out = adata[keep].copy()

    unique_after = set(out.obs[perturbation_col].astype(str).unique())
    dropped = sorted(unique_before - unique_after)

    stats = {
        "n_perts_before": len(unique_before),
        "n_perts_after": len(unique_after),
        "n_perts_dropped": len(dropped),
        "n_cells_dropped": int((~keep).sum()),
        "dropped_perturbations": dropped,
    }
    return out, stats


def normalize_total_log1p(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    counts_layer: str = "counts",
    log1p: bool = True,
) -> ad.AnnData:
    """Library-size normalize to ``target_sum`` counts per cell, then log1p.

    Raw counts are preserved in ``.layers[counts_layer]`` and each cell's
    original library size in ``.obs['total_counts']``. When ``log1p`` is True,
    returned values lie in ``[0, log1p(target_sum)]``; otherwise they lie in
    ``[0, target_sum]``. Cells with zero counts stay all-zero either way.
    """
    out = adata.copy()
    X = _as_csr_float32(out.X)
    out.layers[counts_layer] = X.copy()
    totals = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(
        np.float32(target_sum),
        totals,
        out=np.zeros_like(totals, dtype=np.float32),
        where=totals > 0,
    )
    X = X.multiply(scale[:, None]).tocsr()
    if log1p:
        X.data = np.log1p(X.data)
    out.X = X
    out.obs["total_counts"] = totals
    return out


def select_highly_variable_genes(
    adata: ad.AnnData, n_top_genes: int = 2000
) -> ad.AnnData:
    """Select the ``n_top_genes`` most dispersed genes (Seurat-style).

    Dispersion is ``variance / mean`` computed per gene on the current
    (log-normalized) matrix. Returns a new AnnData subsetted to the selected
    genes, with ``mean``, ``dispersion`` and ``highly_variable == True``
    recorded in ``.var``. Exactly ``min(n_top_genes, n_vars)`` genes are
    returned.
    """
    X = adata.X
    Xs = X if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X, dtype=np.float32))
    mean = np.asarray(Xs.mean(axis=0)).ravel()
    mean_sq = np.asarray(Xs.multiply(Xs).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0.0)
    dispersion = np.divide(var, mean, out=np.zeros_like(var), where=mean > 0)
    n_top = min(int(n_top_genes), adata.n_vars)
    top = np.sort(np.argsort(-dispersion, kind="stable")[:n_top])
    out = adata[:, top].copy()
    out.var["mean"] = mean[top]
    out.var["dispersion"] = dispersion[top]
    out.var["highly_variable"] = True
    return out

