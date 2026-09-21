"""Unit tests for QC filtering, normalization, and HVG selection.

All tests use small synthetic AnnData fixtures with deterministic structure;
the real Perturb-seq dataset is not required to run them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from perturbgpt.data import preprocessing as pp

GENE_NAMES = ["MT-ND1", "MT-ND2"] + [f"GENE{i}" for i in range(2, 10)]


def _fixture_counts() -> np.ndarray:
    """8 cells x 10 genes with known QC behaviour.

    * cells 0-4: healthy (8 detected genes, no mitochondrial counts)
    * cell 5: only 1 detected gene        -> fails a min-genes filter
    * cell 6: ~97% mitochondrial counts   -> fails a mito-fraction filter
    * cell 7: healthy but flagged doublet -> fails doublet removal
    """
    rng = np.random.default_rng(0)
    X = np.zeros((8, 10), dtype=np.float32)
    X[:5, 2:] = rng.poisson(5, size=(5, 8))
    X[5, 5] = 3
    X[6, 0:2] = 100
    X[6, 2:8] = 1
    X[7, 2:] = rng.poisson(5, size=8)
    return X


@pytest.fixture
def adata() -> AnnData:
    obs = pd.DataFrame(
        {"predicted_doublet": [False] * 7 + [True]},
        index=[f"cell{i}" for i in range(8)],
    )
    var = pd.DataFrame(index=GENE_NAMES)
    return AnnData(sparse.csr_matrix(_fixture_counts()), obs=obs, var=var)


# ---------------------------------------------------------------- QC metrics


def test_compute_qc_metrics(adata):
    out = pp.compute_qc_metrics(adata)
    assert out.obs.loc["cell5", "n_genes"] == 1
    assert out.obs.loc["cell5", "total_counts"] == 3
    assert out.obs.loc["cell6", "pct_mito"] == pytest.approx(100.0 * 200 / 206)
    assert out.obs.loc["cell0", "pct_mito"] == 0.0
    # input is not modified
    assert "n_genes" not in adata.obs.columns


# ------------------------------------------------------------- QC filtering


def test_filter_min_genes_removes_expected_cells(adata):
    out = pp.filter_min_genes(adata, min_genes=5)
    assert out.n_obs == 7
    assert "cell5" not in out.obs_names


def test_filter_min_genes_uses_precomputed_column(adata):
    # values in the named column take precedence over recomputation
    adata.obs["n_genes"] = [10, 10, 10, 10, 10, 10, 10, 1]
    out = pp.filter_min_genes(adata, min_genes=5, n_genes_col="n_genes")
    assert out.n_obs == 7
    assert "cell7" not in out.obs_names


def test_filter_min_counts_removes_expected_cells(adata):
    out = pp.filter_min_counts(adata, min_counts=10)
    assert out.n_obs == 7
    assert "cell5" not in out.obs_names


def test_filter_min_counts_uses_precomputed_column(adata):
    # values in the named column take precedence over recomputation
    adata.obs["total_counts"] = [50, 50, 50, 50, 50, 50, 50, 1]
    out = pp.filter_min_counts(adata, min_counts=10, n_counts_col="total_counts")
    assert out.n_obs == 7
    assert "cell7" not in out.obs_names


def test_filter_mito_fraction_removes_expected_cells(adata):
    out = pp.filter_mito_fraction(adata, max_mito_pct=15.0)
    assert out.n_obs == 7
    assert "cell6" not in out.obs_names


def test_filter_doublets_removes_flagged_cells(adata):
    out = pp.filter_doublets(adata)  # auto-detects 'predicted_doublet'
    assert out.n_obs == 7
    assert "cell7" not in out.obs_names


def test_filter_doublets_numeric_multiplet_counts(adata):
    adata.obs["number_of_cells"] = [1, 1, 1, 1, 1, 1, 1, 3]
    out = pp.filter_doublets(adata, column="number_of_cells")
    assert out.n_obs == 7
    assert "cell7" not in out.obs_names


def test_filter_doublets_no_column_is_noop(adata):
    adata_noflag = adata[:, :].copy()
    del adata_noflag.obs["predicted_doublet"]
    out = pp.filter_doublets(adata_noflag)
    assert out.n_obs == adata.n_obs


def test_qc_filters_chain_to_expected_cells(adata):
    out = pp.filter_min_counts(adata, min_counts=10)
    out = pp.filter_min_genes(out, min_genes=5)
    out = pp.filter_mito_fraction(out, max_mito_pct=15.0)
    out = pp.filter_doublets(out)
    assert list(out.obs_names) == [f"cell{i}" for i in range(5)]
    assert adata.n_obs == 8  # original untouched




# ------------------------------------------------------------- normalization


@pytest.fixture
def norm_adata() -> AnnData:
    X = np.array(
        [
            [10, 0, 0, 0, 0],
            [5, 5, 0, 0, 0],
            [0, 0, 0, 0, 0],  # empty cell must not produce NaNs
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(4)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
    return AnnData(X.copy(), obs=obs, var=var)


def test_normalize_total_log1p_value_ranges(norm_adata):
    out = pp.normalize_total_log1p(norm_adata, target_sum=1e4)
    X = out.X.toarray() if sparse.issparse(out.X) else np.asarray(out.X)
    assert np.all(X >= 0)
    assert not np.isnan(X).any()
    # a cell whose counts are all in one gene saturates at log1p(target_sum)
    assert X[0, 0] == pytest.approx(np.log1p(1e4), rel=1e-6)
    assert X.max() <= np.log1p(1e4) + 1e-6
    # reversing log1p recovers the target library size for non-empty cells
    row_sums = np.expm1(X).sum(axis=1)
    assert row_sums[0] == pytest.approx(1e4, rel=1e-4)
    assert row_sums[1] == pytest.approx(1e4, rel=1e-4)
    assert row_sums[3] == pytest.approx(1e4, rel=1e-4)
    assert row_sums[2] == 0.0  # empty cell stays zero


def test_normalize_total_log1p_preserves_counts_layer(norm_adata):
    out = pp.normalize_total_log1p(norm_adata, target_sum=1e4)
    counts = out.layers["counts"]
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    np.testing.assert_array_equal(counts, norm_adata.X)
    assert out.obs["total_counts"].tolist() == [10.0, 10.0, 0.0, 5.0]


# ---------------------------------------------------------------------- HVGs


@pytest.fixture
def hvg_adata() -> AnnData:
    """50 cells x 30 genes; genes g0-g4 are engineered to be most dispersed."""
    rng = np.random.default_rng(42)
    X = rng.poisson(2, size=(50, 30)).astype(np.float32)
    for j in range(5):
        X[:25, j] = 50.0  # half the cells high -> high variance/dispersion
    obs = pd.DataFrame(index=[f"c{i}" for i in range(50)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(30)])
    return AnnData(sparse.csr_matrix(X), obs=obs, var=var)


def test_hvg_selection_returns_requested_count(hvg_adata):
    out = pp.select_highly_variable_genes(hvg_adata, n_top_genes=10)
    assert out.n_vars == 10
    assert out.n_obs == hvg_adata.n_obs
    assert out.var["highly_variable"].all()


def test_hvg_selection_picks_most_dispersed_genes(hvg_adata):
    out = pp.select_highly_variable_genes(hvg_adata, n_top_genes=10)
    assert {f"g{i}" for i in range(5)} <= set(out.var_names)


def test_hvg_selection_caps_at_n_vars(hvg_adata):
    out = pp.select_highly_variable_genes(hvg_adata, n_top_genes=999)
    assert out.n_vars == hvg_adata.n_vars

