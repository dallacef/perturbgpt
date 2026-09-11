"""Unit tests for dataset loading and obs-schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from anndata import AnnData

from perturbgpt.data.loading import (
    SchemaValidationError,
    load_dataset,
    load_h5ad,
    validate_obs_schema,
)

REQUIRED = ["perturbation", "guide_id", "ncounts", "ngenes", "percent_mito"]


def _mini_adata(columns) -> AnnData:
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(3)])
    for col in columns:
        obs[col] = ["x", "y", "z"] if col in ("perturbation", "guide_id") else 1.0
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    return AnnData(np.ones((3, 4), dtype=np.float32), obs=obs, var=var)


def test_validate_obs_schema_passes_when_complete():
    validate_obs_schema(_mini_adata(REQUIRED), REQUIRED)  # must not raise


def test_validate_obs_schema_lists_exactly_missing_columns():
    adata = _mini_adata(["perturbation", "ncounts", "ngenes"])
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_obs_schema(adata, REQUIRED)
    msg = str(excinfo.value)
    assert "'guide_id'" in msg and "'percent_mito'" in msg
    assert "missing 2 expected column(s)" in msg
    assert "Found columns" in msg
    # present columns must not be reported as missing
    assert "['guide_id', 'percent_mito']" in msg


def test_load_h5ad_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_h5ad(tmp_path / "nope.h5ad")


def _write_config(tmp_path, required):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    config = {
        "dataset": {"raw_file": "raw/mini.h5ad"},
        "schema": {"required_obs_columns": required},
    }
    cfg_path = cfg_dir / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return cfg_path


def test_load_dataset_validates_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _mini_adata(["perturbation"]).write_h5ad(raw_dir / "mini.h5ad")
    cfg_path = _write_config(tmp_path, REQUIRED)
    with pytest.raises(SchemaValidationError, match="guide_id"):
        load_dataset(cfg_path)


def test_load_dataset_success(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _mini_adata(REQUIRED).write_h5ad(raw_dir / "mini.h5ad")
    cfg_path = _write_config(tmp_path, REQUIRED)
    adata = load_dataset(cfg_path)
    assert adata.shape == (3, 4)
    for col in REQUIRED:
        assert col in adata.obs.columns
