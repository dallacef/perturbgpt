"""Load the raw Norman et al. 2019 Perturb-seq dataset and validate its schema.

The raw dataset is the curated scPerturb mirror of GEO accession GSE133344
(a single ``.h5ad`` file; see ``configs/data.yaml`` for provenance). Loading
returns an :class:`anndata.AnnData` with raw counts as a float32 CSR matrix in
``.X`` and validates that the expected ``obs`` columns (perturbation label,
guide id, QC metrics) are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import anndata as ad
import numpy as np
import yaml


class SchemaValidationError(ValueError):
    """Raised when an AnnData object is missing expected ``obs`` columns."""


def load_config(config_path: Union[str, Path]) -> dict:
    """Load the dataset/preprocessing configuration YAML."""
    with Path(config_path).open() as fh:
        return yaml.safe_load(fh)


def project_root(config_path: Union[str, Path]) -> Path:
    """Project root, given a config path of the form ``<root>/configs/data.yaml``."""
    return Path(config_path).resolve().parents[1]


def validate_obs_schema(adata: ad.AnnData, required_columns: Iterable[str]) -> None:
    """Check that every column in ``required_columns`` exists in ``adata.obs``.

    Raises
    ------
    SchemaValidationError
        If any expected column is missing. The message lists exactly which
        columns are missing, followed by the columns that were found.
    """
    required = list(required_columns)
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        found = sorted(str(c) for c in adata.obs.columns)
        raise SchemaValidationError(
            f"AnnData.obs is missing {len(missing)} expected column(s): {missing}. "
            f"Found columns: {found}."
        )


def load_h5ad(path: Union[str, Path]) -> ad.AnnData:
    """Read an ``.h5ad`` file, returning counts as a float32 CSR matrix in ``.X``.

    CSR layout makes the cell-wise operations used downstream (QC filtering,
    library-size normalization) much faster than the CSC layout the scPerturb
    file is stored in. Counts are cast to float32 (exact for integer counts
    < 2**24) to halve memory use.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run scripts/download_data.py first."
        )
    adata = ad.read_h5ad(path)
    if hasattr(adata.X, "tocsr"):
        adata.X = adata.X.tocsr().astype(np.float32)
    return adata


def load_dataset(config_path: Union[str, Path]) -> ad.AnnData:
    """Load the raw dataset described by ``config_path`` and validate its schema.

    The required ``obs`` columns are read from
    ``schema.required_obs_columns`` in the config, so the expected schema is
    configuration-driven rather than hard-coded.
    """
    config = load_config(config_path)
    raw_file = Path(config["dataset"]["raw_file"])
    if not raw_file.is_absolute():
        raw_file = project_root(config_path) / raw_file
    adata = load_h5ad(raw_file)
    required = config.get("schema", {}).get("required_obs_columns", [])
    validate_obs_schema(adata, required)
    return adata

pass