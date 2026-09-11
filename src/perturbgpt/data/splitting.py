"""Perturbation-level and combinatorial train/val/test split construction.

Two strategies over ``data/processed/perturbseq.h5ad``:

1. **Perturbation split** — partition unique single-gene perturbation IDs
   70/15/15 into train/val/test with a fixed seed, assigning every cell for
   a given perturbation entirely to one split (no perturbation-ID leakage).

2. **Combinatorial split** — train on all single-gene perturbations plus a
   configurable fraction of two-gene combinations, holding out the rest.
   Each held-out combination is tagged ``"both_seen"`` (both constituent
   genes have single-gene perturbations in training) or
   ``"at_least_one_unseen"`` for stratified evaluation.

Both splits are persisted as JSON so they are reproducible without rerunning
the random partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import yaml

# ---------------------------------------------------------------- constants

CONTROL_LABEL = "control"
COMBO_SEPARATOR = "_"
SEEN_TAG = "both_seen"
UNSEEN_TAG = "at_least_one_unseen"

# ----------------------------------------------------- perturbation parsing


def parse_perturbation(label) -> list[str]:
    """Split a perturbation label into constituent gene symbols.

    ``'control'`` → ``[]``, ``'KLF1'`` → ``['KLF1']``,
    ``'AHR_KLF1'`` → ``['AHR', 'KLF1']``.
    """
    label = str(label)
    if label == CONTROL_LABEL:
        return []
    return label.split(COMBO_SEPARATOR)


def is_single_gene(label) -> bool:
    return len(parse_perturbation(label)) == 1


def is_combination(label) -> bool:
    return len(parse_perturbation(label)) > 1


def is_control(label) -> bool:
    return str(label) == CONTROL_LABEL


# ------------------------------------------------------------- combo tagging


def tag_combination(combo_label, seen_genes: set[str]) -> str:
    """Tag a held-out combination by whether constituents were seen in training.

    Returns ``SEEN_TAG`` if every constituent gene is in ``seen_genes``,
    otherwise ``UNSEEN_TAG``.
    """
    constituents = parse_perturbation(combo_label)
    if all(g in seen_genes for g in constituents):
        return SEEN_TAG
    return UNSEEN_TAG


# --------------------------------------------------------------- dataclasses


@dataclass
class PerturbationSplit:
    """Cell-level train/val/test assignment from a perturbation-level split."""

    train: list[str]
    val: list[str]
    test: list[str]
    train_perts: set[str]
    val_perts: set[str]
    test_perts: set[str]
    seed: int
    params: dict = field(default_factory=dict)
    created: str = ""

    def all_cells(self) -> set[str]:
        return set(self.train) | set(self.val) | set(self.test)

    def assert_no_overlap(self) -> None:
        """Assert no perturbation-ID overlap between train/val/test."""
        assert not (self.train_perts & self.val_perts), (
            f"perturbation overlap train∩val: {self.train_perts & self.val_perts}"
        )
        assert not (self.train_perts & self.test_perts), (
            f"perturbation overlap train∩test: {self.train_perts & self.test_perts}"
        )
        assert not (self.val_perts & self.test_perts), (
            f"perturbation overlap val∩test: {self.val_perts & self.test_perts}"
        )

    def assert_partitions_every_cell(self, adata: ad.AnnData) -> None:
        """Assert every cell in *adata* lands in exactly one split."""
        all_obs = set(adata.obs_names)
        union = self.all_cells()
        missing = all_obs - union
        extra = union - all_obs
        assert not missing, f"{len(missing)} cells not assigned to any split"
        assert not extra, f"{len(extra)} split cells not in adata"
        assert not (set(self.train) & set(self.val)), "cells in train∩val"
        assert not (set(self.train) & set(self.test)), "cells in train∩test"
        assert not (set(self.val) & set(self.test)), "cells in val∩test"

@dataclass
class CombinatorialSplit:
    """Cell-level train/test assignment from a combinatorial split.

    All single-gene perturbation cells go to training; a configurable
    fraction of two-gene combinations is also kept in training, and the
    remainder is held out in ``test``. Each held-out combination is tagged
    in ``held_out_tags`` as ``SEEN_TAG`` or ``UNSEEN_TAG``.
    """

    train: list[str]
    test: list[str]
    held_out_both_seen: list[str]
    held_out_unseen: list[str]
    train_perts: set[str]
    test_perts: set[str]
    held_out_tags: dict[str, str]
    seed: int
    params: dict = field(default_factory=dict)
    created: str = ""

    def all_cells(self) -> set[str]:
        return set(self.train) | set(self.test)

    def assert_no_overlap(self) -> None:
        """Assert no perturbation-ID overlap between train and test."""
        assert not (self.train_perts & self.test_perts), (
            f"perturbation overlap train∩test: {self.train_perts & self.test_perts}"
        )

    def assert_partitions_every_cell(self, adata: ad.AnnData) -> None:
        """Assert every cell in *adata* lands in exactly one split."""
        all_obs = set(adata.obs_names)
        union = self.all_cells()
        missing = all_obs - union
        extra = union - all_obs
        assert not missing, f"{len(missing)} cells not assigned to any split"
        assert not extra, f"{len(extra)} split cells not in adata"
        assert not (set(self.train) & set(self.test)), "cells in train∩test"

# ----------------------------------------------------------- split strategies


def perturbation_split(
    adata: ad.AnnData,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    perturbation_col: str = "perturbation",
    control_split: str = "train",
) -> PerturbationSplit:
    """Partition single-gene perturbation IDs 70/15/15, assign cells wholesale.

    Every cell for a given perturbation goes entirely to one split; control
    cells go to ``control_split``.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"fractions must sum to 1, got {train_frac + val_frac + test_frac}"
    )
    assert control_split in ("train", "val", "test"), (
        f"control_split must be train/val/test, got {control_split!r}"
    )
    perts = adata.obs[perturbation_col].astype(str)
    single_labels = sorted({p for p in perts.unique() if is_single_gene(p)})

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(single_labels))
    labels = np.array(single_labels)[perm]
    n = len(labels)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_perts = set(labels[:n_train])
    val_perts = set(labels[n_train : n_train + n_val])
    test_perts = set(labels[n_train + n_val :])

    ctrl_mask = perts == CONTROL_LABEL
    combo_mask = perts.apply(is_combination)
    train_mask = perts.isin(train_perts) | combo_mask | (
        ctrl_mask if control_split == "train" else False
    )
    val_mask = perts.isin(val_perts) | (
        ctrl_mask if control_split == "val" else False
    )
    test_mask = perts.isin(test_perts) | (
        ctrl_mask if control_split == "test" else False
    )

    return PerturbationSplit(
        train=list(adata.obs_names[train_mask]),
        val=list(adata.obs_names[val_mask]),
        test=list(adata.obs_names[test_mask]),
        train_perts=train_perts,
        val_perts=val_perts,
        test_perts=test_perts,
        seed=seed,
        params={
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "control_split": control_split,
            "perturbation_col": perturbation_col,
        },
        created=datetime.now(timezone.utc).isoformat(),
    )

def combinatorial_split(
    adata: ad.AnnData,
    combo_train_frac: float = 0.50,
    seed: int = 42,
    perturbation_col: str = "perturbation",
) -> CombinatorialSplit:
    """Train on all single-gene perturbations + a fraction of combinations.

    The remaining combinations are held out (test) and tagged by whether
    their constituent genes were seen in training (i.e. have single-gene
    perturbations in the dataset). Control cells always go to train.
    """
    assert 0.0 <= combo_train_frac <= 1.0, (
        f"combo_train_frac must be in [0, 1], got {combo_train_frac}"
    )
    perts = adata.obs[perturbation_col].astype(str)
    single_labels = {p for p in perts.unique() if is_single_gene(p)}
    combo_labels = sorted({p for p in perts.unique() if is_combination(p)})

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(combo_labels))
    labels = np.array(combo_labels)[perm]
    n_train = int(len(labels) * combo_train_frac)
    train_combos = set(labels[:n_train])
    test_combos = set(labels[n_train:])

    held_out_tags = {c: tag_combination(c, single_labels) for c in test_combos}
    held_out_both_seen = sorted(c for c, t in held_out_tags.items() if t == SEEN_TAG)
    held_out_unseen = sorted(c for c, t in held_out_tags.items() if t == UNSEEN_TAG)

    train_perts = single_labels | train_combos
    test_perts = test_combos
    train_mask = perts.isin(train_perts) | (perts == CONTROL_LABEL)
    test_mask = perts.isin(test_combos)

    return CombinatorialSplit(
        train=list(adata.obs_names[train_mask]),
        test=list(adata.obs_names[test_mask]),
        held_out_both_seen=held_out_both_seen,
        held_out_unseen=held_out_unseen,
        train_perts=train_perts,
        test_perts=test_perts,
        held_out_tags=held_out_tags,
        seed=seed,
        params={
            "combo_train_frac": combo_train_frac,
            "perturbation_col": perturbation_col,
        },
        created=datetime.now(timezone.utc).isoformat(),
    )

# ------------------------------------------------------------- persistence


def _split_to_dict(split: Union[PerturbationSplit, CombinatorialSplit]) -> dict:
    """Serialize a split dataclass to a JSON-compatible dict."""
    if isinstance(split, PerturbationSplit):
        kind = "perturbation_split"
        return {
            "kind": kind,
            "seed": split.seed,
            "created": split.created,
            "params": split.params,
            "train": split.train,
            "val": split.val,
            "test": split.test,
            "train_perts": sorted(split.train_perts),
            "val_perts": sorted(split.val_perts),
            "test_perts": sorted(split.test_perts),
        }
    kind = "combinatorial_split"
    return {
        "kind": kind,
        "seed": split.seed,
        "created": split.created,
        "params": split.params,
        "train": split.train,
        "test": split.test,
        "held_out_both_seen": split.held_out_both_seen,
        "held_out_unseen": split.held_out_unseen,
        "held_out_tags": split.held_out_tags,
        "train_perts": sorted(split.train_perts),
        "test_perts": sorted(split.test_perts),
    }


def _dict_to_split(d: dict) -> Union[PerturbationSplit, CombinatorialSplit]:
    """Reconstruct a split dataclass from a persisted dict."""
    kind = d["kind"]
    if kind == "perturbation_split":
        return PerturbationSplit(
            train=d["train"],
            val=d["val"],
            test=d["test"],
            train_perts=set(d["train_perts"]),
            val_perts=set(d["val_perts"]),
            test_perts=set(d["test_perts"]),
            seed=d["seed"],
            params=d.get("params", {}),
            created=d.get("created", ""),
        )
    if kind == "combinatorial_split":
        return CombinatorialSplit(
            train=d["train"],
            test=d["test"],
            held_out_both_seen=d["held_out_both_seen"],
            held_out_unseen=d["held_out_unseen"],
            train_perts=set(d["train_perts"]),
            test_perts=set(d["test_perts"]),
            held_out_tags=d["held_out_tags"],
            seed=d["seed"],
            params=d.get("params", {}),
            created=d.get("created", ""),
        )
    raise ValueError(f"unknown split kind: {kind!r}")


def persist_split(
    split: Union[PerturbationSplit, CombinatorialSplit],
    path: Union[str, Path],
) -> Path:
    """Write a split to *path* as JSON (parent dirs are created).."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(_split_to_dict(split), fh, indent=2)
    return path


def load_persisted_split(
    path: Union[str, Path],
) -> Union[PerturbationSplit, CombinatorialSplit]:
    """Load a split previously written by :func:`persist_split`."""
    with Path(path).open() as fh:
        return _dict_to_split(json.load(fh))


# ------------------------------------------------------------- config helper


def load_splitting_config(config_path: Union[str, Path]) -> dict:
    """Load the ``splitting`` section from a training config YAML."""
    with Path(config_path).open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg["splitting"]

