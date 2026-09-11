"""Tests asserting zero perturbation-ID overlap across train/val/test splits.

These tests are designed to **fail** if the perturbation-level split is ever
replaced by a random cell-level split. The key anti-regression test
(``test_perturbation_split_atomic``) uses a fixture with 5 cells per
perturbation; a cell-level 70/15/15 split has probability ≈ 0.24^10 ≈ 1e-6
of accidentally keeping every perturbation's cells in a single split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from perturbgpt.data.splitting import (
    CONTROL_LABEL,
    SEEN_TAG,
    UNSEEN_TAG,
    combinatorial_split,
    load_persisted_split,
    parse_perturbation,
    persist_split,
    perturbation_split,
    tag_combination,
)

# The fixture has 7 single-gene perturbations and 5 two-gene combos.
# Genes A–G have single-gene perturbations; X and Y do NOT.
SINGLE_GENES = ["A", "B", "C", "D", "E", "F", "G"]
COMBOS = ["A_B", "C_D", "E_F", "X_Y", "A_X"]
CELLS_PER_PERT = 5


def _build_fixture() -> AnnData:
    """65 cells: 7 single-gene × 5 + 5 combos × 5 + 5 control = 65."""
    labels = []
    for g in SINGLE_GENES:
        labels += [g] * CELLS_PER_PERT
    for c in COMBOS:
        labels += [c] * CELLS_PER_PERT
    labels += [CONTROL_LABEL] * CELLS_PER_PERT
    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cell{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(10)])
    X = np.zeros((len(labels), 10), dtype=np.float32)
    return AnnData(X, obs=obs, var=var)


@pytest.fixture
def adata() -> AnnData:
    return _build_fixture()


# ---------------------------------------------------------- parsing helpers


def test_parse_perturbation():
    assert parse_perturbation("control") == []
    assert parse_perturbation("KLF1") == ["KLF1"]
    assert parse_perturbation("AHR_KLF1") == ["AHR", "KLF1"]


# ---------------------------------------------------- perturbation-split tests


def test_perturbation_split_no_overlap(adata):
    split = perturbation_split(adata, seed=42)
    split.assert_no_overlap()


def test_perturbation_split_partitions_every_cell(adata):
    split = perturbation_split(adata, seed=42)
    split.assert_partitions_every_cell(adata)


def test_perturbation_split_atomic(adata):
    """Anti-regression: every perturbation's cells must be in exactly one
    split. A random cell-level split would scatter cells and fail here.
    """
    split = perturbation_split(adata, seed=42)
    split.assert_no_overlap()
    split.assert_partitions_every_cell(adata)

    perts = adata.obs["perturbation"].astype(str)
    train_set, val_set, test_set = set(split.train), set(split.val), set(split.test)
    for pert in perts.unique():
        if pert == CONTROL_LABEL:
            continue
        cells = set(adata.obs_names[perts == pert])
        n_train = len(cells & train_set)
        n_val = len(cells & val_set)
        n_test = len(cells & test_set)
        assert n_train + n_val + n_test == len(cells), (
            f"perturbation {pert!r}: cells lost or duplicated "
            f"(train={n_train}, val={n_val}, test={n_test}, total={len(cells)})"
        )
        n_splits = bool(n_train) + bool(n_val) + bool(n_test)
        assert n_splits == 1, (
            f"perturbation {pert!r} split across {n_splits} splits "
            f"(train={n_train}, val={n_val}, test={n_test}) — "
            "this looks like a cell-level split, not a perturbation-level split!"
        )


def test_perturbation_split_fractions(adata):
    split = perturbation_split(adata, seed=42)
    total = adata.n_obs - CELLS_PER_PERT  # exclude control
    n_ctrl = int((adata.obs["perturbation"] == CONTROL_LABEL).sum())
    # ±1 perturbation tolerance (each perturbation has 5 cells)
    tol_cells = 2 * CELLS_PER_PERT
    assert abs(len(split.train) - n_ctrl - total * 0.70) < tol_cells
    assert abs(len(split.val) - total * 0.15) < tol_cells
    assert abs(len(split.test) - total * 0.15) < tol_cells


def test_perturbation_split_seed_reproducibility(adata):
    s1 = perturbation_split(adata, seed=42)
    s2 = perturbation_split(adata, seed=42)
    assert s1.train == s2.train
    assert s1.val == s2.val
    assert s1.test == s2.test


def test_perturbation_split_different_seeds_differ(adata):
    s1 = perturbation_split(adata, seed=42)
    s2 = perturbation_split(adata, seed=999)
    # With 7 perturbations at 4/1/2, different seeds almost always differ
    assert s1.train_perts != s2.train_perts or s1.test_perts != s2.test_perts



def test_perturbation_split_control_in_train(adata):
    split = perturbation_split(adata, seed=42, control_split="train")
    ctrl_cells = set(adata.obs_names[adata.obs["perturbation"] == CONTROL_LABEL])
    assert ctrl_cells <= set(split.train), "control cells not in train"
    assert not (ctrl_cells & set(split.val)), "control cells leaked to val"
    assert not (ctrl_cells & set(split.test)), "control cells leaked to test"


def test_perturbation_split_control_in_val(adata):
    split = perturbation_split(adata, seed=42, control_split="val")
    ctrl_cells = set(adata.obs_names[adata.obs["perturbation"] == CONTROL_LABEL])
    assert ctrl_cells <= set(split.val), "control cells not in val"


# -------------------------------------------------- combinatorial-split tests


def test_combinatorial_split_no_overlap(adata):
    split = combinatorial_split(adata, seed=42)
    split.assert_no_overlap()


def test_combinatorial_split_partitions_every_cell(adata):
    split = combinatorial_split(adata, seed=42)
    split.assert_partitions_every_cell(adata)


def test_combinatorial_split_all_single_in_train(adata):
    split = combinatorial_split(adata, seed=42)
    for g in SINGLE_GENES:
        cells = set(adata.obs_names[adata.obs["perturbation"] == g])
        assert cells <= set(split.train), f"single-gene {g} cells not all in train"
        assert not (cells & set(split.test)), f"single-gene {g} cells leaked to test"


def test_combinatorial_split_control_in_train(adata):
    split = combinatorial_split(adata, seed=42)
    ctrl_cells = set(adata.obs_names[adata.obs["perturbation"] == CONTROL_LABEL])
    assert ctrl_cells <= set(split.train), "control cells not in train"


def test_combinatorial_split_no_combo_overlap(adata):
    split = combinatorial_split(adata, seed=42)
    train_combos = split.train_perts - set(SINGLE_GENES)
    test_combos = split.test_perts
    assert not (train_combos & test_combos), "combo in both train and test"


def test_combinatorial_split_combo_atomic(adata):
    """Anti-regression: held-out combos must have ALL cells in test."""
    split = combinatorial_split(adata, seed=42)
    perts = adata.obs["perturbation"].astype(str)
    for combo in split.test_perts:
        cells = set(adata.obs_names[perts == combo])
        assert cells <= set(split.test), (
            f"held-out combo {combo!r} cells leaked into train"
        )


def test_combinatorial_split_fraction(adata):
    split = combinatorial_split(adata, seed=42, combo_train_frac=0.5)
    n_total_combos = len(COMBOS)
    n_train_combos = len(split.train_perts - set(SINGLE_GENES))
    n_test_combos = len(split.test_perts)
    assert n_train_combos + n_test_combos == n_total_combos
    # With 5 combos at 50%, expect 2-3 in train
    assert abs(n_train_combos - n_total_combos * 0.5) <= 1



# ------------------------------------------------------- combo tagging tests


def test_tag_combination_both_seen():
    """Both constituent genes are in the seen set → SEEN_TAG."""
    seen = {"A", "B", "C", "D", "E", "F", "G"}
    assert tag_combination("A_B", seen) == SEEN_TAG
    assert tag_combination("C_D", seen) == SEEN_TAG


def test_tag_combination_at_least_one_unseen():
    """At least one constituent gene not in seen set → UNSEEN_TAG."""
    seen = {"A", "B", "C", "D", "E", "F", "G"}
    # X and Y are NOT in seen
    assert tag_combination("X_Y", seen) == UNSEEN_TAG
    # X is not in seen even though A is
    assert tag_combination("A_X", seen) == UNSEEN_TAG


def test_combinatorial_split_tags_match_independent_computation(adata):
    """Tags in the split must match an independent recomputation from scratch."""
    split = combinatorial_split(adata, seed=42)
    seen_genes = set(SINGLE_GENES)
    for combo, tag in split.held_out_tags.items():
        expected = tag_combination(combo, seen_genes)
        assert tag == expected, (
            f"combo {combo!r}: got {tag!r}, expected {expected!r}"
        )


def test_combinatorial_split_tagged_lists_consistent(adata):
    """held_out_both_seen and held_out_unseen must partition test combos."""
    split = combinatorial_split(adata, seed=42)
    all_held = set(split.held_out_both_seen) | set(split.held_out_unseen)
    assert all_held == split.test_perts, "tagged lists don't cover all test combos"
    assert not (set(split.held_out_both_seen) & set(split.held_out_unseen)), (
        "a combo is in both seen and unseen lists"
    )
    for c in split.held_out_both_seen:
        assert split.held_out_tags[c] == SEEN_TAG
    for c in split.held_out_unseen:
        assert split.held_out_tags[c] == UNSEEN_TAG


# ----------------------------------------------------------- persistence test


def test_persist_load_roundtrip(adata, tmp_path):
    s = perturbation_split(adata, seed=42)
    p = persist_split(s, tmp_path / "splits" / "perturbation_split.json")
    loaded = load_persisted_split(p)
    assert isinstance(loaded, type(s))
    assert loaded.train == s.train
    assert loaded.val == s.val
    assert loaded.test == s.test
    assert loaded.train_perts == s.train_perts
    assert loaded.test_perts == s.test_perts
    assert loaded.seed == s.seed


def test_persist_load_roundtrip_combinatorial(adata, tmp_path):
    s = combinatorial_split(adata, seed=42)
    p = persist_split(s, tmp_path / "splits" / "combo_split.json")
    loaded = load_persisted_split(p)
    assert isinstance(loaded, type(s))
    assert loaded.train == s.train
    assert loaded.test == s.test
    assert loaded.held_out_tags == s.held_out_tags
    assert loaded.held_out_both_seen == s.held_out_both_seen

