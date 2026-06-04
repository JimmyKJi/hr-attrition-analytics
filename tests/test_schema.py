"""Phase 1 DoD: schema + split integrity for the IBM HR dataset.

Skips cleanly if the (gitignored) raw data isn't present — run `make data`.
"""
from __future__ import annotations

import pytest

from src.config import HR_CSV, TARGET
from src.data.load import (
    CONSTANT_COLS,
    ID_COLS,
    RAW_SHAPE,
    feature_columns,
    load_clean,
    load_raw,
    train_test,
)

pytestmark = pytest.mark.skipif(
    not HR_CSV.exists(), reason="raw data not present; run `make data`"
)

# A stable subset of columns the rest of the pipeline relies on.
REQUIRED_COLUMNS = {
    "Age", "Attrition", "BusinessTravel", "Department", "DistanceFromHome",
    "Gender", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
    "OverTime", "YearsAtCompany", "YearsSinceLastPromotion",
}


def test_raw_shape():
    assert load_raw().shape == RAW_SHAPE


def test_required_columns_present():
    cols = set(load_raw().columns)
    assert REQUIRED_COLUMNS <= cols


def test_constant_columns_are_constant():
    raw = load_raw()
    for c in CONSTANT_COLS:
        assert raw[c].nunique() == 1, f"{c} expected constant"


def test_clean_drops_constants_and_id():
    clean = load_clean()
    for c in CONSTANT_COLS + ID_COLS:
        assert c not in clean.columns


def test_target_is_binary_and_balanced():
    y = load_clean()[TARGET]
    assert set(y.unique()) == {0, 1}
    assert 0.14 < y.mean() < 0.18  # canonical ~16.1%


def test_feature_partition_is_complete_and_disjoint():
    clean = load_clean()
    cols = feature_columns(clean)
    numeric, categorical = set(cols["numeric"]), set(cols["categorical"])
    assert numeric.isdisjoint(categorical)
    assert numeric | categorical | {TARGET} == set(clean.columns)
    # OverTime / Gender are strings; MonthlyIncome is numeric.
    assert {"OverTime", "Gender", "MaritalStatus"} <= categorical
    assert {"MonthlyIncome", "Age"} <= numeric


def test_split_sizes_and_stratification():
    clean = load_clean()
    tr, te = train_test(clean)
    assert len(tr) + len(te) == len(clean)
    # stratified: train/test attrition rates close to the full rate
    full = clean[TARGET].mean()
    assert abs(tr[TARGET].mean() - full) < 0.02
    assert abs(te[TARGET].mean() - full) < 0.03


def test_split_is_deterministic():
    a1, a2 = train_test()[0], train_test()[0]
    assert a1.equals(a2)


def test_split_has_no_row_leakage():
    clean = load_clean().reset_index().rename(columns={"index": "_rid"})
    tr, te = train_test(clean)
    train_ids, test_ids = set(tr["_rid"]), set(te["_rid"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(clean["_rid"])
