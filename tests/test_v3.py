"""v3 DoD: registry + generic loader integrity for the cross-dataset datasets.

Light checks only (no model fitting) so `make test` stays fast. Each dataset
skips cleanly if its (gitignored) raw file isn't present — run `make v3` (or
`make data` for IBM) to fetch them.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import DATA_RAW
from src.v3.datasets import (
    OUTCOME,
    SPECS,
    TREAT,
    build_causal_frame_v3,
    load_clean_v3,
    sensitive_frame,
    treatment_vector,
)

# Sane base-rate / lever-prevalence windows per dataset (documented in PROGRESS).
EXPECTED = {
    "ibm": dict(base=(0.14, 0.18), lever=(0.20, 0.35)),
    "hr_turnover": dict(base=(0.20, 0.27), lever=(0.18, 0.26)),
    "employee_future": dict(base=(0.30, 0.39), lever=(0.07, 0.14)),
}


def _present(spec) -> bool:
    return (DATA_RAW / spec.filename).exists()


@pytest.mark.parametrize("key", list(SPECS))
def test_loads_with_expected_shape_and_binary_outcome(key):
    spec = SPECS[key]
    if not _present(spec):
        pytest.skip(f"{spec.filename} not present; run `make v3`")
    df = load_clean_v3(spec)
    assert df.shape[0] == spec.expected_rows
    assert OUTCOME in df.columns
    assert set(df[OUTCOME].unique()) == {0, 1}
    lo, hi = EXPECTED[key]["base"]
    assert lo < df[OUTCOME].mean() < hi
    # the lever column survives load (the risk model uses it)
    assert spec.lever_col in df.columns


@pytest.mark.parametrize("key", list(SPECS))
def test_treatment_vector_is_binary_with_sane_prevalence(key):
    spec = SPECS[key]
    if not _present(spec):
        pytest.skip(f"{spec.filename} not present; run `make v3`")
    df = load_clean_v3(spec)
    t = treatment_vector(df, spec)
    assert set(np.unique(t)) <= {0, 1}
    assert len(t) == len(df)
    lo, hi = EXPECTED[key]["lever"]
    assert lo < t.mean() < hi


@pytest.mark.parametrize("key", list(SPECS))
def test_causal_frame_excludes_lever_and_is_numeric(key):
    spec = SPECS[key]
    if not _present(spec):
        pytest.skip(f"{spec.filename} not present; run `make v3`")
    df = load_clean_v3(spec)
    frame, confounders = build_causal_frame_v3(df, spec)
    assert len(frame) == len(df)
    assert {TREAT, OUTCOME} <= set(frame.columns)
    assert len(confounders) > 0
    # the lever source column must not leak into the confounders
    assert spec.lever_col not in confounders
    assert OUTCOME not in confounders and TREAT not in confounders
    # confounder matrix is fully numeric and has no missing values
    X = frame[confounders]
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    assert not X.isna().any().any()


@pytest.mark.parametrize("key", list(SPECS))
def test_sensitive_frame_matches_spec(key):
    spec = SPECS[key]
    if not _present(spec):
        pytest.skip(f"{spec.filename} not present; run `make v3`")
    df = load_clean_v3(spec)
    S = sensitive_frame(df, spec)
    expected = list(spec.sensitive) + (["AgeBand"] if spec.age_col else [])
    assert list(S.columns) == expected
    if not expected:                       # hr_turnover has no demographics
        assert S.shape[1] == 0
