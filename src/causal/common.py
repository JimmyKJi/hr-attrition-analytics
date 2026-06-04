"""Shared definition of the causal problem so identify/uplift/divergence agree.

Treatment: OverTime (binary). Outcome: Attrition (binary). Confounders: every
other feature. The intervention studied is "relieve required overtime"; an
employee's uplift from it is the CATE tau(x) of OverTime on attrition — higher
tau means more attrition averted by removing their overtime.

OverTime is chosen because it is (a) the strongest driver in v1, (b) a clean
binary, and (c) a plausible real policy lever. Continuous levers (income,
promotion cadence) are left to future work; one treatment done rigorously beats
several done loosely.
"""
from __future__ import annotations

import pandas as pd

from src.config import TARGET
from src.data.load import feature_columns

TREATMENT = "OverTime"
OUTCOME = TARGET  # "Attrition", already mapped to {0,1} by load_clean()


def build_causal_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (fully-numeric frame, confounder columns).

    OverTime -> {0,1}; the other string columns are one-hot encoded; numerics
    pass through. Row order is preserved so estimates align across modules.
    """
    df = df.copy()
    df[TREATMENT] = (df[TREATMENT].astype(str).str.strip() == "Yes").astype(int)
    cols = feature_columns(df)  # OverTime is now int -> lands in 'numeric'
    numeric = [c for c in cols["numeric"] if c != TREATMENT]
    categorical = [c for c in cols["categorical"] if c != TREATMENT]

    dummies = pd.get_dummies(df[categorical], drop_first=True, dtype=float)
    frame = pd.concat(
        [df[numeric].astype(float), dummies, df[[TREATMENT, OUTCOME]].astype(int)],
        axis=1,
    )
    confounders = numeric + list(dummies.columns)
    return frame, confounders


def YTX(frame: pd.DataFrame, confounders: list[str]):
    """Split a causal frame into outcome Y, treatment T, confounders X."""
    Y = frame[OUTCOME].to_numpy()
    T = frame[TREATMENT].to_numpy()
    X = frame[confounders].to_numpy()
    return Y, T, X
