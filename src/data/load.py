"""Load and split the IBM HR Analytics dataset.

The raw CSV ships with a UTF-8 BOM and three constant columns that carry no
signal; both are handled here so downstream code sees a clean frame. No model
is fit in this module — only loading and the stratified train/test split — so
there is a single, auditable place where leakage could enter. There is none:
the split is stratified on the target with the fixed seed, and nothing is fit
before it.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import HR_CSV, RANDOM_SEED, TARGET, TEST_SIZE

# Verified constant in this dataset: zero variance, no signal.
CONSTANT_COLS = ["EmployeeCount", "StandardHours", "Over18"]
# Identifier, not a feature.
ID_COLS = ["EmployeeNumber"]

RAW_SHAPE = (1470, 35)


def load_raw() -> pd.DataFrame:
    """Read the raw CSV (BOM-safe) and assert its known shape."""
    if not HR_CSV.exists():
        raise FileNotFoundError(
            f"{HR_CSV} not found. Run `make data` (see DATA_LINEAGE.md)."
        )
    df = pd.read_csv(HR_CSV, encoding="utf-8-sig")  # utf-8-sig strips the BOM
    if df.shape != RAW_SHAPE:
        raise ValueError(f"Expected raw shape {RAW_SHAPE}, got {df.shape}.")
    return df


def load_clean() -> pd.DataFrame:
    """Raw frame minus constants/identifier, with the target mapped to {0,1}."""
    df = load_raw().drop(columns=CONSTANT_COLS + ID_COLS)
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Partition feature columns (everything but the target) into numeric and
    categorical. The dataset's ordinal codes (e.g. JobSatisfaction 1–4) are
    treated as numeric — they are ordered — while the string columns are
    categorical."""
    feats = [c for c in df.columns if c != TARGET]
    categorical = [c for c in feats if df[c].dtype == object]
    numeric = [c for c in feats if c not in categorical]
    return {"numeric": numeric, "categorical": categorical}


def train_test(
    df: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the target. Deterministic for a fixed seed."""
    if df is None:
        df = load_clean()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_clean()
    cols = feature_columns(df)
    tr, te = train_test(df)
    print(f"clean shape: {df.shape}")
    print(f"numeric ({len(cols['numeric'])}): {cols['numeric']}")
    print(f"categorical ({len(cols['categorical'])}): {cols['categorical']}")
    print(f"train/test: {tr.shape} / {te.shape}")
    print(
        f"attrition rate — full {df[TARGET].mean():.4f}, "
        f"train {tr[TARGET].mean():.4f}, test {te[TARGET].mean():.4f}"
    )
