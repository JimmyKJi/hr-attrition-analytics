"""Central paths and the one global random seed.

Every module imports paths and the seed from here so the pipeline is
reproducible and notebooks stay thin.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
PAPER = PROJECT_ROOT / "paper"

# Canonical Kaggle file name for the IBM HR Analytics dataset.
HR_CSV = DATA_RAW / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

RANDOM_SEED = 42
TARGET = "Attrition"
TEST_SIZE = 0.25

# Created on import so a fresh clone can write outputs immediately.
for _d in (DATA_PROCESSED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
