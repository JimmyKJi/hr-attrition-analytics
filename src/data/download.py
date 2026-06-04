"""Fetch the IBM HR Analytics CSV into data/raw/ (gitignored).

Strategy: use the Kaggle CLI if it is configured (authoritative source),
otherwise fall back to a public GitHub mirror. Idempotent — does nothing if a
valid file is already present. Provenance is recorded in DATA_LINEAGE.md.
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request

from src.config import DATA_RAW, HR_CSV

KAGGLE_DATASET = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
# Public mirror of the exact Kaggle file (confirmed 1470x35).
MIRROR_URL = (
    "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/"
    "master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
EXPECTED_LINES = 1471  # 1470 data rows + 1 header


def _looks_valid() -> bool:
    if not HR_CSV.exists():
        return False
    with open(HR_CSV, encoding="utf-8-sig") as f:
        return sum(1 for _ in f) == EXPECTED_LINES


def _via_kaggle() -> bool:
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(DATA_RAW), "--unzip"],
            check=True, capture_output=True,
        )
        return HR_CSV.exists()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _via_mirror() -> bool:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MIRROR_URL, HR_CSV)
    return HR_CSV.exists()


def main() -> None:
    if _looks_valid():
        print(f"Dataset already present and valid: {HR_CSV}")
        return
    print("Fetching IBM HR Analytics dataset...")
    if _via_kaggle():
        source = "Kaggle CLI"
    elif _via_mirror():
        source = "GitHub mirror"
    else:
        sys.exit("Could not fetch dataset; see DATA_LINEAGE.md for manual steps.")
    if not _looks_valid():
        sys.exit(f"Downloaded file failed validation (expected {EXPECTED_LINES} lines).")
    print(f"Downloaded via {source} -> {HR_CSV}")


if __name__ == "__main__":
    main()
