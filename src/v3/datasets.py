"""v3 — dataset registry, download, and a generic loader for cross-dataset work.

v2 studied one synthetic workforce (IBM). v3 asks whether the
prediction-vs-cause divergence is a property of *that fixture* or of the
*method*, by re-running the same pipeline on independent employee-turnover
datasets through one generic code path. This module is the data plumbing:

  * `DatasetSpec`  — a small record describing each dataset's outcome, lever,
    and (where present) sensitive attributes;
  * `download`     — idempotent fetch into `data/raw/` (gitignored; provenance
    in DATA_LINEAGE.md). IBM reuses the existing v1/v2 fetcher;
  * `load_clean_v3`— returns a frame with a canonical binary outcome `Y`, the
    raw lever column kept (so the risk model can use it), and raw sensitive
    columns kept (so the fairness audit can group on them);
  * `treatment_vector` / `sensitive_frame` / `build_causal_frame_v3` — the
    generic equivalents of the IBM-specific helpers in `src/causal/common.py`.

All three datasets are synthetic / benchmark sets with NO ground-truth causal
structure: read every causal number downstream as a method demonstration, never
a finding about a real workforce.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass

import pandas as pd

from src.config import DATA_RAW, HR_CSV
from src.data.download import main as download_ibm

# Canonical column names the generic pipeline writes / reads.
OUTCOME = "Y"      # 1 = left / would leave
TREAT = "T"        # 1 = currently exposed to the lever (e.g. on overtime)


@dataclass(frozen=True)
class DatasetSpec:
    key: str                       # short id, also the figure/CSV key
    label: str                     # human label for tables/figures
    filename: str                  # file under data/raw/
    fmt: str                       # "ibm" | "csv" | "arff"
    url: str | None                # download URL (None for IBM)
    target: str                    # raw outcome column
    target_positive: str           # value (as str) meaning "left / would leave"
    drop_cols: tuple[str, ...]     # ids/constants to drop entirely
    lever_col: str                 # column the treatment is derived from
    lever_kind: str                # "equals" | "threshold_ge"
    lever_value: object            # positive category (equals) or threshold (ge)
    lever_label: str               # human label for the lever
    sensitive: tuple[str, ...]     # raw categorical columns for the audit
    age_col: str | None            # numeric col to band into AgeBand (if any)
    expected_rows: int             # row count after dropping `drop_cols`


SPECS: dict[str, DatasetSpec] = {
    # The v2 fixture, run through the generic path as a reference / consistency
    # check: the generic pipeline must reproduce the v2 divergence on IBM before
    # its verdict on the new datasets is trustworthy.
    "ibm": DatasetSpec(
        key="ibm", label="IBM HR (v2 reference)",
        filename=HR_CSV.name, fmt="ibm", url=None,
        target="Attrition", target_positive="Yes",
        drop_cols=("EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"),
        lever_col="OverTime", lever_kind="equals", lever_value="Yes",
        lever_label="required overtime",
        sensitive=("Gender", "MaritalStatus"), age_col="Age",
        expected_rows=1470,
    ),
    # Classic ~15k HR turnover set. Lever = overwork, the closest analogue to the
    # IBM overtime lever (threshold the monthly-hours column). No demographics,
    # so no fairness audit here.
    "hr_turnover": DatasetSpec(
        key="hr_turnover", label="HR turnover (15k)",
        filename="hr_comma_sep.csv", fmt="csv",
        url="https://raw.githubusercontent.com/aiplanethub/Datasets/master/"
            "HR_comma_sep.csv",
        target="left", target_positive="1",
        drop_cols=(),
        lever_col="average_montly_hours", lever_kind="threshold_ge",
        lever_value=250, lever_label="overwork (>=250 h/mo)",
        sensitive=(), age_col=None,
        expected_rows=14999,
    ),
    # Indian IT 2-year attrition set. Lever = EverBenched (idle on the bench).
    # Has Gender + Age, so the fairness audit replicates here.
    "employee_future": DatasetSpec(
        key="employee_future", label="Employee future (4.7k)",
        filename="employee_future.arff", fmt="arff",
        url="https://www.openml.org/data/download/22125236/dataset",
        target="LeaveOrNot", target_positive="Leave",
        drop_cols=(),
        lever_col="EverBenched", lever_kind="equals", lever_value="Yes",
        lever_label="ever benched",
        sensitive=("Gender",), age_col="Age",
        expected_rows=4653,
    ),
}


def download(spec: DatasetSpec) -> None:
    """Fetch the raw file into data/raw/ if absent. Idempotent."""
    if spec.fmt == "ibm":
        download_ibm()  # existing idempotent fetcher (Kaggle CLI / GitHub mirror)
        return
    path = DATA_RAW / spec.filename
    if path.exists():
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(spec.url, path)


def _read_arff(path) -> pd.DataFrame:
    """Minimal ARFF reader: take column names from @attribute lines, read the
    body after @data as CSV. Robust to the exact header length."""
    attrs: list[str] = []
    data_idx: int | None = None
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.lower().startswith("@attribute"):
            attrs.append(s.split()[1].strip("'\""))
        elif s.lower() == "@data":
            data_idx = i
            break
    if data_idx is None:
        raise ValueError(f"{path}: no @data marker found")
    return pd.read_csv(path, skiprows=data_idx + 1, header=None, names=attrs,
                       skipinitialspace=True)


def load_clean_v3(spec: DatasetSpec) -> pd.DataFrame:
    """Clean frame with canonical outcome `Y` in {0,1}; lever and sensitive
    columns retained. Asserts the expected row count."""
    path = DATA_RAW / spec.filename
    if not path.exists():
        download(spec)
    if spec.fmt == "arff":
        df = _read_arff(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig strips any BOM

    df = df.drop(columns=[c for c in spec.drop_cols if c in df.columns])
    # Strip stray quotes/whitespace from string cells (e.g. ARFF 'New Delhi').
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.strip("'\"").str.strip()

    if df.shape[0] != spec.expected_rows:
        raise ValueError(
            f"{spec.key}: expected {spec.expected_rows} rows, got {df.shape[0]}")

    df[OUTCOME] = (
        df[spec.target].astype(str).str.strip() == str(spec.target_positive)
    ).astype(int)
    if spec.target != OUTCOME:
        df = df.drop(columns=[spec.target])
    return df


def treatment_vector(df: pd.DataFrame, spec: DatasetSpec):
    """Binary 'currently exposed to the lever' vector (0/1)."""
    col = df[spec.lever_col]
    if spec.lever_kind == "equals":
        return (col.astype(str).str.strip() == str(spec.lever_value)).astype(int).to_numpy()
    if spec.lever_kind == "threshold_ge":
        return (col.astype(float) >= float(spec.lever_value)).astype(int).to_numpy()
    raise ValueError(f"unknown lever_kind {spec.lever_kind!r}")


def sensitive_frame(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Sensitive-attribute frame for the fairness audit (empty if none)."""
    s = pd.DataFrame(index=df.index)
    for c in spec.sensitive:
        s[c] = df[c].astype(str)
    if spec.age_col:
        s["AgeBand"] = pd.cut(
            df[spec.age_col], bins=[0, 30, 45, 200], labels=["<30", "30-45", "45+"]
        ).astype(str)
    return s


def feature_cols(df: pd.DataFrame, exclude: tuple[str, ...]) -> dict[str, list[str]]:
    """Partition columns into numeric / categorical, excluding `exclude`."""
    feats = [c for c in df.columns if c not in exclude]
    categorical = [c for c in feats if df[c].dtype == object]
    numeric = [c for c in feats if c not in categorical]
    return {"numeric": numeric, "categorical": categorical}


def build_causal_frame_v3(df: pd.DataFrame, spec: DatasetSpec):
    """Numeric frame + confounder list for the causal model.

    The lever column is *removed from the confounder pool* (for an `equals`
    lever it becomes the treatment; for a `threshold_ge` lever, keeping the
    underlying continuous column would leak the treatment), and the treatment
    `T` and outcome `Y` are appended. Row order is preserved so estimates align
    with the risk scores.
    """
    df = df.copy()
    t = treatment_vector(df, spec)
    drop = {OUTCOME, spec.lever_col}
    conf_df = df.drop(columns=[c for c in drop if c in df.columns])
    cols = feature_cols(conf_df, exclude=())
    dummies = pd.get_dummies(conf_df[cols["categorical"]], drop_first=True, dtype=float)
    frame = pd.concat([conf_df[cols["numeric"]].astype(float), dummies], axis=1)
    confounders = cols["numeric"] + list(dummies.columns)
    frame[TREAT] = t
    frame[OUTCOME] = df[OUTCOME].astype(int).to_numpy()
    return frame, confounders


def main() -> None:
    """Fetch all v3 datasets and print a one-line schema summary for each."""
    for spec in SPECS.values():
        download(spec)
        df = load_clean_v3(spec)
        t = treatment_vector(df, spec)
        print(f"{spec.key:16s} {df.shape[0]:>6d} rows  "
              f"base {df[OUTCOME].mean():.3f}  "
              f"lever[{spec.lever_label}] {t.mean():.3f}  "
              f"sensitive={list(sensitive_frame(df, spec).columns)}")


if __name__ == "__main__":
    main()
