# Data lineage

Raw data is **not committed** (`data/raw/` is gitignored). This file is the
record of what the data is, where it comes from, and how to obtain it.

## v1 — IBM HR Analytics Employee Attrition (Kaggle)

| Field | Value |
| --- | --- |
| Dataset | IBM HR Analytics Employee Attrition & Performance |
| Source | Kaggle: `pavansubhasht/ibm-hr-analytics-attrition-dataset` |
| File | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| Shape | 1,470 rows × 35 columns |
| Target | `Attrition` (Yes/No) — class balance ≈ 16% Yes |
| Nature | **Synthetic.** A fictional dataset created by IBM data scientists. |

**Why "synthetic" matters:** the data was generated, not observed, so it
encodes no real causal structure. All causal/policy estimates in this repo are
therefore demonstrations of *method*. See `FRAMING.md` §7 and the README
caveat.

### How to obtain

Place the CSV at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

- **Automated:** `make data` runs `src/data/download.py`, which fetches the file
  into `data/raw/` (Kaggle CLI if configured, else a documented public mirror).
- **Manual:** download from the Kaggle dataset page and drop it into
  `data/raw/`.

The loader (`src/data/load.py`) verifies shape and schema on read, so a wrong
or truncated file fails loudly rather than silently.

### Leakage notes

- Three columns are constant in this dataset and carry no signal:
  `EmployeeCount`, `StandardHours`, `Over18`. They are dropped on load.
- `EmployeeNumber` is an identifier, not a feature — dropped.
- The train/test split is stratified on `Attrition` with the fixed seed in
  `src/config.py`; all fitting (scaling, models, causal estimators) happens on
  the training split only.

## v2 — AI-layoff-wave sources (flagged stretch, not yet used)

For the future real-data extension (see README): Layoffs.fyi firm-level layoff
events, US WARN Act filings, and LLM-extracted earnings-call AI-capex signals,
plus a bilingual (English + Chinese) East-Asian coverage slice. Each carries its
own selection-bias and measurement caveats, recorded here when that work starts.
