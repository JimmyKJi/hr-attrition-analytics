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

## v3 — cross-dataset replication datasets

Two additional, independent, **synthetic** employee-turnover datasets, used to
test whether the v2 risk-vs-uplift result is a property of the IBM fixture or of
the method. Both are gitignored; `make v3` fetches them via `src/v3/datasets.py`,
which asserts each row count on load.

### HR turnover (`HR_comma_sep.csv`)

| Field | Value |
| --- | --- |
| Source | Public mirror: `raw.githubusercontent.com/aiplanethub/Datasets/master/HR_comma_sep.csv` |
| Shape | 14,999 rows × 10 columns |
| Target | `left` (1 = left) — base rate ≈ 23.8% |
| Lever | overwork: `average_montly_hours >= 250` (≈ 22% exposed) — the closest analogue to the IBM overtime lever |
| Demographics | none — so no fairness audit on this set |
| Nature | **Synthetic** ("HR Analytics" Kaggle set). No real causal structure. |

### Employee future (`employee_future.arff`)

| Field | Value |
| --- | --- |
| Source | OpenML (no-auth): `openml.org/data/download/22125236/dataset` (ARFF) |
| Shape | 4,653 rows × 9 columns |
| Target | `LeaveOrNot` (= "Leave") — base rate ≈ 34.4% |
| Lever | `EverBenched` = "Yes" (≈ 10% exposed) |
| Demographics | `Gender`, `Age` (banded) — supports the fairness replication |
| Nature | **Synthetic** ("Employee Future Prediction" Kaggle set). No real causal structure. |

The ARFF body is plain comma-separated; the loader takes column names from the
`@attribute` header and parses the rows after `@data` (robust to header length),
stripping the stray quotes around `'New Delhi'`. A third candidate (a ~55k-row
promotion dataset) was found but **excluded**: its outcome is *promotion*, not
attrition, so it would need a different policy framing.

## Later idea — AI-layoff-wave sources (not started)

A separate, much more ambitious *real-data* direction (see README): Layoffs.fyi
firm-level layoff events, US WARN Act filings, and LLM-extracted earnings-call
AI-capex signals, plus a bilingual (English + Chinese) East-Asian coverage slice.
Each carries its own selection-bias and measurement caveats, recorded here if and
when that work starts.
