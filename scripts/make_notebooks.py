"""Generate the four thin review notebooks for attrition-risk-vs-uplift.

Thin = no logic reimplemented; each notebook imports from `src/` for the optional
regenerate path and otherwise displays the committed `figures/` artifacts so the
project can be reviewed visually without running anything. The notebooks are
committed *with outputs* — regenerate and re-execute them with:

    python scripts/make_notebooks.py        # rewrite the .ipynb skeletons
    make notebooks                           # rewrite + execute in place

This script is deterministic and idempotent; it is not part of the analysis
pipeline (`make all`).
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

_n = 0
def cid(prefix):
    global _n
    _n += 1
    return f"{prefix}-{_n}"

def md(prefix, text):
    return {"cell_type": "markdown", "id": cid(prefix), "metadata": {}, "source": text}

def code(prefix, text):
    return {"cell_type": "code", "id": cid(prefix), "metadata": {},
            "execution_count": None, "outputs": [], "source": text}

BOOTSTRAP = """import os, sys
from pathlib import Path

# Make the repo root the working dir and importable, whether this notebook is
# launched from notebooks/ or from the repo root.
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Image
pd.set_option("display.width", 120)
print("repo root:", ROOT)"""


def write(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", path)


# ── 01 · EDA ──────────────────────────────────────────────────────────────
write("01_eda.ipynb", [
    md("eda", "# 01 · EDA — the data behind the argument\n"
       "\n> **Synthetic data.** IBM HR Analytics benchmark; every number is "
       "illustrative of method, not a real workforce.\n"
       "\nA *thin* notebook — all logic lives in `src/`. This one loads the "
       "cleaned frame and shows the shape of the problem: the target balance "
       "and the treatment (`OverTime`) prevalence the whole causal argument "
       "rests on."),
    code("eda", BOOTSTRAP),
    code("eda",
         "from src.config import TARGET\n"
         "from src.causal.common import TREATMENT\n"
         "from src.data.load import load_clean, train_test\n"
         "\n"
         "df = load_clean()\n"
         "print('clean frame:', df.shape)\n"
         "print(f'\\n{TARGET} balance (1 = leaves):')\n"
         "print(df[TARGET].value_counts(normalize=True).round(3).to_string())"),
    md("eda", "## Stratified split + the treatment\n"
       "\nThe split is fixed-seed and stratified, so the ~16% base rate is "
       "preserved in both folds (no leakage). `OverTime` is the binary "
       "intervention studied in Phases 3–5b."),
    code("eda",
         "train_df, test_df = train_test(df)\n"
         "print(f'train/test rows: {len(train_df)} / {len(test_df)}')\n"
         "print(f'train {TARGET}: {train_df[TARGET].mean():.3f}   "
         "test {TARGET}: {test_df[TARGET].mean():.3f}')\n"
         "\n"
         "on_ot = (df[TREATMENT].astype(str).str.strip() == 'Yes').mean()\n"
         "print(f'\\n{TREATMENT} prevalence (the lever): {on_ot:.3f}')\n"
         "print('\\nraw attrition by overtime (Phase 3 adjusts this causally):')\n"
         "print(df.groupby(TREATMENT)[TARGET].mean().round(3).to_string())"),
    md("eda", "## Feature mix"),
    code("eda", "print(df.dtypes.value_counts().to_string())"),
    md("eda", "**Takeaway.** ~16% leave; ~25% are on overtime, and raw attrition "
       "is much higher among them. Whether that gap is *causal* — and whether "
       "the highest-risk employees are the ones the lever can move — is Phases "
       "3–4. Next: `02_prediction.ipynb`."),
])

# ── 02 · Prediction ───────────────────────────────────────────────────────
write("02_prediction.ipynb", [
    md("pred", "# 02 · Prediction — who is likely to leave\n"
       "\n> Synthetic data; method demonstration.\n"
       "\nThin wrapper over `src/predict` and `src/interpret`. Regenerate the "
       "artifacts with `make predict`; here we read the committed metric tables "
       "and figures."),
    code("pred", BOOTSTRAP),
    md("pred", "## Cross-validated + held-out metrics\n"
       "\nThe logistic baseline leads and is calibrated, so its scores can "
       "legitimately be read as *risk*."),
    code("pred",
         "print('5-fold CV (train):')\n"
         "display(pd.read_csv('figures/metrics_cv.csv'))\n"
         "print('Held-out test:')\n"
         "display(pd.read_csv('figures/metrics_test.csv'))"),
    md("pred", "## Calibration"),
    code("pred", "Image('figures/calibration.png')"),
    md("pred", "## What drives the *risk* score (SHAP)\n"
       "\nThese attribute risk; whether they are movable *levers* is a causal "
       "question — Phase 3."),
    code("pred", "Image('figures/shap_summary.png')"),
    md("pred", "**Takeaway.** A usable risk *ranking*, not an action plan: the "
       "score says *who*, not *what to do*. Next: `03_causal.ipynb`."),
])

# ── 03 · Causation ────────────────────────────────────────────────────────
write("03_causal.ipynb", [
    md("cau", "# 03 · Causation — effect, uplift, and the divergence\n"
       "\n> Synthetic data; identification assumptions are demonstrative and "
       "probed with refutation tests *and* a formal sensitivity analysis.\n"
       "\nThin wrapper over `src/causal` and `src/policy`. Regenerate with "
       "`make causal && make policy` (the causal forest takes a minute); here "
       "we read the committed results."),
    code("cau", BOOTSTRAP),
    md("cau", "## Identification + refutation (DoWhy)\n"
       "\nBackdoor ATE of `OverTime` on attrition, with three refuters. Read: "
       "placebo new-effect ≈ 0 = good; random-common-cause / subset ≈ estimated "
       "= robust."),
    code("cau", "display(pd.read_csv('figures/refutation.csv').round(4))"),
    md("cau", "## Stress-testing the *unmeasured*-confounding assumption (Ext A)\n"
       "\nThe refuters probe the no-unobserved-confounding assumption only "
       "qualitatively; this quantifies it — the Cinelli–Hazlett robustness value "
       "(**RV 0.24**) and the VanderWeele **E-value 5.13**. Read as method: the "
       "synthetic data has no ground-truth causal structure, so these numbers "
       "show *how* you would test robustness, not that a real overtime effect "
       "is robust."),
    code("cau",
         "display(pd.read_csv('figures/sensitivity.csv').round(4))\n"
         "Image('figures/sensitivity.png')"),
    md("cau", "## Heterogeneous uplift (EconML CausalForestDML)\n"
       "\nThe same intervention helps some employees far more than others — that "
       "spread is what a sensible policy targets."),
    code("cau",
         "cate = pd.read_csv('figures/cate_test.csv')\n"
         "print('columns:', list(cate.columns))\n"
         "display(cate.describe().round(4))"),
    md("cau", "## The divergence result — the figure the project exists for\n"
       "\nRisk ranking ≠ uplift ranking."),
    code("cau",
         "display(pd.read_csv('figures/divergence_stats.csv'))\n"
         "Image('figures/divergence.png')"),
    md("cau", "## Does the divergence generalize? A second lever (Ext B)\n"
       "\nRe-running the whole risk-vs-uplift machinery on an independent lever — "
       "`BusinessTravel = Travel_Frequently` — against the *same* risk score. The "
       "OverTime column reproduces the result above to the digit (a consistency "
       "check); BusinessTravel diverges from risk too. Risk ≠ uplift is not an "
       "overtime artefact."),
    code("cau",
         "display(pd.read_csv('figures/levers.csv').round(3))\n"
         "Image('figures/levers.png')"),
    md("cau", "## Causally-grounded policy simulation\n"
       "\nUplift-targeting beats risk-targeting at every budget; the full causal "
       "redo is 16.0%→11.1%, not v1's naive 16.0%→7.8%."),
    code("cau",
         "display(pd.read_csv('figures/policy_simulation.csv').round(4))\n"
         "Image('figures/policy_simulation.png')"),
    md("cau", "**Takeaway.** Targeting the most *at-risk* is not targeting the "
       "most *influenceable*; acting on risk wastes a third or more of "
       "achievable retention — and the effect is robust to plausible hidden "
       "confounding and reproduces on a second lever. Next: `04_ethics.ipynb`."),
])

# ── 04 · Ethics ───────────────────────────────────────────────────────────
write("04_ethics.ipynb", [
    md("eth", "# 04 · Ethics — who gets acted on, and where it ports\n"
       "\n> Synthetic data; read magnitudes as a method demonstration.\n"
       "\nThin wrapper over `src/ethics`. Regenerate with "
       "`make ethics && make transport`."),
    code("eth", BOOTSTRAP),
    md("eth", "## Fairness audit — who gets flagged (top-20%)\n"
       "\nDP ratio < 0.80 trips the four-fifths rule; EO diff = error-rate gap "
       "vs actual attrition. Risk-targeting amplifies disparity; "
       "uplift-targeting partly corrects it but isn't free."),
    code("eth",
         "display(pd.read_csv('figures/fairness_metrics.csv').round(3))\n"
         "Image('figures/fairness_selection_rates.png')"),
    md("eth", "## The efficacy–fairness frontier (Ext C)\n"
       "\nA single dial λ interpolates the targeting rule from pure-risk (λ=0) to "
       "pure-uplift (λ=1) at a fixed budget. Moving toward uplift is a *Pareto "
       "improvement on the worst-affected group* — it averts more attrition **and** "
       "lifts the weakest four-fifths ratio (0.21→0.47) — but the gains are "
       "attribute-specific (Gender parity erodes) and **no rule clears four-fifths "
       "on every attribute**. The endpoints reproduce the audit above."),
    code("eth",
         "display(pd.read_csv('figures/fairness_frontier.csv').round(3))\n"
         "Image('figures/fairness_frontier.png')"),
    md("eth", "## Transportability — validated where?\n"
       "\nMarkets defined by overtime prevalence (ESS confirms each is a real "
       "sample, not a few upweighted outliers). The risk *ranking* ports "
       "(ROC-AUC holds); the *policy* does not."),
    code("eth",
         "cols = ['market', 'overtime_share', 'ess_share', 'roc_auc', 'brier',\n"
         "        'baseline_rate', 'averted_uplift']\n"
         "display(pd.read_csv('figures/transportability.csv')[cols].round(3))\n"
         "Image('figures/transportability.png')"),
    md("eth", "**Takeaway.** Choosing a targeting rule is a distributive-justice "
       "decision, not a neutral default; and a model validated in one labour "
       "market is not thereby validated for another. See `model_card.md` and "
       "`FRAMING.md` §5–6."),
])

print("done")
