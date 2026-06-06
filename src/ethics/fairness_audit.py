"""Phase 5 — fairness audit of the targeting decision.

The ethical object of study is not the model's accuracy but *who gets acted on*.
A retention budget flags the top-k employees for intervention; this module asks
whether that flag falls unevenly across sensitive attributes — Gender,
MaritalStatus, and an Age band — and whether the choice of *targeting rule*
(risk vs uplift) changes the answer.

For each attribute we report, via fairlearn:
  - demographic-parity difference : gap in selection (flagged) rate across groups
  - demographic-parity ratio      : min/max selection rate (the 80%-rule lens)
  - equalized-odds difference     : gap in TPR/FPR using actual attrition as label

Demographic parity and equalized odds pull in opposite directions when a group's
true attrition rate genuinely differs — that tension *is* the governance
problem, so per-group base rates are plotted alongside the selection rates
rather than hidden. Synthetic data: read the magnitudes as a method
demonstration, not a finding about real workforces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from src.causal.common import TREATMENT
from src.causal.uplift import estimate_cate
from src.config import FIGURES, TARGET
from src.data.load import load_clean, train_test
from src.predict.baselines import Xy, fit_risk_model
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)

SENSITIVE = ["Gender", "MaritalStatus", "AgeBand"]
BUDGET = 0.20  # flag the top 20% for intervention


def _flags(train_df, test_df, budget: float = BUDGET):
    """Binary 'flagged for intervention' vectors under each targeting rule."""
    X_te, _ = Xy(test_df)
    risk = fit_risk_model(train_df, "logit").predict_proba(X_te)[:, 1]
    cate, _ = estimate_cate(train_df, test_df)
    on_ot = (test_df[TREATMENT].astype(str).str.strip() == "Yes").to_numpy()
    realized = cate["causal_forest"] * on_ot

    n = len(risk)
    k = int(round(budget * n))
    flag_risk = np.zeros(n, dtype=int)
    flag_risk[np.argsort(risk)[::-1][:k]] = 1
    flag_uplift = np.zeros(n, dtype=int)
    flag_uplift[np.argsort(realized)[::-1][:k]] = 1
    return {"risk": flag_risk, "uplift": flag_uplift}, k


def _sensitive_frame(test_df) -> pd.DataFrame:
    s = pd.DataFrame(index=test_df.index)
    s["Gender"] = test_df["Gender"].astype(str)
    s["MaritalStatus"] = test_df["MaritalStatus"].astype(str)
    s["AgeBand"] = pd.cut(
        test_df["Age"], bins=[0, 30, 45, 200], labels=["<30", "30-45", "45+"]
    ).astype(str)
    return s


def audit(train_df, test_df):
    y = test_df[TARGET].to_numpy()
    flags, k = _flags(train_df, test_df)
    S = _sensitive_frame(test_df)

    summary_rows = []
    per_group: dict[str, pd.DataFrame] = {}
    for attr in SENSITIVE:
        sf = S[attr].to_numpy()
        # Per-group base attrition rate + selection rate under each rule.
        base = MetricFrame(metrics=selection_rate, y_true=y, y_pred=y,
                           sensitive_features=sf).by_group
        cols = {"base_attrition_rate": base}
        for policy, flag in flags.items():
            mf = MetricFrame(
                metrics={"selection_rate": selection_rate,
                         "tpr": true_positive_rate, "fpr": false_positive_rate},
                y_true=y, y_pred=flag, sensitive_features=sf,
            )
            cols[f"{policy}_selected"] = mf.by_group["selection_rate"]
            summary_rows.append({
                "policy": policy, "attribute": attr,
                "dp_difference": demographic_parity_difference(
                    y, flag, sensitive_features=sf),
                "dp_ratio": demographic_parity_ratio(
                    y, flag, sensitive_features=sf),
                "eo_difference": equalized_odds_difference(
                    y, flag, sensitive_features=sf),
            })
        per_group[attr] = pd.DataFrame(cols)

    return pd.DataFrame(summary_rows), per_group, k


def plot(per_group):
    fig, axes = plt.subplots(1, len(SENSITIVE), figsize=(13, 4.2))
    for ax, attr in zip(axes, SENSITIVE):
        g = per_group[attr]
        x = np.arange(len(g))
        w = 0.27
        ax.bar(x - w, g["base_attrition_rate"] * 100, w, label="actual attrition",
               color="#7f7f7f")
        ax.bar(x, g["risk_selected"] * 100, w, label="risk-targeted",
               color="#d62728")
        ax.bar(x + w, g["uplift_selected"] * 100, w, label="uplift-targeted",
               color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=0, fontsize=8)
        ax.set_title(attr)
        ax.set_ylabel("rate (%)") if attr == SENSITIVE[0] else None
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    fig.suptitle("Who gets flagged for intervention — by group, by targeting rule")
    fig.tight_layout()
    return viz.save(fig, "fairness_selection_rates.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    summary, per_group, k = audit(train_df, test_df)
    path = plot(per_group)

    summary.to_csv(FIGURES / "fairness_metrics.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(f"=== Fairness audit — top-{BUDGET:.0%} flagged for intervention "
          f"(k={k}) ===\n")
    print("DP diff = gap in flagged rate across groups (0 = parity); "
          "DP ratio < 0.80 trips\nthe four-fifths rule; EO diff = gap in "
          "error rates vs actual attrition.\n")
    print(summary.to_string(index=False))

    worst = summary.loc[summary["dp_difference"].idxmax()]
    print(f"\nLargest disparity: {worst['policy']}-targeting on "
          f"{worst['attribute']} — flagged-rate gap {worst['dp_difference']:.0%}, "
          f"ratio {worst['dp_ratio']:.2f}.")
    print(f"\nFigure -> {path}")
    print(
        "\nReading: risk-targeting tracks each group's base attrition rate, so it "
        "concentrates\nthe intervention on the youngest and single employees — "
        "demographic disparity that\njust acting on a risk score would impose. "
        "Uplift-targeting redistributes that flag\ntoward whoever the lever can "
        "actually move; compare the columns to see whether it\nrelieves or "
        "merely relocates the disparity. Either way, the decision to act on\nany "
        "of these scores is a governance choice, not a model output."
    )


if __name__ == "__main__":
    main()
