"""Phase 4 — causally-grounded policy simulation.

Rebuilds the v1 retention counterfactual on CATE estimates instead of the
predictive model, and asks the operational question: given a fixed retention
budget — HR can relieve required overtime for k employees — *whom do you pick?*

Three strategies are compared at each budget:
  - risk-targeted   : top-k by predicted attrition risk r(x)   (the naive policy)
  - uplift-targeted : top-k by realized uplift                 (the causal policy)
  - random          : spend the budget at random              (lower bound)

The realized causal reduction for an employee is tau(x) *only if they are
actually on overtime* (the lever can be pulled); it is 0 otherwise. So a policy
that spends budget on high-risk employees who aren't on overtime buys nothing —
which is exactly why risk-targeting underperforms here.

Baseline is the observed test attrition rate, so the post-policy numbers sit on
the same 16.1% anchor as v1. The v1 "16.1% -> 7.8%" claim is revisited under the
full causal intervention (relieve overtime for everyone who has it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.common import TREATMENT
from src.causal.uplift import estimate_cate
from src.config import FIGURES, TARGET
from src.data.load import load_clean, train_test
from src.predict.baselines import Xy, fit_risk_model
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)

V1_POLICY = 0.078  # v1's naive post-intervention claim (16.1% -> 7.8%)
TABLE_BUDGETS = (0.05, 0.10, 0.20)
PLOT_BUDGETS = np.round(np.arange(0.0, 0.51, 0.025), 3)


def _inputs(train_df, test_df):
    """Per-employee risk r(x), uplift tau(x), and realized reduction."""
    X_te, _ = Xy(test_df)
    risk = fit_risk_model(train_df, "logit").predict_proba(X_te)[:, 1]
    cate, _ = estimate_cate(train_df, test_df)
    tau = cate["causal_forest"]
    on_ot = (test_df[TREATMENT].astype(str).str.strip() == "Yes").to_numpy()
    realized = tau * on_ot  # the lever only moves employees actually on overtime
    return risk, tau, on_ot, realized


def _post_rates(risk, realized, baseline, budgets):
    """Post-policy attrition rate for each strategy across budget fractions."""
    n = len(risk)
    risk_order = np.argsort(risk)[::-1]
    uplift_order = np.argsort(realized)[::-1]
    mean_realized = realized.mean()

    rows = []
    for f in budgets:
        k = int(round(f * n))
        red_risk = realized[risk_order[:k]].sum() / n
        red_uplift = realized[uplift_order[:k]].sum() / n
        red_random = (k / n) * mean_realized  # expected
        rows.append({
            "budget_frac": f, "k": k,
            "reduction_risk": red_risk,
            "reduction_uplift": red_uplift,
            "post_risk": baseline - red_risk,
            "post_uplift": baseline - red_uplift,
            "post_random": baseline - red_random,
            "efficiency_risk_vs_uplift": red_risk / red_uplift if red_uplift > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def simulate(train_df, test_df):
    risk, tau, on_ot, realized = _inputs(train_df, test_df)
    baseline = float(test_df[TARGET].mean())
    n = len(risk)

    table = _post_rates(risk, realized, baseline, TABLE_BUDGETS)
    curve = _post_rates(risk, realized, baseline, PLOT_BUDGETS)

    # Full intervention: relieve overtime for everyone who has it (v1's redo).
    red_full = realized.sum() / n
    stats = {
        "baseline_rate": baseline,
        "share_on_overtime": float(on_ot.mean()),
        "ate_among_on_overtime": float(tau[on_ot].mean()) if on_ot.any() else float("nan"),
        "full_intervention_reduction": float(red_full),
        "full_intervention_post_rate": float(baseline - red_full),
        "v1_naive_post_rate": V1_POLICY,
    }
    return table, curve, stats


def plot(curve, stats):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = curve["budget_frac"] * 100
    ax.plot(x, curve["post_uplift"] * 100, "-o", ms=3, color="#1f77b4",
            label="uplift-targeted (causal)")
    ax.plot(x, curve["post_risk"] * 100, "-o", ms=3, color="#d62728",
            label="risk-targeted (naive)")
    ax.plot(x, curve["post_random"] * 100, "--", color="#7f7f7f",
            label="random")
    ax.axhline(stats["baseline_rate"] * 100, color="k", lw=0.8,
               label=f"baseline {stats['baseline_rate']:.1%}")
    ax.axhline(V1_POLICY * 100, color="#2ca02c", lw=0.8, ls=":",
               label=f"v1 naive claim {V1_POLICY:.1%}")
    ax.set_xlabel("retention budget — share of workforce given the intervention")
    ax.set_ylabel("post-policy attrition rate (%)")
    ax.set_title("Targeting the influenceable beats targeting the risky")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    return viz.save(fig, "policy_simulation.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    table, curve, stats = simulate(train_df, test_df)
    path = plot(curve, stats)

    out = table.copy()
    for c in ["budget_frac", "reduction_risk", "reduction_uplift",
              "post_risk", "post_uplift", "post_random"]:
        out[c] = out[c].round(4)
    out.to_csv(FIGURES / "policy_simulation.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("=== Causally-grounded policy simulation (test set) ===")
    print(f"Baseline attrition rate: {stats['baseline_rate']:.1%}  |  "
          f"on overtime: {stats['share_on_overtime']:.1%}  |  "
          f"mean uplift among them: {stats['ate_among_on_overtime']:+.3f}\n")
    show = table.assign(
        budget=lambda d: (d["budget_frac"] * 100).map(lambda v: f"{v:.0f}%"),
        risk=lambda d: (d["post_risk"] * 100).map(lambda v: f"{v:.1f}%"),
        uplift=lambda d: (d["post_uplift"] * 100).map(lambda v: f"{v:.1f}%"),
        random=lambda d: (d["post_random"] * 100).map(lambda v: f"{v:.1f}%"),
        eff=lambda d: d["efficiency_risk_vs_uplift"].map(lambda v: f"{v:.0%}"),
    )[["budget", "k", "risk", "uplift", "random", "eff"]]
    show.columns = ["budget", "k", "post(risk)", "post(uplift)", "post(random)",
                    "risk/uplift eff"]
    print(show.to_string(index=False))

    print(f"\nFull intervention (relieve overtime for all {stats['share_on_overtime']:.0%} "
          f"who have it):\n  {stats['baseline_rate']:.1%} -> "
          f"{stats['full_intervention_post_rate']:.1%}  "
          f"(causal reduction {stats['full_intervention_reduction']*100:.1f} pp).")
    print(f"  v1's naive predictive sim claimed {stats['baseline_rate']:.1%} -> "
          f"{V1_POLICY:.1%}; the causal redo does NOT reach it — the naive number "
          "overstated\n  what removing overtime can actually do.")
    print(f"\nFigure -> {path}")
    print(
        "\nHeadline: at the same budget, uplift-targeting leaves a lower attrition "
        "rate than\nrisk-targeting — the gap is the retention wasted by ranking on "
        "risk instead of\ninfluenceability."
    )


if __name__ == "__main__":
    main()
