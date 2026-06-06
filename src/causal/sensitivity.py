"""Phase 3 (extension) — sensitivity to unobserved confounding.

The causal claims rest on a *no-unobserved-confounding* assumption that the
refuters in `identify.py` probe only qualitatively (placebo ≈ 0, random-cause /
subset stable). The honest follow-up question a causal reader asks first is
quantitative: **how strong would a confounder we *didn't* measure have to be to
overturn the +0.187 overtime effect?** This module answers it two ways, each a
standard tool stated in its own language:

  1. **Robustness value (Cinelli & Hazlett 2020).** On a linear backdoor fit,
     RV is the partial R² an unobserved confounder must share with *both*
     treatment and outcome to drive the estimate to zero. Reported alongside the
     treatment's own partial R² with the outcome, and benchmarked against the
     single strongest *observed* covariate ("a confounder 1–3× as strong as
     that"). The bias contour plot is the sensemakr-style figure.

  2. **E-value (VanderWeele & Ding 2017).** On the covariate-adjusted risk
     ratio of the average potential outcomes, the E-value is the minimum
     risk-ratio association a confounder must have with *both* treatment and
     outcome — above and beyond the measured covariates — to explain the effect
     away.

Both are computed by hand from a single OLS fit (statsmodels), so the result is
exact and version-independent. Synthetic data: this quantifies the *fragility of
the method's assumption*, not a defended real-world effect.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.causal.common import OUTCOME, TREATMENT, build_causal_frame
from src.config import FIGURES
from src.data.load import load_clean, train_test
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)

warnings.filterwarnings("ignore")

R2_AXIS_MAX = 0.6  # plot partial-R² axes up to 60%


def _design(frame: pd.DataFrame, confounders: list[str], treatment) -> pd.DataFrame:
    """[const, TREATMENT, *confounders] with TREATMENT set to `treatment`
    (a vector for fitting, a scalar 0/1 for counterfactual prediction)."""
    X = frame[confounders].astype(float)
    if np.isscalar(treatment):
        t = pd.Series(float(treatment), index=frame.index, name=TREATMENT)
    else:
        t = pd.Series(np.asarray(treatment, dtype=float), index=frame.index,
                      name=TREATMENT)
    return sm.add_constant(pd.concat([t, X], axis=1), has_constant="add")


def _fit(train_df):
    frame, confounders = build_causal_frame(train_df)
    y = frame[OUTCOME].astype(float).to_numpy()
    design = _design(frame, confounders, frame[TREATMENT].astype(float).to_numpy())
    model = sm.OLS(y, design).fit()
    return model, frame, confounders


def cinelli_hazlett(model) -> dict:
    """Partial R² of the treatment with the outcome, and the robustness value
    RV_{q=1} (the partial R² that kills the point estimate)."""
    t = float(abs(model.tvalues[TREATMENT]))
    dof = int(model.df_resid)
    r2_yd_x = t**2 / (t**2 + dof)
    f = t / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2)
    return {
        "coef": float(model.params[TREATMENT]),
        "se": float(model.bse[TREATMENT]),
        "t_stat": t,
        "dof": dof,
        "partial_r2_treatment_outcome": r2_yd_x,
        "robustness_value": float(rv),
    }


def benchmark_strongest(model, frame, confounders) -> dict:
    """Partial R² (with treatment and with outcome) of the single strongest
    observed covariate — the yardstick the unobserved confounder is sized against."""
    tvals = model.tvalues.drop(["const", TREATMENT], errors="ignore").abs()
    top = str(tvals.idxmax())
    dof_y = int(model.df_resid)
    r2_y_top = float(model.tvalues[top] ** 2 / (model.tvalues[top] ** 2 + dof_y))

    d = frame[TREATMENT].astype(float).to_numpy()
    dmodel = sm.OLS(d, sm.add_constant(frame[confounders].astype(float))).fit()
    dof_d = int(dmodel.df_resid)
    r2_d_top = float(dmodel.tvalues[top] ** 2 / (dmodel.tvalues[top] ** 2 + dof_d))
    return {"covariate": top, "r2_with_treatment": r2_d_top, "r2_with_outcome": r2_y_top}


def e_value(model, frame, confounders) -> dict:
    """E-value on the covariate-adjusted causal risk ratio of the average
    potential outcomes E[Y(1)] / E[Y(0)]."""
    p0 = float(np.clip(model.predict(_design(frame, confounders, 0.0)).mean(), 1e-6, 1 - 1e-6))
    p1 = float(np.clip(model.predict(_design(frame, confounders, 1.0)).mean(), 1e-6, 1 - 1e-6))
    rr = p1 / p0
    rr_star = max(rr, 1.0 / rr)
    ev = rr_star + np.sqrt(rr_star * (rr_star - 1.0))
    return {"p0": p0, "p1": p1, "risk_ratio": rr, "e_value": float(ev)}


def run(train_df) -> dict:
    model, frame, confounders = _fit(train_df)
    return {
        **cinelli_hazlett(model),
        "benchmark": benchmark_strongest(model, frame, confounders),
        **e_value(model, frame, confounders),
        "_model": model,
    }


def plot(res):
    coef, se, dof = res["coef"], res["se"], res["dof"]
    sign = np.sign(coef) or 1.0
    g = np.linspace(0, R2_AXIS_MAX, 300)
    RDZ, RYZ = np.meshgrid(g, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = se * np.sqrt(dof) * np.sqrt(RYZ * RDZ / (1 - RDZ))
    adjusted = coef - sign * bias

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    levels = sorted({0.0, 0.05, 0.10, 0.15, round(abs(coef), 3)})
    cs = ax.contour(RDZ, RYZ, adjusted, levels=levels, colors="#1f77b4", linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f")
    cz = ax.contour(RDZ, RYZ, adjusted, levels=[0.0], colors="#d62728", linewidths=2.2)
    ax.clabel(cz, inline=True, fontsize=8, fmt="effect = 0")

    # observed estimate (no unobserved confounding)
    ax.scatter([0], [0], marker="^", s=70, color="k", zorder=5)
    ax.annotate(f"observed\n{coef:+.3f}", (0, 0), textcoords="offset points",
                xytext=(8, 8), fontsize=8)

    # robustness value on the diagonal
    rv = res["robustness_value"]
    ax.scatter([rv], [rv], marker="D", s=55, color="#6a3d9a", zorder=5)
    ax.annotate(f"RV = {rv:.2f}", (rv, rv), textcoords="offset points",
                xytext=(8, -4), fontsize=8, color="#6a3d9a")

    # benchmark: confounder 1–3× as strong as the strongest observed covariate
    b = res["benchmark"]
    short = b["covariate"].replace("cat__", "").replace("num__", "")
    for k in (1, 2, 3):
        x, y = min(k * b["r2_with_treatment"], R2_AXIS_MAX), min(k * b["r2_with_outcome"], R2_AXIS_MAX)
        ax.scatter([x], [y], s=40, color="#ff7f0e", zorder=5)
        ax.annotate(f"{k}×", (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="#cc6600")

    ax.set_xlim(0, R2_AXIS_MAX)
    ax.set_ylim(0, R2_AXIS_MAX)
    ax.set_xlabel("partial R² of unobserved confounder with TREATMENT (overtime)")
    ax.set_ylabel("partial R² with OUTCOME (attrition)")
    ax.set_title("Sensitivity to unobserved confounding\n"
                 f"(orange = a confounder k× as strong as '{short}')", fontsize=10)
    return viz.save(fig, "sensitivity.png")


def main() -> None:
    df = load_clean()
    train_df, _ = train_test(df)
    res = run(train_df)
    path = plot(res)

    b = res["benchmark"]
    pd.DataFrame([{
        "coef": res["coef"], "se": res["se"], "t_stat": res["t_stat"],
        "partial_r2_treatment_outcome": res["partial_r2_treatment_outcome"],
        "robustness_value": res["robustness_value"],
        "risk_ratio": res["risk_ratio"], "e_value": res["e_value"],
        "benchmark_covariate": b["covariate"],
        "benchmark_r2_with_treatment": b["r2_with_treatment"],
        "benchmark_r2_with_outcome": b["r2_with_outcome"],
    }]).to_csv(FIGURES / "sensitivity.csv", index=False)

    rv = res["robustness_value"]
    print("=== Sensitivity to unobserved confounding (OverTime → attrition) ===\n")
    print(f"Linear backdoor effect: {res['coef']:+.4f} "
          f"(se {res['se']:.4f}, t {res['t_stat']:.1f})")
    print(f"Treatment's own partial R² with the outcome: "
          f"{res['partial_r2_treatment_outcome']:.3f}")
    print(f"\nRobustness value (RV): {rv:.3f}")
    print(f"  → an unobserved confounder must explain ~{rv:.0%} of the residual "
          "variance in\n    BOTH overtime and attrition to drive the effect to "
          "zero.")
    print(f"\nStrongest observed covariate: '{b['covariate']}'")
    print(f"  partial R² with treatment {b['r2_with_treatment']:.3f}, "
          f"with outcome {b['r2_with_outcome']:.3f}")
    cmp = "more" if rv > max(b["r2_with_treatment"], b["r2_with_outcome"]) else "less"
    print(f"  → the confounder would have to be {cmp} explanatory than the "
          "strongest thing we\n    already measured.")
    print(f"\nE-value (adjusted risk ratio {res['risk_ratio']:.2f}): "
          f"{res['e_value']:.2f}")
    print(f"  → confounding would need a risk-ratio association of "
          f"≥{res['e_value']:.2f} with both\n    overtime and attrition, beyond "
          "the measured covariates, to explain it away.")
    print(f"\nFigure -> {path}")
    print(
        "\nReading: the overtime effect is moderately robust — not bulletproof, "
        "but it would\ntake a fairly strong hidden confounder to overturn. On "
        "synthetic data this is a\nstatement about the method's honesty (we can "
        "*quantify* the assumption's fragility),\nnot a defence of a real-world "
        "effect size."
    )


if __name__ == "__main__":
    main()
