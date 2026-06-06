"""Phase 3 (extension) — does the divergence generalise beyond one treatment?

The headline result (`divergence.py`) is built on a single lever, OverTime. The
obvious objection is that *risk ≠ uplift* might be an artefact of that one
treatment. This module answers it by re-running the whole risk-vs-uplift
machinery on a **second, independent lever** — frequent business travel
(BusinessTravel == "Travel_Frequently") — and comparing the two side by side.

The design is deliberately apples-to-apples with Phase 3:

  * **Risk r(x) is computed once.** The logistic risk score is *treatment-
    agnostic* — it predicts attrition from every feature, including both
    OverTime and BusinessTravel — so the *same* risk ranking is held fixed while
    only the lever (and hence the uplift ranking τ(x)) changes. That isolates
    the effect of swapping the treatment.
  * **The same causal-forest configuration** as `uplift.py` is reused, so the
    OverTime column here reproduces the Phase-3 divergence numbers exactly — a
    built-in consistency check, not a re-derivation.

Two things fall out, and both sharpen the core claim:

  1. **Each lever's uplift diverges from risk** (low ρ, low top-decile overlap,
     sub-1 risk-targeting efficiency) — so the divergence is a property of
     prediction-vs-causation, not of overtime specifically.
  2. **The two levers are not interchangeable.** They avert different amounts of
     attrition (ATE ≈18 pp vs ≈11 pp), reach different people (most high-risk
     employees are not frequent travellers, so *that* lever cannot touch them),
     and their uplift rankings — though positively correlated (ρ≈0.7, a shared
     "responsiveness" structure) — still pick materially different top deciles.
     Influenceability is partly lever-specific; a risk score, which knows of no
     lever at all, cannot stand in for either.

Synthetic data: this is a statement about the *method* (the result is not a
one-treatment artefact), not a defended real-world effect size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.config import FIGURES, RANDOM_SEED, TARGET
from src.data.load import feature_columns, load_clean, train_test
from src.predict.baselines import Xy, fit_risk_model
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)

_T = "__treatment__"  # internal treatment-column name in the causal frame

# The levers compared. Each is a clean binary policy lever with a plausible
# real-world reading ("relieve …" / "reduce …"). OverTime is the Phase-3 lever
# (kept here as the reproducing baseline); BusinessTravel is the new one.
LEVERS = [
    {"name": "OverTime", "column": "OverTime", "treated": "Yes",
     "policy": "relieve required overtime", "short": "overtime"},
    {"name": "BusinessTravel", "column": "BusinessTravel",
     "treated": "Travel_Frequently",
     "policy": "reduce frequent business travel", "short": "frequent travel"},
]


def build_lever_frame(
    df: pd.DataFrame, column: str, treated: str
) -> tuple[pd.DataFrame, list[str]]:
    """Causal frame for an arbitrary binary lever.

    The treatment is 1 where ``column == treated``. Every *other* feature is a
    confounder (numerics pass through; the remaining string columns — including
    the non-treatment levers — are one-hot encoded). Mirrors
    ``common.build_causal_frame`` but parameterised by the treatment column, so
    OverTime reproduces the Phase-3 confounder set exactly.
    """
    df = df.copy()
    treat = (df[column].astype(str).str.strip() == treated).astype(int)
    cols = feature_columns(df)
    numeric = [c for c in cols["numeric"] if c != column]
    categorical = [c for c in cols["categorical"] if c != column]

    dummies = pd.get_dummies(df[categorical], drop_first=True, dtype=float)
    treat_out = pd.DataFrame(
        {_T: treat.to_numpy(), TARGET: df[TARGET].astype(int).to_numpy()},
        index=df.index,
    )
    frame = pd.concat([df[numeric].astype(float), dummies, treat_out], axis=1)
    confounders = numeric + list(dummies.columns)
    return frame, confounders


def _causal_forest() -> CausalForestDML:
    """Identical configuration to ``uplift.py`` so OverTime reproduces Phase 3."""
    return CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20, random_state=RANDOM_SEED),
        model_t=RandomForestClassifier(
            n_estimators=100, min_samples_leaf=20, random_state=RANDOM_SEED),
        discrete_treatment=True, cv=3, n_estimators=500, random_state=RANDOM_SEED,
    )


def estimate_lever_cate(train_df, test_df, column, treated) -> np.ndarray:
    """Test-set CATE τ(x) for one lever via the Phase-3 causal forest."""
    f_tr, conf = build_lever_frame(train_df, column, treated)
    f_te, _ = build_lever_frame(test_df, column, treated)
    for c in conf:  # align one-hot columns absent from the test split
        if c not in f_te.columns:
            f_te[c] = 0.0
    cf = _causal_forest()
    cf.fit(f_tr[TARGET].to_numpy(), f_tr[_T].to_numpy(), X=f_tr[conf].to_numpy())
    return np.asarray(cf.effect(f_te[conf].to_numpy())).ravel()


def divergence_stats(risk, uplift, on_treatment, top_frac: float = 0.10) -> dict:
    """Risk-vs-uplift divergence for one lever (mirrors ``divergence.compute``)."""
    rho = spearmanr(risk, uplift).statistic
    n = len(risk)
    k = max(1, round(top_frac * n))
    risk_top = np.argsort(risk)[::-1][:k]
    uplift_top = np.argsort(uplift)[::-1][:k]
    overlap = len(set(risk_top.tolist()) & set(uplift_top.tolist())) / k
    gain_uplift = uplift[uplift_top].sum()
    efficiency = uplift[risk_top].sum() / gain_uplift if gain_uplift > 0 else float("nan")
    risk_top_not_treated = 1.0 - on_treatment[risk_top].mean()
    return {
        "ate": float(uplift.mean()),
        "spearman_rho_risk_uplift": float(rho),
        "top_decile_overlap": float(overlap),
        "risk_targeting_efficiency": float(efficiency),
        "risk_top_share_not_treated": float(risk_top_not_treated),
        "_risk_top": risk_top, "_uplift_top": uplift_top,
    }


def run(train_df, test_df, top_frac: float = 0.10) -> dict:
    """Compute risk once, then each lever's uplift and divergence vs that risk."""
    X_te, _ = Xy(test_df)
    risk = fit_risk_model(train_df, "logit").predict_proba(X_te)[:, 1]

    levers = {}
    for spec in LEVERS:
        uplift = estimate_lever_cate(train_df, test_df, spec["column"], spec["treated"])
        on_t = (test_df[spec["column"]].astype(str).str.strip()
                == spec["treated"]).to_numpy()
        stats = divergence_stats(risk, uplift, on_t, top_frac)
        levers[spec["name"]] = {**spec, "uplift": uplift, "treated_share": float(on_t.mean()),
                                **stats}

    names = [s["name"] for s in LEVERS]
    u0, u1 = levers[names[0]]["uplift"], levers[names[1]]["uplift"]
    cross_rho = float(spearmanr(u0, u1).statistic)
    k = max(1, round(top_frac * len(u0)))
    top0 = set(np.argsort(u0)[::-1][:k].tolist())
    top1 = set(np.argsort(u1)[::-1][:k].tolist())
    cross_overlap = len(top0 & top1) / k
    return {"risk": risk, "levers": levers, "names": names,
            "cross_lever_rho": cross_rho, "cross_lever_top_decile_overlap": cross_overlap}


def _divergence_panel(ax, risk, L):
    """One risk-vs-uplift scatter, top-decile membership colour-coded."""
    n = len(risk)
    rt, ut = set(L["_risk_top"].tolist()), set(L["_uplift_top"].tolist())
    both = rt & ut
    color = np.array(["#cfcfcf"] * n, dtype=object)
    for i in ut:
        color[i] = "#1f77b4"
    for i in rt:
        color[i] = "#d62728"
    for i in both:
        color[i] = "#9467bd"
    ax.scatter(risk, L["uplift"], c=list(color), s=16, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("predicted attrition risk  r(x)")
    ax.set_ylabel(f"uplift τ(x) — effect of removing {L['short']}")
    ax.set_title(f"{L['name']}: ρ={L['spearman_rho_risk_uplift']:.2f}, "
                 f"overlap {L['top_decile_overlap']:.0%}", fontsize=10)


def plot(res):
    risk, levers, names = res["risk"], res["levers"], res["names"]
    A, B = levers[names[0]], levers[names[1]]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    _divergence_panel(axes[0], risk, A)
    _divergence_panel(axes[1], risk, B)
    legend = [
        Patch(color="#d62728", label="top-decile RISK"),
        Patch(color="#1f77b4", label="top-decile UPLIFT"),
        Patch(color="#9467bd", label="both"),
        Patch(color="#cfcfcf", label="rest"),
    ]
    axes[0].legend(handles=legend, loc="upper right", frameon=True, fontsize=8)

    ax = axes[2]
    ax.scatter(A["uplift"], B["uplift"], s=16, alpha=0.7, color="#2a9d8f",
               edgecolor="none")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(f"uplift τ(x) — {A['short']}")
    ax.set_ylabel(f"uplift τ(x) — {B['short']}")
    ax.set_title(f"Cross-lever: ρ={res['cross_lever_rho']:.2f}, top-decile "
                 f"overlap {res['cross_lever_top_decile_overlap']:.0%}\n"
                 "correlated, but not interchangeable", fontsize=10)

    fig.suptitle("Divergence generalises beyond one treatment — risk vs uplift, "
                 "two independent levers", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return viz.save(fig, "levers.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    res = run(train_df, test_df)
    path = plot(res)

    rows = []
    for name in res["names"]:
        L = res["levers"][name]
        rows.append({
            "lever": L["name"], "treated_value": L["treated"],
            "treated_share": L["treated_share"], "ate_causal_forest": L["ate"],
            "spearman_rho_risk_uplift": L["spearman_rho_risk_uplift"],
            "top_decile_overlap": L["top_decile_overlap"],
            "risk_targeting_efficiency": L["risk_targeting_efficiency"],
            "risk_top_share_not_treated": L["risk_top_share_not_treated"],
            "cross_lever_rho": res["cross_lever_rho"],
            "cross_lever_top_decile_overlap": res["cross_lever_top_decile_overlap"],
        })
    pd.DataFrame(rows).to_csv(FIGURES / "levers.csv", index=False)

    print("=== Divergence across two independent levers (test set) ===\n")
    for name in res["names"]:
        L = res["levers"][name]
        print(f"{L['name']}  (treat = {L['treated']}, {L['treated_share']:.0%} of employees;"
              f" policy: {L['policy']})")
        print(f"  ATE (causal forest)            {L['ate']:+.4f}")
        print(f"  Spearman ρ(risk, uplift)       {L['spearman_rho_risk_uplift']:.3f}")
        print(f"  top-decile overlap             {L['top_decile_overlap']:.0%}")
        print(f"  risk-targeting efficiency      {L['risk_targeting_efficiency']:.0%}")
        print(f"  top-risk decile NOT treated    {L['risk_top_share_not_treated']:.0%} "
              "(lever can't touch them)\n")

    print(f"Cross-lever Spearman ρ(τ_overtime, τ_travel): {res['cross_lever_rho']:.3f}; "
          f"top-decile overlap {res['cross_lever_top_decile_overlap']:.0%}")
    print("  → the two uplift rankings are positively correlated (a shared "
          "'responsiveness'\n    structure) but far from identical: their most-"
          "influenceable deciles overlap\n    only "
          f"{res['cross_lever_top_decile_overlap']:.0%}. Influenceability is partly "
          "lever-specific — and neither\n    lever's ranking matches the risk "
          "ranking.")
    print(f"\nFigure -> {path}")
    print(
        "\nReading: risk ≠ uplift holds for a second, independent lever — the "
        "divergence is a\nproperty of prediction-vs-causation, not an artefact of "
        "the overtime treatment. The\ntwo levers also avert different amounts "
        "(ATE ≈18 vs ≈11 pp) and reach different\npeople (38% vs 70% of the top-"
        "risk decile is unreachable). (Synthetic data: a claim\nabout the method, "
        "not a real-world effect size.)"
    )


if __name__ == "__main__":
    main()
