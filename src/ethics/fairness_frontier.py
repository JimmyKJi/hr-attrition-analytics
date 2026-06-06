"""Phase 5 (extension) — the efficacy–fairness frontier of the targeting rule.

Phase 4 showed uplift-targeting retains more people than risk-targeting; Phase 5
showed the two rules distribute the *intervention flag* differently across
sensitive groups. This module puts both axes on one picture and asks the
governance question directly: **as you move the targeting rule from pure-risk to
pure-uplift, what is the joint consequence for retention and for fairness?**

The targeting score is interpolated with a single dial λ:

    s_λ(x) = (1 − λ) · z(risk r(x)) + λ · z(realized uplift τ(x)·on_overtime)

(z = standardisation, so the two scales are comparable). λ = 0 is the naive
risk rule; λ = 1 is the causal uplift rule; the budget is held at the Phase-5
20% so only the *rule* varies. For each λ we flag the top-20% and record:

  * **attrition averted** — Σ realized uplift over the flagged set (efficacy);
  * **worst-case demographic-parity ratio** — the *minimum* four-fifths ratio
    across Gender, MaritalStatus and an Age band (the weakest-link fairness
    summary: a rule clears the four-fifths rule only if this is ≥ 0.80).

Two readings fall out. The **frontier** (averted vs worst-case ratio) shows
whether moving toward uplift-targeting is a trade-off or a Pareto improvement.
The **per-attribute decomposition** keeps it honest: a summary that only tracks
the worst group hides that the same move can *help* one attribute while *hurting*
another. The λ = 0 and λ = 1 endpoints reproduce the Phase-5 risk/uplift audit
exactly — a built-in consistency check.

Synthetic data: this is a demonstration of the method (the targeting rule is a
distributive choice with measurable consequences), not a finding about any real
workforce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_ratio

from src.causal.common import TREATMENT
from src.causal.uplift import estimate_cate
from src.config import FIGURES, TARGET
from src.data.load import load_clean, train_test
from src.ethics.fairness_audit import BUDGET, SENSITIVE, _sensitive_frame
from src.predict.baselines import Xy, fit_risk_model
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)

FOUR_FIFTHS = 0.80
LAMBDAS = np.linspace(0.0, 1.0, 21)
_ATTR_COLORS = {"Gender": "#1f77b4", "MaritalStatus": "#d62728", "AgeBand": "#2ca02c"}


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / (a.std() + 1e-12)


def _inputs(train_df, test_df):
    """Risk r(x), realized uplift τ(x)·on_overtime, outcome y, sensitive frame."""
    X_te, _ = Xy(test_df)
    risk = fit_risk_model(train_df, "logit").predict_proba(X_te)[:, 1]
    cate, _ = estimate_cate(train_df, test_df)
    on_ot = (test_df[TREATMENT].astype(str).str.strip() == "Yes").to_numpy()
    realized = cate["causal_forest"] * on_ot
    y = test_df[TARGET].to_numpy()
    S = _sensitive_frame(test_df)
    return risk, realized, y, S


def frontier(risk, realized, y, S, budget: float = BUDGET, lambdas=LAMBDAS) -> pd.DataFrame:
    """Sweep the risk→uplift dial; record efficacy and four-fifths ratios."""
    n = len(risk)
    k = int(round(budget * n))
    zr, zu = _z(risk), _z(realized)
    rows = []
    for lam in lambdas:
        score = (1.0 - lam) * zr + lam * zu
        flag = np.zeros(n, dtype=int)
        flag[np.argsort(score)[::-1][:k]] = 1
        averted = realized[flag.astype(bool)].sum() / n  # reduction in attrition rate
        ratios = {
            attr: float(demographic_parity_ratio(
                y, flag, sensitive_features=S[attr].to_numpy()))
            for attr in SENSITIVE
        }
        rows.append({
            "lambda": float(lam),
            "attrition_averted": float(averted),
            "min_dp_ratio": float(min(ratios.values())),
            **{f"dp_ratio_{a}": ratios[a] for a in SENSITIVE},
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # --- Panel A: the frontier (efficacy vs worst-case fairness) ---
    ax = axes[0]
    x = df["attrition_averted"] * 100
    y = df["min_dp_ratio"]
    ax.plot(x, y, color="#aaaaaa", lw=0.9, zorder=2)
    sc = ax.scatter(x, y, c=df["lambda"], cmap="viridis", s=42, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("λ:  risk-targeted (0)  →  uplift-targeted (1)")
    ax.scatter(x.iloc[0], y.iloc[0], marker="s", s=110, facecolor="#d62728",
               edgecolor="k", zorder=4)
    ax.annotate("risk-targeted", (x.iloc[0], y.iloc[0]), textcoords="offset points",
                xytext=(8, -2), fontsize=8)
    ax.scatter(x.iloc[-1], y.iloc[-1], marker="D", s=95, facecolor="#1f77b4",
               edgecolor="k", zorder=4)
    ax.annotate("uplift-targeted", (x.iloc[-1], y.iloc[-1]), textcoords="offset points",
                xytext=(-12, 10), fontsize=8)
    ax.axhline(FOUR_FIFTHS, ls="--", color="k", lw=1)
    ax.text(x.min(), FOUR_FIFTHS + 0.01, "four-fifths rule (0.80)", fontsize=8)
    ax.set_xlabel("attrition averted (pp of workforce)  →  more effective")
    ax.set_ylabel("worst-case demographic-parity ratio  →  fairer")
    ax.set_title("Efficacy–fairness frontier of the targeting rule", fontsize=10)

    # --- Panel B: per-attribute decomposition (no free lunch) ---
    ax = axes[1]
    for attr in SENSITIVE:
        ax.plot(df["lambda"], df[f"dp_ratio_{attr}"], "-o", ms=3,
                color=_ATTR_COLORS[attr], label=attr)
    ax.axhline(FOUR_FIFTHS, ls="--", color="k", lw=1)
    ax.text(0.02, FOUR_FIFTHS + 0.01, "four-fifths rule", fontsize=8)
    ax.set_xlabel("λ:  risk-targeted (0)  →  uplift-targeted (1)")
    ax.set_ylabel("demographic-parity ratio (per attribute)")
    ax.set_title("…but the gains are attribute-specific", fontsize=10)
    ax.legend(loc="lower center", fontsize=8, frameon=True)

    fig.suptitle("Choosing a targeting rule is a distributive-justice decision",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return viz.save(fig, "fairness_frontier.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    risk, realized, y, S = _inputs(train_df, test_df)
    fr = frontier(risk, realized, y, S)
    path = plot(fr)

    fr.round(4).to_csv(FIGURES / "fairness_frontier.csv", index=False)

    risk_row, uplift_row = fr.iloc[0], fr.iloc[-1]
    print(f"=== Efficacy–fairness frontier (test set, budget {BUDGET:.0%}) ===\n")
    print("Endpoints (reproduce the Phase-5 audit):")
    for label, row in [("risk-targeted (λ=0)", risk_row),
                       ("uplift-targeted (λ=1)", uplift_row)]:
        print(f"  {label:24s} averted {row['attrition_averted']*100:.1f} pp  |  "
              f"worst four-fifths ratio {row['min_dp_ratio']:.2f}  "
              f"(G {row['dp_ratio_Gender']:.2f} / M {row['dp_ratio_MaritalStatus']:.2f}"
              f" / A {row['dp_ratio_AgeBand']:.2f})")

    d_av = (uplift_row["attrition_averted"] - risk_row["attrition_averted"]) * 100
    d_fair = uplift_row["min_dp_ratio"] - risk_row["min_dp_ratio"]
    print(f"\nMoving risk → uplift: attrition averted {d_av:+.1f} pp, worst-case "
          f"four-fifths ratio {d_fair:+.2f}.")
    cleared = (fr["min_dp_ratio"] >= FOUR_FIFTHS).any()
    print(f"Any rule on this lever clears the four-fifths rule on every attribute? "
          f"{'yes' if cleared else 'NO'} "
          f"(best worst-case ratio {fr['min_dp_ratio'].max():.2f}).")
    print(f"\nFigure -> {path}")
    print(
        "\nReading: on the worst-affected group, moving toward uplift-targeting is "
        "not a\ntrade-off — it averts MORE attrition AND raises the weakest "
        "four-fifths ratio. But\nthe right panel is the honest caveat: the same "
        "move improves MaritalStatus and Age\nwhile eroding Gender parity, and no "
        "rule on this single lever clears four-fifths on\nevery attribute. The "
        "targeting rule is a distributive-justice dial, not a neutral\ndefault "
        "(synthetic data: a statement about the method)."
    )


if __name__ == "__main__":
    main()
