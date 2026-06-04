"""Phase 3 — the divergence result (the figure the project exists for).

Ranks every test employee by predicted RISK r(x) and by estimated UPLIFT tau(x),
and shows the rankings diverge: the highest-risk employees are often not the most
influenceable. Reports Spearman rho, top-decile overlap, a risk-targeting
efficiency (how much retention a risk-targeted policy captures vs an
uplift-targeted one), and an actionability stat (share of the top-risk decile not
even on overtime, for whom this lever does nothing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from src.causal.common import TREATMENT
from src.causal.uplift import estimate_cate
from src.config import FIGURES, TARGET
from src.data.load import load_clean, train_test
from src.predict.baselines import Xy, fit_risk_model
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)


def compute(train_df, test_df, top_frac: float = 0.10):
    X_te, _ = Xy(test_df)
    risk = fit_risk_model(train_df, "logit").predict_proba(X_te)[:, 1]
    cate, _ = estimate_cate(train_df, test_df)
    uplift = cate["causal_forest"]

    rho = spearmanr(risk, uplift).statistic
    n = len(risk)
    k = max(1, round(top_frac * n))
    risk_top = set(np.argsort(risk)[::-1][:k].tolist())
    uplift_top = set(np.argsort(uplift)[::-1][:k].tolist())
    overlap = len(risk_top & uplift_top) / k

    gain_risk = uplift[list(risk_top)].sum()
    gain_uplift = uplift[list(uplift_top)].sum()
    efficiency = gain_risk / gain_uplift if gain_uplift > 0 else float("nan")

    on_ot = (test_df[TREATMENT].astype(str).str.strip() == "Yes").to_numpy()
    risk_top_not_ot = 1.0 - on_ot[list(risk_top)].mean()

    stats = {
        "n_test": n, "k_top_decile": k,
        "spearman_rho": rho,
        "top_decile_overlap": overlap,
        "risk_targeting_efficiency": efficiency,
        "risk_top_share_not_on_overtime": risk_top_not_ot,
    }
    return risk, uplift, risk_top, uplift_top, stats


def plot(risk, uplift, risk_top, uplift_top, stats):
    n = len(risk)
    both = risk_top & uplift_top
    color = np.array(["#cfcfcf"] * n, dtype=object)
    for i in uplift_top:
        color[i] = "#1f77b4"  # high uplift only
    for i in risk_top:
        color[i] = "#d62728"  # high risk only
    for i in both:
        color[i] = "#9467bd"  # both

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(risk, uplift, c=list(color), s=20, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("predicted attrition risk  r(x)")
    ax.set_ylabel("estimated uplift  τ(x)  — effect of removing overtime")
    ax.set_title(
        f"Risk vs uplift diverge — Spearman ρ = {stats['spearman_rho']:.2f}, "
        f"top-decile overlap {stats['top_decile_overlap']:.0%}"
    )
    legend = [
        Patch(color="#d62728", label="top-decile RISK"),
        Patch(color="#1f77b4", label="top-decile UPLIFT"),
        Patch(color="#9467bd", label="both"),
        Patch(color="#cfcfcf", label="rest"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, fontsize=8)
    return viz.save(fig, "divergence.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    risk, uplift, risk_top, uplift_top, stats = compute(train_df, test_df)
    path = plot(risk, uplift, risk_top, uplift_top, stats)

    pd.DataFrame([stats]).to_csv(FIGURES / "divergence_stats.csv", index=False)
    print("=== Divergence: risk ranking vs uplift ranking (test set) ===")
    for k, v in stats.items():
        print(f"  {k:34s} {v:.4f}" if isinstance(v, float) else f"  {k:34s} {v}")
    print(f"\nFigure -> {path}")
    print(
        "\nHeadline: ranking by risk is not ranking by influenceability. "
        f"Only {stats['top_decile_overlap']:.0%} of the top-risk decile is also "
        "top-uplift;\na risk-targeted overtime policy captures just "
        f"{stats['risk_targeting_efficiency']:.0%} of the retention an "
        "uplift-targeted one would,\nand "
        f"{stats['risk_top_share_not_on_overtime']:.0%} of the top-risk decile "
        "isn't even on overtime — the lever can't touch them."
    )


if __name__ == "__main__":
    main()
