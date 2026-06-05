"""Phase 5b — transportability under distribution shift.

A people-analytics model and the policy it implies are fit on one workforce;
multinationals then deploy them across very different labour markets. This module
keeps the trained risk and uplift models fixed and asks what survives when the
*deployment* distribution moves — specifically along the hours-norm axis that the
whole intervention rests on. Each "market" is the test set reweighted to a
different **overtime prevalence**, an understated proxy for labour cultures whose
hours norms differ from the dataset's implicit US-corporate frame:

  - "original"         : the dataset's ~25% on overtime
  - "long-hours"       : ~55% on overtime
  - "normalized-hours" : ~8% on overtime (overtime is rare / not the binding lever)

Reweighting on a single binary keeps the effective sample size high (reported as
ESS), so the comparison is honest rather than an artifact of a few upweighted
outliers.

The result is the project's thesis applied to deployment: the risk *ranking*
ports reasonably (ROC-AUC holds; the mean prediction tracks each base rate, so
Brier moves mostly with prevalence, not a calibration collapse), but the
*policy* does not — the overtime lever averts attrition in proportion to how many
are on overtime, so a retention policy validated where hours are long does almost
nothing where they aren't. Synthetic data: a constructed stress test, not a
measured cross-market effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.causal.common import TREATMENT
from src.causal.uplift import estimate_cate
from src.config import FIGURES, TARGET
from src.data.load import load_clean, train_test
from src.predict.baselines import Xy, fit_risk_model
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (viz sets Agg backend)

BUDGET = 0.20
# Target overtime prevalence per market (None = leave the data as-is).
MARKETS = {"original": None, "long-hours": 0.55, "normalized-hours": 0.08}


def market_weights(on_ot: np.ndarray, target: float | None) -> np.ndarray:
    """Reweight so the on-overtime share equals `target`. Two weight values
    only (on/off overtime), so the effective sample stays large. Sums to n."""
    if target is None:
        return np.ones(len(on_ot))
    p = on_ot.mean()
    return np.where(on_ot == 1, target / p, (1 - target) / (1 - p))


def _ess(w: np.ndarray) -> float:
    """Kish effective sample size as a share of n."""
    return float(w.sum() ** 2 / (np.square(w).sum() * len(w)))


def _weighted_topk_reduction(score, realized, w, budget):
    """Reduction from treating the top-budget weighted share, ranked by `score`."""
    order = np.argsort(score)[::-1]
    cum = np.cumsum(w[order]) / w.sum()
    sel = order[cum <= budget]
    if sel.size == 0:
        sel = order[:1]
    return (w[sel] * realized[sel]).sum() / w.sum()


def _evaluate(y, risk, realized, on_ot, w):
    base = float(np.average(y, weights=w))
    red_uplift = _weighted_topk_reduction(realized, realized, w, BUDGET)
    return {
        "overtime_share": float(np.average(on_ot, weights=w)),
        "ess_share": _ess(w),
        "roc_auc": roc_auc_score(y, risk, sample_weight=w),
        "pr_auc": average_precision_score(y, risk, sample_weight=w),
        "brier": brier_score_loss(y, risk, sample_weight=w),
        "baseline_rate": base,
        "mean_predicted": float(np.average(risk, weights=w)),
        "averted_uplift": red_uplift,  # pp of attrition the policy removes
        "post_uplift": base - red_uplift,
    }


def run(train_df, test_df):
    X_te, _ = Xy(test_df)
    risk = fit_risk_model(train_df, "logit").predict_proba(X_te)[:, 1]
    cate, _ = estimate_cate(train_df, test_df)
    on_ot = (test_df[TREATMENT].astype(str).str.strip() == "Yes").to_numpy()
    realized = cate["causal_forest"] * on_ot
    y = test_df[TARGET].to_numpy()

    results = {}
    for name, target in MARKETS.items():
        w = market_weights(on_ot.astype(float), target)
        results[name] = _evaluate(y, risk, realized, on_ot.astype(float), w)
    return results


def plot(results):
    names = list(results.keys())
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A — the ranking ports (ROC-AUC roughly flat across markets).
    ax1.bar(x, [results[n]["roc_auc"] for n in names], 0.55, color="#1f77b4")
    for xi, n in zip(x, names):
        ax1.annotate(f"{results[n]['roc_auc']:.2f}",
                     (xi, results[n]["roc_auc"] + 0.01), ha="center", fontsize=8)
    ax1.axhline(0.5, color="k", lw=0.6, ls=":")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("The risk ranking ports", fontsize=10)

    # Panel B — the policy does not (attrition averted by the overtime lever).
    averted = [results[n]["averted_uplift"] * 100 for n in names]
    bars = ax2.bar(x, averted, 0.55, color="#d62728")
    for b, v in zip(bars, averted):
        ax2.annotate(f"{v:.1f} pp", (b.get_x() + b.get_width() / 2, v + 0.05),
                     ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("attrition averted by uplift policy (pp)")
    ax2.set_title("…the overtime policy does not", fontsize=10)

    fig.suptitle("Transportability: a model validated in one labour market "
                 "is not validated for another")
    fig.tight_layout()
    return viz.save(fig, "transportability.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    results = run(train_df, test_df)
    path = plot(results)

    pd.DataFrame([{"market": k, **v} for k, v in results.items()]
                 ).to_csv(FIGURES / "transportability.csv", index=False)

    names = list(results.keys())
    print("=== Transportability — markets defined by overtime prevalence ===\n")
    metrics = [
        ("overtime share", "overtime_share", "{:.0%}"),
        ("effective sample (ESS)", "ess_share", "{:.0%}"),
        ("ROC-AUC (ranking)", "roc_auc", "{:.3f}"),
        ("Brier (base-rate bound)", "brier", "{:.3f}"),
        ("baseline attrition", "baseline_rate", "{:.1%}"),
        ("mean predicted risk", "mean_predicted", "{:.1%}"),
        ("attrition averted (uplift)", "averted_uplift", "{:.1%}"),
    ]
    print(f"{'metric':28s}" + "".join(f"{n:>18s}" for n in names))
    for label, key, fmt in metrics:
        print(f"{label:28s}" + "".join(
            f"{fmt.format(results[n][key]):>18s}" for n in names))

    a_long = results["long-hours"]["averted_uplift"] * 100
    a_norm = results["normalized-hours"]["averted_uplift"] * 100
    print(f"\nFigure -> {path}")
    print(
        "\nReading: the risk *ranking* ports — ROC-AUC stays well above chance in "
        "every market\n(ESS confirms each is a real sample, not a few upweighted "
        "outliers). The *policy*\ndoes not: the overtime lever averts "
        f"{a_long:.1f} pp of attrition in the long-hours market but\nonly "
        f"{a_norm:.1f} pp in the normalized-hours one, where few are on overtime "
        "for it to relieve.\nA model that predicts acceptably abroad can still "
        "imply a retention policy that does\nalmost nothing there — so policy "
        "validation, especially, does not transport across\nlabour markets."
    )


if __name__ == "__main__":
    main()
