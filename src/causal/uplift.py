"""Phase 3 — uplift / CATE: how much an intervention can actually move attrition.

Estimates the heterogeneous treatment effect tau(x) of OverTime on attrition
with two estimators — a causal forest (CausalForestDML) and a T-learner — fit on
train and evaluated on test. tau(x) > 0 means overtime raises this employee's
attrition risk, so removing it would retain them. Agreement between the two
estimators is a robustness check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.metalearners import TLearner
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.causal.common import build_causal_frame, YTX
from src.config import FIGURES, RANDOM_SEED
from src.data.load import load_clean, train_test


def _causal_forest() -> CausalForestDML:
    return CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20, random_state=RANDOM_SEED),
        model_t=RandomForestClassifier(
            n_estimators=100, min_samples_leaf=20, random_state=RANDOM_SEED),
        discrete_treatment=True, cv=3, n_estimators=500, random_state=RANDOM_SEED,
    )


def _t_learner() -> TLearner:
    return TLearner(models=RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, random_state=RANDOM_SEED))


def estimate_cate(train_df, test_df) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit on train; return ({estimator: test CATE}, {estimator: ATE})."""
    f_tr, conf = build_causal_frame(train_df)
    f_te, _ = build_causal_frame(test_df)
    # Align test columns to train's confounder set (one-hot can differ).
    for c in conf:
        if c not in f_te.columns:
            f_te[c] = 0.0
    Ytr, Ttr, Xtr = YTX(f_tr, conf)
    _, _, Xte = YTX(f_te, conf)

    cate: dict[str, np.ndarray] = {}
    cf = _causal_forest(); cf.fit(Ytr, Ttr, X=Xtr)
    cate["causal_forest"] = np.asarray(cf.effect(Xte)).ravel()
    tl = _t_learner(); tl.fit(Ytr, Ttr, X=Xtr)
    cate["t_learner"] = np.asarray(tl.effect(Xte)).ravel()
    ate = {k: float(v.mean()) for k, v in cate.items()}
    return cate, ate


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    cate, ate = estimate_cate(train_df, test_df)

    pd.DataFrame(cate).to_csv(FIGURES / "cate_test.csv", index=False)
    cf = cate["causal_forest"]
    rho = spearmanr(cate["causal_forest"], cate["t_learner"]).statistic

    print("ATE — effect of OverTime on P(attrition):")
    for k, v in ate.items():
        print(f"  {k:14s} {v:+.4f}")
    print(f"\nCATE rank agreement (Spearman, forest vs T-learner): {rho:.3f}")
    print(
        f"CATE (forest) — mean {cf.mean():+.4f}, sd {cf.std():.4f}, "
        f"range [{cf.min():+.4f}, {cf.max():+.4f}]"
    )
    print(f"Share with positive uplift: {(cf > 0).mean():.1%}")
    print("\nThat tau(x) varies widely is the point: the effect of the SAME "
          "intervention\ndiffers a lot across employees — Phase 4 targets by it.")


if __name__ == "__main__":
    main()
