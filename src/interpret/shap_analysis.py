"""Phase 2 — SHAP attribution for the interpretable risk model.

Exact Shapley values for the logistic risk model via LinearExplainer (fast and
exact for a linear model), over the one-hot/scaled feature space. The beeswarm
answers "which features drive the *predicted risk*" — distinct from "which
features an intervention can *change*" (Phase 3).
"""
from __future__ import annotations

import numpy as np
import shap

from src.config import RANDOM_SEED
from src.data.load import load_clean, train_test
from src.predict.baselines import Xy, build_preprocessor
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression


def compute_shap(train_df, test_df, n_background=200, n_explain=300):
    pre = build_preprocessor(train_df)
    Xtr, ytr = Xy(train_df)
    Xte, _ = Xy(test_df)

    Xtr_t = pre.fit_transform(Xtr, ytr)
    Xte_t = pre.transform(Xte)
    names = list(pre.get_feature_names_out())

    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED).fit(Xtr_t, ytr)

    rng = np.random.default_rng(RANDOM_SEED)
    bg = Xtr_t[rng.choice(len(Xtr_t), min(n_background, len(Xtr_t)), replace=False)]
    sample = Xte_t[rng.choice(len(Xte_t), min(n_explain, len(Xte_t)), replace=False)]

    explainer = shap.LinearExplainer(clf, bg)
    shap_values = explainer.shap_values(sample)
    return shap_values, sample, names


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)
    shap_values, sample, names = compute_shap(train_df, test_df)

    shap.summary_plot(shap_values, sample, feature_names=names, show=False, max_display=15)
    viz.save(plt.gcf(), "shap_summary.png")

    shap.summary_plot(shap_values, sample, feature_names=names, show=False,
                      plot_type="bar", max_display=15)
    viz.save(plt.gcf(), "shap_bar.png")

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    print("Top SHAP drivers of predicted attrition risk (mean |SHAP|):")
    for i in order:
        print(f"  {names[i]:35s} {mean_abs[i]:.4f}")
    print("\nThese drive the predicted RISK. Whether they are levers an "
          "intervention can move is a causal question — Phase 3.")


if __name__ == "__main__":
    main()
