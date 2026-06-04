"""Phase 2 — predictive baselines (the part everyone does; do it well, move on).

Two models for the *risk* score r(x) = P(attrition | x): an interpretable
logistic regression and a gradient-boosting classifier. Stratified CV
(ROC-AUC, PR-AUC), held-out test metrics, and — crucially — a calibration
check, because a probability is only a "risk" if it is calibrated.

This module deliberately stops at *prediction*. It says who is likely to
leave; it does not say what to do about it. That gap is what Phase 3 (causal
uplift) exists to fill — see FRAMING.md §2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FIGURES, RANDOM_SEED, TARGET
from src.data.load import feature_columns, load_clean, train_test
from src.viz import figures as viz
import matplotlib.pyplot as plt  # noqa: E402  (imported after viz sets Agg backend)


def Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cols = feature_columns(df)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), cols["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cols["categorical"]),
        ]
    )


def build_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    pre = build_preprocessor(df)
    return {
        "logit": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ]),
        "gbm": Pipeline([
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]),
    }


def fit_risk_model(train_df: pd.DataFrame, kind: str = "logit") -> Pipeline:
    """Fit and return one model; used downstream as the canonical risk scorer."""
    model = build_models(train_df)[kind]
    X, y = Xy(train_df)
    return model.fit(X, y)


def cross_validate_models(df: pd.DataFrame) -> pd.DataFrame:
    X, y = Xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for name, model in build_models(df).items():
        res = cross_validate(
            model, X, y, cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        )
        rows.append({
            "model": name,
            "cv_roc_auc": res["test_roc_auc"].mean(),
            "cv_roc_auc_std": res["test_roc_auc"].std(),
            "cv_pr_auc": res["test_pr_auc"].mean(),
            "cv_pr_auc_std": res["test_pr_auc"].std(),
        })
    return pd.DataFrame(rows).set_index("model")


def evaluate_on_test(train_df, test_df) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    Xtr, ytr = Xy(train_df)
    Xte, yte = Xy(test_df)
    rows, probs = [], {}
    for name, model in build_models(train_df).items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        probs[name] = p
        rows.append({
            "model": name,
            "test_roc_auc": roc_auc_score(yte, p),
            "test_pr_auc": average_precision_score(yte, p),
            "test_brier": brier_score_loss(yte, p),
        })
    return pd.DataFrame(rows).set_index("model"), probs


def plot_calibration(test_df, probs: dict[str, np.ndarray]):
    _, yte = Xy(test_df)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed attrition fraction")
    ax.set_title("Calibration — predicted risk vs. observed")
    ax.legend(loc="upper left")
    return viz.save(fig, "calibration.png")


def main() -> None:
    df = load_clean()
    train_df, test_df = train_test(df)

    cv = cross_validate_models(train_df)
    test, probs = evaluate_on_test(train_df, test_df)

    cv.to_csv(FIGURES / "metrics_cv.csv")
    test.to_csv(FIGURES / "metrics_test.csv")
    fig_path = plot_calibration(test_df, probs)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("=== Cross-validated (train, 5-fold stratified) ===")
    print(cv.to_string())
    print("\n=== Held-out test ===")
    print(test.to_string())
    print(f"\nCalibration figure -> {fig_path}")
    print(
        "\nInterpretation: these models tell us WHO is likely to leave "
        "(ranking + calibrated risk).\nThey do NOT tell us what to do about it "
        "— acting on the risk ranking assumes\nrisk = influenceability, which "
        "Phase 3 shows is false. See FRAMING.md §2."
    )


if __name__ == "__main__":
    main()
