"""Phase 3 — identification & refutation (DoWhy).

States the estimand for OverTime -> Attrition under a backdoor adjustment on all
other recorded features. The key assumption — no unobserved confounding beyond
those covariates — is *demonstrative*: the data is synthetic, so this is a
method showcase, not a defended real-world effect. Credibility is probed with
three refuters: a placebo treatment (effect should vanish), a random common
cause (effect should be stable), and a data subset (effect should be stable).
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd

from src.causal.common import OUTCOME, TREATMENT, build_causal_frame
from src.config import FIGURES, RANDOM_SEED
from src.data.load import load_clean, train_test

logging.getLogger("dowhy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def run(train_df) -> tuple[float, pd.DataFrame, str]:
    from dowhy import CausalModel

    frame, confounders = build_causal_frame(train_df)
    model = CausalModel(
        data=frame, treatment=TREATMENT, outcome=OUTCOME, common_causes=confounders
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    try:
        estimate = model.estimate_effect(
            estimand, method_name="backdoor.propensity_score_weighting",
            target_units="ate", method_params={"weighting_scheme": "ips_weight"},
        )
    except Exception:
        estimate = model.estimate_effect(
            estimand, method_name="backdoor.linear_regression", target_units="ate")
    ate = float(estimate.value)

    refuters = {
        "placebo_treatment": dict(
            method_name="placebo_treatment_refuter", placebo_type="permute"),
        "random_common_cause": dict(method_name="random_common_cause"),
        "data_subset": dict(method_name="data_subset_refuter", subset_fraction=0.8),
    }
    rows = []
    for name, kw in refuters.items():
        r = model.refute_estimate(
            estimand, estimate, num_simulations=20, random_seed=RANDOM_SEED, **kw)
        res = getattr(r, "refutation_result", None)
        p = res.get("p_value") if isinstance(res, dict) else None
        if isinstance(p, (list, tuple)):
            p = p[0]
        rows.append({
            "refuter": name,
            "estimated_effect": r.estimated_effect,
            "new_effect": r.new_effect,
            "p_value": p,
        })
    return ate, pd.DataFrame(rows), str(estimand)


def main() -> None:
    df = load_clean()
    train_df, _ = train_test(df)
    ate, refutation, estimand = run(train_df)

    refutation.to_csv(FIGURES / "refutation.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("Identified ATE — effect of OverTime on P(attrition): "
          f"{ate:+.4f}\n")
    print("Refutation (read: placebo new_effect ~0 = good; random/subset "
          "new_effect ~ estimated = robust):")
    print(refutation.to_string(index=False))
    print("\nIdentification assumptions are demonstrative (synthetic data). The "
          "refuters\nand the prediction-vs-causation framing are the contribution, "
          "not the effect size.")


if __name__ == "__main__":
    main()
