# Model card — HR attrition models

A living document. Sections marked _(pending Phase N)_ are filled as the build
reaches them. The model card exists partly to make the **out-of-scope uses**
explicit and binding.

## Model details

- **Models.** (Phase 2) A class-balanced logistic regression (interpretable
  baseline) and a gradient-boosting classifier for the *predictive* risk score;
  (Phase 3) causal/uplift estimators (EconML CausalForestDML + a meta-learner)
  for *treatment effects*.
- **Owner.** Jimmy Kaian Ji. **Repo.** `hr-attrition-analytics`.
- **Data.** IBM HR Analytics benchmark — **synthetic** (see `DATA_LINEAGE.md`).

## Intended use

- A **methodological demonstration** that predictive risk and causal
  influenceability diverge, and that acting on the risk score is both
  ineffective and ethically fraught.
- Teaching / portfolio artifact for prediction-vs-causation and
  fairness-in-deployment.

## Out-of-scope / prohibited uses

- ❌ **Do not use to make individual adverse decisions** (terminations, denied
  promotions, targeted surveillance). The score does not support them, and the
  data is synthetic.
- ❌ Do not treat any number here as an estimate of a real firm's attrition or
  of a real intervention's effect.
- ❌ Do not deploy across a different labour market without re-validation — see
  the transportability result (Phase 5b).

## Metrics

- **Predictive performance.** _(pending Phase 2)_ — ROC-AUC, PR-AUC, calibration.
- **Causal estimates.** _(pending Phase 3)_ — CATE per candidate policy;
  risk-vs-uplift divergence statistic.

## Fairness findings

_(pending Phase 5)_ — demographic-parity and equalized-odds differences across
gender, age band, and marital status, computed on **who gets intervened on**,
not only on classifier error. Disparate-impact flag and interpretation.

## Ethical considerations

- Acting on a flight-risk score can be a **self-fulfilling prophecy** and a form
  of surveillance; the label changes behaviour toward the labelled.
- The risk ranking is the **wrong operational target** (see `FRAMING.md` §2–3),
  so intervening "on the high-risk" wastes effort and may concentrate on the
  wrong people.

## Caveats & limitations

- Synthetic data; observational, demonstrative identification assumptions probed
  with refutation tests (not a claimed real-world effect).
- Single dataset, single (implicit US-corporate) labour context.
