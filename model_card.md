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

- **Predictive performance.** (Phase 2) Logistic regression leads — test
  ROC-AUC 0.81, PR-AUC 0.59, Brier 0.10; reasonably calibrated. GBM ROC-AUC
  0.76. The score is a usable *risk* ranking, not an action plan.
- **Causal estimates.** (Phase 3) OverTime → attrition: backdoor ATE +0.187,
  EconML causal forest +0.185 (T-learner agrees, ρ=0.77); all three DoWhy
  refuters pass. Risk-vs-uplift **divergence**: Spearman ρ=0.53, top-decile
  overlap 27%; risk-targeting captures only 74% of achievable retention.
- **Divergence generalises beyond one lever.** (Phase 3 ext) Re-running the
  risk-vs-uplift machinery on a second, independent lever — BusinessTravel =
  Travel_Frequently (causal-forest ATE +0.106) against the *same* risk score —
  reproduces it: ρ=0.55, top-decile overlap 22%, risk-targeting efficiency 69%
  (the OverTime column reproduces Phase 3 to the digit). The two levers reach
  different people (70% vs 38% of the top-risk decile is unreachable) and their
  uplift rankings are correlated but not interchangeable (cross-lever ρ=0.71,
  overlap 57%) — *risk ≠ uplift* is a property of prediction-vs-causation, not an
  overtime artefact (`src/causal/levers.py`, `figures/levers.png`).
- **Sensitivity to unobserved confounding.** (Phase 3 ext) Cinelli–Hazlett
  robustness value **RV=0.24** and **E-value 5.13**: the workflow that quantifies
  how strong an unmeasured confounder would have to be — explain ~24% of residual
  variance in *both* overtime and attrition (≈12× the strongest measured
  covariate's 0.019), or a risk-ratio association ≥5.1 with both — to move the
  estimate to zero. The synthetic data has no ground-truth causal structure, so
  this demonstrates the sensitivity *method*, not that any real effect is robust
  (`figures/sensitivity.png`).
- **Policy (Phase 4).** At a fixed budget, uplift-targeting beats risk-targeting
  (post-policy 13.2% vs 14.4% at a 10% budget). Full overtime relief is a causal
  16.0%→11.1% — short of v1's naive 7.8% claim.

## Fairness findings

(Phase 5) Audited on **who gets flagged for intervention** (top-20%), across
Gender, MaritalStatus, and an Age band — under both targeting rules.

- **Risk-targeting amplifies demographic disparity.** It fails the four-fifths
  rule badly on MaritalStatus (selection-rate ratio **0.21**, flagged-rate gap
  32 pp — Single employees flagged ~5× more than Divorced) and on Age (ratio
  **0.27**; under-30s over-flagged, 45+ under-flagged relative to their actual
  attrition). Gender is near-parity (ratio 0.89).
- **Uplift-targeting partly corrects it, but isn't free.** Because it targets
  influenceability rather than raw risk, it roughly halves the MaritalStatus and
  Age disparities (ratios 0.59 and 0.48; Age equalized-odds gap 0.59→0.13) — but
  it *introduces* a mild gender disparity (ratio 0.89→0.71).
- **Neither rule clears the four-fifths rule on most attributes.** The honest
  reading: choosing a targeting rule is itself a distributive-justice decision
  with non-obvious, attribute-specific consequences — not a neutral model output.
  See `figures/fairness_selection_rates.png` and `FRAMING.md` §5.
- **Efficacy–fairness frontier (Phase 5 ext).** Interpolating the targeting score
  from pure-risk to pure-uplift (dial λ, fixed 20% budget) traces a frontier:
  moving toward uplift-targeting is a *Pareto improvement on the worst-affected
  group* — attrition averted +1.5 pp (2.9→4.5) **and** the weakest four-fifths
  ratio +0.27 (0.21→0.47). But the gains are attribute-specific (Gender parity
  erodes 0.89→0.71 while Marital/Age improve), and **no rule on this single lever
  clears four-fifths on every attribute** (best worst-case 0.47 < 0.80). The rule
  choice is a distributive dial, not a neutral default
  (`figures/fairness_frontier.png`).

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
