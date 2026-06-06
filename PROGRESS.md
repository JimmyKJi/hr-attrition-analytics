# Progress

Living status for the v2 build (prediction → causation → ethics). Updated at
each phase's Definition of Done (DoD). Computed numbers land in §Results.

Legend: ✅ done · 🔄 in progress · ⬜ not started

| Phase | What | DoD | Status |
| --- | --- | --- | --- |
| 0 | Repo scaffold & framing | structure, README w/ synthetic caveat, FRAMING.md, requirements, Makefile, lineage, model-card skeleton | ✅ |
| 1 | Data loader + schema test | loader, fixed-seed stratified split, no leakage, schema test (1470 rows) passes | ✅ |
| 2 | Predictive baselines | logit + GBM, stratified CV, ROC-AUC/PR-AUC, **calibration**, SHAP summary | ✅ |
| 3 | Causal layer (novel core) | DoWhy identification + refutation; EconML CATE per policy; **risk-vs-uplift divergence** figure + stat | ✅ |
| 4 | Causal policy simulation | uplift-targeted vs risk-targeted reduction table + interpretation | ✅ |
| 5 | Ethics & fairness audit | fairlearn parity/eq-odds, disparate-impact, model card, normative section | ✅ |
| 5b | Transportability check | distribution-shifted test set; perf + policy degradation figure + caveat | ✅ |
| 6 | Writeup + reproduce | paper/writeup.md, `make all`, thin notebooks | ✅ |

## Results (filled as phases complete — do not fabricate)

- **Data (Phase 1):** clean frame 1470×31 (23 numeric, 7 categorical, target);
  stratified split 1102/368 with attrition ≈16.1% preserved in both; 9 schema
  tests pass. Source: confirmed public mirror of the Kaggle file.
- **Predictive baseline (Phase 2):** logit leads — test ROC-AUC 0.814, PR-AUC
  0.588, Brier 0.098 (5-fold CV ROC-AUC 0.847); GBM (HistGradientBoosting) test
  ROC-AUC 0.764. Both reasonably calibrated (`figures/calibration.png`). Top
  SHAP risk drivers: YearsSinceLastPromotion, EnvironmentSatisfaction,
  NumCompaniesWorked, OverTime, JobSatisfaction (`figures/shap_summary.png`).
  The model says *who*, not *what to do* — Phase 3 takes that up.
- **Causal effect (Phase 3):** OverTime → attrition. DoWhy backdoor-adjusted
  ATE **+0.187**; EconML agrees (causal forest +0.185, T-learner +0.191, CATE
  rank-agreement ρ=0.77). All three refuters pass — placebo new-effect ≈0
  (−0.003), random-common-cause and 80%-subset both stable (0.187, 0.192).
  CATE τ(x) ranges [+0.03, +0.40] (sd 0.088): the *same* intervention helps
  some employees ~12× more than others — that spread is what Phase 4 targets.
- **Divergence (Phase 3):** risk ranking ≠ uplift ranking. Spearman
  ρ(risk, τ) = **0.53**; top-decile overlap only **27%**; a risk-targeted
  overtime policy captures just **74%** of the retention an uplift-targeted one
  would; and **38%** of the top-risk decile isn't even on overtime, so the
  lever can't touch them (`figures/divergence.png`). This is the headline:
  targeting the most *at-risk* is not targeting the most *influenceable*.
- **Policy (Phase 4):** at a fixed retention budget (relieve overtime for the
  top-k), uplift-targeting beats risk-targeting at every budget — post-policy
  attrition of **13.2% vs 14.4%** at a 10% budget, **11.6% vs 13.1%** at 20%
  (risk-targeting captures only **54–66%** of the achievable reduction; the rest
  is wasted on high-risk employees the lever can't move). Full intervention
  (relieve overtime for all 25% who have it) is a causal **16.0%→11.1%**
  (−4.9 pp) — it does **not** reach v1's naive **7.8%**: the predictive sim
  overstated the effect (`figures/policy_simulation.png`).
- **Fairness (Phase 5):** audited on *who gets flagged* (top-20%) across gender,
  marital status, age band. **Risk-targeting amplifies disparity** — fails the
  four-fifths rule on marital status (sel-rate ratio **0.21**, gap 32 pp) and age
  (**0.27**); gender near-parity (0.89). **Uplift-targeting roughly halves** the
  marital/age disparities (ratios 0.59, 0.48; age eq-odds gap 0.59→0.13) but
  *introduces* a mild gender gap (0.89→0.71). Neither clears four-fifths on most
  attributes — the targeting rule is itself a distributive choice
  (`figures/fairness_selection_rates.png`, `fairness_metrics.csv`).
- **Transportability (Phase 5b):** markets defined by **overtime prevalence**
  (25% original / 55% long-hours / 8% normalized-hours), reweighting on the single
  binary so the effective sample stays large (ESS **100% / 68% / 86%** — a real
  sample, not a few upweighted outliers). **The risk *ranking* ports:** ROC-AUC
  **0.81 / 0.84 / 0.76**, all well above chance; Brier (0.098 / 0.125 / 0.083)
  moves with the base rate, not a calibration collapse — mean predicted tracks
  each market's prevalence (14.4% vs 16.0%, 20.0% vs 24.2%, 11.2% vs 11.3%).
  **The *policy* does not:** the overtime lever averts **5.8 pp** of attrition in
  the long-hours market but only **1.6 pp** in the normalized-hours one (4.4 pp
  original), because where few are on overtime there is little for it to relieve.
  A model that predicts acceptably abroad can still imply a retention policy that
  does almost nothing there — *a model validated in one labour market is not
  thereby validated for another* (`figures/transportability.png`,
  `transportability.csv`).
- **Writeup + reproduce (Phase 6):** `paper/writeup.md` carries the full arc
  (prediction → why it misleads → causal uplift → divergence → policy →
  ethics/fairness → transportability → conclusion → flagged v2). **`make all`
  reproduces the whole pipeline end-to-end (verified, exit 0)**; the only run-to-
  run movement is last-ULP float noise in the forest and Monte-Carlo jitter in
  the DoWhy refuters — every reported figure is stable to the digits cited. Four
  thin, executed notebooks (`notebooks/01_eda … 04_ethics`) wrap `src/` and embed
  outputs for visual review. All six phases ✅.

## Extensions (on-theme, post-Phase-6)

Sharpening passes that stay on the prediction-vs-causation + ethics theme;
each ships like a phase (branch → commit → main).

- **Confounding sensitivity (Ext A):** the refuters probe the no-unobserved-
  confounding assumption *qualitatively*; this quantifies it. On a linear
  backdoor fit (effect +0.193, t=8.9), the Cinelli–Hazlett **robustness value is
  RV=0.24** and the **E-value 5.13** — a hidden confounder would need to explain
  ~24% of residual variance in *both* overtime and attrition (≈12× the strongest
  measured covariate, EnvironmentSatisfaction at 0.019), or carry a risk-ratio
  link ≥5.1 with both, to drive the effect to zero. Moderately robust; the
  assumption's fragility is now *legible*, not asserted
  (`src/causal/sensitivity.py`, `figures/sensitivity.png`).
- **Second treatment lever (Ext B):** the natural objection to the divergence
  headline is that *risk ≠ uplift* might be an overtime artefact. It isn't. Re-
  running the full risk-vs-uplift machinery on an independent lever —
  **BusinessTravel = Travel_Frequently** (19% of staff; causal-forest ATE
  **+0.106** vs overtime's +0.185) against the *same* treatment-agnostic risk
  score — reproduces the divergence: Spearman ρ(risk, uplift) **0.55** (vs 0.53),
  top-decile overlap **22%** (vs 27%), risk-targeting efficiency **69%** (vs
  74%). The OverTime column reproduces Phase 3 to the digit (ATE +0.185, ρ 0.53,
  overlap 27%, efficiency 74%) — a built-in consistency check. Two extra findings
  sharpen it: (i) **reachability differs** — 70% of the top-risk decile aren't
  frequent travellers (vs 38% not on overtime), so the lever can't touch them;
  (ii) the two uplift rankings are **correlated but not interchangeable** (cross-
  lever ρ 0.71, top-decile overlap 57%), and *neither* matches risk —
  influenceability is partly lever-specific, and a risk score knows of no lever
  at all (`src/causal/levers.py`, `figures/levers.png`, `figures/levers.csv`).

## Environment notes

- Python 3.12 (Anaconda). Auth: HTTPS via osxkeychain (push verified).
- `econml`/`shap` constrain `numpy<2` and an older `scikit-learn` — see
  `requirements.txt`. Exact versions frozen in `requirements.lock` after install.

## v2 (flagged stretch, not started)

2026 AI-layoff-wave extension (Layoffs.fyi + WARN + earnings-call AI-capex,
bilingual East-Asian slice). Year-2, faculty-involved. Spec in README.
