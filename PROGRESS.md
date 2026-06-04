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
| 4 | Causal policy simulation | uplift-targeted vs risk-targeted reduction table + interpretation | ⬜ |
| 5 | Ethics & fairness audit | fairlearn parity/eq-odds, disparate-impact, model card, normative section | ⬜ |
| 5b | Transportability check | distribution-shifted test set; perf + policy degradation figure + caveat | ⬜ |
| 6 | Writeup + reproduce | paper/writeup.md, `make all`, thin notebooks | ⬜ |

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
- **Policy (Phase 4):** _pending_ — naive risk-targeted vs causal
  uplift-targeted attrition reduction (and whether v1's 16.1%→7.8% survives).
- **Fairness (Phase 5):** _pending_ — demographic-parity / equalized-odds
  differences across gender, age band, marital status.
- **Transportability (Phase 5b):** _pending_ — metric + policy degradation under
  shift.

## Environment notes

- Python 3.12 (Anaconda). Auth: HTTPS via osxkeychain (push verified).
- `econml`/`shap` constrain `numpy<2` and an older `scikit-learn` — see
  `requirements.txt`. Exact versions frozen in `requirements.lock` after install.

## v2 (flagged stretch, not started)

2026 AI-layoff-wave extension (Layoffs.fyi + WARN + earnings-call AI-capex,
bilingual East-Asian slice). Year-2, faculty-involved. Spec in README.
