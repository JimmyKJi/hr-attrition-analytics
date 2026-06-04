# Progress

Living status for the v2 build (prediction → causation → ethics). Updated at
each phase's Definition of Done (DoD). Computed numbers land in §Results.

Legend: ✅ done · 🔄 in progress · ⬜ not started

| Phase | What | DoD | Status |
| --- | --- | --- | --- |
| 0 | Repo scaffold & framing | structure, README w/ synthetic caveat, FRAMING.md, requirements, Makefile, lineage, model-card skeleton | 🔄 |
| 1 | Data loader + schema test | loader, fixed-seed stratified split, no leakage, schema test (1470 rows) passes | ⬜ |
| 2 | Predictive baselines | logit + GBM, stratified CV, ROC-AUC/PR-AUC, **calibration**, SHAP summary | ⬜ |
| 3 | Causal layer (novel core) | DoWhy identification + refutation; EconML CATE per policy; **risk-vs-uplift divergence** figure + stat | ⬜ |
| 4 | Causal policy simulation | uplift-targeted vs risk-targeted reduction table + interpretation | ⬜ |
| 5 | Ethics & fairness audit | fairlearn parity/eq-odds, disparate-impact, model card, normative section | ⬜ |
| 5b | Transportability check | distribution-shifted test set; perf + policy degradation figure + caveat | ⬜ |
| 6 | Writeup + reproduce | paper/writeup.md, `make all`, thin notebooks | ⬜ |

## Results (filled as phases complete — do not fabricate)

- **Predictive baseline (Phase 2):** _pending_ — ROC-AUC, PR-AUC, calibration.
- **Divergence (Phase 3):** _pending_ — Spearman ρ(risk, uplift), top-decile
  overlap, wasted-effort %.
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
