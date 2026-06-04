# HR Attrition — Prediction, Causation, and the Ethics of Acting on Scores

**A study of *why predictive HR models mislead intervention* — and the ethics of
acting on attrition risk.** Predictive models rank employees by *risk*; firms
then act as if risk marks *where intervention works*. It doesn't. This project
builds the causal/uplift layer that most attrition notebooks skip, shows that
the **risk ranking and the treatment-effect ranking diverge**, rebuilds the
policy simulation on causal estimates, and audits the fairness of acting on the
score at all.

> ### ⚠️ Synthetic data — read first
> v1 uses the **IBM HR Analytics** benchmark, which is **synthetic**. Every
> causal/policy number here is therefore **illustrative of the method, not an
> estimate of any real-world effect**. That is a feature: this is a
> demonstration of *rigour*, not a deployable HR tool. See
> [Honest guardrails](#honest-guardrails) and [`FRAMING.md`](FRAMING.md).

The full argument lives in **[`FRAMING.md`](FRAMING.md)**. Live status and
computed numbers are in **[`PROGRESS.md`](PROGRESS.md)**.

---

## The thesis in one diagram

```
  predictive model            causal / uplift model
  r(x) = P(leaves | x)        τ(x) = effect of an intervention on x
        │                              │
        ▼                              ▼
  rank by RISK                  rank by UPLIFT (influenceability)
        └──────────────┬───────────────┘
                       ▼
        these rankings DIVERGE  ← the result the project exists for
        (Spearman ρ, top-decile overlap, wasted-effort %)
                       ▼
   acting on risk wastes effort and can harm the wrong people
                       ▼
        fairness audit + model card = ethics, in code
```

## Why this is not another attrition classifier

A logistic regression that predicts who quits is a solved exercise. The value
here is in going two steps further, in the direction of the research question
*what happens when ethics becomes a number an institution acts on*:

1. **Prediction ≠ causation, made rigorous.** A causal/uplift layer
   (`econml` CausalForestDML + meta-learners, identified with `dowhy`) estimates
   *treatment effects*, and a divergence analysis shows the highest-risk
   employees are often not the most *influenceable*.
2. **Causally-grounded policy simulation.** The counterfactual is rebuilt on
   the causal estimates, so the headline reduction is defensible rather than a
   naive re-scoring of the predictive model.
3. **Ethics / fairness / governance, in code.** A `fairlearn` audit of *who
   gets intervened on*, a model card, and the normative argument against acting
   on flight-risk scores.
4. **Transportability.** A distribution-shift stress test: a model validated in
   one labour market is not validated for another.

## Repository layout

```
hr-attrition-analytics/
├── README.md                  # this file
├── FRAMING.md                 # the prediction-vs-causation + ethics argument
├── PROGRESS.md                # phase tracker + computed results
├── DATA_LINEAGE.md            # data sources (raw data is gitignored)
├── model_card.md              # intended use, limits, fairness findings
├── requirements.txt / .lock   # pinned environment
├── Makefile                   # `make help` for all targets
├── src/
│   ├── config.py              # paths + the one random seed
│   ├── data/                  # download + load + split (no leakage)
│   ├── predict/               # logit + GBM, CV, calibration   (Phase 2)
│   ├── interpret/             # SHAP attribution               (Phase 2)
│   ├── causal/                # identify · uplift · divergence  (Phase 3)
│   ├── policy/                # causally-grounded simulation    (Phase 4)
│   ├── ethics/                # fairness audit · transportability (Phase 5/5b)
│   └── viz/                   # shared figure helpers
├── tests/                     # schema + pipeline tests
├── notebooks/                 # thin notebooks (01_eda … 04_ethics)
├── figures/                   # generated figures (committed)
├── paper/                     # writeup
└── hr-attrition-analysis.ipynb  # v1 (the predictive baseline, preserved)
```

## Reproduce

```bash
git clone https://github.com/JimmyKJi/hr-attrition-analytics.git
cd hr-attrition-analytics
python -m venv .venv && source .venv/bin/activate
make setup                 # install dependencies
make data                  # fetch the IBM HR dataset into data/raw/ (gitignored)
make all                   # predict → causal → policy → ethics → transport
make test                  # schema + pipeline tests
```

`make help` lists every target. The single random seed lives in
[`src/config.py`](src/config.py).

---

## v1 — the predictive baseline (preserved)

v1 is a competent, conventional attrition analysis — kept intact in
[`hr-attrition-analysis.ipynb`](hr-attrition-analysis.ipynb) as the baseline the
v2 work interrogates. Its method: EDA → chi-square / t-tests → class-balanced
logistic regression → risk segmentation → a counterfactual policy simulation.

Its central finding — *overtime and promotion-cadence dominate, and a 3-lever
package cuts predicted firm attrition from **16.1% to 7.8%*** — is exactly the
kind of result this project then puts under causal scrutiny (see
[`FRAMING.md`](FRAMING.md) §4): that number is a re-scoring of the predictive
model, and Phase 4 asks whether it survives a *causal* redo.

### v1 key visuals

| Top predictors | Risk segmentation |
| --- | --- |
| ![Top Predictors](Top%20Predictors.png) | ![Risk Segmentation](Risk%20Segmentation.png) |

| Policy simulation (naive) | Key drivers |
| --- | --- |
| ![Policy Simulation](Improvement%20Simulation.png) | ![Key Drivers](Key%20Drivers.png) |

---

## Honest guardrails

- **Synthetic data → method, not truth.** All causal numbers illustrate the
  method. Stated plainly because it is a strength (demonstrated rigour), not a
  weakness to hide.
- **Observational causal inference, stated honestly.** Identification
  assumptions are demonstrative and probed with refutation tests; the
  prediction-vs-causation framing is the contribution, not a claimed effect.
- **No deployable tool is claimed** — this is a critique of naive deployment,
  with the machinery to back it. The model card says: *do not use to make
  individual adverse decisions.*

## v2 (flagged stretch — not started)

Apply the same machinery to the **2026 AI-driven involuntary-attrition wave**
(Layoffs.fyi + WARN filings + earnings-call AI-capex signals), with a
bilingual (English + Chinese) East-Asian coverage slice. Honest caveats
(selection bias, noisy signal extraction, simultaneity of AI-capex and layoffs)
are baked into that spec. This is a Year-2, faculty-involved extension — see
[`PROGRESS.md`](PROGRESS.md).

## About

Built by **Jimmy Kaian Ji** — KCL Philosophy BA. Applying quantitative methods
to workforce and commercial-strategy questions, with a research interest in the
ethics of operationalising judgement into governance metrics.

Related work:
[ESG Ratings and Capital Flows](https://github.com/JimmyKJi/esg-retail-flows-causal)
— a causal-inference study using stacked difference-in-differences and
instrumental variables.

Contact: [linkedin.com/in/jimmy-kaian-ji](https://www.linkedin.com/in/jimmy-kaian-ji/).
