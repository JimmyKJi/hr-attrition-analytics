# Framing: why predictive HR models mislead intervention — and the ethics of acting on attrition scores

> This document is the intellectual spine of the project. The code in `src/`
> instantiates the argument made here; the numbers it produces are reported in
> `PROGRESS.md` and the figures in `figures/`. **The data is synthetic** (see
> §7), so every quantity below is a claim about *method*, not about any real
> workforce.

---

## 1. What predictive HR analytics actually delivers

A standard attrition model — including this project's own v1 — learns a risk
score

$$ r(x) = \Pr(\text{leaves} \mid X = x). $$

It answers *"who is likely to leave?"* well, and the v1 notebook does this
competently: a calibrated logistic model, validated drivers (overtime,
promotion cadence), a clean risk segmentation. Thousands of near-identical
Kaggle notebooks stop here, then take one more step that looks obvious and is
in fact a category error: they **rank employees by $r(x)$ and act on the top of
the list** — "these are our flight risks, intervene on them."

The whole project exists to show why that last step is wrong, what it costs,
and what is ethically at stake when an institution does it anyway.

## 2. Prediction is not causation

Targeting assumes that the people *most likely to leave* are the people an
intervention can *most change*. That is a claim about a **treatment effect**,
and a risk score contains no information about it.

Write it in potential-outcomes terms. For a candidate intervention $T$ (e.g.
removing an overtime requirement), each employee has two potential attrition
outcomes, $Y(1)$ under the intervention and $Y(0)$ without it. What an HR team
can actually move is the **conditional average treatment effect** (the
*uplift*):

$$ \tau(x) = \mathbb{E}[\,Y(0) - Y(1)\mid X = x\,]. $$

The risk score $r(x) = \Pr(Y(0)=1\mid X=x)$ and the uplift $\tau(x)$ are
**different functions of $x$**. They coincide only under assumptions no one
checks. A high-risk employee whose risk is driven by factors the intervention
doesn't touch (e.g. a long commute, a strong outside offer) has high $r$ and
near-zero $\tau$ — visible to the risk model, invisible to it as a target.
Conversely, a moderate-risk employee whose risk is overtime-driven may have
large $\tau$: the intervention is precisely what retains them.

**This is the core novelty.** Phase 3 estimates $\tau(x)$ directly with
causal/uplift methods (`econml`: CausalForestDML plus a meta-learner) under an
explicit identification strategy (`dowhy`: a stated DAG, refutation tests), and
then *measures the divergence between the two rankings*.

## 3. The divergence result (the figure the project exists for)

`src/causal/divergence.py` ranks every employee twice — by $r(x)$ and by
$\hat\tau(x)$ — and quantifies how far apart the rankings are:

- **Spearman rank correlation** between risk and uplift.
- **Top-decile overlap**: of the employees a risk-targeted policy would treat,
  what fraction are also in the high-uplift set an effect-targeted policy would
  treat? (1.0 would mean "risk targeting is fine"; low overlap is the result.)
- **Wasted-effort fraction**: share of risk-targeted interventions landing on
  near-zero-uplift employees.

The computed result (test set; full numbers in `PROGRESS.md`): Spearman
$\rho(r,\hat\tau)=0.53$, top-decile overlap just **27%**, and a risk-targeted
overtime policy captures only **74%** of the retention an uplift-targeted one
would — while **38%** of the top-risk decile isn't even on overtime, so the
lever can't touch them. Headline:
*targeting the highest-risk employees is not the same as targeting the ones an
intervention can actually retain — acting on the predictive score wastes effort
and can harm the wrong people.*

**Not a one-treatment artefact.** The obvious objection is that this divergence
is peculiar to overtime. `src/causal/levers.py` re-runs the entire risk-vs-uplift
machinery on a second, independent lever — *frequent business travel*
(BusinessTravel = Travel_Frequently; causal-forest ATE +0.106 vs overtime's
+0.185) — against the **same** treatment-agnostic risk score, and the divergence
reproduces: $\rho=0.55$, top-decile overlap **22%**, risk-targeting efficiency
**69%**. (The overtime column reproduces Phase 3 to the digit — a built-in
consistency check.) Two further points follow. First, the levers **reach
different people**: 70% of the top-risk decile aren't frequent travellers (vs 38%
not on overtime), so that lever simply can't touch them. Second, the two uplift
rankings are **correlated but not interchangeable** (cross-lever $\rho=0.71$,
top-decile overlap 57%), and *neither* matches the risk ranking — influenceability
is partly lever-specific, while a risk score knows of no lever at all. Risk ≠
uplift is thus a property of prediction-vs-causation, not of one treatment
(`figures/levers.png`).

## 4. Causally-grounded policy simulation

v1's counterfactual ("a 3-lever package cuts firm attrition 16.1% → 7.8%") is
computed by **perturbing inputs to the predictive model and re-scoring** —
which assumes the model's associations are causal. Phase 4 rebuilds the same
simulation on the **causal estimates**, targeting by $\hat\tau$ rather than by
$r$, and contrasts the two:

- naive (risk-targeted, predictive re-scoring) vs.
- causal (uplift-targeted, CATE-based) attrition reduction.

The v1 number survives into the README **only if it survives the causal redo**;
otherwise the honest causal number replaces it. The point is not a bigger
reduction — it is a *defensible* one, and a demonstration that *who you treat*
(influenceable, not merely risky) dominates.

## 5. Ethics: when ethics becomes a number an institution acts on

This is the project's tie to the master-thesis question — *what happens when
ethical judgement is operationalised into a governance metric an organisation
then optimises?* An attrition score is exactly such a number. Acting on it
raises hazards the predictive accuracy says nothing about:

1. **Disparate impact — and the targeting rule as a distributive choice.** If
   $r(x)$ tracks sensitive attributes, then "intervene on the high-risk"
   silently allocates scrutiny and resources unevenly. Phase 5 audits this with
   `fairlearn` (demographic-parity and equalized-odds differences) on **who gets
   flagged for intervention**, not just on classifier error — and finds the
   result depends on *which* rule you act on. Risk-targeting **amplifies**
   demographic disparity: it fails the four-fifths rule badly on marital status
   (selection-rate ratio 0.21 — Single employees flagged ~5× more than Divorced)
   and on age (ratio 0.27, over-flagging under-30s). Uplift-targeting, by
   chasing influenceability rather than raw risk, roughly **halves** those two
   disparities — but it is not free: it *introduces* a mild gender disparity
   (ratio 0.89→0.71), and neither rule clears the four-fifths rule on most
   attributes. The lesson is the thesis in miniature: once ethics is compressed
   into a score and the score is acted on, **the choice of decision rule is
   itself a distributive-justice judgement** with non-obvious, attribute-specific
   consequences — not a neutral technical default. (`figures/fairness_selection_rates.png`.)
   To make that dial explicit, `src/ethics/fairness_frontier.py` interpolates the
   targeting score continuously from pure-risk to pure-uplift and plots the
   **efficacy–fairness frontier**: moving toward uplift-targeting turns out to be
   a *Pareto improvement on the worst-affected group* — it averts more attrition
   (+1.5 pp) *and* raises the weakest four-fifths ratio (0.21→0.47) — yet the same
   move erodes Gender parity (0.89→0.71), and **no rule on this single lever
   clears the four-fifths rule on every attribute** (best worst-case 0.47). The
   frontier is the thesis made quantitative: there is no value of the dial that is
   simply "fair", only positions with different, measurable distributive profiles
   (`figures/fairness_frontier.png`).
2. **Surveillance and the self-fulfilling prophecy.** Labelling someone a
   "flight risk" changes how managers treat them; the label can *produce* the
   exit it predicted. A score acted upon is not a passive measurement.
3. **Acting on associations the evidence can't support.** §2–3 show the risk
   ranking is the *wrong* operational target; §7 notes the data is synthetic
   and the identification assumptions are demonstrative. Making individual
   adverse decisions on this basis is the moral hazard, stated plainly in
   `model_card.md`: *do not use to make individual adverse decisions.*

The contribution here is not a manifesto but **ethics instantiated in code**: a
fairness audit, a model card, and a divergence result that together show *why*
the naive deployment is both ineffective and unjust.

## 6. Transportability: validated where?

A people-analytics model and the policy it implies are fit on one workforce;
multinationals then deploy them across very different labour cultures. Phase 5b
builds a **distribution-shifted** test set (reweighting toward a contrasting
labour profile — different tenure/hours/role mix) and shows that **both
predictive performance and the recommended policy degrade** under the shift.

The understated motivating case is an East-Asian / Hong Kong labour context,
where retention drivers (collectivist obligation, family and face
considerations, hours norms) differ from the dataset's implicit US-corporate
frame. The general claim is market-neutral: *a model validated in one labour
market is not thereby validated for another* — a validity-and-fairness problem
for any cross-border employer.

**v3 confirms this across whole datasets, not just reweighted versions of one.**
Re-running the pipeline on two further turnover datasets, the risk *model* ports
(AUC 0.75–0.82 everywhere) but the lever's averted attrition collapses from
4.5 pp on IBM to 0.5–0.6 pp elsewhere — the policy does not travel even when the
prediction does (`src/v3/`).

## 7. Honest guardrails

- **The v1 data is the IBM HR Analytics benchmark, which is synthetic.** Every
  causal number in this project is therefore *illustrative of the method*, not
  an estimate of a real-world effect. This is stated at the top of the README
  and the model card.
- **Observational causal inference, stated honestly.** Identification
  assumptions are demonstrative; their credibility is probed with refutation
  tests (placebo treatment, random common cause, data subset) and, crucially,
  *quantified* with a formal sensitivity analysis: the Cinelli–Hazlett
  robustness value is **RV=0.24** and the E-value **5.13**. Read these as the
  robustness-quantification *step* of the workflow, not a robustness *result*:
  the IBM data has no ground-truth causal structure, so they cannot show that a
  real overtime effect is robust — they show *how* the no-unobserved-confounding
  assumption would be stress-tested on data where the effect were real (and what
  the procedure returns on this synthetic fixture: a confounder ~12× more
  explanatory than the strongest measured one, or a risk-ratio ≥5.1 with both
  treatment and outcome, would move the estimate to zero). The point is not that
  the effect is *true* — the data is synthetic — but that the assumption's
  fragility is made legible rather than hidden (`src/causal/sensitivity.py`). The
  prediction-vs-causation framing, not a claimed effect size, remains the
  contribution.
- **No deployable tool is claimed.** The claim is a *critique of naive
  deployment*, with the machinery to back it.

The intended arc: v1 is a methodological demonstration on a public benchmark; v2
builds the prediction-vs-causation and ethics study on it; v3 shows those findings
replicate across independent datasets. A separate, later real-data stretch (see
README) would apply the same machinery to the 2026 AI-driven involuntary-attrition
wave, where the same prediction-vs-causation and transportability problems would
recur on real, topical data.
