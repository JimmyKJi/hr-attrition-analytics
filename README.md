# HR Attrition Analytics — Predicting Turnover and Simulating Policy Impact
 
**Research question:** Which employee attributes best predict voluntary attrition, and what is the expected impact of targeted HR policy interventions on firm-level attrition risk?
 
**Finding:** Overtime burden and time-since-last-promotion are the dominant predictors of attrition in this workforce. A simulated package of three HR policy interventions — overtime reduction, promotion-cadence improvement, and targeted satisfaction investment — reduces predicted firm-level attrition risk from **16.1% to 7.8%**, a **~52% relative reduction**.
 
**Method at a glance:** Exploratory analysis → statistical hypothesis testing (chi-square, t-tests) → logistic regression with feature importance → employee-level risk segmentation → counterfactual policy simulation.
 
**Stack:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn.
 
---
 
## Motivation
 
Employee attrition is a major operating cost that most firms manage reactively. Typical HR reporting is descriptive (headline turnover rate, cost-per-hire) and rarely answers the two questions that actually matter: which employees are at highest risk of leaving, and which specific policy changes would most reduce that risk.
 
This project treats attrition as a workforce decision-making problem rather than a pure prediction task. The aim is not just to identify who is likely to leave, but to generate actionable guidance on where HR budget is best spent.
 
## Data
 
Kaggle IBM HR Analytics Employee Attrition dataset (1,470 employees, 35 features). Features span demographic, compensation, performance, and satisfaction dimensions. Target variable: voluntary attrition (binary).
 
## Method
 
**Exploratory analysis.** Attrition rates profiled across department, job role, age band, and tenure. Visual identification of candidate predictors.
 
**Statistical testing.** Chi-square tests for categorical features; independent-sample t-tests for continuous features. Multiple-comparison caveats flagged in the notebook.
 
**Logistic regression.** Binary classifier with feature standardisation and class-weight balancing. Coefficients interpreted as log-odds contributions for each predictor, holding others constant.
 
**Feature importance and risk segmentation.** Predictors ranked by standardised coefficient magnitude. Employees ranked by predicted attrition probability and bucketed into risk tiers (low / medium / high) to support HR prioritisation.
 
**Policy simulation.** Counterfactual analysis: inputs on overtime, promotion cadence, and satisfaction are shifted to hypothetical improved values, holding all other features constant. Model re-scored to produce simulated firm-level attrition risk under the intervention package.
 
## Key findings
 
**Overtime is the single strongest attrition driver.** Employees required to work overtime have materially higher predicted attrition probability than comparable peers who do not.
 
**Promotion cadence matters more than absolute tenure.** Years since last promotion is a stronger predictor of attrition than overall years at company, suggesting the salient signal to employees is "am I progressing?", not "am I established?".
 
**Satisfaction variables reduce risk.** Higher job satisfaction and environment satisfaction scores are associated with lower attrition probability, consistent with standard HR-literature findings.
 
**Simulated intervention package reduces predicted attrition by ~52%.** Combined policy changes — reducing overtime requirement, shortening promotion cadence, and targeted satisfaction improvements — reduce predicted firm-level attrition from 16.1% to 7.8% in the counterfactual scenario.
 
## Key visuals
 
### Top predictors of attrition
![Top Predictors](Top%20Predictors.png)
 
Standardised logistic regression coefficients. Positive values increase attrition probability; negative values reduce it. Overtime and recent-promotion-absence dominate the positive side; satisfaction measures dominate the negative side.
 
### Risk segmentation
![Risk Segmentation](Risk%20Segmentation.png)
 
Employees bucketed into risk tiers by predicted attrition probability. The highest-risk tier concentrates a disproportionate share of expected departures — supporting a targeted retention approach rather than blanket HR spend.
 
### Policy simulation
![Policy Simulation](Improvement%20Simulation.png)
 
Baseline predicted attrition versus simulated intervention-package outcome. Y-axis: firm-level predicted attrition rate.
 
### Key drivers summary
![Key Drivers](Key%20Drivers.png)
 
Synthesised view of the factors that matter most, combining coefficient magnitude with business interpretability.
 
## Management implications
 
Three actionable recommendations follow from the analysis.
 
**Overtime management as a retention lever.** Because overtime is the strongest positive predictor, the highest-leverage intervention is not more compensation but better workload management. Identifying teams with structurally embedded overtime expectations — and addressing staffing, scope, or process root causes — is the most cost-effective attrition-reduction move available.
 
**Promotion cadence transparency.** Because time-since-last-promotion outperforms raw tenure as a predictor, HR should monitor promotion-pipeline blockages as a leading indicator of attrition risk, not just use tenure as a lagging one. Transparent promotion criteria and cadence targets would address the underlying signal.
 
**Targeted risk-tier intervention.** Risk segmentation identifies a concentrated high-risk group. Directing retention budget (skip-level 1:1s, development conversations, retention equity) toward this group yields better marginal return than firm-wide satisfaction surveys or generic engagement programmes.
 
## Reproducibility
 
Clone the repository and run the notebook end-to-end:
 
```bash
git clone https://github.com/JimmyKJi/hr-attrition-analytics.git
cd hr-attrition-analytics
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter lab hr-attrition-analysis.ipynb
```
 
Random seeds are fixed. Results reproduce exactly on any machine with the same library versions.
 
## Repository contents
 
- `hr-attrition-analysis.ipynb` — full notebook with code, tests, modelling, and simulation
- `Top Predictors.png`, `Risk Segmentation.png`, `Improvement Simulation.png`, `Key Drivers.png` — summary charts
- `README.md` — this document
## Limitations and honest notes
 
This is a single-dataset proof-of-concept using the IBM HR Analytics public dataset, not a causally-identified study. Findings should be read as directional signal, not generalisable policy guidance. A production version would need firm-specific data, causal identification (e.g. before/after policy-change comparison within a single firm), and validation against actual retention outcomes rather than just predicted ones.
 
## About
 
Built by Jimmy Kaian Ji — KCL Philosophy BA (Y1, 2026). Interested in applying quantitative methods to workforce and commercial strategy problems.
 
Related work: [ESG Ratings and Capital Flows Causal Inference Study](https://github.com/JimmyKJi/esg-retail-flows-causal) — causal-inference study using stacked difference-in-differences and instrumental variables.
 
Contact: [linkedin.com/in/jimmy-kaian-ji](https://www.linkedin.com/in/jimmy-kaian-ji/).
