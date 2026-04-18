# HR Attrition Analytics Project

This project examines employee attrition as a workforce decision-making problem. Using HR data, it identifies the main factors associated with employee turnover, builds a predictive model of attrition risk, and evaluates how simple policy changes may reduce that risk.

Rather than treating attrition purely as a technical prediction task, the project is framed around a practical business question:

**How can organisations use data to better understand and reduce employee turnover?**

## Project Objective

The objective of this project is to:

- identify the key drivers of employee attrition
- validate those relationships using statistical testing
- build a predictive model estimating attrition probability
- segment employees by risk level
- simulate how organisational policy changes may affect attrition risk

## Kaggle Notebook

The full notebook is available in this repository as:

**`hr-attrition-analysis.ipynb`**

## Methods Used

- Exploratory data analysis
- Statistical hypothesis testing (chi-square tests and t-tests)
- Logistic regression modelling
- Feature importance analysis
- Risk segmentation
- Retention policy simulation

## Key Findings

- Overtime was one of the strongest predictors of employee attrition.
- Longer periods since the last promotion were associated with higher turnover risk.
- Job satisfaction and environment satisfaction were associated with lower attrition probability.
- The predictive model was able to distinguish meaningful differences in workforce risk.
- Simulated policy improvements reduced predicted attrition risk from **16.1% to 7.8%**, representing an estimated reduction of **~52%**.

## Management Implications

The analysis suggests that workforce analytics can support more proactive people and management decisions.

In particular, the results indicate that organisations may be able to reduce attrition risk through:

- better overtime and workload management
- clearer promotion and progression pathways
- stronger attention to employee satisfaction and workplace experience
- earlier identification of higher-risk employee groups

## Key Visuals

### Top Predictors of Attrition
![Top Predictors](Top%20Predictors.png)

This chart shows the strongest positive and negative predictors of employee attrition in the logistic regression model.

### Risk Segmentation
![Risk Segmentation](Risk%20Segmentation.png)

This chart groups employees by predicted attrition risk, showing how the model can support workforce prioritisation.

### Policy Simulation
![Policy Simulation](Improvement%20Simulation.png)

This chart compares baseline attrition risk with the simulated effect of HR policy improvements.

### Key Drivers
![Key Drivers](Key%20Drivers.png)

This visual summarises the most influential factors associated with attrition across the analysis.

## Tools Used

Python  
Pandas  
Scikit-learn  
Matplotlib  
Seaborn  

## Repository Contents

- `hr-attrition-analysis.ipynb` — full notebook
- chart images used to summarise the main findings
- this README overview

## Author

Jimmy Kaian Ji  
King’s College London
