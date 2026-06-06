.PHONY: help setup lock data predict causal policy ethics transport v3 paper paper-html notebooks test all clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies from requirements.txt
	$(PY) -m pip install -r requirements.txt

lock:  ## Freeze exact installed versions to requirements.lock
	$(PY) -m pip list --format=freeze > requirements.lock  # clean name==version (no conda file:// paths)

data:  ## Fetch the IBM HR dataset into data/raw/ (see DATA_LINEAGE.md)
	$(PY) -m src.data.download

predict:  ## Phase 2 — predictive baselines, calibration, SHAP
	$(PY) -m src.predict.baselines
	$(PY) -m src.interpret.shap_analysis

causal:  ## Phase 3 — identification, CATE/uplift, divergence, second lever, confounding sensitivity
	$(PY) -m src.causal.identify
	$(PY) -m src.causal.uplift
	$(PY) -m src.causal.divergence
	$(PY) -m src.causal.levers
	$(PY) -m src.causal.sensitivity

policy:  ## Phase 4 — causally-grounded policy simulation
	$(PY) -m src.policy.simulate

ethics:  ## Phase 5 — fairness / disparate-impact audit + efficacy-fairness frontier
	$(PY) -m src.ethics.fairness_audit
	$(PY) -m src.ethics.fairness_frontier

transport:  ## Phase 5b — transportability / distribution-shift stress test
	$(PY) -m src.ethics.transportability

v3:  ## v3 — cross-dataset replication (downloads 2 extra turnover datasets to data/raw/)
	$(PY) -m src.v3.replicate

paper:  ## Pointer to the writeup
	@echo "Writeup: paper/writeup.md"

paper-html:  ## Render the writeup to a self-contained paper/writeup.html (figures embedded, MathJax)
	$(PY) scripts/build_paper_html.py

notebooks:  ## Regenerate the four review notebooks and execute them in place
	$(PY) scripts/make_notebooks.py
	PYDEVD_DISABLE_FILE_VALIDATION=1 $(PY) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=300 notebooks/0*.ipynb

test:  ## Run the test suite
	$(PY) -m pytest -q

all: data predict causal policy ethics transport  ## Run the full v2 pipeline

clean:  ## Remove caches and generated (figures/) outputs
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f figures/*.png
