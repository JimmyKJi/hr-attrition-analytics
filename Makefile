.PHONY: help setup lock data predict causal policy ethics transport paper test all clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies from requirements.txt
	$(PY) -m pip install -r requirements.txt

lock:  ## Freeze exact installed versions to requirements.lock
	$(PY) -m pip freeze > requirements.lock

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

ethics:  ## Phase 5 — fairness / disparate-impact audit
	$(PY) -m src.ethics.fairness_audit

transport:  ## Phase 5b — transportability / distribution-shift stress test
	$(PY) -m src.ethics.transportability

paper:  ## Pointer to the writeup
	@echo "Writeup: paper/writeup.md"

test:  ## Run the test suite
	$(PY) -m pytest -q

all: data predict causal policy ethics transport  ## Run the full v2 pipeline

clean:  ## Remove caches and generated (figures/) outputs
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f figures/*.png
