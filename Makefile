SHELL = /bin/bash

ifeq ($(OS),Windows_NT)
        ACTIVATE := venv\Scripts\activate
    else
        ACTIVATE := source venv/bin/activate
    endif

CONFIG ?= references/configs/config.yaml
TORCH_INDEX ?= https://download.pytorch.org/whl/cu126
MLFLOW_TRACKING_URI ?= http://localhost:8080
export MLFLOW_TRACKING_URI

.ONESHELL:
virtualenv:
	python3.11 -m venv venv
	@echo "Virtual environment created successfully"

install:
	$(ACTIVATE) && python -m pip install -r requirements.txt

# Exact versions used for the reported results (torch built for CUDA 12.6).
install_lock:
	$(ACTIVATE) && python -m pip install -r reproduction/requirements-lock.txt --extra-index-url $(TORCH_INDEX)

update:
	$(ACTIVATE) && pip install --upgrade -r requirements.txt

start_tracking:
	@echo "Start tracking MlFlow..."
	$(ACTIVATE) && mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///src/models/mlflow.db

start_QA:
	@echo "Start Quality Assurance checks..."
	$(ACTIVATE) && echo "Executing Ruff..." && ruff check .
	$(ACTIVATE) && echo "Executing Mypy..." && mypy src
	$(ACTIVATE) && echo "Executing Bandit..." && bandit -r src

start_testing:
	@echo "Start testing..."
	pytest test/

# One experiment (default: references/configs/config.yaml); override with CONFIG=...
experiment:
	$(ACTIVATE) && python scripts/run_experiment.py --config $(CONFIG) --output-dir outputs

# 5-sample end-to-end check with a 7B model on one GPU, no MLflow needed.
smoke:
	$(ACTIVATE) && MLFLOW_TRACKING_URI= python scripts/run_experiment.py \
		--config reproduction/configs/qwen-umls-7b_sphn.yaml --limit 5 --output-dir outputs/smoke

# Full paper grid (needs 4x80 GB GPUs, ~2.5 h); results in reproduction/RESULTS.md.
reproduce:
	mkdir -p reproduction/logs
	$(ACTIVATE) && PYTHON=python setsid nohup reproduction/run_all.sh > reproduction/logs/run_all.log 2>&1 &
	@echo "Started. Check progress with: make status"

status:
	reproduction/status.sh

# Ablation, energy and native-prompt runs of the paper (4x80 GB GPUs, ~6 h) -> results/revision/gpu
ablations:
	mkdir -p reproduction/logs
	$(ACTIVATE) && PYTHON=python setsid nohup reproduction/run_revision.sh > reproduction/logs/revision_all.log 2>&1 &
	@echo "Started. Log: reproduction/logs/revision_all.log"

# After `reproduce` and `ablations`: baselines, promotion gate, statistics, paper tables
R := results/revision
paper:
	$(ACTIVATE) && set -e
	python scripts/run_baselines.py --output-dir $(R)/baselines
	[ -e $(R)/promotion/gate_report_001.json ] || python scripts/promote.py \
		--champion "$(R)/gpu/qwen2.5-72b_{dataset}_ref" --candidate "$(R)/gpu/qwen2.5-72b_{dataset}_pubmedbert" \
		--champion-config "reproduction/configs_revision/qwen2.5-72b_{dataset}_ref.yaml" \
		--candidate-config "reproduction/configs_revision/qwen2.5-72b_{dataset}_pubmedbert.yaml" \
		--report $(R)/promotion/gate_report_001.json --tracking-uri "sqlite:///$(CURDIR)/$(R)/mlflow/mlflow.db"
	rm -rf $(R)/analysis $(R)/analysis_pooled
	python scripts/analyze_results.py --runs reproduction/outputs/run1 $(R)/baselines $(R)/gpu --family main --output-dir $(R)/analysis
	python scripts/analyze_results.py --runs reproduction/outputs/run1 $(R)/baselines $(R)/gpu --family pooled --output-dir $(R)/analysis_pooled
	python scripts/cluster_sensitivity.py --out $(R)/analysis/cluster_mimic.csv
	python scripts/check_motu_items.py
	python scripts/make_tables.py
	@echo "Tables: docs/FGCS_paper/tables/  Statistics: $(R)/analysis (paper), $(R)/analysis_pooled (sensitivity)"
