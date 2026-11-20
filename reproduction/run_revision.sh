#!/usr/bin/env bash
# Ablation, energy and native-prompt runs of the paper on 4×80 GB GPUs, two lanes.
# All configs of one model run in one process (weights loaded once; loading measured separately).
# Usage (repository root): setsid nohup reproduction/run_revision.sh > reproduction/logs/revision_all.log 2>&1 &
set -uo pipefail
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///$PWD/results/revision/mlflow/mlflow.db}
PY=${PYTHON:-python}; C=reproduction/configs_revision; O=results/revision/gpu; L=reproduction/logs
mkdir -p $O $L results/revision/mlflow

run() {  # run <model-key> <gpus>
  echo "[$(date -Is)] START $1 gpus=$2"
  if CUDA_VISIBLE_DEVICES=$2 $PY scripts/run_experiment.py --config $C/$1_*.yaml --output-dir $O \
       > $L/revision_$1.log 2>&1; then echo "[$(date -Is)] OK    $1"; else echo "[$(date -Is)] FAIL  $1 (see $L/revision_$1.log)"; fi
}

( run qwen2.5-72b 0,1; run qwen-umls-7b 0 & run biomistral-7b-dare 1 & wait ) &
( run llama-3.3-70b 2,3; run med42-70b 2,3; run falcon-40b 2,3 ) &
wait
echo "[$(date -Is)] ALL_DONE"
