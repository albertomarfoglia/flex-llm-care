#!/usr/bin/env bash
# Run every config of one model (all datasets and variants) in a single process,
# so the weights are loaded once.
# Usage: reproduction/run_model.sh <model-key> <cuda-visible-devices> <output-dir>
# Example: reproduction/run_model.sh qwen-umls-7b 0 reproduction/outputs/run1
# Run from the repository root. Set MLFLOW_TRACKING_URI to also log to MLflow.
#
# Before starting, waits until the requested GPUs are free (at most GPU_WAIT_MIN
# minutes, default 60; then gives up), so a run never starts on a GPU still held
# by another process. A failed run is retried once (outputs are overwritten).
set -uo pipefail
MODEL_KEY=$1; GPUS=$2; OUT=$3
PYTHON=${PYTHON:-python}
mkdir -p reproduction/logs
LOG=reproduction/logs/${MODEL_KEY}_$(basename "$OUT").log

gpus_free() {
  local used
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -lt 1000 ] || return 1
  done
}

wait_for_gpus() {
  local deadline=$(( $(date +%s) + ${GPU_WAIT_MIN:-60} * 60 ))
  until gpus_free; do
    [ "$(date +%s)" -lt "$deadline" ] || return 1
    sleep 10
  done
}

for attempt in 1 2; do
  if ! wait_for_gpus; then
    echo "[$(date -Is)] FAIL  ${MODEL_KEY} out=${OUT}: GPUs ${GPUS} still busy after ${GPU_WAIT_MIN:-60} min"
    exit 1
  fi
  echo "[$(date -Is)] START ${MODEL_KEY} gpus=${GPUS} out=${OUT} attempt=${attempt}"
  if CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" scripts/run_experiment.py \
       --config reproduction/configs/${MODEL_KEY}_*.yaml --output-dir "$OUT" > "$LOG" 2>&1; then
    echo "[$(date -Is)] OK    ${MODEL_KEY} out=${OUT}"
    exit 0
  fi
  echo "[$(date -Is)] FAIL  ${MODEL_KEY} out=${OUT} attempt=${attempt} (see $LOG)"
  cp "$LOG" "${LOG%.log}.attempt${attempt}.log"
  sleep 30
done
exit 1
