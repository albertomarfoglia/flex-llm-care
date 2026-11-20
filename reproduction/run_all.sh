#!/usr/bin/env bash
# Full reproduction on 4×80 GB GPUs: two lanes in parallel, then summary.
# run1 = all models; run2 = repeat of the 7B models and Falcon-40B to verify that
# repeated runs are byte-identical. Llama-3.3-70B runs only if HF_KEY grants access.
# Usage (from the repository root): setsid nohup reproduction/run_all.sh > reproduction/logs/run_all.log 2>&1 &
set -uo pipefail
export HF_HUB_DISABLE_IMPLICIT_TOKEN=${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}
R=reproduction; O=$R/outputs; M=$R/run_model.sh
mkdir -p $R/logs
(   # lane A: GPUs 0 and 1
    $M qwen-umls-7b 0 $O/run1 & $M biomistral-7b-dare 1 $O/run1 & wait
    $M qwen-umls-7b 0 $O/run2 & $M biomistral-7b-dare 1 $O/run2 & wait
    $M med42-70b 0,1 $O/run1
) &
(   # lane B: GPUs 2 and 3
    $M qwen2.5-72b 2,3 $O/run1
    $M falcon-40b 2,3 $O/run1
    $M falcon-40b 2,3 $O/run2
    if [ -n "${HF_KEY:-}" ]; then HF_TOKEN=$HF_KEY $M llama-3.3-70b 2,3 $O/run1; fi
) &
wait
${PYTHON:-python} $R/summarize.py > $R/RESULTS.md && echo "[$(date -Is)] wrote $R/RESULTS.md"
echo ALL_DONE
