#!/usr/bin/env bash
# Progress of reproduction/run_all.sh. Run from anywhere.
R="$(cd "$(dirname "$0")" && pwd)"
echo "== Queue"; grep -E 'START|OK|FAIL|wrote|ALL_DONE' "$R/logs/run_all.log" 2>/dev/null || echo "(no run_all.log)"
echo; echo "== Running jobs"; pgrep -af "scripts/run_experiment.py" | grep -v pgrep | sed -E 's#.*configs/([^ ]+)_[a-z]+[^ ]*\.yaml.*--output-dir ([^ ]+).*#\1 -> \2#' | sort -u || true
echo; echo "== Latest progress"
for f in $(ls -t "$R"/logs/*_run*.log 2>/dev/null | head -3); do
  echo "$(basename "$f"): $(tail -c 400 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -1)"
done
echo; echo "== Errors in logs"; grep -l -E "Traceback|\[SKIP\]" "$R"/logs/*_run*.log 2>/dev/null || echo none
echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
echo; if grep -q ALL_DONE "$R/logs/run_all.log" 2>/dev/null; then echo "== DONE: see $R/RESULTS.md"; else echo "== Still running"; fi
