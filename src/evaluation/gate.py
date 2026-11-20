"""
Promotion gate: decide whether a candidate FRMT-M configuration replaces the
champion, using acceptance criteria fixed in advance
(``references/configs/promotion_gate.yaml``).
"""

import json
from pathlib import Path

import numpy as np

from src.evaluation.stats import hits_at, load_items, paired_comparison


def load_run(run_dir: Path) -> dict:
    """Items, per-item generation seconds and logged metrics of one run folder."""
    run_dir = Path(run_dir)
    rows    = [json.loads(line) for line in (run_dir / "predictions.jsonl").open(encoding="utf-8")]
    return {
        "dir":     str(run_dir),
        "items":   load_items(run_dir / "predictions.jsonl"),
        "seconds": [row.get("seconds", 0.0) for row in rows],
        "metrics": json.loads((run_dir / "metrics.json").read_text())["metrics"],
    }


def evaluate_gate(champion: dict[str, dict], candidate: dict[str, dict], criteria: dict) -> dict:
    """
    Apply the criteria to champion and candidate runs on the same datasets.

    Parameters
    ----------
    champion, candidate : dict[str, dict]
        ``{dataset: run}`` as returned by ``load_run``.
    criteria : dict
        The ``criteria`` section of ``promotion_gate.yaml``.

    Returns
    -------
    dict
        Per-criterion values, pass/fail flags and the overall ``decision``
        (``"promote"`` only if every criterion passes, else ``"reject"``).
    """
    if set(champion) != set(candidate):
        raise ValueError("champion and candidate must cover the same datasets")
    alpha  = criteria["max_significant_degradation"]["alpha"]
    report = {"per_dataset": {}, "criteria": {}}
    pooled_champ, pooled_cand = [], []
    degraded = []
    for dataset in sorted(champion):
        ch, ca = champion[dataset]["items"], candidate[dataset]["items"]
        if [it.text for it in ch] != [it.text for it in ca]:
            raise ValueError(f"{dataset}: champion and candidate items differ")
        h_ch, h_ca = hits_at(ch, 3), hits_at(ca, 3)
        cmp_ = paired_comparison(h_ca, h_ch)          # candidate − champion
        worse = cmp_["b_only"] > cmp_["a_only"] and cmp_["p_mcnemar"] < alpha
        degraded.append(worse)
        pooled_champ.append(h_ch)
        pooled_cand.append(h_ca)
        report["per_dataset"][dataset] = {
            "n": len(ch), "champion_top3": float(h_ch.mean()), "candidate_top3": float(h_ca.mean()),
            **cmp_, "significantly_worse": bool(worse)}

    champ_top3 = float(np.concatenate(pooled_champ).mean())
    cand_top3  = float(np.concatenate(pooled_cand).mean())
    fail_rate  = max(run["metrics"].get("failure_rate", 0.0) for run in candidate.values())
    invalid    = max(run["metrics"].get("invalid_output_rate", 0.0) for run in candidate.values())
    lat_champ  = float(np.median([s for run in champion.values() for s in run["seconds"]]))
    lat_cand   = float(np.median([s for run in candidate.values() for s in run["seconds"]]))
    ratio      = lat_cand / lat_champ if lat_champ else float("inf")

    checks = {
        "pooled_top3_improves": {
            "champion": champ_top3, "candidate": cand_top3,
            "passed": (cand_top3 > champ_top3) if criteria["pooled_top3_must_improve"] else True},
        "no_significant_degradation": {
            "alpha": alpha, "degraded_datasets": [d for d, w in zip(sorted(champion), degraded) if w],
            "passed": not any(degraded)},
        "failure_rate": {"value": fail_rate, "max": criteria["max_failure_rate"],
                         "passed": fail_rate <= criteria["max_failure_rate"]},
        "invalid_output_rate": {"value": invalid, "max": criteria["max_invalid_output_rate"],
                                "passed": invalid <= criteria["max_invalid_output_rate"]},
        "median_latency_ratio": {"champion_s": lat_champ, "candidate_s": lat_cand, "ratio": ratio,
                                 "max": criteria["max_median_latency_ratio"],
                                 "passed": ratio <= criteria["max_median_latency_ratio"]},
    }
    report["criteria"] = checks
    report["decision"] = "promote" if all(c["passed"] for c in checks.values()) else "reject"
    return report
