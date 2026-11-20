"""
Unit tests for the promotion gate (synthetic predictions, no model).
"""

import json

import pytest
import yaml

from src.evaluation.gate import evaluate_gate, load_run
from src.evaluation.stats import Item
from src.pipeline import ROOT

CRITERIA = yaml.safe_load((ROOT / "references/configs/promotion_gate.yaml").read_text())["criteria"]


def _run(correct: list[bool], seconds: float = 1.0, failure_rate: float = 0.0) -> dict:
    items = [Item(text=f"t{i}", ground_truth=["a"], ranked=["a"] if ok else ["b", "c", "d"],
                  retrieved=[]) for i, ok in enumerate(correct)]
    return {"items": items, "seconds": [seconds] * len(items),
            "metrics": {"failure_rate": failure_rate, "invalid_output_rate": 0.0}}


def test_better_candidate_is_promoted():
    """Higher pooled top-3, no degradation, same latency → promote."""
    champ = {"d1": _run([True] * 10 + [False] * 10), "d2": _run([True] * 5)}
    cand  = {"d1": _run([True] * 14 + [False] * 6), "d2": _run([True] * 5)}
    report = evaluate_gate(champ, cand, CRITERIA)
    assert report["decision"] == "promote"
    assert report["criteria"]["pooled_top3_improves"]["candidate"] == pytest.approx(19 / 25)


def test_significant_degradation_on_one_dataset_rejects():
    """A significant loss on one dataset rejects even if the pooled score improves."""
    champ = {"d1": _run([True] * 12 + [False] * 28), "d2": _run([True] * 12 + [False] * 0)}
    cand  = {"d1": _run([True] * 40), "d2": _run([False] * 12)}
    report = evaluate_gate(champ, cand, CRITERIA)
    assert report["criteria"]["pooled_top3_improves"]["passed"]
    assert report["criteria"]["no_significant_degradation"]["degraded_datasets"] == ["d2"]
    assert report["decision"] == "reject"


@pytest.mark.parametrize("kwargs,criterion", [
    ({"seconds": 2.0}, "median_latency_ratio"),
    ({"failure_rate": 0.01}, "failure_rate"),
])
def test_operational_criteria_reject(kwargs, criterion):
    """Slower or failing candidates are rejected even when more accurate."""
    champ = {"d1": _run([True] * 5 + [False] * 5)}
    cand  = {"d1": _run([True] * 6 + [False] * 4, **kwargs)}
    report = evaluate_gate(champ, cand, CRITERIA)
    assert not report["criteria"][criterion]["passed"] and report["decision"] == "reject"


def test_mismatched_items_raise():
    """Champion and candidate must be scored on the same items."""
    champ = {"d1": _run([True, False])}
    cand  = {"d1": _run([True])}
    with pytest.raises(ValueError):
        evaluate_gate(champ, cand, CRITERIA)
    with pytest.raises(ValueError):
        evaluate_gate(champ, {"d2": _run([True, False])}, CRITERIA)


def test_load_run(tmp_path):
    """Run folders written by the pipeline are read."""
    row = {"text": "x", "ground_truth": ["a"], "answer": [["A", "a"]], "retrieved_resources": ["a"],
           "seconds": 0.5}
    (tmp_path / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "metrics.json").write_text(json.dumps({"metrics": {"failure_rate": 0.0}}))
    run = load_run(tmp_path)
    assert run["seconds"] == [0.5] and run["items"][0].ranked == ["a"]
