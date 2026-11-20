"""
Unit tests for the evaluation metrics.
"""

import pytest

from src.evaluation.metrics import evaluate
from src.evaluation.rag_metrics import hit_k
from src.models.classifier import PredictionResult


def _result(y_pred, y_true, retrieved, ground_truth=None, error=""):
    return PredictionResult(y_pred=y_pred, y_true=y_true, retrieved_resources=retrieved,
                            ground_truth=ground_truth or [y_true], error=error)


def test_hit_k_accepts_single_and_multiple_answers():
    """Strings and lists of acceptable answers."""
    contexts = [["a", "b"], ["c"], ["a"]]
    assert hit_k(contexts, ["b", "c", "z"]) == pytest.approx(2 / 3)
    assert hit_k(contexts, [["z", "b"], ["z"], ["a"]]) == pytest.approx(2 / 3)
    assert hit_k([], []) == 0.0


def test_evaluate_perfect_predictions():
    """All metrics are 1 for perfect predictions."""
    results = [_result("patient", "patient", ["patient"]), _result("device", "device", ["device"])]
    m = evaluate(results)
    assert m["accuracy"] == m["f1_score"] == m["hit_Rate_k"] == 1.0
    assert m["failure_rate"] == 0.0


def test_evaluate_with_failed_sample_does_not_crash():
    """Regression: a failed sample (empty prediction) used to raise TypeError."""
    results = [
        _result("patient", "patient", ["patient"]),
        _result("", "device", [], error="boom"),
    ]
    m = evaluate(results)
    assert m["accuracy"] == 0.5
    assert m["failure_rate"] == 0.5


def test_hit_rate_uses_ground_truth_not_the_prediction():
    """Hit@k is a retrieval metric: any ground-truth resource retrieved counts."""
    results = [_result("device", "patient", ["device", "encounter"],
                       ground_truth=["patient", "encounter"])]
    assert evaluate(results)["hit_Rate_k"] == 1.0


def test_evaluate_empty_raises():
    """Empty input is an error, not a division by zero."""
    with pytest.raises(ValueError):
        evaluate([])


def test_invalid_output_rate():
    """Unconstrained outputs that are not valid answers are reported separately from failures."""
    bad = _result("", "patient", ["patient"])
    bad.invalid_output = True
    m = evaluate([_result("patient", "patient", ["patient"]), bad])
    assert m["invalid_output_rate"] == 0.5 and m["failure_rate"] == 0.0
