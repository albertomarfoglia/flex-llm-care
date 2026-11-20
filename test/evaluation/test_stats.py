"""
Unit tests for the item-level metrics and paired statistics.
"""

import json

import numpy as np
import pytest

from src.evaluation.stats import (
    Item,
    bootstrap_ci,
    class_report,
    conflicting_inputs,
    error_category,
    hits_at,
    lenient_ranked,
    holm,
    load_items,
    paired_comparison,
    retrieval_hits_at,
    summary,
    top1_labels,
)


def _item(gt, ranked, retrieved=(), text="x"):
    return Item(text=text, ground_truth=list(gt), ranked=list(ranked), retrieved=list(retrieved))


def test_hits_are_multi_label_aware():
    """Any ground-truth resource counts; only the first n predictions are considered."""
    items = [_item(["questionnaire", "questionnaireresponse"], ["questionnaireresponse", "a", "b"]),
             _item(["patient"], ["a", "b", "patient"]),
             _item(["patient"], [])]
    assert hits_at(items, 1).tolist() == [1, 0, 0]
    assert hits_at(items, 3).tolist() == [1, 1, 0]


def test_retrieval_hits_ignore_the_prediction():
    """Retrieval hits depend only on the retrieved candidates."""
    items = [_item(["patient"], ["device"], retrieved=["device", "patient"])]
    assert retrieval_hits_at(items, 1).tolist() == [0]
    assert retrieval_hits_at(items, 2).tolist() == [1]


def test_top1_labels_credit_only_the_first_prediction():
    """A valid resource in position 2 is not credited at top-1."""
    y_true, y_pred = top1_labels([_item(["patient"], ["device", "patient"]),
                                  _item(["a", "b"], ["b"]),
                                  _item(["a"], [])])
    assert y_pred == ["device", "b", "<none>"]
    assert y_true == ["patient", "b", "a"]


def test_class_report_supports_sum_to_n():
    """Per-class supports cover every item."""
    items = [_item(["a"], ["a"]), _item(["b"], ["a"]), _item(["b"], ["b"])]
    rows = {r["class"]: r for r in class_report(items)}
    assert sum(r["support"] for r in rows.values()) == 3
    assert rows["a"]["precision"] == pytest.approx(0.5)
    assert rows["b"]["recall"] == pytest.approx(0.5)


def test_error_categories():
    """Retrieval misses are separated from ranking errors."""
    assert error_category(_item(["a"], ["a"], ["a"])) == "correct"
    assert error_category(_item(["a"], ["b"], ["b", "c"])) == "retrieval_miss"
    assert error_category(_item(["a"], ["b"], ["b", "a"])) == "ranking_error"
    assert error_category(_item(["a"], ["b"])) == "ranking_error"  # no retrieval at all


def test_conflicting_inputs():
    """Same text with disjoint labels is flagged; overlapping labels are not."""
    items = [_item(["a"], [], text="t1"), _item(["b"], [], text="t1"),
             _item(["a", "b"], [], text="t2"), _item(["b"], [], text="t2")]
    assert conflicting_inputs(items) == {"t1"}


def test_summary_keys():
    """Summary reports top-1/top-3 and Hit@k when candidates exist."""
    s = summary([_item(["a"], ["a", "b", "c"], ["a", "b"]), _item(["b"], ["c", "b", "a"], ["c", "b"])])
    assert s["top1_accuracy"] == 0.5 and s["top3_accuracy"] == 1.0 and s["hit_at_k"] == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    """The CI contains the sample mean and does not change with the same seed."""
    values = np.array([1] * 70 + [0] * 30)
    lo, hi = bootstrap_ci(values, n_boot=2000, seed=1)
    assert lo < 0.7 < hi
    assert (lo, hi) == bootstrap_ci(values, n_boot=2000, seed=1)


def test_paired_comparison_counts_discordant_pairs():
    """McNemar uses only discordant pairs; identical methods give p = 1."""
    a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    r = paired_comparison(a, b, n_boot=2000)
    assert (r["a_only"], r["b_only"]) == (8, 0)
    assert r["diff"] == pytest.approx(0.8)
    assert r["p_mcnemar"] == pytest.approx(2 * 0.5 ** 8)
    assert paired_comparison(a, a, n_boot=100)["p_mcnemar"] == 1.0
    with pytest.raises(ValueError):
        paired_comparison(a, a[:5])


def test_holm_matches_hand_computation():
    """Holm: sorted p * (m - rank), made monotone, capped at 1."""
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_load_items_reads_pipeline_output(tmp_path):
    """The predictions.jsonl schema written by the pipeline is parsed."""
    row = {"text": "Patient birthday", "ground_truth": ["patient"],
           "answer": [["A", "patient"], ["", ""]], "retrieved_resources": ["patient", "person"]}
    fp = tmp_path / "predictions.jsonl"
    fp.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (item,) = load_items(fp)
    assert item.ranked == ["patient"] and item.retrieved == ["patient", "person"]


def test_lenient_ranked_tolerates_copied_format_and_names():
    """Doubled braces, literal \\n, letters and resource names are all read."""
    retrieved = ["patient", "device", "encounter"]
    resources = {"patient", "device", "encounter", "observation", "medicationrequest"}
    assert lenient_ranked('{{\\n  "answer": ["b", "A", "z"]\\n}}', retrieved, resources) == [
        "device", "patient"]
    assert lenient_ranked('{"answer": ["Observation", "Medication Request", "foo"]}',
                          retrieved, resources) == ["observation", "medicationrequest"]
    assert lenient_ranked("The answer is the patient.", retrieved, resources) == []
