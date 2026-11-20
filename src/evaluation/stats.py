"""
Item-level metrics and paired statistics for FHIR resource-type selection.

Works on saved ``predictions.jsonl`` rows (or any ranked predictions), so
baselines and LLM runs are scored by exactly the same code. Every item is
multi-label aware: a prediction is correct if it is *any* ground-truth resource.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


@dataclass
class Item:
    """
    One evaluated variable.

    Attributes
    ----------
    text : str
        Variable description given to the model.
    ground_truth : list[str]
        Valid resources, in CSV order.
    ranked : list[str]
        Predicted resources, most relevant first (may be empty on failure).
    retrieved : list[str]
        Candidates shown to the model, in retrieval order (empty for
        methods without retrieval).
    """
    text:         str
    ground_truth: list[str]
    ranked:       list[str]
    retrieved:    list[str]


def load_items(path: Path) -> list[Item]:
    """Read a ``predictions.jsonl`` written by ``src/pipeline.py``."""
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(Item(
                text         = row["text"],
                ground_truth = row["ground_truth"],
                ranked       = [res for _, res in row["answer"] if res],
                retrieved    = row["retrieved_resources"],
            ))
    return items


def hits_at(items: list[Item], n: int) -> np.ndarray:
    """0/1 per item: any of the first ``n`` ranked predictions is a ground-truth resource."""
    return np.array([any(p in it.ground_truth for p in it.ranked[:n]) for it in items], dtype=int)


def retrieval_hits_at(items: list[Item], k: int) -> np.ndarray:
    """0/1 per item: any ground-truth resource among the first ``k`` retrieved candidates."""
    return np.array([any(r in it.ground_truth for r in it.retrieved[:k]) for it in items],
                    dtype=int)


def top1_labels(items: list[Item]) -> tuple[list[str], list[str]]:
    """
    Single-label (y_true, y_pred) for the top-1 prediction.

    ``y_true`` is the predicted resource when it is valid, else the first
    ground-truth resource; ``y_pred`` is the top-1 prediction (``"<none>"``
    on failure). Unlike the top-3 rule of ``select_label``, only the first
    prediction is ever credited.
    """
    y_true, y_pred = [], []
    for it in items:
        pred = it.ranked[0] if it.ranked else "<none>"
        y_pred.append(pred)
        y_true.append(pred if pred in it.ground_truth else it.ground_truth[0])
    return y_true, y_pred


def class_report(items: list[Item]) -> list[dict]:
    """Per-class precision, recall, F1 and support of the top-1 prediction."""
    y_true, y_pred = top1_labels(items)
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [{"class": c, "precision": pi, "recall": ri, "f1": fi, "support": int(si)}
            for c, pi, ri, fi, si in zip(labels, p, r, f, s)]


def top1_confusion(items: list[Item]) -> tuple[list[str], np.ndarray]:
    """Labels and confusion matrix (rows = true, columns = predicted) of the top-1 prediction."""
    y_true, y_pred = top1_labels(items)
    labels = sorted(set(y_true) | set(y_pred))
    return labels, confusion_matrix(y_true, y_pred, labels=labels)


def error_category(item: Item, n: int = 3) -> str:
    """
    ``correct`` if any of the top-``n`` predictions is valid; otherwise
    ``retrieval_miss`` if no valid resource was among the candidates shown
    (only possible for retrieval-based methods), else ``ranking_error``.
    """
    if any(p in item.ground_truth for p in item.ranked[:n]):
        return "correct"
    if item.retrieved and not any(r in item.ground_truth for r in item.retrieved):
        return "retrieval_miss"
    return "ranking_error"


def conflicting_inputs(items: list[Item]) -> set[str]:
    """Descriptions that occur more than once with disjoint ground-truth sets."""
    by_text: dict[str, list[set[str]]] = {}
    for it in items:
        by_text.setdefault(it.text, []).append(set(it.ground_truth))
    return {t for t, gts in by_text.items()
            if len(gts) > 1 and not set.intersection(*gts)}


def summary(items: list[Item]) -> dict[str, float]:
    """Headline metrics of one method on one dataset."""
    y_true, y_pred = top1_labels(items)
    errors = Counter(error_category(it) for it in items)
    out = {
        "n":                len(items),
        "top1_accuracy":    hits_at(items, 1).mean(),
        "top3_accuracy":    hits_at(items, 3).mean(),
        "top1_f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        "top1_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "retrieval_miss":   errors["retrieval_miss"] / len(items),
        "ranking_error":    errors["ranking_error"] / len(items),
    }
    if any(it.retrieved for it in items):
        out["hit_at_k"] = retrieval_hits_at(items, max(len(it.retrieved) for it in items)).mean()
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of per-item values (items resampled with replacement)."""
    values = np.asarray(values, dtype=float)
    rng    = np.random.default_rng(seed)
    means  = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                      seed: int = 0) -> dict[str, float]:
    """
    Compare two methods scored 0/1 on the *same* items.

    Returns the accuracy difference ``a - b`` with a paired bootstrap 95 % CI,
    the discordant counts (``a`` right / ``b`` wrong and vice versa), and the
    two-sided exact McNemar p-value (binomial test on the discordant pairs).
    """
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same items for both methods")
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    p      = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    lo, hi = bootstrap_ci(a - b, n_boot=n_boot, seed=seed)
    return {"diff": float((a - b).mean()), "ci_low": lo, "ci_high": hi,
            "a_only": a_only, "b_only": b_only, "p_mcnemar": float(p)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, in the input order."""
    order    = np.argsort(pvalues)
    m        = len(pvalues)
    adjusted = np.empty(m)
    running  = 0.0
    for rank, idx in enumerate(order):
        running       = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def lenient_ranked(raw_output: str, retrieved: list[str], resources: set[str]) -> list[str]:
    """
    Ranked resources from an *unconstrained* generation, read leniently.

    Doubled braces and literal ``\\n`` (copied from the legacy prompt) are
    tolerated; the quoted items of the ``answer`` list are mapped either as
    candidate letters (A = first retrieved candidate, case-insensitive) or as
    resource names among ``resources``. Anything else is dropped.
    """
    text  = raw_output.replace("{{", "{").replace("}}", "}").replace("\\n", "\n")
    match = re.search(r'"answer"\s*:\s*\[(.*?)\]', text, flags=re.S)
    if not match:
        return []
    ranked = []
    for label in re.findall(r'"([^"]*)"', match.group(1)):
        label = label.strip()
        if len(label) == 1 and label.isalpha():
            idx = ord(label.upper()) - ord("A")
            res = retrieved[idx] if idx < len(retrieved) else None
        else:
            res = label.lower().replace(" ", "") if label.lower().replace(" ", "") in resources else None
        if res and res not in ranked:
            ranked.append(res)
    return ranked


def load_items_lenient(path: Path, resources: set[str]) -> list[Item]:
    """``load_items`` with predictions re-read from ``raw_output`` by ``lenient_ranked``."""
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(Item(
                text         = row["text"],
                ground_truth = row["ground_truth"],
                ranked       = lenient_ranked(row.get("raw_output", ""),
                                              row["retrieved_resources"], resources),
                retrieved    = row["retrieved_resources"],
            ))
    return items
