"""
Evaluation metrics for FHIR resource classification.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.evaluation.rag_metrics import hit_k
from src.models.classifier import PredictionResult


def evaluate(results: list[PredictionResult]) -> dict[str, float]:
    """
    Compute classification metrics for FHIR resource predictions.

    Parameters
    ----------
    results : list[PredictionResult]
        List of prediction results, each containing the model's predicted
        FHIR resource, the ground-truth resource(s), and the retrieved
        candidate resources. Failed samples (``y_pred == ""``) count as wrong.

    Returns
    -------
    dict[str, float]
        Dictionary containing the following metrics:

        - ``accuracy``           : exact-match accuracy over all samples.
        - ``precision_macro``    : macro-averaged precision.
        - ``precision_weighted`` : weighted-averaged precision.
        - ``recall_macro``       : macro-averaged recall.
        - ``recall_weighted``    : weighted-averaged recall.
        - ``f1_score``           : weighted-averaged F1-score.
        - ``f1_macro``           : macro-averaged F1-score.
        - ``hit_Rate_k``         : hit-rate@k, i.e. fraction of samples where
                                   any ground-truth resource appears among the
                                   top-k retrieved candidates (independent of
                                   the LLM prediction).
        - ``failure_rate``       : fraction of samples where inference failed.
        - ``invalid_output_rate``: fraction of samples whose unconstrained output
                                   was not a valid answer (always 0 when constrained).
    """
    if not results:
        raise ValueError("Cannot evaluate an empty list of results")
    actuals = [r.y_true for r in results]
    predictions = [r.y_pred for r in results]
    contexts = [r.retrieved_resources for r in results]
    ground_truths = [r.ground_truth or [r.y_true] for r in results]

    kwargs = {"y_true": actuals, "y_pred": predictions, "zero_division": 0}

    return {
        "accuracy":           accuracy_score(actuals, predictions),
        "precision_macro":    precision_score(**kwargs, average="macro"),
        "precision_weighted": precision_score(**kwargs, average="weighted"),
        "recall_macro":       recall_score(**kwargs, average="macro"),
        "recall_weighted":    recall_score(**kwargs, average="weighted"),
        "f1_score":           f1_score(**kwargs, average="weighted"),
        "f1_macro":           f1_score(**kwargs, average="macro"),
        "hit_Rate_k":         hit_k(contexts, ground_truths),
        "failure_rate":       sum(bool(r.error) for r in results) / len(results),
        "invalid_output_rate": sum(r.invalid_output for r in results) / len(results),
    }
