"""
Evaluation metrics for Retrieval and RAG component.
"""


def hit_k(contexts: list[list[str]], actuals: list[str] | list[list[str]]) -> float:
    """
    Computes the Hit@K metric: the fraction of queries for which the expected
    answer is present in the retrieved context.

    Args:
        contexts: List of retrieved contexts, one per query.
                  Each context is a list of strings (the retrieved documents).
        actuals:  List of expected answers, one per query. An answer may be a
                  single string or a list of acceptable strings (hit if any
                  of them is retrieved).

    Returns:
        A float in [0.0, 1.0] representing the proportion of queries
        for which the expected item was found in the retrieved context
        (0.0 for an empty input).

    Example:
        >>> contexts = [["doc1", "doc2"], ["doc3"], ["doc1", "doc4"]]
        >>> actuals  = ["doc2", "doc3", "doc5"]
        >>> hit_k(contexts, actuals)
        0.6666666666666666
    """
    if not actuals:
        return 0.0
    return sum(
        any(a in context for a in ([actual] if isinstance(actual, str) else actual))
        for actual, context in zip(actuals, contexts)
    ) / len(actuals)
