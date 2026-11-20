"""
Non-LLM baselines for FHIR resource-type selection.

Every baseline maps a variable description to a *full ranking* of the targeted
resources, so it is scored with the same item-level metrics as FRMT-M
(``src/evaluation/stats.py``). None of them uses an LLM.

- ``PriorRanker``    : constant ranking by label frequency (no input used).
- ``BM25Ranker``     : lexical BM25 over the same FHIR R4 knowledge base.
- ``NameMatchRanker``: resources whose name occurs in the description first,
                       then BM25 (a simple hand-written rule baseline).
- ``DenseRanker``    : dense retrieval only (the FRMT-M retriever without the LLM).
"""

import json
import math
import re
from collections import Counter

from src.features.vectorstore import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def _complete(ranking: list[str], resources: list[str]) -> list[str]:
    """Append unranked resources (alphabetically) so every ranking is total."""
    seen = set(ranking)
    return ranking + [r for r in sorted(resources) if r not in seen]


class PriorRanker:  # pylint: disable=too-few-public-methods
    """
    Ranks resources by how often they occur as ground truth in ``label_sets``
    (ties alphabetically), ignoring the input.

    Fitted on the evaluation items themselves it is an *oracle* prior (an upper
    bound for any input-independent method); fitted on other datasets it is a
    fair cross-dataset prior.
    """

    def __init__(self, label_sets: list[list[str]], resources: list[str]):
        counts = Counter(r for labels in label_sets for r in labels)
        self.__ranking = _complete(sorted(counts, key=lambda r: (-counts[r], r)), resources)

    def rank(self, _text: str) -> list[str]:
        """The same ranking for every input."""
        return list(self.__ranking)


class BM25Ranker:  # pylint: disable=too-few-public-methods
    """
    Okapi BM25 over one document per resource (``Loader.docs``).

    Parameters
    ----------
    docs : list[str]
        JSON resource documents as returned by ``Loader.docs``.
    k1, b : float
        Standard BM25 parameters (defaults 1.5 and 0.75, not tuned).
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        parsed           = [json.loads(d) for d in docs]
        self.resources   = [p["resource"].lower() for p in parsed]
        self.__tfs       = [Counter(tokenize(d)) for d in docs]
        self.__lengths   = [sum(tf.values()) for tf in self.__tfs]
        self.__avg_len   = sum(self.__lengths) / len(self.__lengths)
        df               = Counter(t for tf in self.__tfs for t in tf)
        n                = len(docs)
        self.__idf       = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.__k1, self.__b = k1, b

    def scores(self, text: str) -> dict[str, float]:
        """BM25 score of every resource for the query ``text``."""
        query = tokenize(text)
        out   = {}
        for res, tf, length in zip(self.resources, self.__tfs, self.__lengths):
            norm = self.__k1 * (1 - self.__b + self.__b * length / self.__avg_len)
            out[res] = sum(self.__idf.get(t, 0.0) * tf[t] * (self.__k1 + 1) / (tf[t] + norm)
                           for t in query if t in tf)
        return out

    def rank(self, text: str) -> list[str]:
        """Resources by decreasing BM25 score (ties alphabetically)."""
        scores = self.scores(text)
        return sorted(scores, key=lambda r: (-scores[r], r))


def split_camel_case(name: str) -> str:
    """``"MedicationRequest"`` → ``"medication request"``."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


class NameMatchRanker:  # pylint: disable=too-few-public-methods
    """
    Rule baseline: resources whose (space-separated) name occurs in the
    description come first, longer names before shorter ones (so
    "medication request" outranks "medication"); the rest follow in BM25 order.

    Parameters
    ----------
    names : list[str]
        Resource names in CamelCase as in ``FHIR_resources.csv``.
    bm25 : BM25Ranker
        Fallback ranking.
    """

    def __init__(self, names: list[str], bm25: BM25Ranker):
        self.__patterns = sorted(
            ((n.lower(), re.compile(r"\b" + re.escape(split_camel_case(n)) + r"\b"))
             for n in names),
            key=lambda p: -len(p[1].pattern))
        self.__bm25 = bm25

    def rank(self, text: str) -> list[str]:
        """Name matches first, then BM25."""
        lowered = text.lower()
        matched = [res for res, pattern in self.__patterns if pattern.search(lowered)]
        return matched + [r for r in self.__bm25.rank(text) if r not in matched]


class DenseRanker:  # pylint: disable=too-few-public-methods
    """
    Dense retrieval only: resources in the order of their best-matching chunk,
    exactly as ``VectorStore.get_context(every_represented_candidate=True)``
    orders the candidates shown to the LLM, but over all resources.
    """

    def __init__(self, vs: VectorStore, resources: list[str]):
        self.__vs        = vs
        self.__resources = resources

    def rank(self, text: str) -> list[str]:
        """Full ranking by nearest chunk."""
        results = self.__vs.db.similarity_search_with_score(query=text, k=self.__vs.db.index.ntotal)
        ranking: list[str] = []
        for doc, _ in results:
            res = json.loads(doc.page_content)["resource"].lower()
            if res not in ranking:
                ranking.append(res)
        return _complete(ranking, self.__resources)
