"""
Unit tests for the non-LLM baselines (no model download).
"""

import json

from langchain_core.embeddings import DeterministicFakeEmbedding

from src.baselines.baselines import (
    BM25Ranker,
    DenseRanker,
    NameMatchRanker,
    PriorRanker,
    split_camel_case,
    tokenize,
)
from src.features.vectorstore import VectorStore

RESOURCES = ["device", "medication", "medicationrequest", "patient"]


def _doc(resource: str, text: str) -> str:
    return json.dumps({"resource": resource, "description": text, "usage": "", "extra": "",
                       "examples": [], "documentation": []})


DOCS = [
    _doc("device", "a manufactured item used in the provision of healthcare such as a prosthetic knee"),
    _doc("medication", "a drug substance identified by its code and ingredients"),
    _doc("medicationrequest", "an order or request for the supply of a medication and dosage instructions"),
    _doc("patient", "demographics and administrative information about a person receiving care birth date gender"),
]


def test_tokenize_and_camel_case():
    """Tokens are lower-case alphanumerics; CamelCase names become phrases."""
    assert tokenize("Patient's Birth-Date (2020)") == ["patient", "s", "birth", "date", "2020"]
    assert split_camel_case("MedicationRequest") == "medication request"


def test_prior_ranker_is_input_independent_and_total():
    """Most frequent first, ties alphabetical, every resource present."""
    ranker = PriorRanker([["patient"], ["device"], ["patient", "device"], ["patient"]], RESOURCES)
    assert ranker.rank("anything") == ["patient", "device", "medication", "medicationrequest"]
    assert ranker.rank("") == ranker.rank("other")


def test_bm25_ranks_the_matching_document_first():
    """Query terms that occur in one document rank it first."""
    bm25 = BM25Ranker(DOCS)
    assert bm25.rank("prosthetic knee model")[0] == "device"
    assert bm25.rank("patient gender and birth date")[0] == "patient"
    assert sorted(bm25.rank("x")) == sorted(RESOURCES)


def test_name_match_prefers_the_longest_name():
    """'medication request' beats 'medication'; unmatched input falls back to BM25."""
    ranker = NameMatchRanker(["Device", "Medication", "MedicationRequest", "Patient"], BM25Ranker(DOCS))
    assert ranker.rank("Medication request start date")[:2] == ["medicationrequest", "medication"]
    assert ranker.rank("prosthetic knee")[0] == "device"


def test_dense_ranker_returns_distinct_total_ranking():
    """Dense ranking contains every resource exactly once."""
    vs = VectorStore(embeddings=DeterministicFakeEmbedding(size=16), docs=DOCS).build(chunk_size=2)
    ranking = DenseRanker(vs, RESOURCES).rank("patient birthday")
    assert sorted(ranking) == sorted(RESOURCES) and len(ranking) == len(set(ranking))
