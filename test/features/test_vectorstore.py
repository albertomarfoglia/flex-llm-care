"""
Unit tests for VectorStore, using deterministic fake embeddings (no model download).
"""

import json
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.features.vectorstore import INDEX_META_FILE, VectorStore


def _doc(resource: str, n_defs: int = 8) -> str:
    return json.dumps({
        "resource": resource,
        "description": f"{resource} description",
        "usage": f"{resource} usage",
        "extra": "",
        "examples": [f"example {i} of {resource}" for i in range(3)],
        "documentation": [f"field{i}: definition {i} of {resource}" for i in range(n_defs)],
    })


DOCS = [_doc(r) for r in ("patient", "device", "encounter", "observation", "procedure")]


@pytest.fixture(name="vs")
def fixture_vs() -> VectorStore:
    """Vector store with fake embeddings."""
    return VectorStore(embeddings=DeterministicFakeEmbedding(size=32), docs=DOCS)


def test_every_represented_candidate_returns_distinct_resources(vs):
    """At most one chunk per resource, labelled A, B, C, ..."""
    vs.build(chunk_size=2)
    ctx = vs.get_context("patient birthday", k=4, every_represented_candidate=True)
    assert len(ctx["resources"]) == len(set(ctx["resources"])) == 4
    labelled = [line.split(" ", 1)[0] for line in ctx["context"].split("\n")
                if line[:2] in ("A)", "B)", "C)", "D)", "E)")]
    assert labelled == ["A)", "B)", "C)", "D)"]
    assert [line.split(" ")[1].rstrip(":") for line in ctx["context"].split("\n")
            if line[:2] in ("A)", "B)", "C)", "D)")] == ctx["resources"]


def test_k_out_of_range_raises(vs):
    """k is bounded by the 26 labels."""
    vs.build(chunk_size=2)
    with pytest.raises(ValueError):
        vs.get_context("x", k=27)


def test_get_context_before_build_raises(vs):
    """Using an unbuilt store is an error."""
    with pytest.raises(RuntimeError):
        vs.get_context("x", k=3)


def test_load_or_build_reuses_matching_index(vs, tmp_path: Path):
    """An index built with the same settings is loaded, not rebuilt."""
    path = tmp_path / "index"
    vs.load_or_build(path, chunk_size=2)
    meta = json.loads((path / INDEX_META_FILE).read_text())
    assert meta["chunk_size"] == 2
    stamp = (path / "index.faiss").stat().st_mtime_ns
    VectorStore(embeddings=DeterministicFakeEmbedding(size=32), docs=DOCS).load_or_build(
        path, chunk_size=2)
    assert (path / "index.faiss").stat().st_mtime_ns == stamp


def test_load_or_build_rebuilds_stale_index(vs, tmp_path: Path):
    """Regression: an index built with other settings used to be reused silently."""
    path = tmp_path / "index"
    vs.load_or_build(path, chunk_size=2)
    n_small_chunks = vs.db.index.ntotal
    other = VectorStore(embeddings=DeterministicFakeEmbedding(size=32), docs=DOCS)
    with pytest.warns(UserWarning, match="stale"):
        other.load_or_build(path, chunk_size=8)
    assert other.db.index.ntotal < n_small_chunks
    assert json.loads((path / INDEX_META_FILE).read_text())["chunk_size"] == 8
