"""
Unit tests for the CDE-M prototype (no model download, no network).
"""

import io
import json

import pytest

from src.extraction.cdem import CDEM, BioPortalSearch, Candidate, EntityExtractor, LocalTerminology

TEXT = "Patient with diabetes mellitus treated with metformin."


def _fake_factory(model, revision):  # pylint: disable=unused-argument
    if "diseases" in model:
        return lambda text: [{"entity_group": "DISEASE", "score": 0.98, "start": 13, "end": 30},
                             {"entity_group": "DISEASE", "score": 0.30, "start": 0, "end": 7},
                             {"entity_group": "0", "score": 0.99, "start": 31, "end": 38}]
    return lambda text: [{"entity_group": "CHEMICAL", "score": 0.95, "start": 44, "end": 53}]


def test_extractor_merges_models_and_drops_low_scores():
    """Entities from every model, above min_score, non-entity groups dropped, sorted."""
    ents = EntityExtractor(pipeline_factory=_fake_factory)(TEXT)
    assert [(e.mention, e.entity_type) for e in ents] == [
        ("diabetes mellitus", "DISEASE"), ("metformin", "CHEMICAL")]


@pytest.fixture(name="terminology")
def fixture_terminology(tmp_path):
    """Toy local terminology (codes are placeholders, not real ontology codes)."""
    fp = tmp_path / "terms.csv"
    fp.write_text("ontology,code,label,synonyms\n"
                  "TOY,T1,Diabetes mellitus,diabetes|DM\n"
                  "TOY,T2,Metformin,metformin hydrochloride\n"
                  "TOY,T3,Diabetes insipidus,\n", encoding="utf-8")
    return LocalTerminology(fp)


def test_local_terminology_ranks_best_overlap_first(terminology):
    """Exact label beats partial overlap; unrelated terms are not returned."""
    cands = terminology.search("diabetes mellitus")
    assert [c.code for c in cands] == ["T1", "T3"]
    assert terminology.search("aspirin") == []


def test_cdem_links_each_entity_top_ranked_or_with_chooser(terminology):
    """Without a chooser the first candidate is selected; a chooser can override."""
    cdem = CDEM(EntityExtractor(pipeline_factory=_fake_factory), terminology)
    out = cdem.extract(TEXT)
    assert [(c.mention, c.selected.code, c.selection) for c in out] == [
        ("diabetes mellitus", "T1", "top-ranked"), ("metformin", "T2", "top-ranked")]
    def chooser(_mention, _context, cands):
        return len(cands) - 1
    out = CDEM(EntityExtractor(pipeline_factory=_fake_factory), terminology, chooser).extract(TEXT)
    assert out[0].selected.code == "T3" and out[0].selection == "chooser"
    assert cdem.link_phrase("unknown header").selection == "no candidate"
    assert cdem.link_phrase("DiabetesMellitus").mention == "Diabetes Mellitus"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_bioportal_search_parses_and_caches(tmp_path):
    """Results are parsed into candidates; the second identical query hits the cache."""
    payload = {"collection": [
        {"@id": "http://purl.bioontology.org/ontology/SNOMEDCT/73211009",
         "prefLabel": "Diabetes mellitus",
         "links": {"ontology": "https://data.bioontology.org/ontologies/SNOMEDCT"}}]}
    calls = []

    def opener(request, timeout):  # pylint: disable=unused-argument
        calls.append(request)
        assert request.get_header("Authorization").startswith("apikey token=")
        return _Response(json.dumps(payload).encode())

    bp = BioPortalSearch(["SNOMEDCT"], api_key="test-key", cache_dir=tmp_path, opener=opener,
                         min_interval_s=0)
    first = bp.search("diabetes mellitus")
    second = bp.search("diabetes mellitus")
    assert first == second == [Candidate("SNOMEDCT", "73211009", "Diabetes mellitus",
                                         "http://purl.bioontology.org/ontology/SNOMEDCT/73211009",
                                         1, "bioportal")]
    assert len(calls) == 1
    assert "test-key" not in json.dumps([c.__dict__ for c in first])


def test_bioportal_requires_key(monkeypatch):
    """A missing key fails early instead of sending unauthenticated requests."""
    monkeypatch.delenv("BIOPORTAL_API_KEY", raising=False)
    with pytest.raises(ValueError):
        BioPortalSearch(["SNOMEDCT"])
