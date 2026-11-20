"""
Unit tests for the fixed candidate catalog (LLM-only ablation).
"""

import json

from src.data.loader import Loader
from src.data.resources import load_resource_names
from src.features.catalog import build_catalog, first_sentence
from src.pipeline import ROOT


def test_first_sentence():
    """Splits at the first sentence end followed by a capital letter."""
    assert first_sentence("A booking. This may result in X.") == "A booking."
    assert first_sentence("e.g. a value  of 3.5 units") == "e.g. a value of 3.5 units"


def test_catalog_aligns_labels_and_resources():
    """Labels are display names, resources lower-case, both in the same order."""
    docs = [json.dumps({"resource": r, "description": d}) for r, d in
            [("patient", "Demographics. More."), ("medicationrequest", "An order.")]]
    cat = build_catalog(docs, ["MedicationRequest", "Patient"])
    assert cat["labels"] == ["MedicationRequest", "Patient"]
    assert cat["resources"] == ["medicationrequest", "patient"]
    assert cat["context"] == "- MedicationRequest: An order.\n- Patient: Demographics."


def test_repository_catalog_covers_all_65_resources():
    """Every targeted resource has a display name and a non-empty definition."""
    cat = build_catalog(Loader().docs, load_resource_names(ROOT / "data" / "FHIR_resources.csv"))
    assert len(cat["labels"]) == len(set(cat["labels"])) == 65
    assert all(label.lower() == res for label, res in zip(cat["labels"], cat["resources"]))
    assert all(line.split(": ", 1)[1] for line in cat["context"].split("\n"))
