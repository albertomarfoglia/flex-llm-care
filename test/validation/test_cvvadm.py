"""
Unit tests for the CVV-ADM prototype rule and outlier layers (no model download).
"""

import datetime

import numpy as np
import pytest

from src.pipeline import ROOT
from src.validation.cvvadm import CVVADM, EmbeddingOutlierDetector, RuleValidator

TODAY = datetime.date(2026, 10, 23)
GOOD = {"record_id": "s1", "patient_id": "p1", "sex": "F", "birth_date": "1950-01-01",
        "admission_date": "2015-03-01", "discharge_date": "2015-03-20", "amputation_side": "L",
        "amputation_cause": "vascular", "lci_admission_score": 30, "barthel_admission_total": 80,
        "morse_admission_total": 45, "height_m": 1.70, "weight_kg": 70}


@pytest.fixture(name="rules")
def fixture_rules():
    """The repository rule file."""
    return RuleValidator.from_yaml(ROOT / "references/configs/cvvadm_rules.yaml", today=TODAY)


def test_clean_record_has_no_findings(rules):
    """The example record satisfies every rule."""
    assert rules.validate([GOOD]) == []


@pytest.mark.parametrize("change,rule_id", [
    ({"discharge_date": "2015-02-01"}, "ORD-1"),
    ({"sex": "X"}, "SEX-1"),
    ({"lci_admission_score": 70}, "LCI-1"),
    ({"height_m": 17.0}, "HGT-1"),
    ({"height_m": "tall"}, "HGT-1"),
    ({"patient_id": ""}, "REQ-1"),
    ({"discharge_date": "2030-01-01"}, "FUT-1"),
])
def test_each_rule_type_fires(rules, change, rule_id):
    """Every rule type detects its injected anomaly and reports its source."""
    findings = rules.validate([{**GOOD, **change}])
    assert [f.check for f in findings] == [rule_id]
    assert findings[0].source and findings[0].layer == "rule"


def test_invalid_rule_definitions_are_rejected():
    """Rules without a source or with an unknown type are not accepted."""
    with pytest.raises(ValueError):
        RuleValidator([{"id": "X", "type": "range", "severity": "error", "field": "a"}])
    with pytest.raises(ValueError):
        RuleValidator([{"id": "X", "type": "magic", "severity": "error", "source": "s"}])


def _toy_embed(values):
    """Values starting with 'zz' point in another direction; the rest are jittered copies."""
    rng = np.random.default_rng(0)
    vecs = np.array([[0.0, 1.0, 0.0] if v.startswith("zz") else [1.0, *rng.normal(0, 0.05, 2)]
                     for v in values])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_outlier_detector_flags_semantic_outlier():
    """The value far from all others is flagged; the rest are not."""
    records = [{"record_id": str(i), "note": f"fell while walking {i}"} for i in range(12)]
    records.append({"record_id": "x", "note": "zz unrelated"})
    det = EmbeddingOutlierDetector(["note"], embed=_toy_embed, k=3, z_threshold=3.5)
    findings = det.validate(records)
    assert [(f.record_id, f.layer) for f in findings] == [("x", "outlier")]


def test_outlier_detector_skips_small_fields():
    """Fields with too few distinct values are not scored."""
    det = EmbeddingOutlierDetector(["note"], embed=_toy_embed, min_values=8)
    assert not det.validate([{"note": "a"}, {"note": "zz b"}])


def test_cvvadm_combines_layers(rules):
    """The report counts findings per layer."""
    report = CVVADM(rules=rules).validate([GOOD, {**GOOD, "record_id": "s2", "sex": "X"}])
    assert report["n_records"] == 2 and report["by_layer"] == {"rule": 1}
    assert report["findings"][0]["record_id"] == "s2"
