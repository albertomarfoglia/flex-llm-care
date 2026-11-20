"""
Unit tests for the SHACL layer of the CVV-ADM prototype (pySHACL; Jena when installed).
"""

import pytest
from rdflib import RDF

from src.pipeline import ROOT
from src.validation.semantic import (
    EX,
    SemanticValidator,
    jena_shacl_available,
    records_to_rdf,
)

SHAPES = ROOT / "references/configs/cvvadm_shapes.ttl"
TYPES = {"admission_date": "date", "discharge_date": "date", "height_m": "decimal",
         "lci_admission_score": "integer"}
GOOD = {"record_id": "s1", "patient_id": "p1", "sex": "F", "admission_date": "2015-03-01",
        "discharge_date": "2015-03-20", "height_m": "1.70", "lci_admission_score": "30"}
BAD = {"record_id": "s2", "patient_id": "p2", "sex": "X", "admission_date": "2015-03-21",
       "discharge_date": "2015-03-20", "height_m": "17.0", "lci_admission_score": "70"}


def test_records_to_rdf_types_and_concepts():
    """One typed node per record; coded values get an ontology concept."""
    g = records_to_rdf([GOOD], "HospitalStay", datatypes=TYPES,
                       concepts={("sex", "F"): "http://example.org/concept/female"})
    node = EX["HospitalStay/s1"]
    assert (node, RDF.type, EX.HospitalStay) in g
    assert str(g.value(node, EX.hasConcept)) == "http://example.org/concept/female"
    assert g.value(node, EX.height_m).datatype.endswith("decimal")


def _checks(findings):
    return {(f.record_id, f.field, f.check) for f in findings}


@pytest.mark.parametrize("engine", [
    "pyshacl",
    pytest.param("jena", marks=pytest.mark.skipif(not jena_shacl_available(),
                                                   reason="JENA_HOME/Java not available")),
])
def test_shacl_flags_injected_violations(engine):
    """Clean record conforms; each injected violation in the bad record is reported."""
    validator = SemanticValidator(SHAPES, "HospitalStay", datatypes=TYPES, engine=engine)
    findings = validator.validate([GOOD, BAD])
    assert validator.engine_used == engine
    assert {f.record_id for f in findings} == {"s2"}
    assert {
        ("s2", "sex", "InConstraintComponent"),
        ("s2", "admission_date", "LessThanOrEqualsConstraintComponent"),
        ("s2", "height_m", "MaxInclusiveConstraintComponent"),
        ("s2", "lci_admission_score", "MaxInclusiveConstraintComponent"),
    } <= _checks(findings)
    height = next(f for f in findings if f.field == "height_m")
    assert height.severity == "warning"
