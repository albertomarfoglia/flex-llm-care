"""
Semantic layer of the CVV-ADM prototype (revision, October 2026; not evaluated).

Records are converted to an RDF graph (rdflib) and validated against SHACL shapes,
either with Apache Jena's ``shacl`` command (if ``JENA_HOME`` and Java are available)
or with pySHACL. Both produce a standard ``sh:ValidationReport``, parsed into
``Finding`` records. This is **constraint checking**; no OWL/RDFS reasoning is run.
Coded values can be linked to ontology concepts (e.g. BioPortal IRIs from CDE-M) with
``ex:hasConcept``, so that shapes can require or restrict codes.
"""

import os
import shutil
import subprocess  # nosec B404 - runs the local Jena CLI with fixed arguments
import tempfile
from pathlib import Path

from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SH

from src.validation.cvvadm import Finding

EX = Namespace("https://example.org/flex-llm-care/")


def records_to_rdf(records: list[dict], record_class: str, id_field: str = "record_id",
                   datatypes: dict[str, str] | None = None,
                   concepts: dict[tuple[str, str], str] | None = None) -> Graph:
    """
    One node per record (``ex:<class>/<id>``, type ``ex:<class>``), one property per field.

    Parameters
    ----------
    datatypes : dict[str, str]
        Field → ``"decimal" | "date" | "integer" | "string"`` (default string).
    concepts : dict[(field, value), iri]
        Optional ontology concept for coded values, added as ``ex:hasConcept``.
    """
    datatypes, concepts = datatypes or {}, concepts or {}
    xsd = {"decimal": XSD.decimal, "date": XSD.date, "integer": XSD.integer, "string": XSD.string}
    g = Graph()
    g.bind("ex", EX)
    for i, record in enumerate(records):
        node = EX[f"{record_class}/{record.get(id_field, i)}"]
        g.add((node, RDF.type, EX[record_class]))
        for key, value in record.items():
            if key == id_field or value in (None, ""):
                continue
            # strings as plain literals: equal to xsd:string under RDF 1.1, and matched by
            # sh:in lists in both Jena and pySHACL
            kind = datatypes.get(key, "string")
            g.add((node, EX[key], Literal(str(value)) if kind == "string"
                   else Literal(str(value), datatype=xsd[kind])))
            if (key, str(value)) in concepts:
                g.add((node, EX.hasConcept, URIRef(concepts[(key, str(value))])))
    return g


def jena_shacl_available() -> bool:
    """Whether ``$JENA_HOME/bin/shacl`` and a Java runtime are available."""
    jena = os.environ.get("JENA_HOME")
    java = shutil.which("java") or (Path(os.environ.get("JAVA_HOME", "")) / "bin" / "java")
    return bool(jena) and (Path(jena) / "bin" / "shacl").exists() and Path(java).exists()


def run_shacl(data: Graph, shapes_path: Path, engine: str = "auto") -> tuple[str, Graph]:
    """
    Validate ``data`` against the shapes; returns ``(engine_used, report_graph)``.

    ``engine``: ``"jena"``, ``"pyshacl"`` or ``"auto"`` (Jena if available).
    """
    if engine == "auto":
        engine = "jena" if jena_shacl_available() else "pyshacl"
    if engine == "pyshacl":
        # pylint: disable=import-outside-toplevel
        from pyshacl import validate
        _, report, _ = validate(data, shacl_graph=str(shapes_path), inference="none",
                                abort_on_first=False)
        return engine, report
    env = dict(os.environ)
    if "JAVA_HOME" in env:
        env["PATH"] = f"{Path(env['JAVA_HOME']) / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "data.ttl"
        data.serialize(data_path, format="turtle")
        out = subprocess.run(  # nosec B603 - fixed executable and arguments
            [str(Path(env["JENA_HOME"]) / "bin" / "shacl"), "validate",
             "--shapes", str(shapes_path), "--data", str(data_path)],
            capture_output=True, text=True, env=env, check=True, timeout=300)
    report = Graph()
    report.parse(data=out.stdout, format="turtle")
    return engine, report


def report_to_findings(report: Graph, engine: str, shapes_path: Path) -> list[Finding]:
    """One ``Finding`` per ``sh:ValidationResult``."""
    findings = []
    for result in report.subjects(RDF.type, SH.ValidationResult):
        focus = report.value(result, SH.focusNode)
        path = report.value(result, SH.resultPath)
        value = report.value(result, SH.value)
        component = report.value(result, SH.sourceConstraintComponent)
        severity = report.value(result, SH.resultSeverity)
        message = report.value(result, SH.resultMessage)
        findings.append(Finding(
            record_id=str(focus).rsplit("/", 1)[-1] if focus else "",
            field=str(path).rsplit("/", 1)[-1] if path else "",
            layer="shacl",
            check=str(component).rsplit("#", 1)[-1] if component else "",
            severity="error" if severity == SH.Violation else "warning",
            message=str(message) if message else "",
            value=str(value) if value is not None else "",
            source=f"{shapes_path.name} ({engine})"))
    return sorted(findings, key=lambda f: (f.record_id, f.field, f.check))


class SemanticValidator:  # pylint: disable=too-few-public-methods
    """CVV-ADM layer: records → RDF → SHACL validation → findings."""

    def __init__(self, shapes_path: Path, record_class: str, datatypes: dict[str, str] | None = None,
                 engine: str = "auto", concepts: dict[tuple[str, str], str] | None = None):
        self.shapes_path, self.record_class = Path(shapes_path), record_class
        self.datatypes, self.engine, self.concepts = datatypes or {}, engine, concepts or {}
        self.engine_used = ""

    def validate(self, records: list[dict], id_field: str = "record_id") -> list[Finding]:
        """Findings from the SHACL report."""
        graph = records_to_rdf(records, self.record_class, id_field, self.datatypes, self.concepts)
        self.engine_used, report = run_shacl(graph, self.shapes_path, self.engine)
        return report_to_findings(report, self.engine_used, self.shapes_path)
