"""
CVV-ADM prototype: Clinical Variable Validation and Anomaly Detection Module.

**Status: prototype implemented during the journal revision (October 2026).** It was not
used to produce any result of the workshop paper or of the submitted manuscript, and it
has not been evaluated. See README, section "CDE-M and CVV-ADM prototypes".

Three independent layers, each producing ``Finding`` records for expert review:

1. ``RuleValidator``: declarative rules from a YAML file (ranges, required fields,
   allowed values, date order, no future dates). Every rule carries a ``source``; the
   example rules shipped with the repository are illustrative and require clinical review.
2. ``EmbeddingOutlierDetector``: values of free-text / categorical fields are embedded
   with Bio_ClinicalBERT (mean pooling) and flagged when they are far from the other
   values of the same field (robust z-score of the mean cosine distance to the k nearest
   neighbours). This is unsupervised outlier scoring, not guideline reasoning.
3. ``src/validation/semantic.py``: records → RDF, validated against SHACL shapes with
   Apache Jena or pySHACL (constraint checking; no OWL reasoning).

Nothing is corrected automatically; findings are flagged for review.
"""

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

CLINICALBERT = ("emilyalsentzer/Bio_ClinicalBERT", "d5892b39a4adaed74b92212a44081509db72f87b")


@dataclass
class Finding:
    """One flagged value or record."""
    record_id: str
    field:     str
    layer:     str          # "rule" | "outlier" | "shacl"
    check:     str          # rule id, detector name or SHACL constraint
    severity:  str          # "error" | "warning"
    message:   str
    value:     str = ""
    source:    str = ""     # provenance of the rule / shape

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return asdict(self)


def _parse_date(value) -> datetime.date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RuleValidator:  # pylint: disable=too-few-public-methods
    """
    Applies rules of the following types (YAML list under ``rules``):

    - ``range``:    ``field``, ``min`` and/or ``max`` (numeric)
    - ``required``: ``fields``
    - ``allowed``:  ``field``, ``values``
    - ``order``:    ``before``, ``after`` (dates; ``before`` must not be later than ``after``)
    - ``not_future``: ``field`` (date not after ``today``)

    Each rule needs ``id``, ``severity`` and ``source``.
    """

    TYPES = ("range", "required", "allowed", "order", "not_future")

    def __init__(self, rules: list[dict], today: datetime.date | None = None):
        for rule in rules:
            missing = [k for k in ("id", "type", "severity", "source") if k not in rule]
            if missing or rule["type"] not in self.TYPES:
                raise ValueError(f"invalid rule {rule.get('id')}: missing {missing} or unknown type")
        self.rules = rules
        self.today = today or datetime.date.today()

    @classmethod
    def from_yaml(cls, path: Path, **kwargs) -> "RuleValidator":
        """Load rules from a YAML file with a top-level ``rules`` list."""
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["rules"], **kwargs)

    def __check(self, rule: dict, rid: str, record: dict) -> list[Finding]:
        # pylint: disable=too-many-return-statements
        def finding(fld, msg, value=""):
            return Finding(rid, fld, "rule", rule["id"], rule["severity"], msg, str(value),
                           rule["source"])
        kind = rule["type"]
        if kind == "required":
            return [finding(f, "missing required value") for f in rule["fields"]
                    if record.get(f) in (None, "")]
        if kind == "order":
            before, after = _parse_date(record.get(rule["before"])), _parse_date(record.get(rule["after"]))
            if before and after and before > after:
                return [finding(rule["after"], f"{rule['after']} precedes {rule['before']}",
                                f"{before} > {after}")]
            return []
        value = record.get(rule["field"])
        if value in (None, ""):
            return []
        if kind == "range":
            num = _number(value)
            if num is None:
                return [finding(rule["field"], "not a number", value)]
            if ("min" in rule and num < rule["min"]) or ("max" in rule and num > rule["max"]):
                return [finding(rule["field"],
                                f"outside [{rule.get('min', '-inf')}, {rule.get('max', 'inf')}]", value)]
            return []
        if kind == "allowed":
            return [] if str(value) in map(str, rule["values"]) else [
                finding(rule["field"], f"not in {rule['values']}", value)]
        date = _parse_date(value)  # not_future
        if date is None:
            return [finding(rule["field"], "not a date", value)]
        return [finding(rule["field"], "date in the future", value)] if date > self.today else []

    def validate(self, records: list[dict], id_field: str = "record_id") -> list[Finding]:
        """Findings for all records and rules."""
        return [f for i, record in enumerate(records)
                for rule in self.rules
                for f in self.__check(rule, str(record.get(id_field, i)), record)]


def clinicalbert_embedder(model: str = CLINICALBERT[0], revision: str = CLINICALBERT[1],
                          device: str | None = None) -> Callable[[list[str]], np.ndarray]:
    """Mean-pooled, L2-normalised Bio_ClinicalBERT embeddings (loaded once)."""
    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model, revision=revision)  # nosec B615
    enc = AutoModel.from_pretrained(model, revision=revision).to(device).eval()  # nosec B615

    @torch.no_grad()
    def embed(texts: list[str]) -> np.ndarray:
        batch = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        hidden = enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        vec = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(vec, dim=1).cpu().numpy()
    return embed


class EmbeddingOutlierDetector:  # pylint: disable=too-few-public-methods
    """
    Flags values that are semantically unlike the other values of the same field.

    For each field, distinct values are embedded; a value's score is the mean cosine
    distance to its ``k`` nearest other values; values whose robust z-score
    ((score - median) / max(1.4826 * MAD, min_scale)) exceeds ``z_threshold`` are flagged;
    ``min_scale`` (cosine distance) keeps negligible spreads, e.g. near-identical values,
    from turning tiny differences into large scores.

    Parameters
    ----------
    fields : list[str]
        Free-text or categorical fields to score.
    embed : Callable[[list[str]], np.ndarray]
        Returns L2-normalised row vectors (``clinicalbert_embedder()`` by default).
    """

    def __init__(self, fields: list[str], embed: Callable[[list[str]], np.ndarray] | None = None,
                 k: int = 5, z_threshold: float = 3.5, min_values: int = 8,
                 min_scale: float = 0.01):
        self.fields, self.k, self.z, self.min_values = fields, k, z_threshold, min_values
        self.min_scale = min_scale
        self.__embed = embed

    def scores(self, values: list[str]) -> np.ndarray:
        """Robust z-scores of the k-NN mean cosine distance, one per value."""
        if self.__embed is None:
            self.__embed = clinicalbert_embedder()
        vectors = self.__embed(values)
        dist = 1.0 - vectors @ vectors.T
        np.fill_diagonal(dist, np.inf)
        k = min(self.k, len(values) - 1)
        knn = np.sort(dist, axis=1)[:, :k].mean(axis=1)
        scale = max(np.median(np.abs(knn - np.median(knn))) * 1.4826, self.min_scale)
        return (knn - np.median(knn)) / scale

    def validate(self, records: list[dict], id_field: str = "record_id") -> list[Finding]:
        """Findings for values whose z-score exceeds the threshold."""
        findings = []
        for fld in self.fields:
            values = sorted({str(r[fld]) for r in records if r.get(fld) not in (None, "")})
            if len(values) < self.min_values:
                continue
            z = dict(zip(values, self.scores(values)))
            for i, record in enumerate(records):
                value = record.get(fld)
                if value not in (None, "") and z[str(value)] > self.z:
                    findings.append(Finding(str(record.get(id_field, i)), fld, "outlier",
                                            "clinicalbert-knn", "warning",
                                            f"semantic outlier (z={z[str(value)]:.1f})", str(value),
                                            "unsupervised; no clinical source"))
        return findings


class CVVADM:  # pylint: disable=too-few-public-methods
    """Runs the configured layers and returns a combined, review-ready report."""

    def __init__(self, rules: RuleValidator | None = None,
                 outliers: EmbeddingOutlierDetector | None = None, semantic=None):
        self.rules, self.outliers, self.semantic = rules, outliers, semantic

    def validate(self, records: list[dict], id_field: str = "record_id") -> dict:
        """``{"n_records", "findings", "by_layer"}``; ``findings`` are dicts."""
        findings: list[Finding] = []
        for layer in (self.rules, self.outliers, self.semantic):
            if layer is not None:
                findings += layer.validate(records, id_field=id_field)
        by_layer: dict[str, int] = {}
        for f in findings:
            by_layer[f.layer] = by_layer.get(f.layer, 0) + 1
        return {"n_records": len(records), "by_layer": by_layer,
                "findings": [f.to_dict() for f in findings]}
