"""
Demonstration of the CDE-M and CVV-ADM prototypes (revision, October 2026).

**This is a demonstration, not an evaluation.** The records are synthetic (no patient
data), and the injected anomalies are of the same kinds the rules and shapes were written
for, so their detection is expected by construction. The demo shows that the modules run
end to end and what their outputs look like for expert review.

- CVV-ADM: synthetic MOTU-like hospital stays with injected anomalies → rule layer,
  Bio_ClinicalBERT outlier layer, SHACL layer (Apache Jena if ``JENA_HOME`` is set,
  otherwise pySHACL).
- CDE-M: BioBERT NER on a synthetic clinical note and on public MOTU variable
  descriptions, linked to SNOMED CT / LOINC through the BioPortal search API
  (``BIOPORTAL_API_KEY`` in ``.env``; skipped without a key).

Usage:
    JENA_HOME=... JAVA_HOME=... python scripts/demo_prototypes.py --output-dir results/revision/prototypes/demo
"""

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from dotenv import load_dotenv

from src.pipeline import ROOT
from src.validation.cvvadm import CVVADM, EmbeddingOutlierDetector, RuleValidator
from src.validation.semantic import SemanticValidator

TYPES = {"birth_date": "date", "admission_date": "date", "discharge_date": "date",
         "height_m": "decimal", "weight_kg": "decimal", "lci_admission_score": "integer",
         "barthel_admission_total": "integer", "morse_admission_total": "integer"}
CAUSES = ["cancer", "congenital", "infectious", "traumatic", "vascular"]
NOTES = ["fell while walking in the corridor", "slipped in the bathroom without the prosthesis",
         "lost balance when standing up from the chair", "tripped on the stairs during therapy",
         "fell during transfer from bed to wheelchair", "knee buckled while walking outdoors",
         "fell while putting on the prosthesis", "lost balance turning around in the gym"]
# (field, injected value, kind) — anomalies of the kinds covered by the rules and shapes
INJECTIONS = [
    ("discharge_date", "before-admission", "date order"),
    ("sex", "X", "invalid code"),
    ("lci_admission_score", 70, "score out of range"),
    ("height_m", 170.0, "unit error (cm in a metre field)"),
    ("barthel_admission_total", 150, "score out of range"),
    ("discharge_date", "2031-05-01", "future date"),
    ("injury_note", "password reset requested by the billing office", "off-topic free text"),
    ("amputation_cause", "alien abduction", "unknown category"),
]


def synthetic_records(n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """``n`` synthetic hospital stays and the list of injected anomalies (ground truth)."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        birth = datetime.date(1940, 1, 1) + datetime.timedelta(days=rng.randrange(0, 18000))
        admission = datetime.date(2011, 1, 1) + datetime.timedelta(days=rng.randrange(0, 3400))
        records.append({
            "record_id": f"stay{i:03d}", "patient_id": f"p{rng.randrange(1000):04d}",
            "sex": rng.choice("MF"), "birth_date": birth.isoformat(),
            "admission_date": admission.isoformat(),
            "discharge_date": (admission + datetime.timedelta(days=rng.randrange(7, 60))).isoformat(),
            "amputation_side": rng.choice("LR"), "amputation_cause": rng.choice(CAUSES),
            "height_m": round(rng.uniform(1.5, 1.95), 2), "weight_kg": round(rng.uniform(50, 110), 1),
            "lci_admission_score": rng.randrange(0, 57), "barthel_admission_total": rng.randrange(20, 101),
            "morse_admission_total": rng.choice(range(0, 126, 5)),
            "injury_note": rng.choice(NOTES)})
    injected = []
    for (field, value, kind), idx in zip(INJECTIONS, rng.sample(range(n), len(INJECTIONS))):
        if value == "before-admission":
            adm = datetime.date.fromisoformat(records[idx]["admission_date"])
            value = (adm - datetime.timedelta(days=5)).isoformat()
        records[idx][field] = value
        injected.append({"record_id": records[idx]["record_id"], "field": field, "kind": kind,
                         "value": str(value)})
    return records, injected


def run_cvvadm(records: list[dict], injected: list[dict], device: str | None) -> dict:
    """All three layers; detection of each injected anomaly by layer."""
    # pylint: disable=import-outside-toplevel
    from src.validation.cvvadm import clinicalbert_embedder
    semantic = SemanticValidator(ROOT / "references/configs/cvvadm_shapes.ttl", "HospitalStay",
                                 datatypes=TYPES)
    module = CVVADM(
        rules=RuleValidator.from_yaml(ROOT / "references/configs/cvvadm_rules.yaml",
                                      today=datetime.date(2026, 10, 23)),
        outliers=EmbeddingOutlierDetector(["injury_note", "amputation_cause"],
                                          embed=clinicalbert_embedder(device=device), min_values=5),
        semantic=semantic)
    report = module.validate(records)
    report["shacl_engine"] = semantic.engine_used
    flagged = {}
    for f in report["findings"]:
        flagged.setdefault((f["record_id"], f["field"]), set()).add(f["layer"])
    injected_keys = {(i["record_id"], i["field"]) for i in injected}
    report["injected"] = [{**i, "detected_by": sorted(flagged.get((i["record_id"], i["field"]), []))}
                          for i in injected]
    report["flags_on_clean_values"] = sorted(
        f"{rid}.{fld} ({'/'.join(sorted(layers))})" for (rid, fld), layers in flagged.items()
        if (rid, fld) not in injected_keys)
    return report


def run_cdem(device: str | None, cache_dir: Path) -> dict:
    """NER + BioPortal linking on a synthetic note and on public MOTU variable descriptions."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    from src.extraction.cdem import CDEM, BioPortalSearch, EntityExtractor
    if not os.environ.get("BIOPORTAL_API_KEY"):
        return {"skipped": "BIOPORTAL_API_KEY not set"}
    cdem = CDEM(EntityExtractor(device=device),
                BioPortalSearch(["SNOMEDCT", "LOINC"], cache_dir=cache_dir), n_candidates=3)
    note = ("Synthetic note: 72-year-old man with type 2 diabetes mellitus and peripheral arterial "
            "disease, transfemoral amputation for critical limb ischemia; on metformin and warfarin.")
    motu = pd.read_csv(ROOT / "data" / "motu" / "motu.csv")
    rows = motu[motu.Field_name.isin(["Diabetes", "AmputationCause", "Height", "DrugTradename"])]
    out = {"note": {"text": note, "concepts": [c.to_dict() for c in cdem.extract(note)]},
           "variables": []}
    for _, row in rows.iterrows():
        out["variables"].append({
            "field": row.Field_name, "description": row.Field_description,
            "header_link": cdem.link_phrase(row.Field_name, row.Field_description).to_dict(),
            "concepts": [c.to_dict() for c in cdem.extract(row.Field_description)]})
    return out


def main() -> None:
    """Run the demo and write JSON outputs."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-records", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    records, injected = synthetic_records(args.n_records, args.seed)
    (args.output_dir / "synthetic_records.json").write_text(json.dumps(records, indent=1))
    cvv = run_cvvadm(records, injected, args.device)
    (args.output_dir / "cvvadm_report.json").write_text(json.dumps(cvv, indent=1))
    cde = run_cdem(args.device, args.output_dir / "bioportal_cache")
    (args.output_dir / "cdem_output.json").write_text(json.dumps(cde, indent=1))

    print(f"CVV-ADM ({cvv['n_records']} synthetic records, SHACL engine: {cvv['shacl_engine']}): "
          f"findings by layer {cvv['by_layer']}")
    for i in cvv["injected"]:
        print(f"  injected {i['kind']:34s} {i['record_id']}.{i['field']:24s} -> {i['detected_by'] or 'MISSED'}")
    print(f"  flags on non-injected values: {len(cvv['flags_on_clean_values'])} "
          f"{cvv['flags_on_clean_values'][:5]}")
    if "skipped" in cde:
        print("CDE-M:", cde["skipped"])
        return
    print("CDE-M on the synthetic note:")
    for c in cde["note"]["concepts"]:
        sel = c["selected"]
        print(f"  {c['mention']!r:45s} {c['entity_type']:9s} -> "
              f"{(sel['ontology'] + ':' + sel['code'] + ' ' + sel['label']) if sel else c['selection']}")
    print("CDE-M header linking of MOTU variables:")
    for v in cde["variables"]:
        sel = v["header_link"]["selected"]
        print(f"  {v['field']:15s} -> {(sel['ontology'] + ':' + sel['code'] + ' ' + sel['label']) if sel else 'no candidate'}"
              f"   | NER on description: {[c['mention'] for c in v['concepts']]}")


if __name__ == "__main__":
    main()
