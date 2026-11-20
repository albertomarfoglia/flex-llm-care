"""
Reconcile the 157 MOTU variables of the data descriptor (Arcobelli et al., Sci Data 11, 806,
2024; Tables 1-2, open-access full text PMC11271302 in data/motu/, CC BY 4.0)
with the 133 rows of data/motu/motu.csv and the 132 evaluated items.

Usage (repository root): python scripts/check_motu_items.py
"""

import collections
import sys
import xml.etree.ElementTree as ET  # nosec B405 - trusted, saved open-access XML
from pathlib import Path

import pandas as pd

ROOT  = Path(__file__).resolve().parents[1]
XML   = ROOT / "data" / "motu" / "PMC11271302_motu_scidata.xml"
TYPES = {"Integer", "Categorical", "Date", "Decimal", "String", "Questionnaire", "Clinical Test"}
sys.path.insert(0, str(ROOT))


def text(e) -> str:
    """Whitespace-normalised text of an XML element."""
    return " ".join("".join(e.itertext()).split())


def descriptor_variables() -> tuple[list[tuple], list[tuple]]:
    """(Table 1 variables, Table 2 questionnaire items) as (table, label, result, mapping)."""
    tables = list(ET.parse(XML).getroot().iter("table-wrap"))  # nosec B314
    t1, current = [], None
    for tr in tables[0].iter("tr"):
        cells = [text(c) for c in tr]
        idx = [k for k, c in enumerate(cells) if c in ("Mapped", "Not Mapped", "Derived")]
        if not idx:
            continue
        i, lead = idx[0], cells[:idx[0]]
        types = [k for k, c in enumerate(lead) if c in TYPES]
        if (types[-1] if types else len(lead)) == 3:
            current = lead[0]
        label = lead[1] if (types[-1] if types else len(lead)) == 3 else lead[0]
        t1.append((current, label, cells[i], cells[i + 1] if i + 1 < len(cells) else ""))
    groups = [r for r in t1 if r[3] == "Questionnaire" or r[3].startswith("Observation[3]")]
    t2 = []
    for tr in tables[1].iter("tr"):
        cells = [text(c) for c in tr]
        if "Mapped" in cells:
            t2.append(("Hospital Stay", cells[1] if len(cells) >= 7 else cells[0], "Mapped",
                       "Questionnaire"))
    return [r for r in t1 if r not in groups], t2


def category(row: tuple) -> str:
    """Exclusion rule that explains why a descriptor variable is absent from the CSV."""
    if row[2] == "Not Mapped":
        return "not mapped"
    if row[2] == "Derived":
        return "derived (FHIR search/operation)"
    if "–>" in row[3].replace(" ", "") or "->" in row[3].replace(" ", ""):
        return "mapped through a resource reference (->)"
    return "mapped outside the 65 candidates" if row[3].split(".")[0] == "Consent" else "UNEXPLAINED"


def main() -> None:
    """Print the reconciliation."""
    from src.data.loader import Loader  # pylint: disable=import-outside-toplevel
    t1, t2 = descriptor_variables()
    csv = pd.read_csv(ROOT / "data" / "motu" / "motu.csv")
    have = collections.Counter(zip(csv.Table_name, csv.Field_name))
    paper = collections.Counter([(r[0], r[1]) for r in t1 + t2])
    missing = [r for r in t1 + t2 if (paper - have)[(r[0], r[1])] > 0]
    print(f"descriptor: {len(t1)} table variables + {len(t2)} questionnaire items = {len(t1) + len(t2)}")
    print(f"motu.csv rows: {len(csv)}; absent from csv: {len(missing)}; "
          f"csv rows not in descriptor: {dict(have - paper)}")
    for cat, n in collections.Counter(category(r) for r in missing).items():
        print(f"  {n:2d}  {cat}")
    evaluated = Loader().load_dataset("motu/motu.csv")
    print(f"evaluated items after the 65-candidate filter: {len(evaluated)}")


if __name__ == "__main__":
    main()
