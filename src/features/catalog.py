"""
Fixed candidate catalog for mapping without retrieval (``retrieval.mode: none``).

Every targeted resource is listed by name with the first sentence of its
FHIR R4 description, taken from the same knowledge base the retriever indexes,
so the LLM-only ablation differs from RAG only in *which* information is shown.
"""

import json
import re


def first_sentence(text: str) -> str:
    """First sentence of ``text`` (whole text if it has a single sentence)."""
    text  = " ".join(text.split())
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return text[:match.start()] if match else text


def build_catalog(docs: list[str], names: list[str]) -> dict:
    """
    Catalog of all resources in ``docs``.

    Parameters
    ----------
    docs : list[str]
        JSON resource documents as returned by ``Loader.docs``.
    names : list[str]
        Display names in CamelCase (``FHIR_resources.csv``); matched to the
        lower-case document resources case-insensitively.

    Returns
    -------
    dict
        ``context`` (one ``"- Name: definition"`` line per resource,
        alphabetical), ``labels`` (display names, used as answer labels) and
        ``resources`` (lower-case names aligned with ``labels``).
    """
    display = {n.lower(): n for n in names}
    entries = sorted(
        (display.get(p["resource"].lower(), p["resource"]), p["resource"].lower(),
         first_sentence(p.get("description", "")))
        for p in map(json.loads, docs))
    return {
        "context":   "\n".join(f"- {label}: {definition}" for label, _, definition in entries),
        "labels":    [label for label, _, _ in entries],
        "resources": [res for _, res, _ in entries],
    }
