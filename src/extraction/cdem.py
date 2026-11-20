"""
CDE-M prototype: Clinical Data Extraction Module.

**Status: prototype implemented during the journal revision (October 2026).** It was not
used to produce any result of the workshop paper or of the submitted manuscript, and it
has not been evaluated. See README, section "CDE-M and CVV-ADM prototypes".

Pipeline (one call to ``CDEM.extract``):

1. Named entity recognition with BioBERT token-classification models
   (default: diseases and chemicals, ``alvaroalon2/biobert_*_ner``).
2. Candidate ontology concepts for each entity, from the BioPortal search API
   (``BIOPORTAL_API_KEY``) or from a local terminology CSV.
3. Selection of one concept per entity: the top-ranked candidate, or an optional
   chooser (e.g. an LLM with constrained decoding, ``LLMChooser``).

The output is a list of ``Concept`` records with offsets, scores and provenance, for
expert review. Structured variables (header + description) are handled by passing the
description as text; the header can be linked directly with ``link_phrase``.
"""

import csv
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

NER_MODELS = {  # HF id: pinned commit (Apache-2.0)
    "alvaroalon2/biobert_diseases_ner":  "0a4ed7dd4fd3e6db7229e5b801b70eefcb253759",
    "alvaroalon2/biobert_chemical_ner":  "b57181cea10a3943c35cf31bb375ff9217886730",
}
BIOPORTAL_URL = "https://data.bioontology.org/search"
OUTSIDE_LABELS = {"0", "O"}  # non-entity groups emitted by some token-classification models


@dataclass
class Candidate:
    """One ontology concept proposed for a mention."""
    ontology: str
    code:     str
    label:    str
    iri:      str
    rank:     int
    source:   str


@dataclass
class Concept:  # pylint: disable=too-many-instance-attributes
    """An extracted entity and the ontology concept selected for it."""
    mention:     str
    entity_type: str
    start:       int
    end:         int
    ner_score:   float
    ner_model:   str
    selected:    Candidate | None = None
    candidates:  list[Candidate] = field(default_factory=list)
    selection:   str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return asdict(self)


class TerminologySource(Protocol):  # pylint: disable=too-few-public-methods
    """Anything that proposes ranked concepts for a phrase."""

    def search(self, phrase: str, limit: int = 5) -> list[Candidate]:
        """Ranked candidates for ``phrase``."""


def _code_from_iri(iri: str) -> str:
    return iri.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


class BioPortalSearch:  # pylint: disable=too-few-public-methods
    """
    BioPortal search API client with an on-disk cache (one JSON file per query).

    Parameters
    ----------
    ontologies : list[str]
        BioPortal ontology acronyms to search (e.g. ``["SNOMEDCT", "LOINC"]``).
    api_key : str | None
        Defaults to the ``BIOPORTAL_API_KEY`` environment variable. Never logged.
    cache_dir : Path | None
        Cache directory; ``None`` disables caching.
    """

    def __init__(self, ontologies: list[str], api_key: str | None = None,
                 cache_dir: Path | None = None, opener: Callable = urllib.request.urlopen,
                 min_interval_s: float = 0.2):
        self.__key       = api_key or os.environ.get("BIOPORTAL_API_KEY")
        if not self.__key:
            raise ValueError("BioPortal API key missing: set BIOPORTAL_API_KEY")
        self.ontologies  = ontologies
        self.__cache_dir = Path(cache_dir) if cache_dir else None
        self.__opener    = opener
        self.__interval  = min_interval_s
        self.__last_call = 0.0

    def __cache_path(self, query: str) -> Path | None:
        if not self.__cache_dir:
            return None
        digest = hashlib.sha256(query.encode()).hexdigest()[:24]
        return self.__cache_dir / f"{digest}.json"

    def __get(self, query: str) -> dict:
        cache = self.__cache_path(query)
        if cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        wait = self.__interval - (time.monotonic() - self.__last_call)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(
            f"{BIOPORTAL_URL}?{query}",
            headers={"Authorization": f"apikey token={self.__key}", "Accept": "application/json"})
        with self.__opener(request, timeout=30) as response:  # nosec B310 - fixed https URL
            data = json.load(response)
        self.__last_call = time.monotonic()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
        return data

    def search(self, phrase: str, limit: int = 5) -> list[Candidate]:
        """Ranked candidates from BioPortal (empty for an empty phrase)."""
        if not phrase.strip():
            return []
        query = urllib.parse.urlencode({
            "q": phrase, "ontologies": ",".join(self.ontologies), "pagesize": limit,
            "require_exact_match": "false", "display_context": "false", "display_links": "true"})
        out = []
        for rank, item in enumerate(self.__get(query).get("collection", [])[:limit], start=1):
            iri = item.get("@id", "")
            out.append(Candidate(
                ontology=item.get("links", {}).get("ontology", "").rsplit("/", 1)[-1],
                code=_code_from_iri(iri), label=item.get("prefLabel", ""), iri=iri,
                rank=rank, source="bioportal"))
        return out


class LocalTerminology:  # pylint: disable=too-few-public-methods
    """
    Terminology from a CSV with columns ``ontology,code,label[,synonyms]`` (synonyms
    separated by ``|``), ranked by token overlap (Jaccard) with the phrase. Intended for
    offline use and tests; licensed terminologies are not shipped with the repository.
    """

    def __init__(self, path: Path):
        with Path(path).open(encoding="utf-8") as f:
            self.__rows = list(csv.DictReader(f))

    @staticmethod
    def __tokens(text: str) -> set[str]:
        return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}

    def search(self, phrase: str, limit: int = 5) -> list[Candidate]:
        """Candidates ranked by best Jaccard overlap of label or synonyms with ``phrase``."""
        query  = self.__tokens(phrase)
        scored = []
        for row in self.__rows:
            names = [row["label"], *[s for s in row.get("synonyms", "").split("|") if s]]
            best  = max(len(query & self.__tokens(n)) / (len(query | self.__tokens(n)) or 1)
                        for n in names)
            if best > 0:
                scored.append((best, row))
        scored.sort(key=lambda x: (-x[0], x[1]["label"]))
        return [Candidate(ontology=row["ontology"], code=row["code"], label=row["label"],
                          iri=row.get("iri", ""), rank=rank, source="local")
                for rank, (_, row) in enumerate(scored[:limit], start=1)]


class EntityExtractor:  # pylint: disable=too-few-public-methods
    """
    BioBERT NER over one or more token-classification models.

    Parameters
    ----------
    models : dict[str, str | None]
        HF model id → revision. Defaults to ``NER_MODELS``.
    pipeline_factory : Callable | None
        Builds a pipeline per model (injected in tests); defaults to
        ``transformers.pipeline("token-classification", aggregation_strategy="simple")``.
    min_score : float
        Entities below this score are dropped.
    """

    def __init__(self, models: dict[str, str | None] | None = None,
                 pipeline_factory: Callable | None = None, min_score: float = 0.5,
                 device: int | str | None = None):
        self.models    = models or dict(NER_MODELS)
        self.min_score = min_score
        if pipeline_factory is None:
            # pylint: disable=import-outside-toplevel
            from transformers import pipeline

            def pipeline_factory(model, revision):
                return pipeline("token-classification", model=model, revision=revision,
                                aggregation_strategy="simple", device=device)
        self.__pipelines = {m: pipeline_factory(m, r) for m, r in self.models.items()}

    def __call__(self, text: str) -> list[Concept]:
        """Entities found in ``text`` by every model, sorted by position."""
        found = []
        for model, nlp in self.__pipelines.items():
            for ent in nlp(text):
                if float(ent["score"]) < self.min_score or str(ent["entity_group"]) in OUTSIDE_LABELS:
                    continue
                found.append(Concept(
                    mention=text[ent["start"]:ent["end"]], entity_type=str(ent["entity_group"]),
                    start=int(ent["start"]), end=int(ent["end"]), ner_score=float(ent["score"]),
                    ner_model=model))
        return sorted(found, key=lambda c: (c.start, -c.ner_score))


class LLMChooser:  # pylint: disable=too-few-public-methods
    """
    Optional selection of one candidate with an instruction-tuned LLM and constrained
    decoding (one letter among the shown candidates), reusing the FRMT-M grammar.
    Loading a 7B model needs a GPU; use ``CDEM(chooser=None)`` to rank by retrieval only.
    """

    PROMPT = ("Text: \"{context}\"\nMention: \"{mention}\"\n"
              "Which ontology concept matches the mention?\n{options}\n"
              "Answer with JSON {{\"answer\": [\"<letter>\"]}}.")

    def __init__(self, model: str = "BioMistral/BioMistral-7B-DARE",
                 revision: str | None = "588fca722fbffc46eab97cbe610d4203f303add8"):
        # pylint: disable=import-outside-toplevel
        import outlines
        from outlines.types import Regex
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.models.classifier import answer_regex
        tok = AutoTokenizer.from_pretrained(model, revision=revision)  # nosec B615
        hf  = AutoModelForCausalLM.from_pretrained(model, revision=revision,  # nosec B615
                                                   dtype="auto", device_map="auto")
        self.__generator = outlines.from_transformers(hf, tok)
        self.__regex     = lambda labels: Regex(answer_regex(labels, 1))

    def __call__(self, mention: str, context: str, candidates: list[Candidate]) -> int:
        labels  = [chr(ord("A") + i) for i in range(len(candidates))]
        options = "\n".join(f"{l}) {c.label} [{c.ontology}:{c.code}]"
                            for l, c in zip(labels, candidates))
        out = self.__generator(self.PROMPT.format(context=context, mention=mention, options=options),
                               output_type=self.__regex(labels), max_new_tokens=12, do_sample=False)
        return labels.index(json.loads(out)["answer"][0])


class CDEM:  # pylint: disable=too-few-public-methods
    """
    Extraction + ontology linking.

    Parameters
    ----------
    extractor : Callable[[str], list[Concept]]
        Typically an ``EntityExtractor``.
    terminology : TerminologySource
        ``BioPortalSearch`` or ``LocalTerminology``.
    chooser : Callable | None
        ``(mention, context, candidates) -> index``; ``None`` selects the top-ranked candidate.
    n_candidates : int
        Candidates retrieved per mention.
    """

    def __init__(self, extractor: Callable[[str], list[Concept]], terminology: TerminologySource,
                 chooser: Callable | None = None, n_candidates: int = 5):
        self.extractor, self.terminology = extractor, terminology
        self.chooser, self.n_candidates  = chooser, n_candidates

    def link_phrase(self, phrase: str, context: str = "") -> Concept:
        """
        Link a whole phrase (e.g. a column header) without NER. CamelCase and
        underscores are split into words first (``AmputationCause`` → ``amputation cause``).
        """
        words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", phrase).replace("_", " ").strip()
        concept = Concept(mention=words, entity_type="PHRASE", start=0, end=len(phrase),
                          ner_score=1.0, ner_model="")
        return self.__select(concept, context or phrase)

    def __select(self, concept: Concept, context: str) -> Concept:
        concept.candidates = self.terminology.search(concept.mention, self.n_candidates)
        if not concept.candidates:
            concept.selection = "no candidate"
        elif self.chooser is None:
            concept.selected, concept.selection = concept.candidates[0], "top-ranked"
        else:
            idx = self.chooser(concept.mention, context, concept.candidates)
            concept.selected, concept.selection = concept.candidates[idx], "chooser"
        return concept

    def extract(self, text: str) -> list[Concept]:
        """Entities in ``text``, each with candidates and a selected concept (or none)."""
        return [self.__select(c, text) for c in self.extractor(text)]
