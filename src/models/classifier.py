"""
Instruction-tuned LLM classifier for FHIR resource mapping.
"""

import json
import re
import time
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import outlines
from huggingface_hub import snapshot_download
from outlines.inputs import Chat
from outlines.types import Regex
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

from src.features.vectorstore import VectorStore

DECODING_MODES = ("greedy", "model_default")
PROMPT_FORMATS = ("legacy", "chat")
"""``legacy``: the template string is passed as-is (Outlines may wrap it in the
model's chat template; ``scripts/render_prompts.py`` shows the resulting text). ``chat``: the template
is the user message and the model's own chat template is applied exactly once."""

DEFAULT_SYSTEM_PROMPT = ("You are a strict medical classification model.\n"
                         "You must follow the output rules exactly.")


@dataclass
class PredictionResult:  # pylint: disable=too-many-instance-attributes
    """
    Dataclass holding a single FHIR resource classification result.

    Attributes
    ----------
    question : str
        Full instantiated prompt sent to the model, including context
        and input description.
    contexts : list[str]
        Raw context chunks retrieved from the vector store, one per
        retrieved candidate.
    retrieved_resources : list[str]
        Ordered list of FHIR resource names returned by the retriever,
        in the same order as ``contexts`` (empty without retrieval).
    answer : list[tuple[str, str]]
        Model output as (label, resource) pairs, where the label is the
        candidate label in the prompt and the resource its name.
    y_pred : str
        Final predicted FHIR resource name (``""`` if inference failed).
    y_true : str
        Ground-truth FHIR resource name the prediction is scored against.
    ground_truth : list[str]
        All valid FHIR resource names for the sample.
    error : str
        Error message if inference failed for this sample, else ``""``.
    invalid_output : bool
        Unconstrained generation only: the output was not a JSON answer with
        ``n_output`` labels taken from the candidates.
    rendered_prompt : str
        Exact text given to the model (after any chat template).
    raw_output : str
        Generated text before parsing.
    prompt_tokens, output_tokens : int
        Token counts of ``rendered_prompt`` and of the generated text.
    seconds : float
        Wall time of the generation call.
    """
    question:            str                   = ""
    contexts:            list[str]             = field(default_factory=list)
    retrieved_resources: list[str]             = field(default_factory=list)
    answer:              list[tuple[str, str]] = field(default_factory=list)
    y_pred:              str                   = ""
    y_true:              str                   = ""
    ground_truth:        list[str]             = field(default_factory=list)
    error:               str                   = ""
    invalid_output:      bool                  = False
    rendered_prompt:     str                   = ""
    raw_output:          str                   = ""
    prompt_tokens:       int                   = 0
    output_tokens:       int                   = 0
    seconds:             float                 = 0.0


def select_label(y_preds: list[str], ground_truth: list[str]) -> tuple[str, str]:
    """
    Reduce the model's ranked predictions and the ground-truth set to one
    (y_pred, y_true) pair for single-label metrics.

    A sample is correct if any predicted resource is a ground-truth resource:
    ``y_pred`` is then the first such prediction and ``y_true`` equals it.
    Otherwise ``y_pred`` is the top prediction and ``y_true`` the first
    ground-truth resource. The result never depends on set/hash ordering.

    Parameters
    ----------
    y_preds : list[str]
        Predicted resource names, most relevant first (may be empty).
    ground_truth : list[str]
        Valid resource names, in a deterministic order.

    Returns
    -------
    tuple[str, str]
        ``(y_pred, y_true)``.
    """
    predicted = next((p for p in y_preds if p in ground_truth), y_preds[0] if y_preds else "")
    y_true    = predicted if predicted in ground_truth else ground_truth[0]
    return predicted, y_true


def answer_regex(labels: list[str], n_output: int) -> str:
    """
    Regex for ``{"answer": ["A", "B", ...]}`` with exactly ``n_output``
    *distinct* letters taken from ``labels``.

    JSON-schema ``uniqueItems`` is not enforced by Outlines, so distinct
    answers are encoded as an alternation over all ordered selections.
    """
    if not 1 <= n_output <= len(labels):
        raise ValueError(f"n_output={n_output} must be between 1 and the number of "
                         f"candidates ({len(labels)})")
    options = (
        ", ".join(f'"{letter}"' for letter in combo)
        for combo in permutations(labels, n_output)
    )
    return r'\{"answer": \[(' + "|".join(re.escape(o) for o in options) + r')\]\}'


def answer_regex_with_repeats(labels: list[str], n_output: int) -> str:
    """
    Regex for ``{"answer": [...]}`` with exactly ``n_output`` labels taken from
    ``labels``, repeats allowed.

    Used for large candidate sets (e.g. all 65 resources without retrieval),
    where enumerating distinct selections is intractable; repeated answers are
    removed after generation and counted.
    """
    if n_output < 1:
        raise ValueError(f"n_output={n_output} must be at least 1")
    one = "(" + "|".join(re.escape(f'"{label}"') for label in labels) + ")"
    return r'\{"answer": \[' + ", ".join([one] * n_output) + r'\]\}'


def parse_free_answer(text: str) -> list[str] | None:
    """
    Labels from unconstrained output: the ``answer`` list of the first JSON
    object in ``text`` that has one, else ``None``.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("answer"), list):
            return [str(a) for a in obj["answer"]]
    return None


class Classifier:  # pylint: disable=too-many-instance-attributes
    """
    Loads a causal LM and runs structured FHIR resource classification inference
    using constrained decoding via Outlines to guarantee valid JSON output.

    Parameters
    ----------
    name : str
        HuggingFace model identifier or local path (e.g. ``"meta-llama/Llama-3.2-1B"``).
    n_output : int
        Number of distinct FHIR resource candidates to return per sample.
    prompt_template : str | Path
        Path to a prompt template file, or the template text itself.
        The template **must** contain ``{context}`` and ``{description}``.
    cache_dir : str | None, optional
        Local directory for caching HuggingFace model weights. ``None``
        (default) uses the standard HuggingFace cache (``HF_HOME``).
    revision : str | None, optional
        Model revision (commit hash, branch or tag) to load. Pin a commit
        hash for reproducible experiments.
    decoding : str, optional
        ``"greedy"`` (default, deterministic) or ``"model_default"`` (use the
        model's own ``generation_config``, which may sample).
    max_new_tokens : int, optional
        Generation budget, by default ``50``.
    prompt_format : str, optional
        ``"legacy"`` (default) or ``"chat"``, see ``PROMPT_FORMATS``.
    system_prompt : str, optional
        System message for ``prompt_format="chat"``.
    constrained : bool, optional
        Constrain the output with a grammar (default ``True``).
    stop_on_tokenizer_eos : bool, optional
        Also stop generation at the tokenizer's EOS token (default ``False``).
        Some models (e.g. Llama3-Med42-70B) end their answers with a token that
        their ``generation_config`` does not list as EOS, so generation runs on
        to ``max_new_tokens`` after the grammar is complete; answers are the
        same, only runtime and energy change.

    Raises
    ------
    FileNotFoundError
        If ``prompt_template`` looks like a file path that does not exist.
    ValueError
        If the template lacks a placeholder or an option is unknown.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        name:            str,
        n_output:        int,
        prompt_template: str | Path,
        cache_dir:       str | None = None,
        revision:        str | None = None,
        decoding:        str = "greedy",
        max_new_tokens:  int = 50,
        prompt_format:   str = "legacy",
        system_prompt:   str = DEFAULT_SYSTEM_PROMPT,
        constrained:     bool = True,
        stop_on_tokenizer_eos: bool = False,
    ):
        self.__prompt_template = ""
        self.__n_output        = n_output
        self.__prompt_format   = prompt_format
        self.__system_prompt   = system_prompt
        self.__constrained     = constrained
        self.__generation_kwargs: dict = {}
        self.__output_types: dict[tuple[str, ...], Regex] = {}
        self.__tokenizer = None
        hf_logging.set_verbosity_error()
        self.__tokenizer = self.__load_tokenizer(name, cache_dir, revision)
        hf_model         = self.__load_model(name, cache_dir, revision,
                                             self.__tokenizer.eos_token_id)
        self._model_eos = hf_model.generation_config.eos_token_id
        self.configure(n_output, prompt_template, decoding, max_new_tokens,
                       prompt_format, system_prompt, constrained, stop_on_tokenizer_eos)
        self.model_revision: str | None = self.__resolve_revision(name, revision, cache_dir)
        self.__generator = outlines.from_transformers(
            model                  = hf_model,
            tokenizer_or_processor = self.__tokenizer,
        )
        self.__system_role_ok = self.__supports_system_role(self.__tokenizer)

    def configure(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_output:        int,
        prompt_template: str | Path,
        decoding:        str = "greedy",
        max_new_tokens:  int = 50,
        prompt_format:   str = "legacy",
        system_prompt:   str = DEFAULT_SYSTEM_PROMPT,
        constrained:     bool = True,
        stop_on_tokenizer_eos: bool = False,
    ) -> "Classifier":
        """
        Change the prompt / output / decoding settings without reloading the
        model weights. Parameters are as in the constructor.
        """
        if decoding not in DECODING_MODES:
            raise ValueError(f"decoding must be one of {DECODING_MODES}, got {decoding!r}")
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, "
                             f"got {prompt_format!r}")
        self.__prompt_template = self.load_prompt(prompt_template)
        self.__n_output        = n_output
        self.__prompt_format   = prompt_format
        self.__system_prompt   = system_prompt
        self.__constrained     = constrained
        self.__output_types    = {}
        self.__generation_kwargs = {"max_new_tokens": max_new_tokens}
        if decoding == "greedy":
            self.__generation_kwargs.update(do_sample=False, temperature=None,
                                            top_p=None, top_k=None)
        tokenizer = getattr(self, "_Classifier__tokenizer", None)  # unset before the model loads
        if stop_on_tokenizer_eos and tokenizer is not None:
            self.__generation_kwargs["eos_token_id"] = self.eos_token_ids(
                getattr(self, "_model_eos", None), tokenizer.eos_token_id)
        return self

    @staticmethod
    def eos_token_ids(model_eos: int | list[int] | None, tokenizer_eos: int) -> list[int]:
        """EOS ids of the model's generation config plus the tokenizer's EOS (deduplicated)."""
        ids = [] if model_eos is None else (model_eos if isinstance(model_eos, list) else [model_eos])
        return list(dict.fromkeys([*ids, tokenizer_eos]))

    @staticmethod
    def __resolve_revision(name: str, revision: str | None, cache_dir: str | None) -> str | None:
        """Exact commit hash of the loaded weights (``None`` for local model paths)."""
        if Path(name).exists():
            return None
        try:
            return Path(snapshot_download(name, revision=revision, cache_dir=cache_dir,
                                          local_files_only=True,
                                          allow_patterns=["config.json"])).name
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            return revision

    @staticmethod
    def __load_tokenizer(name: str, cache_dir: str | None, revision: str | None) -> AutoTokenizer:
        tokenizer           = AutoTokenizer.from_pretrained(  # nosec B615 - revision is configurable
            name, cache_dir=cache_dir, revision=revision)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def __load_model(
        name:         str,
        cache_dir:    str | None,
        revision:     str | None,
        pad_token_id: int,
    ) -> AutoModelForCausalLM:
        model                     = AutoModelForCausalLM.from_pretrained(  # nosec B615
            name,
            cache_dir=cache_dir,
            revision=revision,
            dtype="auto",
            device_map="auto",
        )
        model.config.pad_token_id = pad_token_id
        return model

    @staticmethod
    def __supports_system_role(tokenizer) -> bool:
        """Whether the tokenizer's chat template accepts a system message."""
        if not getattr(tokenizer, "chat_template", None):
            return False
        try:
            tokenizer.apply_chat_template([{"role": "system", "content": "s"},
                                           {"role": "user", "content": "u"}], tokenize=False)
            return True
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            return False

    @staticmethod
    def load_prompt(source: str | Path) -> str:
        """Return the template text for a template path or template string."""
        text = source
        if isinstance(source, Path) or not ("{context}" in source or "\n" in source):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Prompt template not found: {path.resolve()}")
            text = path.read_text(encoding="utf-8")
        missing = [p for p in ("{context}", "{description}") if p not in text]
        if missing:
            raise ValueError(f"Prompt template is missing placeholder(s): {missing}")
        return text

    @staticmethod
    def chat_messages(system: str, user: str, system_role_ok: bool) -> list[dict]:
        """Chat messages; the system text is prepended to the user turn if unsupported."""
        if not system:
            return [{"role": "user", "content": user}]
        if system_role_ok:
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return [{"role": "user", "content": f"{system}\n\n{user}"}]

    def __build_prompt(self, context: str, description: str) -> str | Chat:
        text = (
            self.__prompt_template
            .replace("{context}", context)
            .replace("{description}", description)
        )
        if self.__prompt_format == "legacy":
            return text
        if not getattr(self.__tokenizer, "chat_template", None):
            return f"{self.__system_prompt}\n\nUser: {text}\nAssistant:"
        return Chat(self.chat_messages(self.__system_prompt, text, self.__system_role_ok))

    def __output_type(self, labels: list[str], distinct: bool) -> Regex | None:
        if not self.__constrained:
            return None
        key = (*labels, str(distinct))
        if key not in self.__output_types:
            regex = (answer_regex(labels, self.__n_output) if distinct
                     else answer_regex_with_repeats(labels, self.__n_output))
            self.__output_types[key] = Regex(regex)
        return self.__output_types[key]

    @staticmethod
    def parse_predictions(context: str, generated: dict) -> list[tuple[str, str]]:
        """Map generated letters to resource names using the labelled context."""
        def parse_y_pred(letter: str) -> str | None:
            for line in context.split("\n"):
                if line.startswith(f"{letter})"):
                    parts = line.split(") ", 1)
                    if len(parts) > 1:
                        return parts[1].split(": ")[0].lower()
            return None

        return [
            (letter, pred)
            if (pred := parse_y_pred(letter.upper())) is not None
            else ('', '')
            for letter in generated["answer"]
        ]

    @staticmethod
    def __split_context_documents(context: str) -> list[str]:
        return re.split(r"(?m)^[A-Z]+\)\s*", context)[1:]

    def __n_tokens(self, text: str) -> int:
        return len(self.__tokenizer(text, add_special_tokens=False)["input_ids"])

    def __answer(self, ctx: dict, generated: str) -> tuple[list[tuple[str, str]], bool]:
        """(label, resource) pairs and whether the raw output was invalid."""
        if self.__constrained:
            labels = json.loads(generated)["answer"]
        else:
            labels = parse_free_answer(generated)
            if labels is None:
                return [], True
        if "labels" in ctx:  # catalog: labels are resource names
            mapping = dict(zip(ctx["labels"], ctx["resources"]))
            answer  = [(lab, mapping[lab]) if lab in mapping else ("", "") for lab in labels]
        else:
            answer  = self.parse_predictions(ctx["context"], {"answer": labels})
        invalid = len(labels) != self.__n_output or any(not res for _, res in answer)
        return answer, invalid and not self.__constrained

    def predict(  # pylint: disable=too-many-locals
        self,
        vs:                          VectorStore | None,
        dataset:                     list[dict],
        k:                           int  = 10,
        every_represented_candidate: bool = False,
        catalog:                     dict | None = None,
    ) -> list[PredictionResult]:
        """
        Run inference over a dataset and return structured results.

        Parameters
        ----------
        vs : VectorStore | None
            Vector store used to retrieve the top-k FHIR resource candidates
            for each sample (unused when ``catalog`` is given).
        dataset : list[dict]
            Samples with ``text`` and ``ground_truth`` (as returned by
            ``Loader.load_dataset``).
        k : int, optional
            Number of candidate resources to retrieve per sample, by default ``10``.
        every_represented_candidate : bool, optional
            If ``True``, at most one chunk per resource type, by default ``False``.
        catalog : dict | None, optional
            Fixed candidate list shown for every sample instead of retrieval
            (``context``, ``resources``, ``labels``; see
            ``src/features/catalog.py``). Answers are labels from
            ``labels``; repeats are allowed by the grammar and removed.

        Returns
        -------
        list[PredictionResult]
            One result per sample. If inference fails for a sample, the result
            has ``y_pred == ""`` (counted as wrong) and ``error`` set.
        """
        results = []

        for item in tqdm(dataset, desc="Predicting"):
            ground_truth = item["ground_truth"]
            try:
                if catalog is not None:
                    ctx, labels = catalog, catalog["labels"]
                else:
                    ctx    = vs.get_context(
                        item["text"],
                        k=k,
                        every_represented_candidate=every_represented_candidate,
                    )
                    labels = vs.labels[:len(ctx["resources"])]
                prompt    = self.__build_prompt(ctx["context"], item["text"])
                rendered  = self.__generator.type_adapter.format_input(prompt)
                start     = time.perf_counter()
                output_type = self.__output_type(labels, distinct=catalog is None)
                generated = self.__generator(prompt, output_type=output_type,
                                             **self.__generation_kwargs)
                seconds   = time.perf_counter() - start
                answer, invalid = self.__answer(ctx, generated)
                ranked    = list(dict.fromkeys(res for _, res in answer if res))
                y_pred, y_true = select_label(ranked, ground_truth)

                results.append(PredictionResult(
                    question            = rendered,
                    contexts            = [] if catalog is not None
                                          else self.__split_context_documents(ctx["context"]),
                    retrieved_resources = [] if catalog is not None else ctx["resources"],
                    answer              = answer,
                    y_pred              = y_pred,
                    y_true              = y_true,
                    ground_truth        = ground_truth,
                    invalid_output      = invalid,
                    rendered_prompt     = rendered,
                    raw_output          = generated,
                    prompt_tokens       = self.__n_tokens(rendered),
                    output_tokens       = self.__n_tokens(generated),
                    seconds             = seconds,
                ))
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                print(f"[SKIP] item '{item['text'][:50]}' → {e}")
                results.append(PredictionResult(question=item["text"], y_true=ground_truth[0],
                                                ground_truth=ground_truth, error=str(e)))

        return results
