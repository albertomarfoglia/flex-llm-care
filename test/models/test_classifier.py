"""
Unit tests for the model-independent parts of the classifier.
"""

import json
import re
from pathlib import Path

import pytest

from src.models.classifier import (
    Classifier,
    answer_regex,
    answer_regex_with_repeats,
    parse_free_answer,
    select_label,
)


class TestSelectLabel:
    """Reduction of ranked predictions to a single (y_pred, y_true) pair."""

    def test_hit_on_first_prediction(self):
        """Top prediction in ground truth is correct."""
        assert select_label(["patient", "person"], ["patient"]) == ("patient", "patient")

    def test_hit_on_lower_ranked_prediction(self):
        """Any of the n predictions counts (top-n matching)."""
        assert select_label(["device", "patient"], ["patient"]) == ("patient", "patient")

    def test_miss_uses_top_prediction_and_first_ground_truth(self):
        """A miss is scored against the first ground-truth resource."""
        assert select_label(["device", "person"], ["patient", "encounter"]) == ("device", "patient")

    @pytest.mark.parametrize("ground_truth", [
        ["questionnaire", "questionnaireresponse"],
        ["questionnaireresponse", "questionnaire"],
    ])
    def test_multi_label_hit_is_correct_for_any_ground_truth_order(self, ground_truth):
        """Regression: the original rule could score this as wrong depending on hash order."""
        y_pred, y_true = select_label(["questionnaire", "questionnaireresponse", "x"], ground_truth)
        assert y_pred == y_true == "questionnaire"

    def test_empty_predictions(self):
        """No valid prediction is a miss, not a crash."""
        assert select_label([], ["patient"]) == ("", "patient")


class TestAnswerRegex:
    """Constrained-decoding grammar."""

    LABELS = tuple("ABCDEF")

    def _matches(self, text: str, n: int = 3) -> bool:
        return re.fullmatch(answer_regex(list(self.LABELS), n), text) is not None

    def test_accepts_distinct_candidate_letters(self):
        """Valid answers parse as JSON with n distinct letters."""
        text = '{"answer": ["C", "A", "F"]}'
        assert self._matches(text)
        assert json.loads(text)["answer"] == ["C", "A", "F"]

    def test_rejects_repeated_letters(self):
        """Repeated letters would turn top-3 into top-1/2."""
        assert not self._matches('{"answer": ["A", "A", "B"]}')

    def test_rejects_letters_outside_the_candidates(self):
        """Only the labels shown in the prompt are allowed."""
        assert not self._matches('{"answer": ["A", "B", "G"]}')

    def test_rejects_wrong_cardinality(self):
        """Exactly n letters."""
        assert not self._matches('{"answer": ["A", "B"]}')
        assert self._matches('{"answer": ["B"]}', n=1)

    def test_n_output_larger_than_candidates_raises(self):
        """Cannot ask for more distinct answers than candidates."""
        with pytest.raises(ValueError):
            answer_regex(["A", "B"], 3)


class TestLoadPrompt:
    """Prompt template loading."""

    def test_reads_file(self, tmp_path: Path):
        """A path to a template file is read."""
        fp = tmp_path / "p.txt"
        fp.write_text("C: {context}\nD: {description}")
        assert Classifier.load_prompt(fp) == "C: {context}\nD: {description}"
        assert Classifier.load_prompt(str(fp)) == "C: {context}\nD: {description}"

    def test_missing_file_raises(self, tmp_path: Path):
        """Regression: a missing path used to be silently used as the prompt text."""
        with pytest.raises(FileNotFoundError):
            Classifier.load_prompt(str(tmp_path / "missing.txt"))

    def test_inline_template(self):
        """A template string is accepted as-is."""
        assert Classifier.load_prompt("{context} -> {description}") == "{context} -> {description}"

    def test_missing_placeholder_raises(self, tmp_path: Path):
        """Templates must contain both placeholders."""
        fp = tmp_path / "p.txt"
        fp.write_text("only {context}")
        with pytest.raises(ValueError):
            Classifier.load_prompt(fp)

    def test_repository_prompts_are_valid(self):
        """The shipped prompts load."""
        for name in ("prompt.txt", "prompt_top_3.txt", "prompt_top_3_norag.txt", "prompt_top_3_chat.txt"):
            Classifier.load_prompt(Path("references/configs/prompts") / name)


def test_parse_predictions_maps_letters_to_resources():
    """Letters are resolved against the labelled context."""
    context = "A) patient: x\nB) device: y\nC) encounter: z"
    assert Classifier.parse_predictions(context, {"answer": ["B", "A"]}) == [
        ("B", "device"), ("A", "patient")]
    assert Classifier.parse_predictions(context, {"answer": ["Z"]}) == [("", "")]


class TestAnswerRegexWithRepeats:
    """Grammar for large candidate sets (LLM-only catalog)."""

    def test_accepts_names_and_repeats(self):
        """Any n labels from the list, repeats allowed."""
        regex = answer_regex_with_repeats(["Patient", "Observation"], 3)
        assert re.fullmatch(regex, '{"answer": ["Patient", "Patient", "Observation"]}')
        assert not re.fullmatch(regex, '{"answer": ["Patient", "Device", "Observation"]}')
        assert not re.fullmatch(regex, '{"answer": ["Patient"]}')


class TestParseFreeAnswer:
    """Parsing of unconstrained generations."""

    @pytest.mark.parametrize("text,expected", [
        ('{"answer": ["A", "C", "B"]}', ["A", "C", "B"]),
        ('Sure! {"answer": ["b"]} done', ["b"]),
        ('{"x": 1} then {"answer": ["D"]}', ["D"]),
        ("The answer is A", None),
        ('{"answer": "A"}', None),
        ('{"answer": ["A", ', None),
    ])
    def test_cases(self, text, expected):
        """First JSON object with an answer list wins; anything else is None."""
        assert parse_free_answer(text) == expected


def test_chat_messages_merge_system_when_unsupported():
    """Templates without a system role get the system text in the user turn."""
    assert Classifier.chat_messages("S", "U", True) == [
        {"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert Classifier.chat_messages("S", "U", False) == [{"role": "user", "content": "S\n\nU"}]
    assert Classifier.chat_messages("", "U", True) == [{"role": "user", "content": "U"}]


class _FakeTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class _FakeAdapter:
    @staticmethod
    def format_input(prompt):
        return prompt if isinstance(prompt, str) else str(prompt.messages)


class _FakeGenerator:
    type_adapter = _FakeAdapter()

    def __init__(self, output):
        self.output, self.calls = output, []

    def __call__(self, prompt, output_type=None, **kwargs):
        self.calls.append((prompt, output_type, kwargs))
        return self.output


class _FakeVS:
    labels = list("ABCDEF")

    @staticmethod
    def get_context(_text, k, every_represented_candidate):
        return {"context": "A) patient: p\nB) device: d\nC) encounter: e",
                "resources": ["patient", "device", "encounter"]}


def _fake_classifier(output, constrained=True, n_output=3, prompt_format="legacy"):
    clf = Classifier.__new__(Classifier)
    clf.configure(n_output, "{context}\n{description}", prompt_format=prompt_format,
                  constrained=constrained)
    # pylint: disable=protected-access,attribute-defined-outside-init
    clf._Classifier__tokenizer      = _FakeTokenizer()
    clf._Classifier__generator      = _FakeGenerator(output)
    clf._Classifier__system_role_ok = False
    return clf


DATASET = [{"text": "Patient birthday", "ground_truth": ["patient"]}]


def test_predict_rag_constrained_records_prompt_tokens_and_time():
    """Default path: grammar used, letters mapped, provenance fields filled."""
    clf = _fake_classifier('{"answer": ["B", "A", "C"]}')
    (r,) = clf.predict(_FakeVS(), DATASET, k=3)
    assert [res for _, res in r.answer] == ["device", "patient", "encounter"]
    assert (r.y_pred, r.y_true, r.invalid_output) == ("patient", "patient", False)
    assert r.rendered_prompt.endswith("Patient birthday") and r.prompt_tokens > 0
    assert r.output_tokens == 4 and r.seconds >= 0
    assert r.raw_output == '{"answer": ["B", "A", "C"]}'
    assert clf._Classifier__generator.calls[0][1] is not None  # pylint: disable=protected-access


def test_predict_unconstrained_flags_invalid_output():
    """No grammar; unparsable output is invalid and gives no prediction."""
    clf = _fake_classifier("I think it is the patient resource.", constrained=False)
    (r,) = clf.predict(_FakeVS(), DATASET, k=3)
    assert clf._Classifier__generator.calls[0][1] is None  # pylint: disable=protected-access
    assert r.invalid_output and r.y_pred == "" and not r.error

    clf = _fake_classifier('{"answer": ["A", "Z", "B"]}', constrained=False)
    (r,) = clf.predict(_FakeVS(), DATASET, k=3)
    assert r.invalid_output and r.y_pred == "patient"


def test_predict_catalog_maps_names_and_drops_repeats():
    """LLM-only mode: labels are resource names; repeats are removed; no retrieval recorded."""
    catalog = {"context": "- Device: d\n- Patient: p", "labels": ["Device", "Patient"],
               "resources": ["device", "patient"]}
    clf = _fake_classifier('{"answer": ["Device", "Device", "Patient"]}')
    (r,) = clf.predict(None, DATASET, catalog=catalog)
    assert r.answer == [("Device", "device"), ("Device", "device"), ("Patient", "patient")]
    assert (r.y_pred, r.retrieved_resources, r.contexts) == ("patient", [], [])


def test_predict_chat_format_without_template_uses_plain_turns():
    """Tokenizers without a chat template get a plain User/Assistant prompt."""
    clf = _fake_classifier('{"answer": ["A", "B", "C"]}', prompt_format="chat")
    (r,) = clf.predict(_FakeVS(), DATASET, k=3)
    assert "\n\nUser: " in r.rendered_prompt and r.rendered_prompt.endswith("Assistant:")


def test_eos_token_ids_adds_tokenizer_eos_once():
    """Generation also stops at the tokenizer EOS; ids are not duplicated."""
    assert Classifier.eos_token_ids(128001, 128009) == [128001, 128009]
    assert Classifier.eos_token_ids([128001, 128009], 128009) == [128001, 128009]
    assert Classifier.eos_token_ids(None, 2) == [2]
