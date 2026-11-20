"""
Render the exact text each model receives for a few items (no model weights loaded).

Reproduces the Classifier's prompt construction (template + retrieved context)
and the formatting Outlines applies to a string prompt: if the tokenizer has a
chat template, the prompt is wrapped as a single user message. Writes one file
per model to ``--output-dir`` together with a summary of how the hard-coded
Llama-3 header tokens of the template are tokenized.

Example
-------
python scripts/render_prompts.py --output-dir results/revision/prompt_rendering
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
import yaml
from outlines.models.transformers import _check_hf_chat_template
from transformers import AutoTokenizer

from src.data.loader import Loader
from src.features.vectorstore import VectorStore
from src.models.classifier import Classifier
from src.pipeline import load_config, resolve_path

LLAMA_TOKENS = ("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")


def render(template: str, context: str, description: str) -> str:
    """Same substitution as ``Classifier.__build_prompt``."""
    return template.replace("{context}", context).replace("{description}", description)


def main() -> None:  # pylint: disable=too-many-locals
    """Render prompts for every model in the reproduction configs."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", type=Path,
                        default=sorted(Path("reproduction/configs").glob("*_sphn.yaml")))
    parser.add_argument("--n-items", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    cfg0   = load_config(args.configs[0])
    loader = Loader()
    items  = loader.load_dataset(cfg0["dataset"]["name"])[:args.n_items]
    vs     = VectorStore(embeddings=cfg0["retrieval"]["embeddings"], docs=loader.docs,
                         revision=cfg0["retrieval"]["embeddings_revision"], device="cpu")
    retrieval = cfg0["retrieval"]
    vs.load_or_build(resolve_path(retrieval["index_path"]), chunk_size=retrieval["chunk_size"])
    contexts = [vs.get_context(it["text"], k=retrieval["k"],
                               every_represented_candidate=True)["context"] for it in items]

    report = []
    for cfg_path in args.configs:
        model = yaml.safe_load(cfg_path.read_text())["model"]
        tok   = AutoTokenizer.from_pretrained(model["name"], revision=model.get("revision"))
        template = Classifier.load_prompt(resolve_path(model.get("prompt_template",
                                                                 cfg0["model"]["prompt_template"])))
        wrapped = _check_hf_chat_template(tok)
        texts = []
        for item, ctx in zip(items, contexts):
            text = render(template, ctx, item["text"])
            if wrapped:
                text = tok.apply_chat_template([{"role": "user", "content": text}],
                                               tokenize=False, add_generation_prompt=True)
            texts.append(text)
        special = set(tok.all_special_tokens) | set(tok.get_added_vocab())
        report.append({
            "model":                    model["name"],
            "chat_template_applied":    wrapped,
            "llama_tokens_are_special": {t: t in special for t in LLAMA_TOKENS},
            "begin_of_text_occurrences": texts[0].count("<|begin_of_text|>"),
            "prompt_tokens_item0":      len(tok(texts[0], add_special_tokens=False)["input_ids"]),
        })
        name = model["name"].replace("/", "__")
        (args.output_dir / f"{name}.txt").write_text(
            f"\n\n{'=' * 20} NEXT ITEM {'=' * 20}\n\n".join(texts), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
