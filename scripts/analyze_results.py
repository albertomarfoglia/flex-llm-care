"""
Score saved predictions of LLM runs and baselines with the same item-level
metrics, and compare every method with a reference method on the same items.

Inputs are run folders containing ``predictions.jsonl``; the method and dataset
are taken from the folder name: ``<method>_<dataset>`` for LLM grid runs
(``reproduction/outputs/run1``) and ``<method>__<dataset>`` for baselines.

Outputs (CSV + Markdown) are written to ``--output-dir``, which must not exist.

Example
-------
python scripts/analyze_results.py \\
    --runs reproduction/outputs/run1 results/revision/baselines \\
    --reference qwen2.5-72b --family main --output-dir results/revision/analysis

The paired tests reported in the paper (Table 5 and Section 5.6) use the
15-comparison family fixed with ``--family`` (see ``MAIN_FAMILY`` and the README).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
import pandas as pd

from src.data.resources import load_resources
from src.evaluation.stats import (
    bootstrap_ci,
    class_report,
    conflicting_inputs,
    error_category,
    hits_at,
    holm,
    load_items,
    load_items_lenient,
    paired_comparison,
    summary,
    top1_confusion,
)

DATASETS = ("motu_original", "mimic_original", "motu", "mimic", "sphn")
HITK_MAX = 20

# Holm family of the main comparison (paper, Section 5.6): the reference against the five
# other LLMs, the five baselines of Table 5, two further retrieval-only rankers, seeded
# sampling for two models and BioMistral with a single-answer prompt. Fixed when the first
# analysis was run, before the ablation runs existed.
MAIN_FAMILY = [
    "llama-3.3-70b", "med42-70b", "falcon-40b", "qwen-umls-7b", "biomistral-7b-dare",
    "dense_medembed", "bm25", "namematch", "prior_oracle", "prior_cross",
    "dense_minilm", "dense_pubmedbert", "qwen2.5-72b_sampled", "qwen-umls-7b_sampled",
    "biomistral-7b-dare_top1",
]
# Sensitivity analysis of the paper: one family with every comparison of the main analysis and of
# the ablations (35 per dataset and metric); the Med42 stop-token rerun is a runtime check only.
POOLED_EXCLUDE = ["med42-70b_eosfix"]


def _split_name(folder: str) -> tuple[str, str] | None:
    if "__" in folder:
        method, dataset = folder.split("__", 1)
        return (method, dataset) if dataset in DATASETS else None
    # LLM grid: <model>_<dataset>[_<variant>]; model keys contain no underscore
    tokens = folder.split("_")
    for width in (2, 1):  # "motu_original" before "motu"
        dataset = "_".join(tokens[1:1 + width])
        if len(tokens) > width and dataset in DATASETS:
            return "_".join([tokens[0], *tokens[1 + width:]]), dataset
    return None


def collect(run_dirs: list[Path]) -> dict[tuple[str, str], list]:
    """
    {(method, dataset): items} for every folder with predictions. Unconstrained
    runs (method ending in ``_free``) are also scored leniently as ``<method>_lenient``.
    """
    resources = set(load_resources(Path(__file__).resolve().parents[1] / "data" / "FHIR_resources.csv"))
    runs = {}
    for root in run_dirs:
        for folder in sorted(p for p in root.iterdir() if (p / "predictions.jsonl").exists()):
            key = _split_name(folder.name)
            if key:
                runs[key] = load_items(folder / "predictions.jsonl")
                if key[0].endswith("_free"):
                    runs[(f"{key[0]}_lenient", key[1])] = load_items_lenient(
                        folder / "predictions.jsonl", resources)
    return runs


def summary_table(runs: dict) -> pd.DataFrame:
    """Headline metrics with 95 % bootstrap CIs."""
    rows = []
    for (method, dataset), items in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = {"dataset": dataset, "method": method, **summary(items)}
        for n in (1, 3):
            row[f"top{n}_ci_low"], row[f"top{n}_ci_high"] = bootstrap_ci(hits_at(items, n))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(runs: dict, reference: str, pairs: list[tuple[str, str]] | None = None,
                 family: list[str] | None = None) -> pd.DataFrame:
    """
    Reference minus every other method (or, if ``pairs`` is given, ``a`` minus
    ``b`` for each pair), per dataset and metric; Holm within (dataset, metric).

    ``family`` restricts the reference comparisons to the listed methods, so the
    Holm family is fixed explicitly instead of depending on which run folders
    are passed with ``--runs``.
    """
    rows = []
    for dataset in DATASETS:
        comparisons = (pairs if pairs is not None else
                       [(reference, m) for (m, ds) in sorted(runs)
                        if ds == dataset and m != reference and (family is None or m in family)])
        comparisons = [(a, b) for a, b in comparisons if (a, dataset) in runs and (b, dataset) in runs]
        for n in (1, 3):
            block = []
            for ref_name, method in comparisons:
                ref, items = runs[(ref_name, dataset)], runs[(method, dataset)]
                if [it.text for it in items] != [it.text for it in ref]:
                    raise ValueError(f"{method}/{dataset}: items differ from the reference")
                block.append({"dataset": dataset, "metric": f"top{n}_accuracy",
                              "reference": ref_name, "method": method,
                              **paired_comparison(hits_at(ref, n), hits_at(items, n))})
            for row, p in zip(block, holm([r["p_mcnemar"] for r in block])):
                row["p_holm"] = p
            rows.extend(block)
    return pd.DataFrame(rows)


def hitk_table(runs: dict) -> pd.DataFrame:
    """Hit@k (k = 1..HITK_MAX) of the retrieval-only rankings."""
    rows = []
    for (method, dataset), items in sorted(runs.items()):
        if method.startswith("dense_") or method in ("bm25", "namematch"):
            rows += [{"dataset": dataset, "method": method, "k": k,
                      "hit_at_k": hits_at(items, k).mean()} for k in range(1, HITK_MAX + 1)]
    return pd.DataFrame(rows)


def error_table(runs: dict) -> pd.DataFrame:
    """Error categories (top-3) per method and dataset, plus conflicting-input items."""
    rows = []
    for (method, dataset), items in sorted(runs.items()):
        conflicting = conflicting_inputs(items)
        cats = [error_category(it) for it in items]
        rows.append({"dataset": dataset, "method": method, "n": len(items),
                     **{c: cats.count(c) for c in ("correct", "retrieval_miss", "ranking_error")},
                     "errors_on_conflicting_inputs": sum(
                         c != "correct" and it.text in conflicting for c, it in zip(cats, items))})
    return pd.DataFrame(rows)


def _family(arg: list[str] | None, runs: dict, reference: str) -> list[str] | None:
    """Resolve ``--family``: 'main', 'pooled', explicit method names, or None (all)."""
    if arg == ["main"]:
        return MAIN_FAMILY
    if arg == ["pooled"]:
        return sorted({m for m, _ in runs} - {reference} - set(POOLED_EXCLUDE))
    return arg


def main() -> None:
    """Write all tables."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", default="qwen2.5-72b")
    parser.add_argument("--detail", nargs="*", default=["qwen2.5-72b", "llama-3.3-70b"],
                        help="Methods for per-class tables and confusion matrices.")
    parser.add_argument("--pairs", nargs="*", default=[],
                        help="Extra comparisons 'a:b' (a minus b), written to paired_pairs.csv.")
    parser.add_argument("--family", nargs="*", default=None,
                        help="Methods compared with the reference in paired.csv (one Holm family per "
                             "dataset and metric): 'main' (paper), 'pooled' (sensitivity "
                             "analysis), or method names; default: every other method found.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs = collect(args.runs)

    tables = {
        "summary":  summary_table(runs),
        "paired":   paired_table(runs, args.reference,
                                 family=_family(args.family, runs, args.reference)),
        "hit_at_k": hitk_table(runs),
        "errors":   error_table(runs),
    }
    if args.pairs:
        tables["paired_pairs"] = paired_table(runs, args.reference,
                                              [tuple(p.split(":", 1)) for p in args.pairs])
    for name, df in tables.items():
        df.to_csv(args.output_dir / f"{name}.csv", index=False)
        (args.output_dir / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f"),
                                                     encoding="utf-8")

    for (method, dataset), items in runs.items():
        if method not in args.detail:
            continue
        pd.DataFrame(class_report(items)).to_csv(
            args.output_dir / f"per_class__{method}__{dataset}.csv", index=False)
        labels, cm = top1_confusion(items)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            args.output_dir / f"confusion__{method}__{dataset}.csv")
    print(f"{len(runs)} runs scored → {args.output_dir}")


if __name__ == "__main__":
    main()
