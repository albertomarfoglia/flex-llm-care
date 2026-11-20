"""
Run the non-LLM baselines on every evaluation dataset.

Each (baseline, dataset) pair is written to ``<output-dir>/<baseline>__<dataset>/``
as ``predictions.jsonl`` (same schema as ``src/pipeline.py``, ``answer`` holding
the full ranking) plus ``meta.json``. Existing folders are never overwritten.

Example
-------
python scripts/run_baselines.py --output-dir results/revision/baselines
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
import pandas as pd

from src.baselines.baselines import BM25Ranker, DenseRanker, NameMatchRanker, PriorRanker
from src.data.loader import Loader
from src.features.vectorstore import VectorStore
from src.pipeline import ROOT, resolve_path

# (label, CSV relative to data/); label sets are identical across description variants
DATASETS = {
    "motu":           "motu_enriched.csv",
    "mimic":          "mimic_enriched.csv",
    "sphn":           "sphn/sphn.csv",
    "motu_original":  "motu/motu.csv",
    "mimic_original": "mimic/mimic.csv",
}
FAMILY = {"motu": "motu", "mimic": "mimic", "sphn": "sphn",
          "motu_original": "motu", "mimic_original": "mimic"}

# Embedding models available in the local HF cache, revisions pinned.
EMBEDDERS = {
    "medembed":   ("abhinand/MedEmbed-large-v0.1", "963121bfb9c625475f65b08fb54990ce9c4e7a1a",
                   "models/faiss_index"),
    "pubmedbert": ("NeuML/pubmedbert-base-embeddings", "d6eaca8254bc229f3ca42749a5510ae287eb3486",
                   "models/faiss_index_pubmedbert"),
    "minilm":     ("sentence-transformers/all-MiniLM-L6-v2",
                   "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
                   "models/faiss_index_minilm"),
}
K_SHOWN = 6  # candidates FRMT-M shows to the LLM


def _resource_names(resources: list[str]) -> list[str]:
    df = pd.read_csv(ROOT / "data" / "FHIR_resources.csv")
    return sorted({r.strip() for r in df["Resource"].dropna() if r.strip().lower() in resources})


def _write(out_dir: Path, dataset: list[dict], rankings: list[list[str]], retrieved: bool,
           meta: dict) -> None:
    out_dir.mkdir(parents=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item, ranking in zip(dataset, rankings):
            f.write(json.dumps({
                "text":                item["text"],
                "ground_truth":        item["ground_truth"],
                "answer":              [["", r] for r in ranking],
                "retrieved_resources": ranking[:K_SHOWN] if retrieved else [],
            }) + "\n")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:  # pylint: disable=too-many-locals
    """Run all baselines; skip pairs whose output folder already exists."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision/baselines"))
    parser.add_argument("--embedders", nargs="+", default=list(EMBEDDERS), choices=list(EMBEDDERS))
    args = parser.parse_args()

    loader    = Loader()
    resources = sorted(json.loads(d)["resource"].lower() for d in loader.docs)
    data      = {name: loader.load_dataset(csv) for name, csv in DATASETS.items()}
    bm25      = BM25Ranker(loader.docs)

    rankers: dict[str, tuple[object, bool, dict]] = {
        "bm25":      (bm25, False, {"k1": 1.5, "b": 0.75}),
        "namematch": (NameMatchRanker(_resource_names(resources), bm25), False, {}),
    }
    for key in args.embedders:
        model, revision, index = EMBEDDERS[key]
        vs = VectorStore(embeddings=model, docs=loader.docs, revision=revision)
        vs.load_or_build(resolve_path(index), chunk_size=6)
        rankers[f"dense_{key}"] = (DenseRanker(vs, resources), True,
                                   {"embeddings": model, "revision": revision, "chunk_size": 6})

    for name, dataset in data.items():
        family = FAMILY[name]
        others = [it["ground_truth"] for n, ds in data.items()
                  if FAMILY[n] != family and not n.endswith("_original") for it in ds]
        per_dataset = {
            **rankers,
            "prior_oracle": (PriorRanker([it["ground_truth"] for it in dataset], resources), False,
                             {"fit_on": "the evaluated items themselves (upper bound)"}),
            "prior_cross":  (PriorRanker(others, resources), False,
                             {"fit_on": "the other two datasets"}),
        }
        for method, (ranker, retrieved, params) in per_dataset.items():
            out_dir = args.output_dir / f"{method}__{name}"
            if out_dir.exists():
                print(f"[skip] {out_dir} exists")
                continue
            start    = time.time()
            rankings = [ranker.rank(item["text"]) for item in dataset]
            _write(out_dir, dataset, rankings, retrieved, {
                "method": method, "dataset": DATASETS[name], "n_items": len(dataset),
                "params": params, "runtime_s": time.time() - start})
            print(f"[done] {out_dir} ({len(dataset)} items)")


if __name__ == "__main__":
    main()
