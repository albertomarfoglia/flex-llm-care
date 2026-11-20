"""
Summarise reproduction/outputs/run*/<run_id>/ into Markdown:

1. results of run1 next to the values reported in the paper;
2. determinism check: run1 vs run2 predictions and metrics, byte for byte.

Usage: python reproduction/summarize.py [--outputs reproduction/outputs]
"""

import argparse
import hashlib
import json
from pathlib import Path

# (F1, precision_weighted, recall_weighted, co2_kg) as printed in the paper.
PAPER = {
    ("llama-3.3-70b", "motu"): (0.72, 0.92, 0.64, 0.15),
    ("llama-3.3-70b", "mimic"): (0.81, 0.87, 0.78, 0.16),
    ("llama-3.3-70b", "sphn"): (0.86, 0.96, 0.80, 0.20),
    ("med42-70b", "motu"): (0.50, 0.88, 0.47, 0.38),
    ("med42-70b", "mimic"): (0.82, 0.88, 0.78, 0.75),
    ("med42-70b", "sphn"): (0.84, 0.93, 0.78, 0.32),
    ("falcon-40b", "motu"): (0.47, 0.88, 0.39, 0.49),
    ("falcon-40b", "mimic"): (0.64, 0.78, 0.56, 0.17),
    ("falcon-40b", "sphn"): (0.62, 0.82, 0.53, 0.07),
    ("biomistral-7b-dare", "motu"): (0.48, 0.87, 0.38, 0.11),
    ("biomistral-7b-dare", "mimic"): (0.43, 0.76, 0.35, 0.07),
    ("biomistral-7b-dare", "sphn"): (0.44, 0.62, 0.38, 0.04),
    ("qwen-umls-7b", "motu"): (0.60, 0.90, 0.52, 0.15),
    ("qwen-umls-7b", "mimic"): (0.75, 0.85, 0.70, 0.10),
    ("qwen-umls-7b", "sphn"): (0.76, 0.90, 0.69, 0.05),
    ("qwen2.5-72b", "motu"): (0.52, 0.89, 0.48, 0.37),
    ("qwen2.5-72b", "mimic"): (0.82, 0.90, 0.78, 0.21),
    ("qwen2.5-72b", "sphn"): (0.85, 0.95, 0.78, 0.09),
}
MODELS = ("llama-3.3-70b", "med42-70b", "falcon-40b", "biomistral-7b-dare", "qwen-umls-7b",
          "qwen2.5-72b")
VARIANTS = {"": "top-3, greedy", "_sampled": "top-3, model sampling (seed 42)",
            "_top1": "top-1, greedy"}


def md5(path: Path) -> str:
    """File checksum."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    """Print the Markdown report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("reproduction/outputs"))
    args = parser.parse_args()
    run1 = args.outputs / "run1"

    print("## Results (run1) vs paper\n")
    print("| Model | Dataset | Setting | Acc | F1 (w) | Prec (w) | Rec (w) | Hit@6 | CO2 kg "
          "| Paper F1 / P / R / CO2 | ΔF1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for ds in ("motu", "mimic", "sphn"):
            paper = PAPER[(model, ds)]
            paper_s = " / ".join(f"{v:.2f}" for v in paper)
            found = False
            for suffix, label in VARIANTS.items():
                fp = run1 / f"{model}_{ds}{suffix}" / "metrics.json"
                if not fp.exists():
                    continue
                found = True
                s = json.loads(fp.read_text())
                m = s["metrics"]
                print(f"| {model} | {ds} | {label} | {m['accuracy']:.2f} | {m['f1_score']:.2f} "
                      f"| {m['precision_weighted']:.2f} | {m['recall_weighted']:.2f} "
                      f"| {m['hit_Rate_k']:.2f} | {s['co2_kg']:.3f} | {paper_s} "
                      f"| {m['f1_score'] - paper[0]:+.2f} |")
            if not found:
                print(f"| {model} | {ds} | – | – | – | – | – | – | – | {paper_s} | not run |")

    print("\n## Determinism: run1 vs run2 (separate processes)\n")
    print("| Run | predictions.jsonl identical | metrics identical |")
    print("|---|---|---|")
    run2 = args.outputs / "run2"
    for d in sorted(run2.glob("*/metrics.json")) if run2.exists() else []:
        other = run1 / d.parent.name
        if not (other / "metrics.json").exists():
            continue
        same_pred = md5(d.parent / "predictions.jsonl") == md5(other / "predictions.jsonl")
        same_metrics = (json.loads(d.read_text())["metrics"]
                        == json.loads((other / "metrics.json").read_text())["metrics"])
        print(f"| {d.parent.name} | {'yes' if same_pred else 'NO'} | "
              f"{'yes' if same_metrics else 'NO'} |")

    print("\n## Provenance (run1)\n")
    print("| Run | model revision | runtime s | GPUs |")
    print("|---|---|---|---|")
    for fp in sorted(run1.glob("*/metrics.json")):
        s = json.loads(fp.read_text())
        p = s["provenance"]
        print(f"| {s['run_name']} | {p['model_revision']} | {s['runtime_s']:.0f} "
              f"| {len(p['environment']['gpus'])}× {', '.join(sorted(set(p['environment']['gpus'])))} |")


if __name__ == "__main__":
    main()
