"""
Sensitivity of the MIMIC-IV results to the statistical unit.

The main analysis treats variables as independent. Variables of the same source
table may be correlated, so this script repeats the bootstrap resampling whole
tables (cluster bootstrap) instead of variables, for:

* the top-1 / top-3 accuracy of the reference configuration;
* its paired difference from retrieval alone (same six candidates).

Only MIMIC-IV is analysed: MOTU has 5 tables with 116 of 133 variables in one,
so a table-level bootstrap is not informative, and SPHN has no table structure.

Usage: python scripts/cluster_sensitivity.py [--out results/revision/analysis/cluster_mimic.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from src.evaluation.stats import bootstrap_ci, hits_at, load_items

ROOT      = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/revision/gpu/qwen2.5-72b_mimic_ref/predictions.jsonl"
RETRIEVAL = ROOT / "results/revision/baselines/dense_medembed__mimic/predictions.jsonl"
DATASET   = ROOT / "data/mimic_enriched.csv"


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 10_000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the item mean, resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=float)
    ids    = np.unique(clusters)
    sums   = np.array([values[clusters == c].sum() for c in ids])
    sizes  = np.array([(clusters == c).sum() for c in ids])
    rng    = np.random.default_rng(seed)
    draw   = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means  = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main() -> None:
    """Compute item-level and table-level CIs and write them as a CSV."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/revision/analysis/cluster_mimic.csv")
    args = parser.parse_args()

    ref, ret = load_items(REFERENCE), load_items(RETRIEVAL)
    data     = pd.read_csv(DATASET)
    texts    = data["Field_description"].tolist()
    if [it.text for it in ref] != texts or [it.text for it in ret] != texts:
        raise SystemExit("predictions are not aligned with the dataset rows")
    tables = data["Table_name"].to_numpy()

    rows = []
    for n in (1, 3):
        series = {f"reference top-{n}": hits_at(ref, n),
                  f"reference - retrieval top-{n}": hits_at(ref, n) - hits_at(ret, n)}
        for name, values in series.items():
            rows.append({"quantity": name, "estimate": values.mean(),
                         "item_ci": bootstrap_ci(values),
                         "table_ci": cluster_bootstrap_ci(values, tables)})
    out = pd.DataFrame([{"quantity": r["quantity"], "estimate": round(r["estimate"], 3),
                         "item_ci_low": round(r["item_ci"][0], 3),
                         "item_ci_high": round(r["item_ci"][1], 3),
                         "table_ci_low": round(r["table_ci"][0], 3),
                         "table_ci_high": round(r["table_ci"][1], 3)} for r in rows])
    out.insert(1, "n_items", len(texts))
    out.insert(2, "n_tables", len(np.unique(tables)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
