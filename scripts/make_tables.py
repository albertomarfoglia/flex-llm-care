"""
Generate the LaTeX tables of the revised manuscript from saved run outputs, so that
every number in the paper is traceable to predictions on disk.

Tables (written to ``--output-dir``, default ``docs/FGCS_paper/tables``):
  datasets.tex   evaluation data (Table tab:datasets)
  results.tex    LLMs and baselines, top-1 / top-3 with 95 % CIs (tab:results)
  ablations.tex  one-factor ablations vs each model's reference, paired tests (tab:ablations)
  energy.tex     runtime, tokens and energy per item (tab:energy)
  secondary.tex  macro-F1 and the weighted metrics with top-3 scoring (tab:secondary)

Example
-------
python scripts/make_tables.py --grid reproduction/outputs/run1 \\
    --baselines results/revision/baselines --gpu results/revision/gpu
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd

from src.data.loader import Loader
from src.evaluation.stats import (
    bootstrap_ci,
    hits_at,
    holm,
    load_items,
    load_items_lenient,
    paired_comparison,
    summary,
)
from src.data.resources import load_resources
from src.pipeline import ROOT

DATASETS  = {"motu": "MOTU", "mimic": "MIMIC-IV", "sphn": "SPHN"}
CSV       = {"motu": "motu_enriched.csv", "mimic": "mimic_enriched.csv", "sphn": "sphn/sphn.csv"}
GROUND_TRUTH = {
    "motu":  r"Expert mapping of the MOTU harmonization project~\cite{arcobelliFHIRstandardizedDataCollection2024c}",
    "mimic": r"MIMIC-IV on FHIR~\cite{bennettMIMICIVFHIRConverting2023a}",
    "sphn":  r"SPHN-to-FHIR concept map~\cite{klopfenstein_mapping_2023}",
}
MODELS    = {
    "qwen2.5-72b":        "Qwen2.5-72B-Instruct",
    "llama-3.3-70b":      "Llama-3.3-70B-Instruct",
    "med42-70b":          "Llama3-Med42-70B",
    "falcon-40b":         "Falcon-40B-Instruct",
    "qwen-umls-7b":       "Qwen-UMLS-7B-Instruct",
    "biomistral-7b-dare": "BioMistral-7B-DARE",
}
BASELINES = {
    "dense_medembed": "Retrieval only (MedEmbed)",
    "bm25":           "BM25",
    "namematch":      "Name matching + BM25",
    "prior_oracle":   "Class prior (test items)",
    "prior_cross":    "Class prior (other datasets)",
}
ABLATIONS = {  # variant suffix -> label
    "norag":      "LLM only (no retrieval)",
    "free":       "Unconstrained output (strict)",
    "free_lenient": "Unconstrained output (lenient)",
    "k3":         "$k=3$",
    "k10":        "$k=10$",
    "k15":        "$k=15$",
    "pubmedbert": "PubMedBERT retriever",
    "original":   "Original descriptions",
    "chat":       "Native chat format",
}
PENDING = r"\PENDING{run not available}"


def fmt(x: float) -> str:
    """Two decimals without leading zero issues."""
    return f"{x:.2f}"


def acc_ci(items: list, n: int) -> str:
    """'0.61 (0.53--0.70)'."""
    hits   = hits_at(items, n)
    lo, hi = bootstrap_ci(hits)
    return f"{fmt(hits.mean())} \\footnotesize({fmt(lo)}--{fmt(hi)})"


class Runs:  # pylint: disable=too-few-public-methods
    """Lazy access to run folders: grid (published configs), baselines, revision GPU runs."""

    def __init__(self, grid: Path, baselines: Path, gpu: Path):
        self.grid, self.baselines, self.gpu = grid, baselines, gpu
        self.resources = set(load_resources(ROOT / "data" / "FHIR_resources.csv"))

    def folder(self, kind: str, method: str, dataset: str, variant: str = "") -> Path:
        """Folder of one run."""
        if kind == "grid":
            return self.grid / f"{method}_{dataset}"
        if kind == "baseline":
            return self.baselines / f"{method}__{dataset}"
        return self.gpu / f"{method}_{dataset}_{variant}"

    def items(self, kind: str, method: str, dataset: str, variant: str = "") -> list | None:
        """Items of a run, or ``None`` if it does not exist."""
        lenient = variant == "free_lenient"
        fp = self.folder(kind, method, dataset, "free" if lenient else variant) / "predictions.jsonl"
        if not fp.exists():
            return None
        return load_items_lenient(fp, self.resources) if lenient else load_items(fp)

    def metrics(self, kind: str, method: str, dataset: str, variant: str = "") -> dict | None:
        """metrics.json of a run, or ``None``."""
        fp = self.folder(kind, method, dataset, variant) / "metrics.json"
        return json.loads(fp.read_text()) if fp.exists() else None


def datasets_table(runs: Runs) -> str:
    """Table tab:datasets."""
    loader, rows = Loader(), []
    for key, name in DATASETS.items():
        raw   = pd.read_csv(ROOT / "data" / CSV[key])
        items = loader.load_dataset(CSV[key])
        counts = pd.Series([r for it in items for r in it["ground_truth"]]).value_counts()
        tables = raw["Table_name"].nunique() if "Table_name" in raw else "--"
        ret = runs.items("baseline", "dense_medembed", key)
        rows.append(
            f"{name} & {len(raw)} & {len(items)} & {tables} & {len(counts)} & "
            f"{sum(len(it['ground_truth']) > 1 for it in items)} & "
            f"\\textit{{{counts.index[0].capitalize()}}} ({counts.iloc[0] / len(items):.0%}) & "
            f"{fmt(hits_at(ret, 6).mean()) if ret else PENDING} & ".replace("%", "\\%")
            + GROUND_TRUTH[key] + r" \\")
    return (r"""\begin{table*}[t]
\centering
\begin{tabularx}{\textwidth}{@{}lrrrrrlr>{\raggedright\arraybackslash}X@{}}
\toprule
Dataset & Rows & Items & Tables & Classes & Multi & Largest class & Hit@6 & Ground truth \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\caption{Evaluation data. Rows: mapped variables in the source mapping; Items: variables with at least one ground-truth resource among the 65 candidates; Tables: source tables; Classes: resources occurring in the ground truth; Multi: items with more than one valid resource; Largest class: share of items for which that resource is valid; Hit@6: share of items with a valid resource among the six candidates retrieved by FRMT-M; Ground truth: source of the reference mapping.}
\label{tab:datasets}
\end{table*}
""")


def results_table(runs: Runs) -> str:
    """Table tab:results: six LLMs (published configuration) and baselines."""
    lines = []
    for method, label in [*MODELS.items(), *BASELINES.items()]:
        kind = "grid" if method in MODELS else "baseline"
        cells = []
        for ds in DATASETS:
            items = runs.items(kind, method, ds)
            cells += [acc_ci(items, 1), acc_ci(items, 3)] if items else [PENDING, PENDING]
        if method == "dense_medembed":
            lines.append(r"\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return (r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{MOTU ($n=132$)} & \multicolumn{2}{c}{MIMIC-IV ($n=142$)} & \multicolumn{2}{c}{SPHN ($n=64$)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Method & Top-1 & Top-3 & Top-1 & Top-3 & Top-1 & Top-3 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\caption{Resource-type selection accuracy with 95\% bootstrap confidence intervals. LLMs: FRMT-M with the published configuration (MedEmbed retrieval, $k=6$, shared prompt template, greedy constrained decoding); the reference configuration is Qwen2.5-72B-Instruct. Baselines rank all 65 candidates without an LLM; the class prior fitted on the test items is an optimistic reference.}
\label{tab:results}
\end{table*}
""")


def _reference(runs: Runs, model: str, ds: str) -> list | None:
    return runs.items("grid", model, ds)


def ablation_rows(runs: Runs) -> list[dict]:
    """All one-factor comparisons with paired statistics (Holm within dataset × metric)."""
    rows = []
    for model in MODELS:
        for variant in ABLATIONS:
            for ds in DATASETS:
                ref = _reference(runs, model, ds)
                if variant == "original":
                    if ds == "sphn":
                        continue
                    items = runs.items("gpu", model, f"{ds}_original", "ref")
                else:
                    items = runs.items("gpu", model, ds, variant)
                if ref is None or items is None:
                    continue
                if [it.ground_truth for it in ref] != [it.ground_truth for it in items]:
                    raise ValueError(f"{model}/{variant}/{ds}: items not aligned")
                for n in (1, 3):
                    rows.append({"model": model, "variant": variant, "dataset": ds, "n": n,
                                 "value": hits_at(items, n).mean(),
                                 **paired_comparison(hits_at(items, n), hits_at(ref, n))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = np.nan
        for _, idx in df.groupby(["dataset", "n"]).groups.items():
            df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_mcnemar"].tolist())
    return df.to_dict("records")


def ablations_table(runs: Runs, out_csv: Path) -> str:
    """Table tab:ablations: top-1 / top-3 and paired differences vs the model's reference."""
    rows = ablation_rows(runs)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if not rows:
        return r"\begin{table*}[t]\centering " + PENDING + r"\caption{Ablations.}\label{tab:ablations}\end{table*}" + "\n"
    by = {(r["model"], r["variant"], r["dataset"], r["n"]): r for r in rows}

    def cell(model, variant, ds, n):
        r = by.get((model, variant, ds, n))
        if r is None:
            return "--"
        star = "$^{*}$" if r["p_holm"] < 0.05 else ""
        return f"{fmt(r['value'])} ({r['diff']:+.2f}{star})"

    lines = []
    for model, mlabel in MODELS.items():
        variants = [v for v in ABLATIONS if any(k[:2] == (model, v) for k in by)]
        if not variants:
            continue
        ref_cells = []
        for ds in DATASETS:
            ref = _reference(runs, model, ds)
            ref_cells += [fmt(hits_at(ref, 1).mean()), fmt(hits_at(ref, 3).mean())] if ref else ["--", "--"]
        lines += [r"\midrule", f"\\multicolumn{{7}}{{l}}{{\\textit{{{mlabel}}}}} \\\\",
                  r"\quad Reference & " + " & ".join(ref_cells) + r" \\"]
        for v in variants:
            lines.append(f"\\quad {ABLATIONS[v]} & " + " & ".join(
                cell(model, v, ds, n) for ds in DATASETS for n in (1, 3)) + r" \\")
    return (r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{MOTU} & \multicolumn{2}{c}{MIMIC-IV} & \multicolumn{2}{c}{SPHN} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Configuration & Top-1 & Top-3 & Top-1 & Top-3 & Top-1 & Top-3 \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\caption{One-factor ablations. Each cell gives the accuracy and, in parentheses, the paired difference from the same model's reference configuration (published configuration of Table~\ref{tab:results}); $^{*}$: Holm-adjusted exact McNemar $p<0.05$ (correction within each dataset and metric over all comparisons in the table). ``Original descriptions'' is evaluated on MOTU and MIMIC-IV only (SPHN has no enriched version). Lenient scoring reads unconstrained outputs tolerantly (candidate letters or resource names).}
\label{tab:ablations}
\end{table*}
""")


def energy_table(runs: Runs, out_csv: Path) -> str:
    """Table tab:energy: per-item runtime, tokens and energy (instrumented revision runs)."""
    rows = []
    for model, label in MODELS.items():
        variant = "ref" if runs.metrics("gpu", model, "motu", "ref") else "chat"
        ms = [runs.metrics("gpu", model, ds, variant) for ds in DATASETS]
        if not all(ms):
            continue
        n   = sum(m["n_samples"] for m in ms)
        inf = [m["energy"]["inference"] for m in ms]
        # model loading is measured once per process, in whichever config ran first
        load = next((json.loads(fp.read_text()) for fp in sorted(runs.gpu.glob(f"{model}_*/metrics.json"))
                     if not json.loads(fp.read_text())["model_reused"]), None)
        preds = [json.loads(l) for ds in DATASETS
                 for l in (runs.folder("gpu", model, ds, variant) / "predictions.jsonl").open()]
        rows.append({
            "model": label, "variant": variant, "items": n,
            "load_s": load["load_s"] if load else None,
            "load_gpu_kwh": load["energy"]["load"].get("gpu_energy_kwh") if load else None,
            "gen_s_median": float(np.median([p["seconds"] for p in preds])),
            "prompt_tokens": float(np.mean([p["prompt_tokens"] for p in preds])),
            "output_tokens": float(np.mean([p["output_tokens"] for p in preds])),
            "gpu_wh_per_item": 1000 * sum(e["gpu_energy_kwh"] for e in inf) / n,
            "total_wh_per_item": 1000 * sum(e["energy_kwh"] for e in inf) / n,
            "g_co2_per_item": 1000 * sum(e["co2_kg"] for e in inf) / n,
            "carbon_intensity": inf[0].get("carbon_intensity_kg_per_kwh"),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    dagger = r"$^{\dagger}$"
    lines = [f"{r['model']}{dagger if r['variant'] == 'chat' else ''} & "
             f"{r['load_s'] / 60 if r['load_s'] else float('nan'):.0f} & {r['gen_s_median']:.2f} & "
             f"{r['prompt_tokens']:.0f} & {r['output_tokens']:.1f} & {r['gpu_wh_per_item']:.2f} & "
             f"{r['total_wh_per_item']:.2f} & {r['g_co2_per_item']:.2f} \\\\" for r in rows]
    ci = rows[0]["carbon_intensity"] if rows else None
    return (r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}
\toprule
Model & Load (min) & Time/item (s) & Prompt tok. & Output tok. & GPU Wh/item & Total Wh/item & g CO$_2$e/item \\
\midrule
""" + ("\n".join(lines) if lines else PENDING + r" \\") + r"""
\bottomrule
\end{tabular*}
\caption{Runtime, tokens and energy per item over the 338 items (inference only; model loading reported separately; time per item is the median generation time; tok.: mean tokens per item), measured with CodeCarbon on 4$\times$H100 (one GPU for 7B models, two otherwise), local inference. Carbon intensity """
            + (f"{ci:.3f}" if ci else r"\PENDING{}") + r""" kg CO$_2$e/kWh (CodeCarbon, Emilia-Romagna, Italy). Total energy includes machine-level CPU and RAM energy shared with concurrent runs. $^{\dagger}$Measured on the native-format configuration (no instrumented run of the published configuration).}
\label{tab:energy}
\end{table*}
""")


def secondary_table(runs: Runs) -> str:
    """Table tab:secondary: macro-F1 of top-1 and the weighted metrics with top-3 scoring."""
    lines = []
    for method, label in MODELS.items():
        cells = []
        for ds in DATASETS:
            items, m = runs.items("grid", method, ds), runs.metrics("grid", method, ds)
            if items is None or m is None:
                cells += ["--"] * 3
                continue
            cells += [fmt(summary(items)["top1_f1_macro"]), fmt(m["metrics"]["f1_score"]),
                      fmt(m["metrics"]["precision_weighted"])]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return (r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{MOTU} & \multicolumn{3}{c}{MIMIC-IV} & \multicolumn{3}{c}{SPHN} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Model & F1$_{\mathrm{macro}}$@1 & F1$_{w}$ & P$_{w}$ & F1$_{\mathrm{macro}}$@1 & F1$_{w}$ & P$_{w}$ & F1$_{\mathrm{macro}}$@1 & F1$_{w}$ & P$_{w}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\caption{Secondary metrics. F1$_{\mathrm{macro}}$@1: macro-averaged F1 of the first suggestion (each resource class weighted equally). F1$_{w}$ and P$_{w}$: weighted F1 and precision with the top-3 scoring (a top-3 hit sets predicted and true label to the matching resource), reported as secondary metrics only; weighted recall equals top-3 accuracy.}
\label{tab:secondary}
\end{table*}
""")


def main() -> None:
    """Write all tables."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=Path("reproduction/outputs/run1"))
    parser.add_argument("--baselines", type=Path, default=Path("results/revision/baselines"))
    parser.add_argument("--gpu", type=Path, default=Path("results/revision/gpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/FGCS_paper/tables"))
    parser.add_argument("--csv-dir", type=Path, default=Path("results/revision/tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    runs = Runs(args.grid, args.baselines, args.gpu)
    tables = {
        "datasets":  datasets_table(runs),
        "results":   results_table(runs),
        "ablations": ablations_table(runs, args.csv_dir / "ablations.csv"),
        "energy":    energy_table(runs, args.csv_dir / "energy.csv"),
        "secondary": secondary_table(runs),
    }
    for name, tex in tables.items():
        (args.output_dir / f"{name}.tex").write_text(
            f"% Generated by scripts/make_tables.py — do not edit by hand.\n{tex}", encoding="utf-8")
    print(f"wrote {len(tables)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()
