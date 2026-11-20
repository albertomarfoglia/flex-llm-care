"""
One LLMOps promotion step: compare a candidate FRMT-M configuration with the
champion under the frozen acceptance criteria, write an auditable gate report,
and record the decision in MLflow (tracking + Model Registry aliases).

Run folders are given as templates with ``{dataset}``; the datasets are those
listed in the gate file. The report is never overwritten.

Example
-------
python scripts/promote.py \\
    --champion  "results/revision/gpu/qwen2.5-72b_{dataset}_ref" \\
    --candidate "results/revision/gpu/qwen2.5-72b_{dataset}_pubmedbert" \\
    --champion-config  "reproduction/configs_revision/qwen2.5-72b_{dataset}_ref.yaml" \\
    --candidate-config "reproduction/configs_revision/qwen2.5-72b_{dataset}_pubmedbert.yaml" \\
    --report results/revision/promotion/gate_report_001.json \\
    --tracking-uri sqlite:///results/revision/mlflow/mlflow.db
"""

import argparse
import datetime
import hashlib
import json
import subprocess  # nosec B404 - only `git rev-parse`
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
import yaml

from src.evaluation.gate import evaluate_gate, load_run
from src.pipeline import ROOT

REGISTERED_MODEL = "frmt-m"


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,  # nosec
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _mlflow_run_id(client, run_dir: str) -> str | None:
    """MLflow run logged by the pipeline for ``run_dir`` (tag ``run_dir``), if any."""
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string=f"tags.run_dir = '{run_dir}'", max_results=1)
    return runs[0].info.run_id if runs else None


def record_in_mlflow(report: dict, args: argparse.Namespace, datasets: list[str]) -> dict:
    """Log the promotion run and move the registry aliases; returns MLflow ids."""
    # pylint: disable=import-outside-toplevel
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    linked = {role: {d: _mlflow_run_id(client, report["inputs"][role]["runs"][d]) for d in datasets}
              for role in ("champion", "candidate")}

    mlflow.set_experiment("FRMT-M promotion")
    with mlflow.start_run(run_name=Path(args.report).stem) as run:
        mlflow.log_params({"champion": args.champion, "candidate": args.candidate})
        mlflow.log_metrics({
            "champion_pooled_top3":  report["criteria"]["pooled_top3_improves"]["champion"],
            "candidate_pooled_top3": report["criteria"]["pooled_top3_improves"]["candidate"],
            "median_latency_ratio":  report["criteria"]["median_latency_ratio"]["ratio"]})
        mlflow.set_tags({"decision": report["decision"],
                         **{f"{role}.{d}": rid or "" for role, ids in linked.items()
                            for d, rid in ids.items()}})
        mlflow.log_artifact(args.gate)
        mlflow.log_artifact(args.report)
        for role in ("champion", "candidate"):
            for d in datasets:
                mlflow.log_artifact(getattr(args, f"{role}_config").format(dataset=d),
                                    artifact_path=f"frmt_config_{role}")

    try:
        client.get_registered_model(REGISTERED_MODEL)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            REGISTERED_MODEL, description="FRMT-M configuration (retriever, model, prompt, decoding)")

    def new_version(role: str) -> str:
        version = client.create_model_version(
            REGISTERED_MODEL, source=f"{run.info.artifact_uri}/frmt_config_{role}",
            run_id=run.info.run_id, tags={"role_at_registration": role,
                                          "gate_report": str(args.report)})
        return version.version

    try:
        champion_version = client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    except mlflow.exceptions.MlflowException:
        champion_version = new_version("champion")  # bootstrap: first champion
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", champion_version)
    candidate_version = new_version("candidate")
    client.set_registered_model_alias(REGISTERED_MODEL, "candidate", candidate_version)
    client.set_model_version_tag(REGISTERED_MODEL, candidate_version, "gate_decision",
                                 report["decision"])
    if report["decision"] == "promote":
        client.set_registered_model_alias(REGISTERED_MODEL, "previous_champion", champion_version)
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", candidate_version)
    return {"promotion_run_id": run.info.run_id, "linked_runs": linked,
            "champion_version_before": champion_version, "candidate_version": candidate_version}


def main() -> None:
    """Evaluate the gate, write the report, record it in MLflow."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--gate", default=str(ROOT / "references/configs/promotion_gate.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"{args.report} exists; gate reports are never overwritten")

    gate     = yaml.safe_load(Path(args.gate).read_text())
    datasets = gate["datasets"]
    runs     = {role: {d: load_run(Path(getattr(args, role).format(dataset=d))) for d in datasets}
                for role in ("champion", "candidate")}
    report   = evaluate_gate(runs["champion"], runs["candidate"], gate["criteria"])
    report["inputs"] = {
        "gate_file": args.gate, "gate_md5": _md5(args.gate), "git_commit": _git_commit(),
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **{role: {"runs": {d: r["dir"] for d, r in rs.items()},
                  "predictions_md5": {d: _md5(Path(r["dir"]) / "predictions.jsonl")
                                      for d, r in rs.items()}}
           for role, rs in runs.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.tracking_uri:
        report["mlflow"] = record_in_mlflow(report, args, datasets)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"decision": report["decision"],
                      **{k: v["passed"] for k, v in report["criteria"].items()}}, indent=2))


if __name__ == "__main__":
    main()
