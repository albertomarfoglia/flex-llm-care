"""
Run one or more FHIR-mapping experiments from YAML configs.

Each config is one run (written to ``<output-dir>/<model>-<dataset>-<n_output>/``
and, if a tracking URI is given, logged to MLflow). The LLM is loaded only once
for consecutive configs with an identical ``model`` section.

Examples
--------
python scripts/run_experiment.py --config references/configs/config.yaml
python scripts/run_experiment.py --config reproduction/configs/qwen-umls-7b_{motu,mimic,sphn}.yaml \\
       --seed 42 --output-dir reproduction/outputs/seed42 --tracking-uri http://localhost:8080
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from dotenv import load_dotenv

from src.pipeline import ExperimentRunner, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"),
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI; "
                             "unset = no MLflow logging).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the seed in the config(s).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N samples (smoke test).")
    return parser.parse_args()


def main() -> None:
    """Run every config given on the command line."""
    args = parse_args()
    load_dotenv(".env")
    configs = [load_config(p) for p in args.config]   # validate all before running any
    runner = ExperimentRunner(output_dir=args.output_dir, tracking_uri=args.tracking_uri,
                              seed=args.seed, limit=args.limit)
    for cfg in configs:
        summary = runner.run(cfg)
        print(summary["run_name"], json.dumps(summary["metrics"], indent=2),
              f"\nfailed={summary['n_failed']} runtime={summary['runtime_s']:.0f}s "
              f"model_reused={summary['model_reused']}")


if __name__ == "__main__":
    main()
