"""
End-to-end experiment pipeline: config → data → vectorstore → inference →
metrics → artefacts (local files and, optionally, MLflow).

Used by ``scripts/run_experiment.py`` and ``notebooks/run.ipynb`` so that both
entry points run exactly the same code.
"""

import copy
import csv
import hashlib
import json
import os
import platform
import random
import time
import warnings
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

# Must be set before CUDA is initialised for deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import numpy as np
import torch
import yaml

from src.data.loader import Loader
from src.data.resources import load_resource_names
from src.evaluation.metrics import evaluate
from src.features.catalog import build_catalog
from src.features.vectorstore import LABELS, VectorStore
from src.models.classifier import (
    DECODING_MODES,
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_FORMATS,
    Classifier,
    PredictionResult,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "project_name": "FHIR Classification",
    "experiment_name": "baseline",
    "run_id": None,
    "seed": 42,
    "retrieval": {
        "mode": "rag",
        "embeddings": "abhinand/MedEmbed-large-v0.1",
        "embeddings_revision": None,
        "k": 6,
        "every_represented_candidate": True,
        "chunk_size": 6,
        "index_path": "models/faiss_index",
    },
    "model": {
        "revision": None,
        "n_output": 3,
        "prompt_template": "references/configs/prompts/prompt_top_3.txt",
        "decoding": "greedy",
        "max_new_tokens": 50,
        "cache_dir": None,
        "prompt_format": "legacy",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "constrained": True,
        "stop_on_tokenizer_eos": False,
    },
}

RETRIEVAL_MODES = ("rag", "none")

TRACKED_PACKAGES = ("torch", "transformers", "outlines", "outlines_core", "accelerate",
                    "sentence-transformers", "langchain-community", "langchain-huggingface",
                    "faiss-cpu", "scikit-learn", "codecarbon", "mlflow", "numpy", "pandas")


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path: absolute paths unchanged, relative ones against the repo root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _merge(defaults: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    """
    Load a YAML experiment config, fill in defaults and validate it.

    Relative paths in the config (``model.prompt_template``,
    ``retrieval.index_path``, ``model.cache_dir``) are resolved against the
    repository root, so results do not depend on the working directory.
    """
    cfg = _merge(DEFAULTS, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Raise ``ValueError`` for configs that cannot produce meaningful results."""
    if not cfg.get("dataset", {}).get("name"):
        raise ValueError("config: dataset.name is required")
    if not cfg["model"].get("name"):
        raise ValueError("config: model.name is required")
    k, n_output = cfg["retrieval"]["k"], cfg["model"]["n_output"]
    if cfg["retrieval"]["mode"] not in RETRIEVAL_MODES:
        raise ValueError(f"config: retrieval.mode must be one of {RETRIEVAL_MODES}")
    if cfg["retrieval"]["mode"] == "rag":
        if not 1 <= k <= len(LABELS):
            raise ValueError(f"config: retrieval.k must be in 1..{len(LABELS)}, got {k}")
        if not 1 <= n_output <= k:
            raise ValueError(f"config: model.n_output must be in 1..retrieval.k ({k}), got {n_output}")
    elif n_output < 1:
        raise ValueError(f"config: model.n_output must be at least 1, got {n_output}")
    if cfg["model"]["decoding"] not in DECODING_MODES:
        raise ValueError(f"config: model.decoding must be one of {DECODING_MODES}")
    if cfg["model"]["prompt_format"] not in PROMPT_FORMATS:
        raise ValueError(f"config: model.prompt_format must be one of {PROMPT_FORMATS}")
    if not resolve_path(cfg["model"]["prompt_template"]).is_file():
        raise ValueError(f"config: prompt template not found: {cfg['model']['prompt_template']}")


def set_seed(seed: int) -> None:
    """Seed every RNG used by the pipeline and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def environment_info() -> dict:
    """Software and hardware description stored with every run."""
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpus": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def visible_gpu_ids() -> list[int] | None:
    """Physical GPU indices from ``CUDA_VISIBLE_DEVICES`` (``None`` = not set or not numeric)."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return None
    try:
        return [int(g) for g in value.split(",") if g.strip()]
    except ValueError:
        return None


LOCATION_COLUMNS = ("longitude", "latitude")


def scrub_location(csv_path: Path) -> int:
    """
    Blank the geolocation columns that CodeCarbon writes to ``emissions*.csv`` (the
    server's coordinates); country and region, which determine the carbon intensity, are
    kept. Returns the number of rows changed. Missing files are ignored.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    fields = list(rows[0])
    changed = 0
    for row in rows:
        if any(row.get(c) for c in LOCATION_COLUMNS):
            changed += 1
            for c in LOCATION_COLUMNS:
                if c in row:
                    row[c] = ""
    if changed:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def energy_summary(data, n_items: int) -> dict:
    """kWh / kg CO2e (total and per component) of one CodeCarbon measurement."""
    if data is None:
        return {}
    energy = data.energy_consumed
    return {
        "duration_s":          data.duration,
        "energy_kwh":          energy,
        "gpu_energy_kwh":      data.gpu_energy,
        "cpu_energy_kwh":      data.cpu_energy,
        "ram_energy_kwh":      data.ram_energy,
        "co2_kg":              data.emissions,
        "carbon_intensity_kg_per_kwh": data.emissions / energy if energy else None,
        "wh_per_item":         1000 * energy / n_items if n_items else None,
        "gpu_wh_per_item":     1000 * data.gpu_energy / n_items if n_items else None,
        "country":             data.country_name,
        "region":              data.region,
        "gpu_model":           data.gpu_model,
        "gpu_count":           data.gpu_count,
    }


def run_name(cfg: dict) -> str:
    """``run_id`` from the config, else ``<model>-<dataset>-<n_output>`` (as in the notebook)."""
    return cfg.get("run_id") or (
        f'{cfg["model"]["name"]}-{cfg["dataset"]["name"]}-{cfg["model"]["n_output"]}')


def _write_outputs(out_dir: Path, summary: dict, dataset: list[dict],
                   results: list[PredictionResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item, r in zip(dataset, results):
            row = asdict(r)
            row.pop("question")
            row["text"] = item["text"]
            f.write(json.dumps(row) + "\n")


class ExperimentRunner:  # pylint: disable=too-few-public-methods
    """
    Runs experiments from config files, reusing the loaded LLM across
    consecutive configs whose ``model`` sections are identical.

    Parameters
    ----------
    output_dir : Path
        Directory under which ``<run name>/`` folders are written.
    tracking_uri : str | None
        MLflow tracking URI; ``None`` disables MLflow logging.
    seed : int | None
        Overrides the ``seed`` in each config when given.
    limit : int | None
        Only use the first ``limit`` samples (smoke tests).
    """

    def __init__(self, output_dir: Path, tracking_uri: str | None = None,
                 seed: int | None = None, limit: int | None = None):
        self.output_dir   = Path(output_dir)
        self.tracking_uri = tracking_uri
        self.seed         = seed
        self.limit        = limit
        self.__classifier: Classifier | None = None
        self.__model_key:  str | None = None

    def __get_classifier(self, model_cfg: dict) -> tuple[Classifier, bool]:
        """Load the model, or reuse the loaded weights and only change the settings."""
        cfg = dict(model_cfg)
        cfg["prompt_template"] = resolve_path(cfg["prompt_template"])
        if cfg.get("cache_dir"):
            cfg["cache_dir"] = str(resolve_path(cfg["cache_dir"]))
        weights = {k: cfg.pop(k) for k in ("name", "revision", "cache_dir")}
        key = json.dumps(weights, sort_keys=True)
        if key == self.__model_key and self.__classifier is not None:
            return self.__classifier.configure(**cfg), True
        self.__classifier = None  # free GPU memory before loading another model
        self.__classifier, self.__model_key = Classifier(**weights, **cfg), key
        return self.__classifier, False

    def run(self, cfg: dict) -> dict:
        """Run one experiment described by an already-loaded config."""
        # pylint: disable=import-outside-toplevel,too-many-locals
        from codecarbon import EmissionsTracker
        seed = self.seed if self.seed is not None else cfg["seed"]
        set_seed(seed)

        name    = run_name(cfg)
        out_dir = self.output_dir / name.replace("/", "__")
        out_dir.mkdir(parents=True, exist_ok=True)

        # data
        loader  = Loader()
        dataset = loader.load_dataset(cfg["dataset"]["name"])
        if self.limit:
            dataset = dataset[:self.limit]

        # candidates: retrieval (vector store) or the fixed catalog of all resources
        retrieval = cfg["retrieval"]
        vs, catalog = None, None
        if retrieval["mode"] == "rag":
            vs = VectorStore(embeddings=retrieval["embeddings"], docs=loader.docs,
                             revision=retrieval["embeddings_revision"])
            vs.load_or_build(resolve_path(retrieval["index_path"]),
                             chunk_size=retrieval["chunk_size"])
        else:
            catalog = build_catalog(loader.docs,
                                    load_resource_names(ROOT / "data" / "FHIR_resources.csv"))

        # model loading and inference are measured separately, on this process's GPUs only
        def tracker(output_file: str) -> EmissionsTracker:
            return EmissionsTracker(project_name=cfg["project_name"], output_dir=str(out_dir),
                                    output_file=output_file, gpu_ids=visible_gpu_ids(),
                                    allow_multiple_runs=True, log_level="error")

        load_tracker = tracker("emissions_load.csv")
        load_tracker.start()
        start = time.time()
        classifier, model_reused = self.__get_classifier(cfg["model"])
        load_s = time.time() - start
        load_emissions = load_tracker.stop() or 0.0

        inference_tracker = tracker("emissions.csv")
        inference_tracker.start()
        start = time.time()
        set_seed(seed)  # same RNG state at generation time whether or not the model was reused
        results = classifier.predict(
            vs      = vs,
            dataset = dataset,
            k       = retrieval["k"],
            every_represented_candidate = retrieval["every_represented_candidate"],
            catalog = catalog,
        )
        inference_s = time.time() - start
        inference_emissions = inference_tracker.stop() or 0.0
        for log in ("emissions_load.csv", "emissions.csv"):
            scrub_location(out_dir / log)
        runtime_s = load_s + inference_s
        emissions = load_emissions + inference_emissions

        # metrics
        metrics = evaluate(results)
        summary = {
            "run_name":     name,
            "config":       cfg,
            "seed":         seed,
            "n_samples":    len(dataset),
            "n_failed":     sum(bool(r.error) for r in results),
            "model_reused": model_reused,
            "runtime_s":    runtime_s,
            "co2_kg":       emissions,
            "load_s":       load_s,
            "inference_s":  inference_s,
            "inference_s_per_item": inference_s / len(dataset) if dataset else None,
            "energy": {
                "gpu_ids":   visible_gpu_ids(),
                "load":      energy_summary(getattr(load_tracker, "final_emissions_data", None),
                                            len(dataset)),
                "inference": energy_summary(getattr(inference_tracker, "final_emissions_data", None),
                                            len(dataset)),
            },
            "tokens": {
                "prompt_mean": float(np.mean([r.prompt_tokens for r in results])),
                "output_mean": float(np.mean([r.output_tokens for r in results])),
                "generation_s_median": float(np.median([r.seconds for r in results])),
            },
            "metrics":      metrics,
            "provenance": {
                "model_revision": classifier.model_revision,
                "dataset_md5":    _md5(ROOT / "data" / cfg["dataset"]["name"]),
                "prompt_md5":     _md5(resolve_path(cfg["model"]["prompt_template"])),
                "docs_md5":       hashlib.md5(json.dumps(loader.docs).encode(), usedforsecurity=False).hexdigest(),
                "environment":    environment_info(),
            },
        }
        _write_outputs(out_dir, summary, dataset, results)
        if self.tracking_uri:
            try:
                self.__log_mlflow(cfg, summary, results, dataset, out_dir)
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                # files in out_dir are the primary record; tracking must not abort a grid
                summary["mlflow_error"] = str(e)
                warnings.warn(f"MLflow logging failed for {name}: {e}")
                (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
        return summary

    def __log_mlflow(self, cfg: dict, summary: dict, results: list[PredictionResult],
                     dataset: list[dict], out_dir: Path) -> None:
        # pylint: disable=import-outside-toplevel
        import mlflow

        from src.evaluation.logging import (
            log_configuration,
            log_predictions_as_artifact,
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(cfg["project_name"])
        with mlflow.start_run(run_name=summary["run_name"]):
            log_configuration(cfg)
            mlflow.log_params({"seed": summary["seed"], "limit": self.limit,
                               "model_revision": summary["provenance"]["model_revision"]})
            mlflow.log_metric("co2_kg", summary["co2_kg"])
            mlflow.log_metrics(summary["metrics"])
            inference = summary["energy"]["inference"]
            operational = {
                "load_s":                summary["load_s"],
                "inference_s":           summary["inference_s"],
                "inference_s_per_item":  summary["inference_s_per_item"],
                "generation_s_median":   summary["tokens"]["generation_s_median"],
                "prompt_tokens_mean":    summary["tokens"]["prompt_mean"],
                "output_tokens_mean":    summary["tokens"]["output_mean"],
                "inference_energy_kwh":  inference.get("energy_kwh"),
                "inference_gpu_wh_per_item": inference.get("gpu_wh_per_item"),
            }
            mlflow.log_metrics({k: v for k, v in operational.items() if v is not None})
            mlflow.set_tags({"run_dir": str(out_dir), "model_reused": summary["model_reused"]})
            log_predictions_as_artifact(results, dataset)
            for name in ("metrics.json", "predictions.jsonl", "emissions.csv", "emissions_load.csv"):
                if (out_dir / name).exists():
                    mlflow.log_artifact(str(out_dir / name))
