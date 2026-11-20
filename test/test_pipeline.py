"""
Unit tests for config handling in src/pipeline.py (no model loading).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.pipeline import (
    ROOT,
    energy_summary,
    load_config,
    resolve_path,
    run_name,
    scrub_location,
    visible_gpu_ids,
)


def _write(tmp_path: Path, cfg: dict) -> Path:
    fp = tmp_path / "cfg.yaml"
    fp.write_text(yaml.safe_dump(cfg))
    return fp


MINIMAL = {"dataset": {"name": "sphn/sphn.csv"}, "model": {"name": "some/model"}}


def test_defaults_are_filled_in(tmp_path):
    """Missing keys get the documented defaults."""
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg["seed"] == 42
    assert cfg["model"]["decoding"] == "greedy"
    assert cfg["retrieval"]["k"] == 6
    assert run_name(cfg) == "some/model-sphn/sphn.csv-3"


def test_relative_paths_resolve_against_repo_root(tmp_path, monkeypatch):
    """Results must not depend on the working directory."""
    monkeypatch.chdir(tmp_path)
    assert resolve_path("references/configs/prompts/prompt.txt") == (
        ROOT / "references/configs/prompts/prompt.txt")
    load_config(_write(tmp_path, MINIMAL))   # default prompt path found from another CWD


def test_repository_default_config_is_valid():
    """The shipped config loads and validates."""
    cfg = load_config(ROOT / "references" / "configs" / "config.yaml")
    assert cfg["model"]["revision"]


@pytest.mark.parametrize("override", [
    {"retrieval": {"k": 27}},
    {"model": {"name": "m", "n_output": 7}},
    {"model": {"name": "m", "decoding": "beam"}},
    {"model": {"name": "m", "prompt_template": "does/not/exist.txt"}},
    {"dataset": {"name": ""}},
])
def test_invalid_configs_raise(tmp_path, override):
    """Configs that cannot produce meaningful results are rejected up front."""
    cfg = {**MINIMAL, **override}
    if "model" in override:
        cfg["model"] = {**MINIMAL["model"], **override["model"]}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, cfg))


def test_llm_only_mode_skips_retrieval_limits(tmp_path):
    """Without retrieval, n_output is not bounded by k."""
    cfg = load_config(_write(tmp_path, {**MINIMAL, "retrieval": {"mode": "none", "k": 6},
                                        "model": {"name": "m", "n_output": 10}}))
    assert cfg["retrieval"]["mode"] == "none"


@pytest.mark.parametrize("override", [
    {"retrieval": {"mode": "bm25"}},
    {"model": {"name": "m", "prompt_format": "xml"}},
])
def test_invalid_new_options_raise(tmp_path, override):
    """Unknown retrieval modes and prompt formats are rejected."""
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, {**MINIMAL, **override}))


def test_new_options_default_to_published_behaviour(tmp_path):
    """Published configs run unchanged: RAG, legacy prompt, constrained output."""
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert (cfg["retrieval"]["mode"], cfg["model"]["prompt_format"], cfg["model"]["constrained"]) == (
        "rag", "legacy", True)


@pytest.mark.parametrize("value,expected", [("2,3", [2, 3]), ("", None), ("GPU-abc", None)])
def test_visible_gpu_ids(monkeypatch, value, expected):
    """CodeCarbon is restricted to the GPUs of the process when they are numeric."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert visible_gpu_ids() == expected


def test_energy_summary_per_item():
    """Per-item energy and carbon intensity are derived from the totals."""
    data = SimpleNamespace(duration=10.0, energy_consumed=0.002, gpu_energy=0.0015, cpu_energy=0.0004,
                           ram_energy=0.0001, emissions=0.0006, country_name="Italy",
                           region="emilia-romagna", gpu_model="H100", gpu_count=2)
    e = energy_summary(data, 4)
    assert e["wh_per_item"] == pytest.approx(0.5) and e["gpu_wh_per_item"] == pytest.approx(0.375)
    assert e["carbon_intensity_kg_per_kwh"] == pytest.approx(0.3)
    assert energy_summary(None, 4) == {}


def test_scrub_location_blanks_coordinates_only(tmp_path):
    """Latitude/longitude are blanked; other columns, including region, are untouched."""
    fp = tmp_path / "emissions.csv"
    fp.write_text("timestamp,region,longitude,latitude,emissions\n"
                  "t1,emilia-romagna,11.3387,44.4938,0.01\n", encoding="utf-8")
    assert scrub_location(fp) == 1
    assert fp.read_text(encoding="utf-8").splitlines() == [
        "timestamp,region,longitude,latitude,emissions", "t1,emilia-romagna,,,0.01"]
    assert scrub_location(fp) == 0 and scrub_location(tmp_path / "missing.csv") == 0
