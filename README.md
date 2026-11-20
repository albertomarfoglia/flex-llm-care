# FLEX-LLM-CARE

Code, data and configurations for the article *An LLMOps Framework for Clinical Data
Standardization: Design and Evaluation of an LLM-Based FHIR Mapping Assistant* (citation to be added on
publication).

The evaluated component is **FRMT-M**. Given the free-text description of a clinical variable, it
returns three ranked FHIR R4 resource types out of 65 candidates. It works in four steps:

1. retrieve six candidate resources from the FHIR R4 specification (MedEmbed embeddings, FAISS);
2. show them to an instruction-tuned LLM;
3. constrain the answer with a grammar (Outlines);
4. decode greedily.

The repository also contains:
- the non-LLM baselines, the ablations and the statistical analysis;
- the LLMOps tooling (MLflow tracking, promotion gate, CodeCarbon energy logging);
- unevaluated prototypes of the two other modules, CDE-M and CVV-ADM.

No patient data are used: the evaluation sets are variable-level mapping tables.

## Requirements

| | |
|---|---|
| OS, Python | Linux, Python 3.11 |
| GPUs | 7B models: 1 × ≥ 24 GB. 40–72B models: 2 × 80 GB (bf16, no quantization). The paper used 4 × H100 80 GB. |
| Disk | ≈ 550 GB for the model weights |
| Hugging Face | A token (`HF_KEY`) only for the gated `meta-llama/Llama-3.3-70B-Instruct`; all other models are public |
| Optional | Java 17+ and Apache Jena (CVV-ADM SHACL engine); BioPortal API key (CDE-M online linking) |

Secrets go in a git-ignored `.env` file (`HF_KEY=...`, `BIOPORTAL_API_KEY=...`). `HF_HOME` selects
the model cache.

## Quick check (≈ 5 min)

```bash
make virtualenv && make install_lock && source venv/bin/activate   # exact package versions (torch cu126)
make start_testing      # 156 unit and data tests, no GPU
make smoke              # 5 items end to end with a 7B model on one GPU
```

## Reproducing the paper

Run from the repository root. All outputs are git-ignored.

| Step | Command | Output | Cost |
|---|---|---|---|
| 1. Main grid: 6 LLMs × 3 datasets, plus repeated runs for the determinism check | `make reproduce` (progress: `make status`) | `reproduction/outputs/run1`, `run2` | 4 × 80 GB GPUs, ≈ 3 h |
| 2. Ablations, native prompt format, energy | `make ablations` | `results/revision/gpu` (62 runs) | 4 × 80 GB GPUs, ≈ 6 h |
| 3. Baselines, promotion gate, statistics, tables | `make paper` | `results/revision/{baselines,promotion,analysis,analysis_pooled,tables}`; LaTeX tables in `docs/FGCS_paper/tables/` | 1 GPU for the retrieval baselines, then CPU, ≈ 5 min |

Model and embedding revisions are pinned in every configuration file. Greedy decoding with seeded
generators and deterministic kernels gives byte-identical predictions on the same hardware and
library versions; other GPUs or drivers may change a few low-margin answers.

### Where each result comes from

| Paper element | Produced by | File |
|---|---|---|
| Evaluation-data, accuracy, ablation, energy and secondary-metric tables | `scripts/make_tables.py` | `docs/FGCS_paper/tables/*.tex` (created locally); ablation and energy CSVs in `results/revision/tables/` |
| Paired tests, bootstrap CIs, error analysis, Hit@k (Sections 5.4–5.8) | `scripts/analyze_results.py --family main` | `results/revision/analysis/` |
| Sensitivity analysis with the pooled Holm family (Section 5.4) | `scripts/analyze_results.py --family pooled` | `results/revision/analysis_pooled/` |
| Table-level (cluster) bootstrap for MIMIC-IV (Section 5.4) | `scripts/cluster_sensitivity.py` | `results/revision/analysis/cluster_mimic.csv` |
| MOTU item counts (157 → 133 → 132, Section 5.2) | `scripts/check_motu_items.py` | printed |
| Promotion loop (Section 5.10) | `scripts/promote.py`, criteria in `references/configs/promotion_gate.yaml` | `results/revision/promotion/gate_report_001.json`, MLflow registry in `results/revision/mlflow/` |
| Prompt figure; exact text each tokenizer receives | `references/configs/prompts/`; `scripts/render_prompts.py --configs <yaml> --output-dir <dir>` | |
| Supplementary material: Hit@k curves, per-class results and confusion matrices | `scripts/analyze_results.py` | `results/revision/analysis/{hit_at_k,per_class__*,confusion__*}.csv` |
| Expert-effort estimate (Section 5.11) | retrospective estimate; no code or measurements | |

## Running single experiments

```bash
python scripts/run_experiment.py --config reproduction/configs/qwen2.5-72b_{motu,mimic,sphn}.yaml \
       --output-dir outputs [--limit 5] [--tracking-uri sqlite:///mlflow.db]
```

Configurations sharing a model load it once. Each run writes three files to
`<output-dir>/<run_id>/`:
- `predictions.jsonl`: per item, the input, candidates, rendered prompt, raw output, prediction, tokens and time;
- `metrics.json`: metrics, energy and provenance (model commits, data/prompt/KB checksums, package versions, hardware);
- CodeCarbon logs.

Main configuration keys (`references/configs/config.yaml`; generated variants in
`reproduction/configs*/`):

| Key | Paper setting | Other values |
|---|---|---|
| `dataset.name` | `motu_enriched.csv`, `mimic_enriched.csv`, `sphn/sphn.csv` | `motu/motu.csv`, `mimic/mimic.csv` (original descriptions) |
| `retrieval.mode`, `retrieval.k` | `rag`, `6` | `none` (all 65 resources listed), any `k` ≤ 26 |
| `retrieval.embeddings` | `abhinand/MedEmbed-large-v0.1` | any sentence-transformers model |
| `model.name`, `model.revision` | see `reproduction/configs/` | |
| `model.n_output` | `3` | |
| `model.constrained` | `true` | `false` (free generation, parsed afterwards) |
| `model.prompt_format` | `legacy` | `chat` (each model's own chat template) |
| `model.decoding` | `greedy` | `model_default` (seeded sampling) |

## Data (`data/`)

| File | Items | Source and notes |
|---|---|---|
| `motu_enriched.csv` (`motu/motu.csv`: original descriptions) | 132 | MOTU data descriptor (Arcobelli et al., *Sci Data* 2024, CC BY 4.0); its open-access full text is in `data/motu/` for `check_motu_items.py` |
| `mimic_enriched.csv` (`mimic/mimic.csv`) | 142 | Field-to-resource pairs from the MIMIC-IV on FHIR mappings page (https://kind-lab.github.io/mimic-fhir/mimic-fhir-mappings.html); no PhysioNet access needed |
| `sphn/sphn.csv` | 64 | SPHN-to-FHIR concept map (Klopfenstein et al., MIE 2023) |
| `Docs/` | 65 resources | Snapshot of the HL7 FHIR R4 specification pages (knowledge base); `src/data/scraper.py` re-scrapes it |
| `FHIR_resources.csv` | | The 65 candidate resources (Clinical, Base, Specialized; maturity > 0) |

Item counts are those left after removing rows whose ground truth lies outside the candidate set.
The enriched descriptions were produced before this study with an instruction-tuned LLM whose
identity and prompt were not recorded; the original descriptions are included for the ablation.

## Repository layout

```
src/            pipeline.py (single runner), data/, features/ (retrieval), models/ (LLM + grammar),
                evaluation/ (metrics, statistics, promotion gate), baselines/, extraction/ (CDE-M),
                validation/ (CVV-ADM)
scripts/        experiments, baselines, analysis, tables, promotion, prototype demo
reproduction/   configuration generators, generated configs (configs/: main grid;
                configs_revision/: ablations, energy and promotion runs), GPU runners,
                requirements-lock.txt
references/     default config, prompts, promotion criteria, CVV-ADM rules and SHACL shapes
data/           evaluation sets and FHIR knowledge base
test/           pytest suite (run by GitHub Actions)
notebooks/      run.ipynb (interactive entry point to the same runner)
prometheus.py   exports MLflow metrics to Prometheus (prototype)
```

## CDE-M and CVV-ADM prototypes (not evaluated)

These modules make the framework design concrete. They were not used for any reported result and
have not been evaluated.

- **CDE-M** (`src/extraction/cdem.py`): BioBERT NER, then concept search in BioPortal or a local
  terminology CSV, then selection of the top-ranked concept or optionally by an LLM.
- **CVV-ADM** (`src/validation/`): three layers, none of which corrects anything automatically:
  - declarative rules with a documented source for each rule (`references/configs/cvvadm_rules.yaml`);
  - a Bio_ClinicalBERT outlier score;
  - SHACL validation of an RDF representation, with Apache Jena or pySHACL (`references/configs/cvvadm_shapes.ttl`).

  There is no OWL reasoning and no clinical-guideline content.

Demo on synthetic data:

```bash
python scripts/demo_prototypes.py --output-dir outputs/prototypes --device cuda:0
# optional: JENA_HOME=<apache-jena dir> for the Jena engine; BIOPORTAL_API_KEY for online linking
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 Invalid user token` for public models | `export HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (stale token in `$HF_HOME/token`) |
| `GatedRepoError` for Llama-3.3 | Request access on the model page; set `HF_KEY` |
| 70B model loading takes ≈ 35 min | Disk-bound; configurations of the same model in one call load it once |
| `make start_tracking` | Optional MLflow UI on port 8080 (SQLite backend) |

## Licence

Code: MIT (`LICENSE`). Files in `data/` keep the terms of their sources:
- FHIR specification: HL7, CC0;
- MOTU: CC BY 4.0;
- MIMIC-IV on FHIR mappings and SPHN map: see the sources above.
