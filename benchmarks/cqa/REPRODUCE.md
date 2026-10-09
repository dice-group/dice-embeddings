# Reproducing the final CQA results

The paper and thesis report 24 frozen studies, 13 on UQ-23 and 11 on +H, listed with their recipes,
adapters and measured cost in [`reproduction/registry.py`](reproduction/registry.py). One command reproduces
one study or all of them. Stages skip work whose output exists, so repeating a command resumes it, and
`--dry-run` prints every command that would run.

## Requirements

- This repository installed with `pip install -e '.[dev]'` (Python 3.11+), and NVIDIA GPUs for fits, oracles,
  verification and test runs (a KGFM worker uses up to 12 GiB GPU and 24 GiB host memory, QTO up to 32 GiB host memory).
  With `--image IMG` (from `python -m benchmarks.cqa plus_h build --image IMG`) the benchmark steps run in the
  pinned Docker runtime; fits, oracle exports, analyses and rendering stay on the host.
- Datasets and checkpoints download on first use; if the +H authors' host refuses the models, run
  `python -m benchmarks.cqa plus_h setup --models-archive iscqa-compl-models.zip` once.
- Baseline oracles need `--upstream DIR`, one pinned checkout per repository (the dry run prints the
  `git clone` lines), in an environment with the authors' dependencies. The UQ-23 per-graph and QTO studies
  first train their GNN-QE and FB15k QTO weights ([details](README.md#ultraquery-comparison-baselines)).

## Tables and figures from saved reports (CPU, minutes)

```bash
python -m benchmarks.cqa render --reports ROOT -o tables.tex --figures figures [--thesis THESIS_DIR]
```

## One study

```bash
python -m benchmarks.cqa reproduce --list            # studies, entries, prerequisites, measured test hours
python -m benchmarks.cqa reproduce plus_h-kgfm-b64 --gpus 0 1 2 3 --workers-per-gpu 2 [--image IMG] [--dry-run]
```

A study runs `train`, `manifest`, `prepare`, `oracles`, `verify`, `run`, `report` and `difficulty`
(`--stages` selects). `reproduce kgfm-b64` selects both suites, and patterns such as `'plus_h-*'` work.
Unless `--fit` is given, studies use the shipped adapters of [`benchmarks/adapters`](../adapters).

## Everything at once

```bash
python -m benchmarks.cqa reproduce --all --gpus 0 1 2 3 --workers-per-gpu 2 --upstream UPSTREAM \
  [--fit] [--kgicl-datasets KG-ICL/datasets.zip] [--thesis THESIS_DIR]
```

`--all` adds `combine` (one report per suite), `analyses` and `render`. `--fit` first refits all 78 adapters
into `ROOT/adapters` and uses them; their training data is regenerated and checked against pinned
fingerprints, and refits of the reference ULTRA and TRIX adapters matched the shipped weights to within 2e-6.
Single steps share the interface:

```bash
python -m benchmarks.cqa fit brackets/ultra/global.json --output adapters    # fit --list shows all 78
python -m benchmarks.cqa analysis calibration-profile --backbone kgicl --output calibration-kgicl.json
```

Analyses: `calibration-profile`, `observed-links`, `adapter-weights`, `negation-probe`, `pretraining-overlap`
(`--pretraining kgicl --kgicl-datasets ZIP`) and `answer-classes` (FB15k/FB15k237 answers by cheapest
grounding, with each method's MRR from the rank traces below `--results ROOT`).

## Outputs

`ROOT` (`--output`, default `results/final`) holds one directory per study (`recipes.json`, `manifest.json`,
`bundle/`, `verified-bundle/`, `test/ENTRY/` results and rank traces, `reports/`, `difficulty/`),
`{ultraquery,plus_h}-all-reports/`, `analyses/`, `paper/tables.tex` with `paper/figures/`, and `adapters/` with `--fit`.

## Compute budget (approximate)

Test runs, as the summed wall time of all entries on 4 NVIDIA H100 NVL 94 GB GPUs, up to two jobs per GPU:

| Suite | Entry-hours | Studies (entry-hours) |
|---|---:|---|
| UQ-23 | 414 | kgfm-b64 140.6, ablations 44.9, facts-none 42.3, global 24.2, uqlp 17.8, target-fit 24.7, kgicl 21.1, kgicl-seeds 49.9, kgicl-facts-none 21.8, kgicl-global 12.0, baselines 5.1, pergraph 9.2, qto 0.7 |
| +H | 54 | kgfm-b64 18.1, ablations 5.7, facts-none 5.2, global 3.3, target-fit 3.5, kgicl 1.6, kgicl-seeds 3.7, kgicl-facts-none 1.9, kgicl-global 1.0, baselines 8.9, qto 1.3 |

Verification adds about one GPU-minute per KGFM integration check (748 checks) plus oracle export and parity
for the 90 baseline entries. `--fit` adds about 100 GPU-hours, mostly the 52 target-fit adapters (the backbone
scoring recorded in the adapters' `training` metadata alone is 47 h for those and 14 h for the 22 source
fits). Training the UltraQuery comparison weights is not included. Reports, analyses and tables take minutes
to an hour each on CPU. The complete rerun took about three days on the four GPUs.

## Review experiments (2026-10-08)

The registry's `REVIEW_STUDIES` and `REVIEW_FITS` hold the experiments added after the review of 2026-10-07:
UltraQuery with an opt-in observed-fact traversal (`baselines-traversal`), adapters refitted without observed facts
(`kgfm-b64-gamma0`), the training-free calibrations of CQD-Hybrid and QTO (`kgfm-b64-minmax`, `kgfm-b64-softmax-degree`),
a KG-ICL adapter with wider bounds (`kgfm-b64-kgicl-wide`), UltraQuery's weights as a frozen backbone
(`kgfm-b64-uqweights`), UltraQuery LP with ULTRA 4g (`pergraph-lp-4g`) and adapters fitted on target validation queries
(`kgfm-b64-target-valid`). They are selected by name (`reproduce kgfm-b64-minmax`) and are not part of `--all` until
their results are in the paper. Their adapters live in `benchmarks/adapters/review` (`fit review/ultra/gamma0.json`).
`Experiments/screens/latency.py` measures inference cost and beam-width sensitivity on validation queries, and
`Experiments/screens/flock_subset.py` evaluates Flock on a fixed sample of test queries.

## Equivalence with the frozen runs

`reproduce --all --compare-manifests DIR` compares each resolved manifest with the frozen `DIR/STUDY/manifest.json`;
`tests/final_studies.json` pins them for the tests. All 24 agree in every scientific field (adapters by SHA-256).
By design, bracket and KG-ICL adapters now come from `benchmarks/adapters` (same bytes), the frozen KG-ICL entries
keep the notes of the then provisional recipe, and QTO, stopped after `1p` in the two baseline studies, comes
from the `qto` studies.
