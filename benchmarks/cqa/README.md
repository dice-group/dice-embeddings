# Complex query answering benchmarks

Reproducible evaluation of complex query answering (CQA) on two suites:

| Suite | Datasets | Query types | Answer filters |
|---|---|---|---|
| `ultraquery` | The 23 released UltraQuery datasets: 3 transductive, 9 inductive with new entities, 11 WikiTopics with new entities and relations | 14 (9 positive, 5 negated) | released |
| `plus_h` | **+H**: FB15k237+H, NELL995+H and ICEWS18+H from [Gregucci et al.](https://arxiv.org/abs/2410.12537), built so that complex queries cannot be reduced to simpler link prediction | 16 (adds `4p`, `4i`) | corrected, with released controls |

Each suite evaluates published methods (UltraQuery, GNN-QE, QTO, CQD, CQD-Hybrid,
ConE, CLMPT) through native DICE ports and the ULTRA and TRIX foundation models
with learned score adapters, each paired with an identity-adapter control. Every
prediction is scored under two tie policies: sort order, and the exact
expectation under random ordering of tied scores.

A study freezes its inputs, query batches and source code, checks validation
predictions against the pinned upstream implementations, and only then runs
the test split. [PROTOCOL.md](PROTOCOL.md) states the evaluation protocol and
every known deviation from the publications.

## Quick start

Everything runs through one command line from the repository root:

```bash
python -m benchmarks.cqa {plus_h,ultraquery} COMMAND [options]
```

Commands run in the current environment, which needs DICE and PyTorch. With
`--image IMAGE` they run in the pinned Docker runtime instead; the host then
needs only Python 3, Docker and, for GPUs, the NVIDIA Container Toolkit.
Containers have no network access, read inputs from `--input-root` (default:
the repository) read-only, and write only below `--output`; `--gpu cdi`
selects CDI device access and `--gpu none` runs on CPU.

Exploratory evaluation of any subset needs no preparation:

```bash
python -m benchmarks.cqa plus_h evaluate --output results/cqd-negation \
  --methods cqd cqd-hybrid --datasets FB15k237+H --query-types negation
```

The full verified workflow for one suite:

```bash
export IMAGE=dicee/cqa:local STUDY="$PWD/results/plus-h"
python -m benchmarks.cqa plus_h build --image "$IMAGE"            # one image serves both suites
python -m benchmarks.cqa plus_h prepare --image "$IMAGE" --output "$STUDY"
# Export one validation oracle per published-method entry (see Verification).
python -m benchmarks.cqa plus_h verify --image "$IMAGE" --output "$STUDY" --references "$PWD/references"
python -m benchmarks.cqa plus_h run --image "$IMAGE" --output "$STUDY"
python -m benchmarks.cqa plus_h report --image "$IMAGE" --results "$STUDY/test" --output "$STUDY/reports"
```

| Command | Purpose |
|---|---|
| `setup` | Download the selected datasets and released checkpoints and install the shipped adapters; `--check` only reports missing files and checksum mismatches |
| `evaluate` | Direct evaluation of recipe subsets, without a frozen study |
| `prepare` | Resolve recipes and freeze inputs, query batches and source in `STUDY/bundle` |
| `verify` | Check every entry against its oracle in a fresh process and pin the evidence in `STUDY/verified-bundle` |
| `run` | Run the test split of a verified study into `STUDY/test`, or a validation pilot into `STUDY/pilot` with `--phase pilot` |
| `report` | Tables, paired adapter/filter/graph effects and comparisons with published scores, from completed results |
| `difficulty-report` | Answer-level difficulty of completed +H results, on CPU |
| `train` | Train the UltraQuery comparison baselines with pinned author code |
| `build` | Build the runtime image, or with `--training BASE_IMAGE` the CPU author-training image |

`setup` runs on the host with only the standard library. `evaluate`, `prepare`
and `run` call it first; `--no-setup` instead requires existing inputs.
Downloads are streamed with bounded buffers and disk-space checks, released
checkpoints are checked by SHA-256, and existing files are never replaced. If
the +H authors' model host rejects automated downloads, pass
`--models-archive iscqa-compl-models.zip`. `--dry-run` prints resolved recipes,
input sources or the Docker command without reading inputs or writing files.

## Recipes and selection

A recipe fixes method, checkpoint, dataset, query types, query batches and
execution options. Each suite directory holds its public recipes:

| File | Contents |
|---|---|
| `plus_h/baselines.json` | The seven published methods on the three +H datasets (21 entries) |
| `plus_h/kgfm_adapters.json` | ULTRA and TRIX with 14-type adapters (6 entries, each with an identity control) |
| `plus_h/published_results.json` | Scores reported by the +H authors, for `report` |
| `ultraquery/baselines.json` | Native UltraQuery on all 23 datasets |
| `ultraquery/kgfm_adapters.json` | ULTRA and TRIX with `2i`/`3i`-trained adapters on all 23 datasets |
| `ultraquery/kgfm_14types.json` | The same backbones with the 14-type +H adapters |
| `ultraquery/comparisons.json` | ULTRA link-prediction weights, an incoming-relation heuristic and QTO |
| `ultraquery/trained_baselines.json` | QTO on FB15k and inductive GNN-QE, after `train` |

Defaults are `baselines.json` and `kgfm_adapters.json`; `--manifests FILE ...`
replaces them. Adapter weights shared by both suites are in
[`benchmarks/adapters`](../adapters). Recipe files list shared `defaults` once
per method, and each entry only what differs; `{dataset}` in an ID is filled in.
[`tests/recipe_fingerprints.json`](tests/recipe_fingerprints.json) pins every
resolved entry, so a recipe change is always visible in review.

Selectors combine freely and apply to every command that reads recipes:

- `--methods`, `--datasets`, `--entries`: omit them or pass `all` for every recipe.
- `--query-types`: individual types and the groups `epfo`, `negation` and `all`,
  for example `--query-types epfo 2in`. Omit it to keep each recipe's coverage.
- `--split valid` and `--max-queries-per-shape N` give reproducible samples.

The +H CQD and CQD-Hybrid recipes cover all 16 types, using signed-atom (CQD-A)
negation with +H scoring; `--atomic-negation` enables it for custom recipes.
Query types and execution options are chosen when a study is prepared and
cannot change on `run`; `run` accepts method, dataset and entry selectors
within the frozen study. Repeating a command resumes completed work.

## Verification

`verify` compares each published method's validation predictions with an
oracle exported from its pinned upstream checkout (`REFERENCES` in
[`dicee/query_answering/catalog.py`](../../dicee/query_answering/catalog.py)).
Export one oracle per entry in an environment with the authors' dependencies,
matching Python, PyTorch, CUDA, GPU, driver, precision and threads:

```bash
python benchmarks/cqa/verification/export_reference.py \
  --bundle "$STUDY/bundle" --input-root "$PWD" --entry cqd-FB15k237+H \
  --upstream "$UPSTREAM_CQD" --device cuda --output references/cqd-FB15k237+H.pt
```

The exporter imports no DICE code. Acceptance needs scores within tolerance and
identical candidate order and ties for every query. ULTRA/TRIX entries need no
oracle: an independent implementation checks calibration, pruning, composition
and identity controls on captured backbone logits; the backbones' own parity is
tested separately. Evidence binds inputs, query batches, source, image and
hardware; a change to any of them needs a new study. For the bounded profile,
`--comparison-references` additionally compares CQD with dense oracles.

The default `bounded` profile runs CQD and CQD-Hybrid in 8 GB of GPU memory
(`row_batch_size` and `final_batch_size` 32, no reference batching) with
unchanged beams; `--profile reference` keeps dense upstream batching. Oracles
must come from a study prepared with the same profile.

## Ablations

Each ablation is a separate study with its own verification.

- **Inference graph** (+H): `prepare --inference-graphs train train+valid`.
  Graph-based methods use train+valid facts by default; ConE, CLMPT and CQD do
  not read the graph and run once. IDs gain `-graph-train` or `-graph-train-valid`.
- **Observed facts** (ULTRA/TRIX): `prepare --observed-facts none atomic` or
  `all`, with unchanged adapter weights. IDs gain `-facts-MODE`.
- **Answer filters** (+H): every prediction is also scored with the released
  filters (`-released-filters`); `--answer-filter released` prepares a study
  with released filters only.
- **Adapters**: every ULTRA/TRIX entry has a `without-adapter` control scored
  from the same backbone batches.

`report` writes paired effects with query-bootstrap confidence intervals
conditional on the frozen weights: `adapter-effects`, `filter-effects` and
`graph-effects`.

## Hardware and parallelism

`--hardware-profile h100` (on `prepare` or `evaluate`) is a starting point for
H100 80 GB and H100 NVL GPUs: ULTRA/TRIX use atomic batches of 16, a 4 GiB
budget for calibrated scores, 2 GiB of raw scores on the GPU, and 256/512 MiB
relation/projection caches. `--kgfm-batch-size N` overrides the batch. Other
methods keep their settings. These settings can change floating-point results,
so they are frozen with the study and verified on the target GPU.

`run` and `evaluate` accept `--gpus 0 1 2 3`: each entry runs in a fresh worker
that sees one GPU, longest methods first, with results identical to a
sequential run. `--workers-per-gpu 2` overlaps one worker's host work with
another's GPU work; with the H100 profile a KGFM worker needs about 6 GiB. Keep
one worker per GPU on GPUs shared with other jobs.

## UltraQuery comparison baselines

`trained_baselines.json` needs weights trained locally with pinned author code
(QTO and InductiveQE checkouts below `Experiments/query-baselines/upstream` in
the input root). Training defaults to CPU and the authors' hyperparameters,
selects checkpoints on validation MRR only, and records code revisions, inputs,
environment and checkpoint hashes in `training.json`:

```bash
python -m benchmarks.cqa ultraquery build --image "$IMAGE-training" --training "$IMAGE"
python -m benchmarks.cqa ultraquery train --image "$IMAGE-training" \
  --output "$PWD/checkpoints/ultraquery_benchmark-trained" --methods qto inductive-gnnqe
```

Each job also writes a `manifest.json` for its checkpoint; at the default
output path above, `trained_baselines.json` finds the checkpoints directly.

Inductive GNN-QE requires the complete corrected v2.0 InductiveQE archives
([Zenodo 7306046](https://zenodo.org/records/7306046)); every file is checked
against [`inductive-v2-files.json`](../ultraquery/inductive-v2-files.json).
`--smoke` trains a tiny model for one update, and final runs reject it. The
author runtime applies three mechanical TorchDrug build fixes
([`training/torchdrug_compat.py`](training/torchdrug_compat.py)) without
changing kernel arithmetic.

## Paper tables and figures

```bash
python -m benchmarks.cqa.paper [REPORT.json ...] -o tables.tex [--figures DIR]
```

renders nine LaTeX tables from saved reports, using only the standard library;
without reports it prints the planned layout with every score missing ("-").
`--figures` also exports the paper figures (needs matplotlib). Scores are never
imputed and confidence intervals are copied, not estimated.

## Tests

```bash
python -m pytest benchmarks
```

runs the harness, protocol and paper tests on CPU without datasets. Tests that
need upstream checkouts or released data are skipped unless configured:
`DICEE_CQD_REFERENCE`, `DICEE_ULTRAQUERY_REFERENCE_ROOT`,
`DICEE_INDUCTIVE_REFERENCE_ROOT` and `DICEE_ULTRAQUERY_DATA_ROOT` point to the
pinned checkouts and the extracted UltraQuery data.
