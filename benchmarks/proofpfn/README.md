# ProofPFN benchmark

Zero-shot link prediction on frozen graph sets, by the evaluation protocol of the ProofPFN research runs
([PROTOCOL.md](PROTOCOL.md)). The benchmark ranks every test fact of a graph in both directions with ProofPFN families
(three seeds of one arm) and the native reference models, and writes one row file per graph whose format the
registered analysis reads. The golden test G6 checks that its ranks equal the research harness's bit for bit on three
development graphs.

## Quick start

Everything runs from the repository root:

```bash
python -m benchmarks.proofpfn list --set dev121 --view main
python -m benchmarks.proofpfn check --set gate --family S:/runs/S/training/{seed}/export-24000.pt
python -m benchmarks.proofpfn run --set dev121 --view main --out rows \
  --family S:/runs/S/training/{seed}/export-24000.pt --references trix,ultra3g,flockoff
python -m benchmarks.proofpfn cells --set gate --out queries.csv
python -m benchmarks.proofpfn fv1-models --out models-fv1.json
```

| Command | Purpose |
|---|---|
| `list` | the graphs of a manifest view (`--view`, `--graphs`, `--codes`) |
| `check` | CPU only: the graphs' hash checks and counts, the systems' files and pins; no score |
| `run` | rank the graphs with the systems; one row file per graph |
| `cells` | the query cells of the registered analysis (`queries.csv` and its `.meta.json`) |
| `fv1-models` | the `--models` file of the registered `fv1_tables.py`, from `manifests/models.json` |

A family is `--family [KEY=]ARM:TEMPLATE`. `ARM` is one of S, R, M, N, HT, CU and UL, `TEMPLATE` the path of a clean
checkpoint (`proofpfn-checkpoint-v1`) or of a research export, with `{seed}` for the seeds 42, 43 and 44. Research
exports convert on load (`dicee.synthetic_kg.checkpoint.from_research_export`, EMA weights, after its inertness
checks). `KEY` defaults to the arm's registered key (`rec10axlepcunoa` for S, `pcur`, `pcum`, `pcun`, `pcuht`,
`rec10axlecunoa`, `rec10axleulnoa`), under which every file must match the research export pins of `models.json`. Any
other lower-case key runs a test family without pins. References: `trix`, `ultra3g` and `flockoff` (official Flock),
with their checkpoints below `--checkpoint-root` (default `checkpoints/`).

`run` options: `--device` (default `cuda`; deterministic CUDA needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the command
sets when it is unset), `--shard i/k`, `--parts`, `--save-minutes`, `--max-minutes`, `--data-root` (the folder that the
manifest paths are relative to, default the repository root), `--flock-passes-in-sequence`. Exit codes: 0 done, 2 wrong
input, 3 a graph ran out of CUDA memory (its record is in `.errors/` of the row folder), 76 stopped at `--max-minutes`,
143 stopped by SIGTERM (the part file is saved).

## Graph sets

| Manifest | Graphs | Hashes checked before a graph loads | Test rows | Source |
|---|---|---|---|---|
| `dev121` | 121 audited development graphs; views `main` (91, with weights), `contamination` (29), `apart` (UMLS) | content of train.txt and test.txt; file SHA-256 where the suite manifest records the folder | `first-1000` rule | `graph-audit-v1/graphs.json`, `stats/inventory.json` |
| `t1` | 54 graphs of the standard zero-shot suite; views `trans`, `ind_e`, `ind_er` | file SHA-256 of the three splits, counts | all | `p0-suite-v1/manifest.json` |
| `gate` | UMLS, `ultra-suite-v1/NELLInductive_v1`, `inductive/grail-WN18RR_v1` | file SHA-256 and content | `first-1000` rule (all rows here) | the audit and the golden records |

The audit excludes grail-WN18RR_v1 as a copy of `ultra-suite-v1/WN18RRInductive_v1`, so it is in `gate` and not in
`dev121`. Each manifest lists its inputs and their SHA-256 under `sources`; `manifests.build_dev121`, `build_t1` and
`build_gate` derive the files from them (`tests/test_manifests.py` rebuilds them when the inputs are available).

Blind sets. A manifest with `"blind": true` opens only with `--release NAME` naming it, and only such a manifest may name
a folder below `KGs/fresh-v1` or `KGs/lockbox3`. This branch holds no blind manifest; the fresh-v1 and block-3 manifests
are added after the frozen read.

## Systems and row files

| Key | System | Models |
|---|---|---|
| a family key | ProofPFN family of one arm | three seeds, `log_softmax` per model, family view = mean over the seeds |
| `trix` | TRIX, `trix/entity_prediction.pth` | one |
| `ultra3g` | ULTRA-3g, `ultra_3g.pth` | one |
| `flockoff` | Flock, `flock/flock_entity.pth`, the graph's official walk count and passes, seed 42 | one; the passes of a query form one microbatch |

A row file `<out>/<collection>/<keys joined by '-'>/<graph>.json` holds per test fact and direction the ranks of every key
(PROTOCOL.md): `F`, `F:gt2`, `F:raw`, `F:raw:gt2` per system and `F.seedS:raw`, `F.seedS`, `F.seedS:gt2` per seed of a
family. It also holds `metrics`, `seconds`, the research record `allgraph` (graph, file hashes, sample, systems, model
hashes, Flock's setting), `metrics_views` (both directions, tail queries, head queries) and `provenance` (code hashes,
versions, device, checkpoint records). The model hash of a family seed is its research export's SHA-256 where there is
one, so clean and research rows compare with the same pins.

Sharding (T1): `--shard i/k` ranks block i of k contiguous blocks of the relation-ordered query groups into
`.parts/<graph>/shard-i-of-k.json`; a part saves every `--save-minutes`, resumes after its last stored group, and the run
that completes the last part writes the row file (run a finished shard again to merge). Ranks do not depend on the
sharding.

## The registered analysis

`registered/` holds the six registered analysis files of the fresh-v1 test, byte-identical to their freeze anchors
(`registered/README.md`): `fv1_analysis.py`, `test_fv1_analysis.py`, `fv1_tables.py`, `test_fv1_tables.py`,
`fv1_drops.py`, `fv1_devrows.py`. They run as scripts, unchanged:

```bash
python -B benchmarks/proofpfn/registered/fv1_tables.py --rows ROWS --manifest MANIFEST --block3 BLOCK3 \
  --models models-fv1.json --out-dir TABLES
python -B benchmarks/proofpfn/registered/fv1_analysis.py analyse --mrr TABLES/mrr.csv --queries queries.csv \
  --ranks TABLES/ranks.csv --gates gates.json --dropped dropped.json --manifest MANIFEST --cost TABLES/cost.csv --out result.json
```

Ruff does not check them (`registered/ruff.toml`) and pytest does not collect their test scripts (`conftest.py`);
`tests/test_registered.py` checks their SHA-256, runs their fast checks as subprocesses and, with `PROOFPFN_SIMULATE=1`,
their simulations (about 9 minutes).

## Tests and goldens

`python -m pytest benchmarks/proofpfn` runs the tests: about two minutes on the CPU, plus about seven for G6 where its
GPU and goldens are present. Goldens (`tests/golden/README.md`):

| Golden | Test | Needs |
|---|---|---|
| G6, ranks | every key on the gate graphs equals the research rows: families S, CU, UL, HT of the G4 exports and the three references | CUDA on the golden GPU (RTX 4070 Ti SUPER), `CUBLAS_WORKSPACE_CONFIG`, `PROOFPFN_GOLDEN` (the exports); about 10 minutes |
| G7, cells | `queries.csv` of the gate graphs byte-identical | the gate graphs |
| G8, chain | simulated rows -> registered `fv1_tables.py` -> registered `fv1_analysis.py`, every output byte-identical | none |

`PROOFPFN_GOLDEN` names the proofpfn-golden folder with the large goldens; tests that need them skip without it. The
goldens hold for Python 3.12.2, torch 2.5.1, numpy 1.26.4 and scipy 1.13.0.

## Not here

The N validity gate (held-out worlds of the priors) and the external adapters (KGPFN, MERIT, AnyBURL) follow in later
phases. `models.json` pins the MERIT checkpoints and lists the KGPFN files under `external` for them.
