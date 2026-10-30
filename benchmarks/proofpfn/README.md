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
python -m benchmarks.proofpfn adapter anyburl --set dev121 --view main --out rows --code /path/to/AnyBURL
```

| Command | Purpose |
|---|---|
| `list` | the graphs of a manifest view (`--view`, `--graphs`, `--codes`) |
| `check` | CPU only: the graphs' hash checks and counts, the systems' files and pins; no score |
| `run` | rank the graphs with the systems; one row file per graph |
| `cells` | the query cells of the registered analysis (`queries.csv` and its `.meta.json`) |
| `fv1-models` | the `--models` file of the registered `fv1_tables.py`, from `manifests/models.json` |
| `adapter` | an external baseline (`kgpfn`, `merit`, `anyburl`) on the graphs, one row file per graph ([External baselines](#external-baselines)) |

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

## External baselines

`adapters/` runs three external baselines by the logic of their research runners and writes row files that the registered
`fv1_tables.py` reads. Their code, checkpoints and runtimes stay outside the repository: each adapter takes its code
folder from `--code` or an environment variable and checks it, and its files, against the pins of `manifests/models.json`
(`external`; for MERIT also `references`). Rows, ranks and the record follow [PROTOCOL.md](PROTOCOL.md), section External
baselines. `--check` checks the setup, the pins and the graphs without a run; a graph whose external step fails gets
an error record in `.errors/` (exit code 3); work files stay in `<out>/.work` (or `--work`), a dot folder that the tables
skip. Tests that need the external code skip without it.

| Adapter | Row file | Keys | Research runner |
|---|---|---|---|
| `kgpfn` | `<out>/<collection>/kgpfn/<graph>.json` | `kgpfn`, `kgpfn:raw`; no model hash entry | `kgpfn-ref-v1-runtime/kgpfn_ag.py`, `kgpfn_eval.py` |
| `merit` | `<out>/<collection>/merit1024-merit2025-merit2026-merit4000-merit8000/<graph>.json` | `merit<S>`, `merit<S>:gt2`, `merit<S>:raw`, `merit<S>:raw:gt2`; model hash `merit<S>` | `allgraph-v1/ag_merit.py`, `merit-ref-v1-runtime/merit_ref.py` |
| `anyburl` | `<out>/<collection>/anyburl/<graph>.json` | `anyburl`, `anyburl:raw`; no model hash entry | `anyburl-ref-v1-runtime/anyburl_run.py` |

None of the three local copies holds a licence file. The repository contains none of their code and none of their
checkpoints; check the terms of each upstream project before you copy or publish its files.

### KGPFN

- Code: a clone of github.com/HKUST-KnowComp/KGPFN at `4f1bc61dcc998cf0672deaca148899b505797af6` (`--code` or
  `$KGPFN_CODE`). Checkpoints: `kgpfn_icl_3g.pth` and `tabicl-regressor-v2-20260212.ckpt` (`--checkpoints` or
  `$KGPFN_CHECKPOINT_DIR`), as KGPFN's `script/download.py` fetches them.
- Runtime: KGPFN's own environment (its `requirements.txt`: torch 2.5.1+cu121, torch-geometric, tabicl, and a CUDA
  toolkit for the JIT build of its rspmm kernel). `--python` or `$KGPFN_PYTHON` is the command that starts its Python,
  split like shell words. The host runs `adapters/kgpfn_worker.py` with it as `<command> -B -P kgpfn_worker.py ...`;
  the environment sets `PYTHONPATH` to the code folder and names the folders that a container must mount
  (`KGPFN_CODE`, `KGPFN_CHECKPOINT_DIR`, `KGPFN_WORKER_DIR`, `KGPFN_WORK`). A wrapper for the image `kgfm/kgpfn:af415c33`
  (which is built on an NVIDIA CUDA image under the NVIDIA Deep Learning Container License) mounts them at their host
  paths:

  ```bash
  #!/bin/bash
  exec docker run --rm --network none --gpus '"device=0"' --memory=8g --memory-swap=8g --cpus=4 --user "$(id -u):$(id -g)" \
    -e PYTHONPATH -e PYTHONDONTWRITEBYTECODE -e TORCH_EXTENSIONS_DIR -e MPLCONFIGDIR -e WANDB_MODE -e HF_HUB_OFFLINE \
    -e KGPFN_CODE -e KGPFN_CHECKPOINT_DIR -e HOME=/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
    -v "$KGPFN_CODE:$KGPFN_CODE:ro" -v "$KGPFN_CHECKPOINT_DIR:$KGPFN_CHECKPOINT_DIR:ro" -v "$KGPFN_WORKER_DIR:$KGPFN_WORKER_DIR:ro" \
    -v "$KGPFN_WORK:$KGPFN_WORK" -w "$PWD" kgfm/kgpfn:af415c33 python3.12 "$@"
  ```

- Settings: the answer-blind context (the query's answer is hidden from the context sampler), context seed 0, 20
  positives and the largest of 80, 40 and 20 negatives that fills every ranked query; batch 64, 32, 16 or 8 by the entity
  count (memory only, but the batch changes the context draws); after CUDA out of memory one more attempt with half the
  batch. Small graphs (entities x ranked facts up to 2.5 million) verify the worker's ranks against the benchmark's
  ranker on the dumped scores. `--gpu-gb` caps the CUDA allocator.
- The frozen run (`fresh-v1-runtime/fv1_worker.py`, `fv1_refs.py`) ran `kgpfn_eval.py` in the portable files of the
  same image (`~/venvs/kgpfn-recovered-20261027`, with `PYTHONHOME`) on kiowl's H100 NVL, the host steps in the run's
  Python. KGPFN's scores, and so some ranks, depend on the GPU type: `tests/golden/kgpfn_probe_ranks.json` holds the
  ranks of the probe graphs per GPU.

### MERIT

- Code: a clone of github.com/cgregucci/MERIT at `480e0e4dd0c8919b14bc69fa621b2e1072b4b00d` (`--code` or `$MERIT_HOME`);
  the adapter uses its `kgfm/models/merit.py` (`RelTransformerMERIT`) and `kgfm/tasks.py` (`build_motif_tensor`), both
  pinned. Checkpoints: the five release checkpoints `merit/ckpts/merit/merit_s{1024,2025,2026,4000,8000}.pth` of the
  release asset `ckpts.zip`, below `--checkpoint-root` (default `checkpoints/`).
- Runtime: MERIT's package imports only with its requirements (torch-scatter, torch-geometric, easydict, jinja2, PyYAML;
  MERIT's `requirements.txt`) on top of the benchmark's. The adapter checks the setup, then runs the scoring in a child
  process: `--python` or `$MERIT_PYTHON` (default: the current interpreter), with the repository and the MERIT folder
  on its `PYTHONPATH`. `--device` (default `cuda`) is the scoring device; the CPU runs MERIT's `mha` attention instead of
  compiled flex attention (smoke tests only).
- Settings: dicee's ULTRA entity tower with MERIT's relation Transformer and motif tensor; noise seed 0, the noise of a
  query seeded by its head entity and converted relation; a fresh compiler cache per graph, a recompilation is an error.
  The five systems share one row file and the benchmark's views (`F`, `F:gt2`, `F:raw`, `F:raw:gt2`).
- The frozen run ran `ag_merit.py` through `fv1_harness.py` in kiowl's run environment (torch 2.5.1+cu124), where it
  loaded the two MERIT files with stand-in modules for the missing torch-scatter. `tests/golden/merit_check_ranks.json`
  holds the raw ranks of `merit-ref-v1/check` (kiowl).

### AnyBURL

- Code: `AnyBURL-23-1x-java11.jar` (built from the official sources with `javac --release 11`), the sources archive and
  the supplementary material `suppl/` of the KR 2026 paper in one folder (`--code` or `$ANYBURL_HOME`). Runtime: Java 11
  (`--java`, default `java`) and `systemd-run --user` (each Java step runs under `MemoryMax=8G`; `--no-scope` turns it
  off).
- Settings: trained per graph, not zero-shot. Learn with the KR setting of `suppl/exp/fb237/config-learn.properties`
  (checked): 30 seconds, `MAX_LENGTH_CYCLIC` 1, zero rules, thresholds 2, 4 threads; Apply with `TOP_K_OUTPUT`
  min(1000, entities); AnyBURL's own Eval as a check of the rank rebuild. Learn's rules depend on the time budget and the
  threads, so two runs can differ; Apply with a stored rule file gives the same lists.
- The frozen run ran `anyburl_run.py --view ag` through `fv1_refs.py` on kiowl's CPU, with Temurin 11.0.28.

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
| AnyBURL rebuild | the rows and native ranks of a stored ranking file of NELLInductive_v1 equal the research `rebuild`; Apply with the stored rule file gives them again | the gate graph; for Apply `$ANYBURL_HOME` and Java 11 |
| KGPFN probe ranks | the worker's ranks of three probe graphs equal the research runner's on the same GPU type | `$KGPFN_CODE`, `$KGPFN_CHECKPOINT_DIR`, `$KGPFN_PYTHON`, a GPU of the fixture |
| MERIT check ranks | the raw ranks of the five systems on two graphs equal `merit-ref-v1/check` | `$MERIT_HOME`, `$MERIT_PYTHON`, an H100 NVL |

`PROOFPFN_GOLDEN` names the proofpfn-golden folder with the large goldens; tests that need them skip without it. The
goldens hold for Python 3.12.2, torch 2.5.1, numpy 1.26.4 and scipy 1.13.0.

## Not here

The N validity gate (held-out worlds of the priors) follows in a later phase.
