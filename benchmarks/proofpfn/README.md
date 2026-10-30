# ProofPFN benchmark

Zero-shot link prediction on frozen graph sets, by the evaluation protocol of the ProofPFN research runs
([PROTOCOL.md](PROTOCOL.md)). The benchmark ranks every test fact of a graph in both directions with ProofPFN families
(three seeds of one arm) and the native reference models, and writes one row file per graph whose format the
registered analysis reads. It also runs the N validity gate and three external baselines. The golden test G6 checks
that its ranks equal the research harness's bit for bit on three development graphs.

## Quick start

Everything runs from the repository root:

```bash
python -m benchmarks.proofpfn list --set dev121 --view main
python -m benchmarks.proofpfn check --set gate --family S:/runs/S/training/{seed}/export-24000.pt
python -m benchmarks.proofpfn run --set dev121 --view main --out rows \
  --family S:/runs/S/training/{seed}/export-24000.pt --references trix,ultra3g,flockoff
python -m benchmarks.proofpfn cells --set gate --out queries.csv
python -m benchmarks.proofpfn gate --family S:/runs/S/training/{seed}/export-24000.pt \
  --family N:/runs/N/training/{seed}/export-24000.pt --out gate.json
python -m benchmarks.proofpfn fv1-models --out models-fv1.json
python -m benchmarks.proofpfn adapter anyburl --set dev121 --view main --out rows --code /path/to/AnyBURL
```

| Command | Purpose |
|---|---|
| `list` | the graphs of a manifest view (`--view`, `--graphs`, `--codes`) |
| `check` | CPU only: the graphs' hash checks and counts, official Flock's setting, the systems' files and pins, no score |
| `run` | rank the graphs with the systems, one row file per graph |
| `cells` | the query cells of the registered analysis (`queries.csv` and its `.meta.json`) |
| `gate` | the N validity gate: S and N on held-out worlds of the prior ([N validity gate](#n-validity-gate)) |
| `fv1-models` | the `--models` file of the registered `fv1_tables.py`, from `manifests/models.json` |
| `adapter` | an external baseline (`kgpfn`, `merit`, `anyburl`) on the graphs, one row file per graph ([External baselines](#external-baselines)) |

`python -m benchmarks.proofpfn <command> --help` lists the options of a command.

A family is `--family [KEY=]ARM:TEMPLATE`. `ARM` is one of S, R, M, N, HT, CU and UL, `TEMPLATE` the path of a clean
checkpoint (`proofpfn-checkpoint-v1`) or of a research export, with `{seed}` for the seeds 42, 43 and 44. Research
exports convert on load (`dicee.synthetic_kg.checkpoint.from_research_export`, EMA weights, after its inertness
checks). `KEY` defaults to the arm's registered key (`rec10axlepcunoa` for S, `pcur`, `pcum`, `pcun`, `pcuht`,
`rec10axlecunoa`, `rec10axleulnoa`). Under a registered key, each file must hold the pinned research export of its seed
(`models.json`), with the EMA weights of the pinned update. Any other lower-case key runs a test family without pins.
References: `trix`, `ultra3g` and `flockoff` (official Flock), with their checkpoints below `--checkpoint-root`
(default `checkpoints/`).

`run` options: `--device` (default `cuda`, deterministic CUDA needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the command
sets when it is unset), `--shard i/k`, `--parts`, `--save-minutes`, `--max-minutes`, `--data-root` (the folder that the
manifest paths are relative to, default the repository root), `--flock-passes-in-sequence`. `check` reports each graph
that fails a check and goes on with the next one.

Exit codes: 0 done, 2 wrong input (one `error:` line), 3 a graph ran out of CUDA memory or an adapter failed on a graph
(its record is in `.errors/` of the row folder), 76 stopped at `--max-minutes`, 143 stopped by SIGTERM (with `--parts`
or a shard, the part file is saved). An internal error ends with a traceback and exit code 1.

## Graph sets

| Manifest | Graphs | Hashes checked before a graph loads | Test rows | Source |
|---|---|---|---|---|
| `dev121` | 121 audited development graphs, views `main` (91, with weights), `contamination` (29), `apart` (UMLS) | content of train.txt and test.txt, file SHA-256 where the suite manifest records the folder | `first-1000` rule | `graph-audit-v1/graphs.json`, `stats/inventory.json` |
| `t1` | 54 graphs of the standard zero-shot suite, views `trans`, `ind_e`, `ind_er` | file SHA-256 of the three splits, counts | all | `p0-suite-v1/manifest.json` |
| `gate` | UMLS, `ultra-suite-v1/NELLInductive_v1`, `inductive/grail-WN18RR_v1` | file SHA-256 and content | `first-1000` rule (all rows here) | the audit and the golden records |

The audit excludes grail-WN18RR_v1 as a copy of `ultra-suite-v1/WN18RRInductive_v1`, so it is in `gate` and not in
`dev121`. Each manifest lists its inputs and their SHA-256 under `sources`. The builders of `tests/test_manifests.py`
derive the files from them and compare them with the committed files when the inputs are available.

Blind sets. A manifest with `"blind": true` opens only with `--release NAME` naming it. Any other manifest is refused
when it names a blind folder: a path with a part `fresh-v1` or `lockbox3`, or a folder that resolves (links followed)
below `KGs/fresh-v1` or `KGs/lockbox3`. This branch holds no blind manifest. The fresh-v1 and block-3 manifests are added
after the frozen read.

## Systems and row files

| Key | System | Models |
|---|---|---|
| a family key | ProofPFN family of one arm | three seeds, `log_softmax` per model, family view = mean over the seeds |
| `trix` | TRIX, `trix/entity_prediction.pth` | one |
| `ultra3g` | ULTRA-3g, `ultra_3g.pth` | one |
| `flockoff` | Flock, `flock/flock_entity.pth`, the graph's official walk count and passes, seed 42 | one, the passes of a query form one microbatch |

A row file `<out>/<collection>/<keys joined by '-'>/<graph>.json` holds per test fact and direction the ranks of every key
(PROTOCOL.md): `F`, `F:gt2`, `F:raw`, `F:raw:gt2` per system and `F.seedS:raw`, `F.seedS`, `F.seedS:gt2` per seed of a
family. It also holds `metrics`, `seconds`, the research record `allgraph` (graph, file hashes, sample, systems, model
hashes, Flock's setting), `metrics_views` (both directions, tail queries, head queries) and `provenance` (code hashes,
versions, device, checkpoint records). The model hash of a family seed is its research export's SHA-256 where there is
one, so clean and research rows compare with the same pins.

Sharding (T1): `--shard i/k` ranks block i of k contiguous blocks of the relation-ordered query groups into
`.parts/<graph>/shard-i-of-k.json`. A part saves every `--save-minutes` and resumes after its last stored group. The run
that completes the last part writes the row file (run a finished shard again to merge). Ranks do not depend on the
sharding. A part of another run (other systems or graph files) stops the run with a message.

## N validity gate

`gate` runs the manipulation check of arm N (registration, amendment A1, section 8.4). The families of S and N score
intact held-out worlds of S's prior, and the reading `n_validity` (`pass` or `fail`) is the entry of `gates.json` that
the registered analysis reads. [PROTOCOL.md](PROTOCOL.md), section N validity gate, states the rules. Options:
`--per-family` (1 to 100 held-out worlds per family, default 100), `--families` (the reading needs all five) and
`--device` (default `cuda`). The command prints D and SE per family and writes the record to `--out`, with the
per-world values and a provenance record.

## External baselines

`adapters/` runs three external baselines by the logic of their research runners and writes row files that the registered
`fv1_tables.py` reads. Their code, checkpoints and runtimes stay outside the repository. Each adapter takes its code
folder from `--code` or an environment variable and checks it, and its files, against the pins of `manifests/models.json`
(`external`, for MERIT also `references`). Rows, ranks and the record follow [PROTOCOL.md](PROTOCOL.md), section External
baselines. `--check` checks the setup (with a trial start of KGPFN's Python and of MERIT's child process), the pins and
the graphs without a run. A graph on which an adapter fails (an external step, or an error with its traceback) gets an
error record in `.errors/` next to its row file, and the run goes on with the next graph. The command then exits with
code 3. Work files stay in `<out>/.work` (or `--work`), a dot folder that the tables skip. Tests that need the external
code skip without it.

| Adapter | Row file | Keys | Research runner |
|---|---|---|---|
| `kgpfn` | `<out>/<collection>/kgpfn/<graph>.json` | `kgpfn`, `kgpfn:raw`, no model hash entry | `kgpfn-ref-v1-runtime/kgpfn_ag.py`, `kgpfn_eval.py` |
| `merit` | `<out>/<collection>/merit1024-merit2025-merit2026-merit4000-merit8000/<graph>.json` | `merit<S>`, `merit<S>:gt2`, `merit<S>:raw`, `merit<S>:raw:gt2`, model hash `merit<S>` | `allgraph-v1/ag_merit.py`, `merit-ref-v1-runtime/merit_ref.py` |
| `anyburl` | `<out>/<collection>/anyburl/<graph>.json` | `anyburl`, `anyburl:raw`, no model hash entry | `anyburl-ref-v1-runtime/anyburl_run.py` |

None of the three local copies holds a licence file. The repository contains none of their code and none of their
checkpoints. Check the terms of each upstream project before you copy or publish its files.

### KGPFN

- Code: a clone of github.com/HKUST-KnowComp/KGPFN at `4f1bc61dcc998cf0672deaca148899b505797af6` (`--code` or
  `$KGPFN_CODE`). The 54 files that the worker loads (its modules and the sources of the rspmm kernel) are pinned by
  SHA-256 (`external.kgpfn.imported` of `models.json`). The setup checks them, and a run refuses a worker that loaded
  other files. Checkpoints: `kgpfn_icl_3g.pth` and `tabicl-regressor-v2-20260212.ckpt` (`--checkpoints` or
  `$KGPFN_CHECKPOINT_DIR`), as KGPFN's `script/download.py` fetches them.
- Runtime: KGPFN's own environment (its `requirements.txt`: torch 2.5.1+cu121, torch-geometric, tabicl, and a CUDA
  toolkit for the JIT build of its rspmm kernel). `--python` or `$KGPFN_PYTHON` is the command that starts its Python,
  split like shell words. The host runs `adapters/kgpfn_worker.py` with it as `<command> -B -P kgpfn_worker.py ...`.
  The environment sets `PYTHONPATH` to the code folder, keeps the kernel build cache (`TORCH_EXTENSIONS_DIR`) in the
  work root, and names the folders that a container mounts (`KGPFN_CODE`, `KGPFN_CHECKPOINT_DIR`, `KGPFN_WORKER_DIR`,
  `KGPFN_WORK`). A wrapper for the image `kgfm/kgpfn:af415c33` (which is built on an NVIDIA CUDA image under the NVIDIA
  Deep Learning Container License) mounts them at their host paths:

  ```bash
  #!/bin/bash
  exec docker run --rm --network none --gpus '"device=0"' --memory=8g --memory-swap=8g --cpus=4 --user "$(id -u):$(id -g)" \
    -e PYTHONPATH -e PYTHONDONTWRITEBYTECODE -e TORCH_EXTENSIONS_DIR -e MPLCONFIGDIR -e WANDB_MODE -e HF_HUB_OFFLINE \
    -e KGPFN_CODE -e KGPFN_CHECKPOINT_DIR -e HOME=/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
    -v "$KGPFN_CODE:$KGPFN_CODE:ro" -v "$KGPFN_CHECKPOINT_DIR:$KGPFN_CHECKPOINT_DIR:ro" -v "$KGPFN_WORKER_DIR:$KGPFN_WORKER_DIR:ro" \
    -v "$KGPFN_WORK:$KGPFN_WORK" -w "$PWD" kgfm/kgpfn:af415c33 python3.12 "$@"
  ```

- Settings: the answer-blind context (the query's answer is hidden from the context sampler), context seed 0, 20
  positives and the largest of 80, 40 and 20 negatives that fills every ranked query. The batch is 64, 32, 16 or 8 by
  the entity count (memory only, but the batch changes the context draws). After CUDA out of memory, one more attempt
  runs with half the batch, and a worker run stops after 24 hours. Small graphs (entities x ranked facts up to 2.5
  million) verify the worker's ranks against the benchmark's ranker on the dumped scores. `--gpu-gb` caps the CUDA
  allocator.
- The frozen run (`fresh-v1-runtime/fv1_worker.py`, `fv1_refs.py`) ran `kgpfn_eval.py` in the portable files of the
  same image (with `PYTHONHOME`) on kiowl's H100 NVL, the host steps in the run's Python. KGPFN's scores, and so some
  ranks, depend on the GPU type: `tests/golden/kgpfn_probe_ranks.json` holds the ranks of the probe graphs per GPU.

### MERIT

- The MERIT adapter is unverified: it never ran with MERIT's code. It is verified when
  `test_check_graph_ranks_equal_the_research_ranks` (`tests/test_adapter_merit.py`) passes on an H100 NVL. Its tests
  without MERIT's code use a stand-in relation model.
- Code: a clone of github.com/cgregucci/MERIT at `480e0e4dd0c8919b14bc69fa621b2e1072b4b00d` (`--code` or `$MERIT_HOME`).
  The adapter uses its `kgfm/models/merit.py` (`RelTransformerMERIT`) and `kgfm/tasks.py` (`build_motif_tensor`), both
  pinned. Checkpoints: the five release checkpoints `merit/ckpts/merit/merit_s{1024,2025,2026,4000,8000}.pth` of the
  release asset `ckpts.zip`, below `--checkpoint-root` (default `checkpoints/`).
- Runtime: MERIT's environment, the benchmark's requirements with torch 2.5.1+cu124, torch-scatter 2.1.2,
  torch-geometric 2.4.0, easydict, jinja2 and PyYAML (MERIT's `requirements.txt`). The adapter checks the setup, then
  runs the scoring in a child process: `--python` or `$MERIT_PYTHON` (default: the current interpreter), with the
  repository and the MERIT folder on its `PYTHONPATH`. The child stops after 6 hours per graph of the run. `--device`
  (default `cuda`) is the scoring device. The CPU runs MERIT's `mha` attention instead of compiled flex attention
  (smoke tests only).
- Settings: dicee's ULTRA entity tower with MERIT's relation Transformer and motif tensor. Noise seed 0, the noise of a
  query seeded by its head entity and converted relation. A fresh compiler cache per graph, and a recompilation is an
  error. The five systems share one row file and the benchmark's views (`F`, `F:gt2`, `F:raw`, `F:raw:gt2`).
- The frozen run ran `ag_merit.py` through `fv1_harness.py` in kiowl's run environment (torch 2.5.1+cu124), where it
  loaded the two MERIT files with stand-in modules for the missing torch-scatter. `tests/golden/merit_check_ranks.json`
  holds the raw ranks of `merit-ref-v1/check` (kiowl).

### AnyBURL

- Code: `AnyBURL-23-1x-java11.jar` (built from the official sources with `javac --release 11`), the sources archive and
  the supplementary material `suppl/` of the KR 2026 paper in one folder (`--code` or `$ANYBURL_HOME`). Runtime: Java 11
  (`--java`, default `java`) and `systemd-run --user`: each Java step runs under `MemoryMax=8G` (`--no-scope` turns it
  off), stops after 6 hours, and stops with the adapter.
- Settings: trained per graph, not zero-shot. Learn runs with the KR setting of `suppl/exp/fb237/config-learn.properties`
  (checked): 30 seconds, `MAX_LENGTH_CYCLIC` 1, zero rules, thresholds 2, 4 threads. Apply runs with `TOP_K_OUTPUT`
  min(1000, entities), and AnyBURL's own Eval checks the rank rebuild. Learn's rules depend on the time budget and the
  threads, so two runs can differ. Apply with a stored rule file gives the same lists.
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

Seam. The writer of each input of the registered analysis:

- The row files (`ROWS`): `run` and `adapter`.
- `models-fv1.json`: `fv1-models`.
- `queries.csv`: `cells`.
- `gates.json`: `n_validity` is the reading of `gate`. `kgpfn_parity` is the frozen record of the pre-read parity check
  (registration, section 7.3). The check needs KGPFN's native context mode on the T1 probes, outside the answer-blind
  adapter, so the benchmark has no command for it.
- `dropped.json`: the registered `fv1_drops.py` on the dispatcher records of the frozen run. The clean `run` and
  `adapter` commands write error records to `.errors/` instead.
- `MANIFEST` and `BLOCK3`: the fresh-v1 and block-3 manifests, added after the frozen read.

`fv1_devrows.py` (the development table of `fv1_analysis.py precision`) is kept as a freeze anchor. It finds its inputs
relative to its folder in the research tree (`K = HERE.parent`), so it runs only there.

Ruff does not check the six files (`extend-exclude` in `pyproject.toml`), and pytest does not collect their test scripts
(`conftest.py`). `tests/test_registered.py` checks their SHA-256, runs their fast checks as subprocesses and, with
`PROOFPFN_SIMULATE=1`, their simulations (about 9 minutes).

## Tests and goldens

`python -m pytest benchmarks/proofpfn` runs the tests: a few minutes on the CPU, plus about 9 minutes for G6 where its
GPU and goldens are present. Goldens (`tests/golden/README.md`):

| Golden | Test | Needs |
|---|---|---|
| G6, ranks | every key on the gate graphs equals the research rows: families S, CU, UL, HT of the G4 exports and the three references (about 9 minutes), and the per-seed raw ranks of S, CU and UL on UMLS on the CPU | `PROOFPFN_GOLDEN` (the exports), and for the GPU CUDA on the golden GPU (RTX 4070 Ti SUPER) with `CUBLAS_WORKSPACE_CONFIG` |
| G7, cells | `queries.csv` of the gate graphs byte-identical | the gate graphs |
| G8, chain | simulated rows -> registered `fv1_tables.py` -> registered `fv1_analysis.py`, every output byte-identical | Python 3.12.2 with numpy 1.26.4 |
| G9, gate | the per-world MRRs, D, SE and the reading of the research gate on the G4 exports of S and N at update 20 | `PROOFPFN_GOLDEN` (the exports) |
| sample | the `first-1000` sample of FB15k-237, YAGO3-10 and FB15k equals the research `ag_eval.sample_positions` | the three graphs |
| AnyBURL rebuild | the rows and native ranks of a stored ranking file of NELLInductive_v1 equal the research `rebuild`, and Apply with the stored rule file gives them again | the gate graph, for Apply also `$ANYBURL_HOME` and Java 11 |
| KGPFN probe ranks | the worker's ranks of three probe graphs equal the research runner's on the same GPU type | `$KGPFN_CODE`, `$KGPFN_CHECKPOINT_DIR`, `$KGPFN_PYTHON`, a GPU of the fixture |
| MERIT check ranks | the raw ranks of the five systems on two graphs equal `merit-ref-v1/check` | `$MERIT_HOME`, `$MERIT_PYTHON`, an H100 NVL |

`PROOFPFN_GOLDEN` names the proofpfn-golden folder with the large goldens. Tests that need them skip without it. The
goldens are bitwise in the recorded environment: Python 3.12.2, torch 2.5.1 (CUDA 11.8 on an RTX 4070 Ti SUPER for
G6 on the GPU), numpy 1.26.4 and scipy 1.13.0. Elsewhere the GPU test of G6 skips. The registered `fv1_analysis.py`
refuses every other Python and numpy version, so its fast checks, G8 and its simulations skip there. The table checks,
the anchors and the pins run everywhere.
