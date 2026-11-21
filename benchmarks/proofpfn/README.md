# ProofPFN benchmark

Zero-shot link prediction on frozen graph sets, by the evaluation protocol of the ProofPFN research runs
([PROTOCOL.md](PROTOCOL.md)). The benchmark ranks every test fact of a graph in both directions with ProofPFN families
(three seeds of one arm) and the native reference models, and writes one row file per graph. It also runs the N
validity gate. The golden test G6 checks that its ranks equal the research harness's bit for bit on three development
graphs.

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
```

| Command | Purpose |
|---|---|
| `list` | the graphs of a manifest view (`--view`, `--graphs`, `--codes`) |
| `check` | CPU only: the graphs' hash checks and counts, official Flock's setting, the systems' files and pins, no score |
| `run` | rank the graphs with the systems, one row file per graph |
| `cells` | the query cells of the registered analysis (`queries.csv` and its `.meta.json`) |
| `gate` | the N validity gate: S and N on held-out worlds of the prior ([N validity gate](#n-validity-gate)) |

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

Exit codes: 0 done, 2 wrong input (one `error:` line, also for a selection without graphs or a device that the machine
lacks), 3 a graph ran out of CUDA memory (its record is in `.errors/` of the row folder), 76 stopped at `--max-minutes`,
143 stopped by SIGTERM (with `--parts` or a shard, the part file is saved). An internal error ends with a traceback and
exit code 1.

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

## Tests and goldens

`python -m pytest benchmarks/proofpfn` runs the tests: a few minutes on the CPU, plus about 7 minutes for G6 where its
GPU and goldens are present. Goldens (`tests/golden/README.md`):

| Golden | Test | Needs |
|---|---|---|
| G6, ranks | every key on the gate graphs equals the research rows: families S, CU, UL, HT of the G4 exports and the three references (about 7 minutes), and the per-seed raw ranks of S, CU and UL on UMLS on the CPU | `PROOFPFN_GOLDEN` (the exports), and for the GPU CUDA on the golden GPU (RTX 4070 Ti SUPER) with `CUBLAS_WORKSPACE_CONFIG` |
| G7, cells | `queries.csv` of the gate graphs byte-identical | the gate graphs |
| G9, gate | the per-world MRRs, D, SE and the reading of the research gate on the G4 exports of S and N at update 20 | `PROOFPFN_GOLDEN` (the exports) |
| sample | the `first-1000` sample of FB15k-237, YAGO3-10 and FB15k equals the research `ag_eval.sample_positions` | the three graphs |

`PROOFPFN_GOLDEN` names the proofpfn-golden folder with the large goldens. Tests that need them skip without it. The
goldens are bitwise in the recorded environment: Python 3.12.2, torch 2.5.1 (CUDA 11.8 on an RTX 4070 Ti SUPER for
G6 on the GPU) and numpy 1.26.4. Elsewhere the GPU test of G6 skips. G7 does not depend on the scipy version
([PROTOCOL.md](PROTOCOL.md), section Cells).
