# ProofPFN benchmark

Zero-shot link prediction on frozen graph sets. The benchmark ranks every test fact of a graph in both directions with
pretrained checkpoints of this package and with the released reference models, and writes one result file per graph
and system. [PROTOCOL.md](PROTOCOL.md) states the evaluation rules. A second command scores checkpoints on fresh worlds
of the synthetic prior.

## Quick start

Everything runs from the repository root:

```bash
python -m benchmarks.proofpfn list --set ultra54 --view trans
python -m benchmarks.proofpfn check --set ultra54 --model 'PCUltra-S=runs/PCUltra-S/{seed}/export-24000-ema.pt' --references trix,ultra3g,flock
python -m benchmarks.proofpfn run --set ultra54 --out rows \
  --model 'PCUltra-S=runs/PCUltra-S/{seed}/export-24000-ema.pt' --references trix,ultra3g,flock
python -m benchmarks.proofpfn prior --model 'PCUltra-S=runs/PCUltra-S/{seed}/export-24000-ema.pt' \
  --model 'PCUltra-N=runs/PCUltra-N/{seed}/export-24000-ema.pt' --out prior.json
```

| Command | Purpose |
|---|---|
| `list` | the graphs of a manifest view (`--view`, `--graphs`) |
| `check` | CPU only: the graphs' hash checks and counts, official Flock's setting, the systems' files, no score |
| `run` | rank the graphs with the systems, one result file per graph and system |
| `prior` | score checkpoints on held-out worlds of the synthetic prior, per world family ([Worlds of the prior](#worlds-of-the-prior)) |

`python -m benchmarks.proofpfn <command> --help` lists the options of a command.

A model is `--model NAME=TEMPLATE`. `TEMPLATE` is the path of a checkpoint (`proofpfn-checkpoint-v1`), which carries its
own model options. With the field `{seed}` it names one checkpoint per seed of `--seeds` (default 42, 43, 44), and each
gets the label `NAME.seed<S>`. Without the field it names one checkpoint with the label `NAME`. References: `trix`,
`ultra3g` and `flock` (official Flock), with their released checkpoints below `--checkpoint-root` (default
`checkpoints/`): `trix/entity_prediction.pth`, `ultra_3g.pth` and `flock/flock_entity.pth`. Each reference file must
have the SHA-256 of the released file.

`run` options: `--device` (default `cuda`; deterministic CUDA needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the command
sets when it is unset), `--data-root` (the folder that the manifest paths are relative to, default the repository root)
and `--flock-passes-in-sequence`. A system whose result file of a graph exists is not run again. `check` reports each
graph that fails a check and goes on with the next one.

Exit codes: 0 done, 2 wrong input (one `error:` line, also for a selection without graphs or a device that the machine
lacks), 3 a system ran out of CUDA memory on a graph (the other graphs and systems are done), 143 stopped by SIGTERM. An
internal error ends with a traceback and exit code 1.

## Graph sets

| Manifest | Graphs | Hashes checked before a graph loads | Test rows |
|---|---|---|---|
| `ultra54` | the 54 graphs of the standard zero-shot suite, built under the official ULTRA conventions; views `trans` (13), `ind_e` (18), `ind_er` (23) | file SHA-256 of the three splits, counts | all |
| `heldout44` | the 44 graphs of the held-out collection of the thesis; views `published` (16) and `converted` (28, from public sources) | file SHA-256 of the three splits, counts | all 1,000 of each test file, which holds the sample of the collection |

The graph files are not in the repository. A manifest names each folder relative to `--data-root`, with `train.txt`,
`valid.txt` and `test.txt` (three whitespace-separated names per line). `--set` also takes the path of a manifest file
of your own, in the format of `manifests/ultra54.json`: per graph `name`, `path`, `view`, `size` (distinct entities and
distinct triples of `train.txt`) and `sha256` (per split, or `content_sha256` for `train` and `test`); `sample_rule` is
`all` or `first-1000`.

## Systems and result files

| Label | System | Setting |
|---|---|---|
| `NAME.seed<S>` or `NAME` | a checkpoint of this package (`dicee.synthetic_kg.checkpoint`) | the model options of the checkpoint, one query per call |
| `trix` | TRIX, the released checkpoint | one query per call |
| `ultra3g` | ULTRA-3g, the released checkpoint | one query per call |
| `flock` | Flock, the released checkpoint | the graph's official walk count and passes, sampling seed 42; the passes of a query form one microbatch |

A result file `<out>/<manifest>/<label>/<graph>.json` holds the graph, the system's record and the SHA-256 of its file,
`metrics` (MRR and Hits@1, 3, 10 for `both` directions, `tail` queries and `head` queries) and the rows as three lists
of equal length: `index` (the position in `test.txt`), `head` (a head query) and `rank`.

## Worlds of the prior

`prior` scores checkpoints on intact worlds of the synthetic prior that no training order reaches: up to 100 worlds
(`--per-family`) of each of the families `rec1`, `attr2`, `list1`, `categorical2` and `plain` (`--families`). It prints
the MRR per family and model name and writes the record to `--out`, with the values per world
([PROTOCOL.md](PROTOCOL.md), section Worlds of the prior). The thesis uses it for the structural null: PCUltra-S against
PCUltra-N.

## Tests

`python -m pytest benchmarks/proofpfn` runs the tests: about a minute on the CPU. Fixtures are in `tests/fixtures/`
(its README lists them), and the pins of the gate graphs and of the evaluation on worlds of the prior are in
`tests/fixtures/proofpfn/pins.json` of the repository root.

| Test | Content | Needs |
|---|---|---|
| gate | sizes and row order of UMLS, NELLInductive_v1 and grail-WN18RR_v1 | the three graphs |
| sample | the `first-1000` sample of FB15k-237, YAGO3-10 and FB15k | the three graphs |
| manifests | `ultra54` and `heldout44` rebuilt from their input files | `PROOFPFN_MANIFEST_INPUTS` (the folder with the input files) |

Tests whose graphs or input files are missing skip.
