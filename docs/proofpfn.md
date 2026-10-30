# ProofPFN in DICE

ProofPFN pretrains [ULTRA](ultra.md) only on knowledge graphs that a synthetic prior generates, in the manner of
prior-data fitted networks (PFNs). Each update draws one world from the prior. A world holds the true facts of a
generated graph, an observed part of them (the context) and queries about the hidden part. The model attaches the
context as its graph and scores every entity for each query. The target of a query is uniform over all true answers
that the generator knows and the context does not show (the RB-W target). After pretraining, the model ranks the test
facts of real graphs without fine-tuning (zero-shot).

The main model, PCUltra, is ULTRA with two relation-graph options: count weighting and pair channels. This guide
covers three parts:

- `dicee/synthetic_kg/`: the prior, the seven arms, the pretraining loop and the checkpoint format.
- `dicee_pretrain_synthetic`: the command for pretraining, for the conversion of research exports and for the real
  source graphs.
- `benchmarks/proofpfn/`: the zero-shot benchmark, the N validity gate, the adapters of external baselines and the
  registered analysis files.

The code is a port of the research code of the thesis runs. Golden references of the research code test it bit for
bit (see [Testing](#testing)).

## Install

Install the package from a clone of the repository:

```bash
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu   # CPU only
pip install -e '.[dev]'                                                          # with CUDA
```

- The install provides the console script `dicee_pretrain_synthetic`.
- In a clone without an install, run `python -m dicee.scripts.pretrain_synthetic` from the repository root instead.
- The benchmark runs as `python -m benchmarks.proofpfn` from the repository root.
- The benchmark's `cells` command needs scipy. The `dev` extra installs it.
- The reference systems of the benchmark read the released checkpoints below `checkpoints/`: TRIX
  (`trix/entity_prediction.pth`), ULTRA-3g (`ultra_3g.pth`) and Flock (`flock/flock_entity.pth`). Their guides list
  the downloads: [ULTRA](ultra.md), [TRIX](trix.md), [Flock](flock.md).
- The goldens are bitwise in the recorded environment: Python 3.12.2, torch 2.5.1 with AVX-512 CPU kernels and 2
  threads, numpy 1.26.4 and scipy 1.13.0 ([Testing](#testing)). Elsewhere the training and GPU golden tests skip.

## Model options

Two options of `ULTRA` (`dicee/models/ultra.py`, fields of `dicee.config.Namespace`) extend its relation graph:

| Option | Values | Effect |
|---|---|---|
| `ultra_relation_weighting` | `binary` (default), `normalized` | `normalized` multiplies every relation-graph message by the normalized incidence count of its edge (count weighting) |
| `ultra_pair_channels` | `off` (default), `on` | `on` adds two relation-graph edge types for the entity pairs that two relations share, in the same and in the reverse orientation (pair channels). The six relation tables get 6 rows instead of 4. Requires `normalized` |

The defaults give official ULTRA, so the official checkpoints and every existing use of ULTRA stay unchanged.
`dicee/models/_relation_weights.py` computes the weights on the CPU in float64. Pretraining and inference share this
code.

`GraphKGE` (`dicee/models/graph_model.py`) has three methods for pretraining:

- `internal_graph` builds the inference graph of `set_graph` without a model, for example in a worker process.
- `attach_graph` attaches such a graph, optionally with ULTRA's relation graph (`ultra.relation_graph`).
- `forward_query_facts` scores every entity for the query facts on the whole attached graph. It hides no edge.
  Pretraining masks the context answers in its loss instead.

## Arms

An arm is a preset of `dicee/synthetic_kg/arms.py`. It fixes the prior, the two ULTRA options and the target. Every
arm trains ULTRA with dimension 64 and 6 layers, with the seeds 42, 43 and 44.

| Arm | Prior (order name) | Count weighting | Pair channels | Target |
|---|---|---|---|---|
| S | the synthetic mixture of the world families plain, structured, statistical, categorical2, rec1, attr2 and list1 (`REC10A-XLE`) | on | on | RB-W |
| R | real subgraph worlds of FB15k-237, WN18RR and CoDEx-M (`RO2`) | on | on | RB-W |
| M | S's world families at half their shares, and real subgraph worlds at 0.5 (`M2-REC10A-XLE`) | on | on | RB-W |
| N | S's prior, with the structural null in the four record families (categorical2, rec1, attr2, list1) | on | on | RB-W |
| HT | S's prior | on | on | hard: one answer per query, drawn from the RB-W target |
| CU | S's prior | on | off | RB-W |
| UL | S's prior | off | off | RB-W |

- S, R, M, N and HT are PCUltra. CU (CountULTRA) has the count weighting only. UL is plain ULTRA.
- S, HT, CU and UL train on the same worlds in the same order for each seed.
- N has the same world order as S. Its record worlds carry the structural null.
- R has the same world order for every seed.
- The structural null swaps the tails of fact pairs within each relation of a record world, before the observation
  (swap budget 3). It keeps each relation's fact count and each entity's in-degree and out-degree per relation.
- The world families and their shares are the research `--mix` strings, verbatim (`arms.SYNTHETIC_MIX`,
  `arms.MIXED_MIX`).
- The presets pin the research digests of the 24,000-slot world orders of the seeds 42, 43 and 44. A run refuses an
  order that differs from its pin.
- `dicee/synthetic_kg/recipe.py` holds the shared training values. AdamW runs with learning rate 5e-4 and weight decay
  0.01. The learning rate stays constant for 20,000 updates and decays linearly to zero over 4,000 more.
- The EMA of the weights has decay 0.999. An update backpropagates its query rows in chunks of 2.
- The initial weights come from the seeded stream `capacity-prior-v1:<seed>:initial:d64` (`arms.initial_model`).
- The registered analysis also reads rows of the arms S0, DT and PV of an older architecture. These rows come from the
  frozen run. This package does not train or load these arms.

## Pretraining

```bash
# Arm S, seed 42, on the GPU: 24,000 updates
dicee_pretrain_synthetic train --arm S --seed 42 --out runs/S/42

# Arms R and M also read the real source graphs
dicee_pretrain_synthetic train --arm R --seed 42 --out runs/R/42 --real_sources data/real-sources

# A smoke run: 2 updates on the CPU, a checkpoint after each
dicee_pretrain_synthetic train --arm S --seed 42 --out runs/smoke --device cpu --workers 0 --stop_after 2 --exports 1,2
```

A model family of the benchmark needs the seeds 42, 43 and 44 of one arm. Train each seed into its own folder.

| Option | Default | Meaning |
|---|---|---|
| `--arm` | required | S, R, M, N, HT, CU or UL |
| `--seed` | required | the run seed |
| `--out` | required | the run folder |
| `--device` | `cuda` | the training device |
| `--workers` | 4 | worker processes that generate and prepare the worlds ahead (0: the training process prepares them) |
| `--threads` | 2 | torch threads of the training process |
| `--real_sources` | `$PROOFPFN_REAL_SOURCES` | the folder of the real source graphs (arms R and M) |
| `--save_every` | 250 | updates between two writes of `last.pt` |
| `--allow_code_change` | off | resume although the sources of the run, the torch version or the GPU changed |
| `--stop_after` | none | test option: stop after this update and write `last.pt` |
| `--exports` | `4000,8000,16000,20000,24000` | test option: the export updates; update 24,000 is always exported |

Files of a run folder:

| File | Content |
|---|---|
| `run.json` | the run record (arm, seed, prior, model options, schedule) and its provenance (versions, device, GPU, SHA-256 of the sources and of the real source graphs) |
| `export-<N>-raw.pt`, `export-<N>-ema.pt` | checkpoints of the raw and of the EMA weights after update N |
| `last.pt` | the latest training state: model, AdamW state, EMA and history |
| `progress.json` | the last logged update, every 50 updates and at the end |
| `.lock` | the lock that the training process holds while it trains |

Pretraining writes checkpoints at the export updates and after update 24,000. The registered models are the EMA weights
of update 24,000.

### Resume

- If the run folder holds a `last.pt`, the command resumes from it. To resume, run the same command again.
- A resumed run gives the same updates as an uninterrupted run on the same torch version and device.
- SIGINT and SIGTERM stop the run after the current update. The command then writes `last.pt` and exits with code 3.
- The command refuses a folder that holds another run (another arm, seed or option), and a folder that another process
  trains into (the run holds the folder's `.lock`).
- If a source file of `dicee/synthetic_kg` or of the ULTRA model, the torch version or the GPU changed since the start,
  the command refuses to resume. Pass `--allow_code_change` to resume anyway; the result then can differ from the
  uninterrupted run.
- Exit codes: 0 done (also after `--stop_after`), 1 error, 2 wrong input (one `error:` line), 3 stopped by a signal.

### Real source graphs

Arms R and M build their real worlds from the train splits of FB15k-237, WN18RR and CoDEx-M. The command `sources` builds
the source files from the split files of the three graphs:

```bash
dicee_pretrain_synthetic sources --graphs KGs --out data/real-sources
```

```
data/real-sources/
  FB15k-237/data.pt
  WN18RR/data.pt
  CoDEx-M/data.pt
```

- `--graphs` names a folder with `train.txt`, `valid.txt` and `test.txt` of each graph in `<name>/`. `KGs.zip` holds
  FB15k-237 and WN18RR.
- CoDEx-M comes from the repository tsafavi/codex at commit `3132e426c2a6b643b70bad679905a3a6270be440`. Copy its
  files `data/triples/codex-m/train.txt`, `valid.txt` and `test.txt` into `KGs/CoDEx-M/`. The train file has the SHA-256
  `d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556`.
- Each `data.pt` holds the splits as id tensors, the entity names and the relation names. The command writes each file
  as the research builders did and refuses a file whose SHA-256 differs from its pin (`recipe.REAL_SOURCE_SHA256`).
- Give the folder to a run with `--real_sources` or with the environment variable `PROOFPFN_REAL_SOURCES`. Before
  training, the run checks each file against its pin.
- The repository holds none of the three graphs. Check the licence of each graph at its upstream source: FB15k-237
  (Toutanova and Chen, 2015), WN18RR (Dettmers et al., 2018) and CoDEx-M (Safavi and Koutra, 2020).

### Devices and determinism

- Every update runs under `determinism.deterministic_execution`: deterministic algorithms, float32 matmul precision
  `highest` and no TF32 in cuDNN.
- Deterministic cuBLAS needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA starts. The command sets it when it is
  unset.
- The updates do not depend on the number of workers. An update draws no global random number.
- CPU results depend on the torch thread count (`--threads`) and on torch's CPU kernels
  (`torch.backends.cpu.get_cpu_capability()`).
- CPU and CUDA results differ from the first update on. CUDA results can also differ between GPU types.
- A run uses one device. Select the GPU with `CUDA_VISIBLE_DEVICES`. Without a GPU, pass `--device cpu`.

## Checkpoints

Pretraining writes checkpoints in the format `proofpfn-checkpoint-v1`. `torch.load(path, weights_only=True)` reads it:

```python
{'format': 'proofpfn-checkpoint-v1', 'model_name': 'ULTRA', 'model': <state dict>,
 'model_config': {'ultra_dim': 64, 'ultra_num_layers': 6, 'ultra_relation_weighting': ..., 'ultra_pair_channels': ...},
 'arm': 'S', 'seed': 42, 'updates': 24000, 'weights': 'ema', 'real_training_updates': 0,
 'order_sha256': <digest of the run's world order>, 'provenance': {...}}
```

- The state dictionary has ULTRA's 82 keys. With pair channels, the six relation tables have 6 rows.
- `model_config` holds the arm's ULTRA options. `weights` is `raw` or `ema`.
- `provenance` records the origin of the weights: the run record with the versions and source hashes, or the research
  export with its SHA-256.

The following code loads a checkpoint and scores a small graph:

```python
import torch
from dicee.synthetic_kg.checkpoint import load_model

model = load_model("runs/S/42/export-24000-ema.pt")      # the arm's ULTRA, frozen, on the CPU
facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])  # (head, relation, tail) ids of the inference graph
model.set_graph(facts, num_entities=3, num_relations=2)
with torch.no_grad():
    tail_scores = model.forward_k_vs_all(torch.tensor([[0, 1]]))        # (head, relation): a score per entity
    head_scores = model.forward_k_vs_all_heads(torch.tensor([[1, 2]]))  # (relation, tail): a score per entity
```

- `load_model` checks the checkpoint, builds ULTRA with its `model_config` and loads the weights.
  `model_from_checkpoint` does the same for a checkpoint dictionary that is already loaded.
- `ULTRA(checkpoint['model_config']).load_pretrained(path)` also reads the format.
- `ULTRA.load_pretrained` refuses a checkpoint whose `model_config` records other options than the model's.
- The `dicee` command line loads a checkpoint with `--ultra_checkpoint` and the arm's options
  `--ultra_relation_weighting` and `--ultra_pair_channels` ([ULTRA](ultra.md#relation-graph-options)).
- `benchmarks/kgfm_zero_shot.py` takes no ProofPFN checkpoint, unlike the design: a third zero-shot path next to the
  `dicee` command line and the benchmark has no use in the thesis.

### Research exports

The research trainer wrote `export-<N>.pt` files in its own layout. `convert` turns one into a checkpoint of an arm:

```bash
dicee_pretrain_synthetic convert --export export-24000.pt --arm S --out S-42.pt
```

- The command loads the research export with `weights_only=True`.
- It checks that the export's run record equals the arm's options and that its order equals the pinned order.
- It checks that the parts it drops are inert. These parts are the context and role adapters, the three marker
  buffers and, for CU, rows 4 and 5 of the relation tables.
- Inert adapters have zero output layers and equal raw and EMA tensors. The markers hold their research values, and
  CU's rows 4 and 5 are zero.
- The checkpoint keeps ULTRA's tensors. Its `provenance` records the export's file name and SHA-256.
- `--weights ema` (the default) takes the EMA weights. `--weights raw` takes the raw weights.
- If a check fails, the command writes nothing and exits with code 2.
- The benchmark also accepts research exports directly. It converts them on load with the same checks.

## Zero-shot evaluation

The benchmark `benchmarks/proofpfn/` ranks the test facts of frozen graph sets by the protocol of the research runs.
Its [README](../benchmarks/proofpfn/README.md) lists the commands, the graph sets and the row files. Its
[protocol](../benchmarks/proofpfn/PROTOCOL.md) states each rule with the research module that it reproduces.

```bash
# Hash checks, counts and the model files on the CPU; no score
python -m benchmarks.proofpfn check --set dev121 --graphs UMLS \
  --family mine=S:runs/S/{seed}/export-24000-ema.pt --references trix,ultra3g,flockoff

# Rank the test facts of UMLS with the model family and the references
python -m benchmarks.proofpfn run --set dev121 --graphs UMLS --out rows \
  --family mine=S:runs/S/{seed}/export-24000-ema.pt --references trix,ultra3g,flockoff
```

- A model family is `--family [KEY=]ARM:TEMPLATE`. The benchmark fills `{seed}` with 42, 43 and 44.
- The template names checkpoints or research exports.
- Without a key, the family takes the arm's registered key, for example `rec10axlepcunoa` for S.
- Under a registered key, every file must match the research export pins of `manifests/models.json`.
- For other checkpoints, choose a key of lower-case letters and digits, for example `mine`.
- `run` scores on CUDA by default. For a run on the CPU, pass `--device cpu`.
- The row file of this example is `rows/top/mine-trix-ultra3g-flockoff/UMLS.json`. The registered analysis reads
  this format.

## N validity gate

The N validity gate is a manipulation check of arm N. It asks whether N scores below S on intact worlds of every
nulled world family and close to S on plain worlds. The fresh-v1 registration fixes its rule (amendment A1, section
8.4). The command `gate` applies the rule:

```bash
python -m benchmarks.proofpfn gate --family mys=S:runs/S/{seed}/export-24000-ema.pt \
  --family myn=N:runs/N/{seed}/export-24000-ema.pt --out gate.json
```

- The gate scores the model families of S and N on intact held-out worlds of S's prior, seed 42.
- No training order reaches these worlds. The gate takes up to 100 worlds of each nulled world family and of plain.
- Each model scores every query of a world against all entities, through the training-mode path of pretraining.
- The value of a world is its mean reciprocal rank. The rank excludes the other true answers and the anchor, with
  ties at their mean rank.
- Per world family, D is the mean over S's seeds minus the mean over N's seeds. Its standard error SE combines the
  spread over worlds and over seeds (`gate.compare`).
- The reading `n_validity` is `pass` when D > 2 SE on every nulled world family and |D| <= 2 SE on plain.
- Options: `--per-family` (held-out worlds per world family, at most 100), `--families` (the reading needs all five)
  and `--device` (default `cuda`).
- The command prints D and SE per world family and writes the record to `--out`.

## External baselines

Three adapters run external baselines on the graphs of the benchmark: KGPFN, MERIT and AnyBURL. Their code,
checkpoints and runtimes stay outside the repository. Each adapter checks its files against the pins of
`manifests/models.json` and writes row files in the benchmark's format. Before a run, check the setup with
`--check`. The [README](../benchmarks/proofpfn/README.md#external-baselines) documents the setup and the settings of
each adapter. Official Flock is a native reference system of the benchmark (`flockoff`), not an adapter.

## Reproducing the thesis results

- The thesis rows of the fresh-v1 test come from the frozen run of the research code. The development rows also come
  from the research code.
- The fresh-v1 registration fixed the graphs, models, protocol and analysis before the run (commit `73c94830`, tag
  `freeze/fresh-v1`).
- The six registered files in `benchmarks/proofpfn/registered/` turn row files into the registered analysis.
- `fv1_tables.py` writes the tables `mrr.csv`, `ranks.csv` and `cost.csv`. `fv1_analysis.py analyse` writes the
  result. `fv1_drops.py` writes `dropped.json` from the dispatcher records of the frozen run.
- `fv1_devrows.py` writes the development table of `fv1_analysis.py precision`. The two test scripts check the others.
- The registered files are byte-identical to the freeze anchors of the registration. A test checks their SHA-256.
  Their byte-identical outputs need Python 3.12.2 and numpy 1.26.4.
- The [README](../benchmarks/proofpfn/README.md#the-registered-analysis) shows the commands of the two main scripts.
- The benchmark writes three of their inputs. `fv1-models` writes the `--models` file, `cells` writes `queries.csv`
  and `gate` gives the `n_validity` entry of `gates.json`.
- The `kgpfn_parity` entry of `gates.json` comes from the KGPFN parity check of the frozen run (registration section
  7.3).
- The ported code reproduces the research code bit for bit on the goldens. Its row files have the research format, so
  the registered files read them like the research rows.
- To evaluate the registered models, pass their pinned research exports under the registered keys. The checkpoints
  that `convert` makes from these exports pass the pins too.
- The fresh-v1 and block-3 manifests are not in the repository. They follow after the frozen read. A blind manifest
  opens only with `--release NAME`.
- The research code and its notes stay outside this branch. They are in the branch `proofpfn-research` with the tags
  `research/proofpfn-20261029` and `freeze/fresh-v1`. The snapshot repository `proofpfn-research-snapshot` holds the
  runtime folders of the research runs.

## Testing

```bash
# Fast tests, without the slow golden checks
python -m pytest tests/test_ultra_relation_options.py tests/test_graph_attach.py tests/test_synthetic_*.py \
  benchmarks/proofpfn -m "not slow"

# Every test, with the large goldens and the real source graphs
export PROOFPFN_GOLDEN=/path/to/proofpfn-golden
export PROOFPFN_REAL_SOURCES=/path/to/real-sources
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest tests/test_ultra_relation_options.py tests/test_graph_attach.py tests/test_synthetic_*.py benchmarks/proofpfn
```

Golden references (goldens) record what the research code computes at small sizes:

| Golden | Content | Rule |
|---|---|---|
| G1 | the world orders of every arm and seed | bitwise |
| G2 | worlds of every world family and their prepared tensors | bitwise |
| G3 | the initial weights | bitwise |
| G4 | the first 20 or 100 updates on the CPU: loss, gradient norm, weights, EMA, AdamW state | bitwise |
| G5 | the research exports and the checkpoints converted from them | bitwise per tensor |
| G6 | every row key of the research harness on three development graphs | bitwise |
| G7 | the query cells of the registered analysis | byte-identical |
| G8 | simulated rows through the registered `fv1_tables.py` and `fv1_analysis.py` | byte-identical |
| G9 | the N validity gate | equal values |
| G10 | the first updates of the research trainer on the GPU | bitwise on the golden GPU |

- Small goldens are in `tests/fixtures/proofpfn/` and `benchmarks/proofpfn/tests/golden/`. Their READMEs list the
  sources and SHA-256 of the files.
- Large goldens (research exports, full row files, world tensors) stay outside the repository. Tests read them from
  the folder in `PROOFPFN_GOLDEN` and skip without it.
- `PROOFPFN_REAL_SOURCES` names the real source folder for the tests of the real worlds and of arms R and M.
- G6, G10 and the CUDA scores of converted exports come from an NVIDIA GeForce RTX 4070 Ti SUPER. On other GPUs these
  tests skip. They also skip without `CUBLAS_WORKSPACE_CONFIG`.
- The KGPFN probe ranks exist per GPU type. The MERIT check ranks need an H100 NVL.
- pytest runs the tests marked `slow` by default, as in the rest of the repository. `-m "not slow"` deselects them.
- `PROOFPFN_SIMULATE=1` adds the simulations of the registered analysis tests (about 9 minutes).
- The CPU training goldens (G4) are bitwise in the recorded environment only: torch 2.5.1 with AVX-512 CPU kernels
  (`torch.backends.cpu.get_cpu_capability()`) and 2 threads. Other CPU kernels change the last bits of the backward pass
  and of AdamW, so elsewhere these tests skip. A smoke test then compares the losses of two updates within a relative
  tolerance.
- The CUDA goldens also need torch 2.5.1 with CUDA 11.8. Elsewhere they skip.
- In the recorded environment, a failing golden test means that the port differs from the research code. Then fix the
  code. Leave the golden and its comparison unchanged.

## Known limits

- The MERIT adapter imports MERIT's package, which needs torch-scatter and torch-geometric. Without them, the adapter
  cannot run and its tests skip.
- CUDA results can differ between GPU types. The GPU goldens hold for the RTX 4070 Ti SUPER only. KGPFN's scores, and
  so some of its ranks, differ between GPU types.
- The graph files of the blind test sets (fresh-v1 and block 3) are never in the repository. Their manifests follow
  after the frozen read.
- The real source graphs of arms R and M are not in the repository. The command `sources` builds them from their split
  files.
- AnyBURL learns its rules within a time budget, so two runs can give different rules.
