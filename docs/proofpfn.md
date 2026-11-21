# ProofPFN in DICE

ProofPFN pretrains [ULTRA](ultra.md) on knowledge graphs that a synthetic prior generates, in the manner of
prior-data fitted networks (PFNs). Each update draws one world from the prior. A world holds the true facts of a
generated graph, an observed part of them (the context) and queries about the hidden part. The model attaches the
context as its graph and scores every entity for each query. The target of a query is uniform over all true answers
that the generator knows and the context does not show (the RB-W target). After pretraining, the model ranks the test
facts of real graphs without fine-tuning (zero-shot).

The main model is PCUltra: ULTRA with count-weighted relation-graph edges and pair channels. This guide covers four
parts:

- [PCUltra](#pcultra): the model, `--model PCUltra`.
- [Model variants](#model-variants) and [pretraining](#pretraining): `dicee/synthetic_kg/` and the command
  `dicee_pretrain_synthetic`.
- [Zero-shot evaluation](#zero-shot-evaluation): the benchmark `benchmarks/proofpfn/`.
- [Query answering](#query-answering): a pretrained checkpoint as the backbone of `dicee.query_answering`.

## Install

```bash
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu   # CPU only
pip install -e '.[dev]'                                                          # with CUDA
```

- The install provides the console script `dicee_pretrain_synthetic`. Its command `train` runs on Linux and macOS: the
  lock of a run folder uses `fcntl`.
- In a clone without an install, run `python -m dicee.scripts.pretrain_synthetic` from the repository root instead.
- The benchmark runs as `python -m benchmarks.proofpfn` from the repository root.

## PCUltra

Two options of `ULTRA` (`dicee/models/ultra.py`, fields of `dicee.config.Namespace`) extend its relation graph:

| Option | Values | Effect |
|---|---|---|
| `ultra_relation_weighting` | `binary` (default), `normalized` | `normalized` multiplies every relation-graph message by the normalized incidence count of its edge (count weighting) |
| `ultra_pair_channels` | `off` (default), `on` | `on` adds two relation-graph edge types for the entity pairs that two relations share, in the same and in the reverse orientation (pair channels). The six relation tables get 6 rows instead of 4. Requires `normalized` |

The defaults give official ULTRA, so the official checkpoints and every existing use of ULTRA stay unchanged. ULTRA
with count weighting only is CountULTRA. `PCUltra` (`dicee/models/pcultra.py`) is `ULTRA` with both options on, as a
model of its own. It has the same parameters and the same state dictionary.

Use a pretrained checkpoint on a graph of your own:

```bash
python -m dicee --model PCUltra --dataset_dir KGs/UMLS \
  --ultra_checkpoint checkpoints/PCUltra-S-42.pt --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test \
  --path_to_store_single_run Experiments/pcultra-zero-shot
```

```python
from dicee import KGE

kge = KGE(path="Experiments/pcultra-zero-shot")
tail_scores = kge.predict_missing_tail_entity("entity_a", "relation_r")
```

PCUltra computes in another way than `ULTRA` with the two options, with the same result up to float32 rounding:

- It builds the relation graph and its weights from sparse products. The edges, their order and their float64 weights
  equal those of `ultra.relation_graph`, and the memory grows with the number of relation-graph edges.
- At inference, each layer of the relation model computes its weighted message sum as one sparse matrix product. It
  stores no message for each edge. On AristoV4, whose relation graph has 7.1 million edges, one query takes about
  22 ms and 0.5 GiB of GPU memory with PCUltra, and about 161 ms and 5.5 GiB with the option form (RTX 4070 Ti SUPER,
  query batch 1, 60 queries).
- Training, and inference with `--graph_inference_backend torch`, use the weighted layer of the option form. Then the
  scores are bitwise equal.
- Scores differ by float32 rounding, so a rank can differ where two candidates have nearly equal scores. Pretraining
  and the benchmark use the option form.

## Model variants

A model variant is an architecture and its training worlds. `dicee/synthetic_kg/variants.py` holds the variants of the
thesis as presets. Each preset fixes the prior, the two ULTRA options and the target.

| Variant | Model | Training worlds | Target |
|---|---|---|---|
| `PCUltra-S` | PCUltra | synthetic worlds of the seven world families | RB-W |
| `PCUltra-R` | PCUltra | real subgraph worlds of FB15k-237, WN18RR and CoDEx-M | RB-W |
| `PCUltra-M` | PCUltra | synthetic worlds at half their shares, and real subgraph worlds at 0.5 | RB-W |
| `CountULTRA-S`, `-R`, `-M` | CountULTRA | as PCUltra-S, -R and -M | RB-W |
| `ULTRA-S`, `-R`, `-M` | ULTRA | as PCUltra-S, -R and -M | RB-W |
| `PCUltra-HT` | PCUltra | as PCUltra-S | hard: one answer per query, drawn from the RB-W target |
| `PCUltra-N` | PCUltra | as PCUltra-S, with the structural null in four world families | RB-W |

The synthetic prior is a mixture of seven world families (`priors.FAMILY_TITLES`):

| World family | Name in the code | Share |
|---|---|---|
| Rule | `plain` | 0.2925 |
| Structured | `structured` | 0.0758 |
| Statistical | `statistical` | 0.0817 |
| Categorical | `categorical2` | 0.2700 |
| Attribute-dependency | `attr2` | 0.0900 |
| Record | `rec1` | 0.0900 |
| List | `list1` | 0.1000 |

- Every variant trains ULTRA with dimension 64 and 6 layers. The seeds of the thesis are 42, 43 and 44.
- The variants of one prior train on the same worlds in the same order for each seed. The order of the real worlds is
  the same for every seed.
- The structural null swaps the tails of fact pairs within each relation of a world, before the observation (swap
  budget 3). It keeps each relation's fact count and each entity's in-degree and out-degree per relation. It applies
  to the Categorical, Attribute-dependency, Record and List families.
- The names in the code, the mixture strings and the seeded streams fix the worlds. Each preset pins the digest of its
  world order of 24,000 slots for the seeds 42, 43 and 44, and a run refuses an order that differs from its pin.
- `dicee/synthetic_kg/recipe.py` holds the shared training values. AdamW runs with learning rate 5e-4 and weight decay
  0.01. The learning rate stays constant for 20,000 updates and decays linearly to zero over 4,000 more.
- The EMA of the weights has decay 0.999. An update backpropagates its query rows in chunks of 2.

## Pretraining

```bash
# PCUltra-S, seed 42, on the GPU: 24,000 updates
dicee_pretrain_synthetic train --variant PCUltra-S --seed 42 --out runs/PCUltra-S/42

# The variants with real worlds also read the real source graphs
dicee_pretrain_synthetic train --variant PCUltra-M --seed 42 --out runs/PCUltra-M/42 --real_sources data/real-sources

# A smoke run: 2 updates on the CPU, a checkpoint after each
dicee_pretrain_synthetic train --variant PCUltra-S --seed 42 --out runs/smoke --device cpu --workers 0 --stop_after 2 --exports 1,2
```

| Option | Default | Meaning |
|---|---|---|
| `--variant` | required | a name of the table above |
| `--seed` | required | the run seed |
| `--out` | required | the run folder |
| `--device` | `cuda` | the training device |
| `--workers` | 4 | worker processes that generate and prepare the worlds ahead (0: the training process prepares them) |
| `--threads` | 2 | torch threads of the training process |
| `--real_sources` | `$PROOFPFN_REAL_SOURCES` | the folder of the real source graphs (the variants with real worlds) |
| `--save_every` | 250 | updates between two writes of `last.pt` |
| `--allow_code_change` | off | resume although the sources of the run, the torch version or the GPU changed |
| `--stop_after` | none | test option: stop after this update and write `last.pt` |
| `--exports` | `4000,8000,16000,20000,24000` | test option: the export updates; update 24,000 is always exported |

Files of a run folder:

| File | Content |
|---|---|
| `run.json` | the run record (variant, seed, prior, model options, schedule) and its provenance (versions, device, GPU, SHA-256 of the sources and of the real source graphs) |
| `export-<N>-raw.pt`, `export-<N>-ema.pt` | checkpoints of the raw and of the EMA weights after update N |
| `last.pt` | the latest training state: model, AdamW state, EMA and history |
| `progress.json` | the last logged update, every 50 updates and at the end |
| `.lock` | the lock that the training process holds while it trains |

The thesis models are the EMA weights of update 24,000.

### Resume

- If the run folder holds a `last.pt`, the command resumes from it. To resume, run the same command again.
- A resumed run gives the same updates as an uninterrupted run on the same torch version and device.
- SIGINT and SIGTERM stop the run after the current update. The command then writes `last.pt` and exits with code 3.
- The command refuses a folder that holds another run (another variant, seed or option), and a folder that another
  process trains into (the run holds the folder's `.lock`).
- If a source file of `dicee/synthetic_kg` or of the ULTRA model, the torch version or the GPU changed since the start,
  the command refuses to resume. Pass `--allow_code_change` to resume anyway; the result then can differ from the
  uninterrupted run.
- Exit codes: 0 done (also after `--stop_after`), 1 error, 2 wrong input (one `error:` line), 3 stopped by a signal.

### Real source graphs

The real subgraph worlds come from the train splits of FB15k-237, WN18RR and CoDEx-M. The command `sources` builds
the source files from the split files of the three graphs:

```bash
dicee_pretrain_synthetic sources --graphs KGs --out data/real-sources
```

- `--graphs` names a folder with `train.txt`, `valid.txt` and `test.txt` of each graph in `<name>/`. `KGs.zip` holds
  FB15k-237 and WN18RR.
- CoDEx-M comes from the repository tsafavi/codex at commit `3132e426c2a6b643b70bad679905a3a6270be440`. Copy its
  files `data/triples/codex-m/train.txt`, `valid.txt` and `test.txt` into `KGs/CoDEx-M/`.
- The command reads the split files with dicee's `KG`, which loads the GPT-2 encoding of tiktoken. Without that encoding
  in tiktoken's cache, the first run needs the network.
- A source file pins its content: the names, the key order and the split tensors (`real.content_digest`,
  `recipe.REAL_SOURCE_CONTENT_SHA256`). The command refuses split files with other triples and writes nothing then.
- Give the folder to a run with `--real_sources` or with the environment variable `PROOFPFN_REAL_SOURCES`. Before
  training, the run checks the content of each file. `run.json` records the SHA-256 of each file.
- The repository holds none of the three graphs. Check the licence of each graph at its upstream source: FB15k-237
  (Toutanova and Chen, 2015), WN18RR (Dettmers et al., 2018) and CoDEx-M (Safavi and Koutra, 2020).

### Devices and determinism

- Every update runs under `determinism.deterministic_execution`: deterministic algorithms, float32 matmul precision
  `highest` and no TF32 in cuDNN.
- Deterministic cuBLAS needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA starts. The command sets it when it is
  unset.
- The updates do not depend on the number of workers. An update draws no global random number.
- CPU results depend on the torch thread count (`--threads`) and on torch's CPU kernels. CPU and CUDA results differ
  from the first update on. CUDA results can also differ between GPU types and torch versions.
- A run uses one device. Select the GPU with `CUDA_VISIBLE_DEVICES`. Without a GPU, pass `--device cpu`.

## Checkpoints

Pretraining writes checkpoints in the format `proofpfn-checkpoint-v1`. `torch.load(path, weights_only=True)` reads it:

```python
{'format': 'proofpfn-checkpoint-v1', 'model_name': 'ULTRA', 'model': <state dict>,
 'model_config': {'ultra_dim': 64, 'ultra_num_layers': 6, 'ultra_relation_weighting': ..., 'ultra_pair_channels': ...},
 'variant': 'PCUltra-S', 'seed': 42, 'updates': 24000, 'weights': 'ema', 'real_training_updates': 0,
 'order_sha256': <digest of the run's world order>, 'provenance': {...}}
```

- The state dictionary has ULTRA's 82 keys. With pair channels, the six relation tables have 6 rows.
- `model_config` holds the ULTRA options of the variant. `weights` is `raw` or `ema`.
- `provenance` records the origin of the weights.

The following code loads a checkpoint and scores a small graph:

```python
import torch
from dicee.synthetic_kg.checkpoint import load_model

model = load_model("checkpoints/PCUltra-S-42.pt")        # ULTRA with the options of the variant, frozen, on the CPU
facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])  # (head, relation, tail) ids of the inference graph
model.set_graph(facts, num_entities=3, num_relations=2)
with torch.no_grad():
    tail_scores = model.forward_k_vs_all(torch.tensor([[0, 1]]))        # (head, relation): a score per entity
    head_scores = model.forward_k_vs_all_heads(torch.tensor([[1, 2]]))  # (relation, tail): a score per entity
```

- `load_model` checks the checkpoint, builds ULTRA with its `model_config` and loads the weights.
- `ULTRA(checkpoint['model_config']).load_pretrained(path)` and `PCUltra({}).load_pretrained(path)` also read the
  format. They refuse a checkpoint whose `model_config` records other options than the model's.
- The `dicee` command line loads a checkpoint with `--ultra_checkpoint`, for `--model PCUltra` or for `--model ULTRA`
  with the two options ([ULTRA](ultra.md#relation-graph-options)).

### Checkpoints of the thesis

The thesis has 33 models: the eleven variants with the seeds 42, 43 and 44. Each checkpoint holds the EMA weights of
update 24,000 and has the name `<variant>-<seed>.pt`, for example `PCUltra-S-42.pt`. Its `provenance` records the
SHA-256 of the training export that it comes from.

## Zero-shot evaluation

The benchmark `benchmarks/proofpfn/` ranks the test facts of frozen graph sets. Its
[README](../benchmarks/proofpfn/README.md) lists the commands, the graph sets and the result files. Its
[protocol](../benchmarks/proofpfn/PROTOCOL.md) states each rule.

```bash
# Hash checks, counts and the model files on the CPU; no score
python -m benchmarks.proofpfn check --set ultra54 --view trans \
  --model 'PCUltra-S=checkpoints/PCUltra-S-{seed}.pt' --references trix,ultra3g,flock

# Rank the test facts of the 54 graphs with the three seeds and the references
python -m benchmarks.proofpfn run --set ultra54 --out rows \
  --model 'PCUltra-S=checkpoints/PCUltra-S-{seed}.pt' --references trix,ultra3g,flock
```

| Graph set | Graphs |
|---|---|
| `ultra54` | the 54 graphs of the standard zero-shot suite, in the views `trans`, `ind_e` and `ind_er` |
| `heldout44` | the 44 graphs of the held-out collection of the thesis, in the views `published` and `converted` |

- A model is `--model NAME=TEMPLATE`. The template names a checkpoint. With `{seed}`, the benchmark fills in the seeds
  of `--seeds` (default 42, 43 and 44) and labels each checkpoint `NAME.seed<S>`.
- A checkpoint carries its own model options.
- `run` scores on CUDA by default. For a run on the CPU, pass `--device cpu`.
- The result file of seed 42 on CoDExSmall in this example is `rows/ultra54/PCUltra-S.seed42/CoDExSmall.json`.
- A system whose result file exists is not run again.
- The reference systems read the released checkpoints below `checkpoints/`: TRIX (`trix/entity_prediction.pth`),
  ULTRA-3g (`ultra_3g.pth`) and Flock (`flock/flock_entity.pth`). Their guides list the downloads:
  [ULTRA](ultra.md), [TRIX](trix.md), [Flock](flock.md).
- A result file holds the ranks of its graph and its metrics: MRR and Hits@1, 3 and 10 for both directions, for the
  tail queries and for the head queries.
- The thesis tables give, per graph set, the mean over the graphs of the MRR of both directions, and then the mean and
  the sample standard deviation over the three seeds.
- The graph files are not in the repository. A manifest names each folder and the SHA-256 of its three split files,
  and a graph with another hash does not load.

### Worlds of the prior

The command `prior` scores checkpoints on intact worlds of the synthetic prior that no training order reaches. The
thesis uses it for the structural null: PCUltra-S against PCUltra-N.

```bash
python -m benchmarks.proofpfn prior --model 'PCUltra-S=checkpoints/PCUltra-S-{seed}.pt' \
  --model 'PCUltra-N=checkpoints/PCUltra-N-{seed}.pt' --out prior.json
```

- The worlds are held-out worlds of the synthetic prior, seed 42: up to 100 of each of the world families `rec1`,
  `attr2`, `list1`, `categorical2` and `plain`.
- Each model scores every query of a world against all entities, through the training-mode path of pretraining.
- The value of a world is its mean reciprocal rank. The rank excludes the other true answers and the anchor, with
  ties at their mean rank.
- The command prints the MRR per world family and model name, and writes the values per world to `--out`.

## Query answering

A pretrained checkpoint is also a frozen backbone for complex query answering (`dicee.query_answering`,
[CQA guide](guides/multi_hop_queries.md)). The method name is `pcultra-adapter`.

```python
from dicee.query_answering import QueryScoreAdapter
from dicee.query_answering.method_evaluation import evaluate_method, load_kgfm_backbone

checkpoint = "checkpoints/PCUltra-S-42.pt"
model, record = load_kgfm_backbone("pcultra", checkpoint)
QueryScoreAdapter.rule(model).save("rule.json")   # the rule calibration; it needs no training

recipe = dict(method="pcultra-adapter", checkpoint=checkpoint, dataset="FB15k237+H", root="data/cqa",
              adapters={"product": "rule.json"}, operators={"2p": "product", "2i": "product"},
              query_types=["2p", "2i"], selection_protocol="source-validation",
              options={"beam_size": 64})
report = evaluate_method(recipe, output="results/pcultra-rule", device="cuda")
```

- `load_kgfm_backbone` checks the checkpoint format and builds ULTRA with the options of the checkpoint. The result
  file records the variant, the seed and the update of the checkpoint under `inference.backbone`.
- The rule calibration turns each score row into memberships with a softmax over the tails times the number of
  observed tails, capped below one. Candidates at the cap keep the order of their scores. It reads the raw scores.
- The sigmoid calibrations are the learned adapter and the control without an adapter. The backbone is trained with a
  softmax over all entities, so the level of a score row is free and raw scores can saturate a sigmoid.
- The recipe option `"score_input": "log_softmax"` makes the sigmoid calibrations read the log-softmax of each row.
- A learned adapter must be fitted with the score input of its recipe.
  `python -m dicee.query_answering fit --score-input log_softmax` fits one.

## Testing

```bash
python -m pytest tests/test_ultra_relation_options.py tests/test_graph_attach.py tests/test_pcultra.py \
  tests/test_synthetic_*.py benchmarks/proofpfn -m "not slow"
```

- The worlds, the world orders, the prepared tensors, the initial weights and the first training updates are pinned
  in `tests/fixtures/proofpfn/pins.json`. A change of a generator changes a digest, and the test fails.
- The pinned training updates are bitwise only for torch 2.5.1 with AVX-512 CPU kernels and 2 threads. Elsewhere these
  tests skip, and a smoke test compares the losses of two updates within a tolerance.
- `PROOFPFN_REAL_SOURCES` names the real source folder for the tests of the real worlds. Without it they skip.
- pytest runs the tests marked `slow` by default, as in the rest of the repository. `-m "not slow"` deselects them.

## Known limits

- CUDA results can differ between GPU types and torch versions. On another GPU than the one of the thesis runs, a
  rank can differ by a few places where two candidates have nearly equal scores.
- The graph files of the benchmark and the real source graphs are not in the repository.
- TRIX and Flock trained on the prior, the external baselines (KGPFN, MERIT, AnyBURL) and the fine-tuning study of the
  thesis are not part of this package.
