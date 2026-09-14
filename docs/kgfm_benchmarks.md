# KGFM evaluation protocol

The [README table](../README.md#kgfm-link-prediction) covers **test-set entity prediction without fine-tuning** using the
authors' released checkpoints. The inference graph contains only the target
dataset's training triples and their inverse edges. Validation and test triples
are excluded from the inference graph. We rank all entities for both head and
tail prediction, filtering other known positives from train, validation, and
test while retaining the query's correct answer.

All three checkpoints below were pretrained on FB15k-237, WN18RR, and CoDEx
Medium. Results on datasets outside that mixture measure zero-shot transfer;
rows marked **Yes** measure frozen inference on a graph used in pretraining.
The pretraining mixtures are documented in the official
[ULTRA](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/config/transductive/pretrain_3g.yaml),
[TRIX](https://github.com/yuchengz99/TRIX/blob/7596e14eefefe89e61396205a0550172cadeddb0/config/pretrain_entity.yaml),
and [Flock](https://github.com/jw9730/flock/blob/f35103d25a78bdf4075de5c673a51de4979aa4d7/src_entity/config/pretrain_3g.yaml) configurations.

**Measured in DICE:** populated rows are completed evaluations of the full test
set. `—` means the evaluation has not completed, not zero. Exact metrics,
checkpoint/split hashes, source hashes, hardware, and configurations are saved
locally in each run's output directory under `Experiments/`. Generated JSON
reports are excluded from version control. The existing link
prediction benchmark results remain in their separate section.

**Bold** marks the best completed result for each dataset and metric, including
ties at the displayed precision.

## Checkpoints and evaluation settings

For timings against the official implementations, see the
[KGFM inference speed comparison](kgfm_inference.md). Local result
records retain the code and settings with which they were measured.

| Model | Official checkpoint | DICE guide |
|---|---|---|
| ULTRA-3g | [ultra_3g.pth](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/ckpts/ultra_3g.pth) | [ULTRA](ultra.md) |
| TRIX | [entity_prediction.pth](https://github.com/yuchengz99/TRIX/blob/7596e14eefefe89e61396205a0550172cadeddb0/entity_prediction.pth) | [TRIX](trix.md) |
| Flock | [flock_entity.pth](https://github.com/jw9730/flock/blob/f35103d25a78bdf4075de5c673a51de4979aa4d7/checkpoints/flock_entity.pth) | [Flock](flock.md) |

The [benchmark runner](../benchmarks/kgfm_zero_shot.py) uses DICE's existing dataset
loader and filtered ranking function, with an explicit inference device and no
training or optimizer. It verifies that checkpoint weights remain unchanged and
saves progress after each evaluation batch. Repeating an identical invocation
resumes it; a completed run returns its saved result. Use a new output directory
for a fresh run or changed settings.

The initial ULTRA/TRIX UMLS and Countries runs used an AMD Ryzen 7 7800X3D CPU;
Flock and the larger graph runs use an NVIDIA GeForce RTX 4070 Ti SUPER GPU.
Flock samples walks on CPU and executes its neural network on GPU. Runs use four
PyTorch CPU threads, float32, and 128 test triples per evaluation batch (8 for
the initial ULTRA/TRIX UMLS configuration). Query
batch sizes are recorded per run to bound memory use. GPU runner invocations
disable TF32. Some GPU runs overlap, so recorded wall times are operational
measurements rather than a controlled comparison of model speed.

Download the checkpoints using the links above, then run, for example:

```bash
python benchmarks/kgfm_zero_shot.py --model Flock --dataset UMLS \
  --device cuda:0 --query-batch-size 1 --batch-size 128 --threads 4 \
  --walk-num 128 --test-samples 1 --seed 42 \
  --output Experiments/kgfm-zero-shot/UMLS/Flock

python benchmarks/kgfm_zero_shot.py --model ULTRA --dataset WN18RR \
  --device cuda:0 --query-batch-size 8 --batch-size 128 --threads 4 \
  --output Experiments/kgfm-zero-shot/WN18RR/ULTRA

python benchmarks/kgfm_zero_shot.py --model TRIX --dataset WN18RR \
  --device cuda:0 --query-batch-size 4 --batch-size 128 --threads 4 \
  --output Experiments/kgfm-zero-shot/WN18RR/TRIX
```

Use `--device cpu` for CPU evaluation. Keep the original `train.txt`, `valid.txt`,
and `test.txt` splits. Evaluation ranks all entities with the existing DICE
evaluator. The original runs used `sort` tie handling; the current README reports
the [pessimistic rerun](#pessimistic-rerun-2026-09-10). The earlier CLI
runs used `--num_epochs 0 --scoring_technique NegSample --eval_model test`;
`NegSample` selects head-and-tail evaluation, which still ranks all entities.

Flock's initial setting uses a single sampling seed and one prediction per query,
with 128 base walks of length 128 per refinement and six refinements. This is a
single-run result, not a mean across seeds or a reproduction of the authors'
dataset-specific inference ensembles. Record query ordering and batching with
the sampling settings, since they affect the walks drawn. See the
[Flock sampling notes](flock.md#checkpoints-and-zero-shot-evaluation).

The runner saves `configuration.json`, `command.txt`, `progress.json`,
`eval_report.json`, and `result.json` alongside indexed splits and vocabularies.
The two initial CLI UMLS runs retain their original configurations and reports;
their saved model tensors were checked against the released checkpoint tensors.
To add a result, retain the completed run in the ignored `Experiments/`
directory and update the corresponding README row, including bold highlights
for the best metrics. Do not commit generated benchmark JSON files.

## Prediction tie strategies

Set `--eval_tie_policy` when evaluating a model. Ties mean **exactly equal
prediction scores**, before metric rounding. Known positive alternatives are
filtered first; the query's correct answer remains a candidate.

| Policy | Target rank within a tied group | Example: three candidates tied for first |
|---|---|---|
| `sort` (default) | Existing DICE `torch.sort` position | Implementation-dependent; not uniform random |
| `optimistic` | Best possible rank | 1 |
| `random` | Uniformly sampled integer rank | 1, 2, or 3 with equal probability |
| `pessimistic` | Worst possible rank | 3 |

For example, add `--eval_tie_policy random --eval_tie_seed 42` to a DICE
command. In Python, set `args.eval_tie_policy = "pessimistic"` on a
`dicee.config.Namespace`. The options apply to head/tail, reciprocal, relation,
BPE, and ensemble evaluation. Loaded models also support
`kge.eval_lp_performance(triples, tie_policy="random", tie_seed=42)`;
without overrides they use the saved configuration. Standalone functions such
as `dicee.evaluation.link_prediction.evaluate_lp` accept `tie_policy` and
`tie_seed` keywords (defaults: `"sort"` and `0`).

Random evaluation uses its own CPU generator and does not consume the model's
or training's random stream. `eval_tie_seed` defaults to `random_seed`; each
split evaluation restarts that stream. Reproducing random ranks requires the
same scores, filters, seed, and query order. MRR and Hits are calculated from
the sampled integer ranks, so random ties are not average-rank evaluation.

The KGFM runner accepts `--tie-policy` and `--tie-seed` and saves the policy,
seed, and random generator state for resumable runs. Use a new output directory
when changing a run's settings. The README table uses `pessimistic`; the
comparison below preserves the original `sort` MRR values.

## Pessimistic rerun (2026-09-10)

The README table reports a full rerun of all 21 model/dataset pairs with
`--tie-policy pessimistic`. A target receives the worst position among exactly
equal scores after filtering other known positives. The original `sort` MRR
values are retained below for comparison.

The rerun uses the same released checkpoint files and dataset splits, verified
by SHA-256 against the original runs. All weights remain unchanged. Runs use
PyTorch 2.5.1, float32 with TF32 disabled, four CPU threads, and the optimized
inference implementation. CUDA runs use Triton 3.1.0 through the `auto` backend,
including fused message passing and ranking where applicable. Inference caches,
query reuse for ULTRA/TRIX, and Flock's compact state, scripted sampler, packed
transfers, and prefetching are enabled; optional dense-update compilation is off.

The device and query microbatch size match each original run:

| Dataset | ULTRA-3g device / query batch | TRIX device / query batch | Flock device / query batch |
|---|---|---|---|
| Countries-S1 / S2 / S3 | CPU / 8 | CPU / 8 | CUDA / 1 |
| UMLS | CPU / 8 | CPU / 8 | CUDA / 1 |
| WN18RR | CUDA / 8 | CUDA / 4 | CUDA / 1 |
| YAGO3-10 | CUDA / 2 | CUDA / 2 | CUDA / 1 |
| FB15k-237 | CUDA / 4 | CUDA / 2 | CUDA / 1 |

Evaluation batches contain 128 test triples and use seed 42, except the initial
ULTRA/TRIX UMLS configuration, which uses batches of 8 and seed 1. Flock retains
128 base walks, length 128, six refinements, and one prediction per query.
Keeping Flock's batch boundaries preserves its sampled walks.

For example:

```bash
python benchmarks/kgfm_zero_shot.py --model Flock --dataset WN18RR \
  --device cuda:0 --query-batch-size 1 --batch-size 128 --threads 4 \
  --walk-num 128 --test-samples 1 --seed 42 --tie-policy pessimistic \
  --output Experiments/kgfm-pessimistic-20260910/WN18RR/Flock
```

The following differences compare the rerun against the original README run.
The original run predates the inference optimizations, so these differences
are not a strict isolation of tie handling: floating-point reduction order can
also affect close scores. Each local record retains its exact source hashes
and settings. Values are rounded to six decimals in this comparison and four
in the README. Differences are calculated before rounding.

| Dataset | Model | Original sort MRR | Pessimistic MRR | Difference |
|---|---|---:|---:|---:|
| YAGO3-10 | ULTRA-3g | 0.479963 | 0.479930 | -0.000033 |
| YAGO3-10 | TRIX | 0.409400 | 0.409393 | -0.000007 |
| YAGO3-10 | Flock | 0.399830 | 0.399828 | -0.000001 |
| FB15k-237 | ULTRA-3g | 0.369266 | 0.369259 | -0.000007 |
| FB15k-237 | TRIX | 0.361815 | 0.361814 | -0.000001 |
| FB15k-237 | Flock | 0.311554 | 0.311553 | -0.000001 |
| WN18RR | ULTRA-3g | 0.370754 | 0.369149 | -0.001604 |
| WN18RR | TRIX | 0.508263 | 0.506535 | -0.001728 |
| WN18RR | Flock | 0.530301 | 0.530276 | -0.000025 |
| UMLS | ULTRA-3g | 0.696007 | 0.696007 | 0.000000 |
| UMLS | TRIX | 0.725622 | 0.725622 | 0.000000 |
| UMLS | Flock | 0.776795 | 0.776795 | 0.000000 |
| Countries-S1 | ULTRA-3g | 0.937500 | 0.937500 | 0.000000 |
| Countries-S1 | TRIX | 0.927083 | 0.927083 | 0.000000 |
| Countries-S1 | Flock | 0.927083 | 0.927083 | 0.000000 |
| Countries-S2 | ULTRA-3g | 0.871528 | 0.871528 | 0.000000 |
| Countries-S2 | TRIX | 0.885417 | 0.885417 | 0.000000 |
| Countries-S2 | Flock | 0.885417 | 0.885417 | 0.000000 |
| Countries-S3 | ULTRA-3g | 0.235401 | 0.235401 | 0.000000 |
| Countries-S3 | TRIX | 0.362490 | 0.362490 | 0.000000 |
| Countries-S3 | Flock | 0.253328 | 0.253328 | 0.000000 |

All result records, progress files, and indexed datasets remain in the ignored
`Experiments/kgfm-pessimistic-20260910/` directory. Generated benchmark JSON
files are not committed.
