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
in the [result records](../benchmarks/results/kgfm-zero-shot). The existing link
prediction benchmark results remain in their separate section.

**Bold** marks the best completed result for each dataset and metric, including
ties at the displayed precision. Highlights update as the remaining runs finish.

## Checkpoints and evaluation settings

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
PyTorch CPU threads, float32, and 128 test triples per evaluation batch. Query
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
evaluator, including its sort-position handling of tied scores. The earlier CLI
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
The [result publisher](../benchmarks/update_kgfm_results.py) copies completed records
into the repository and fills only their KGFM table rows:

```bash
python benchmarks/update_kgfm_results.py --runs-dir Experiments/kgfm-zero-shot
```

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
when changing a run's settings. The README table retains its original `sort`
policy; its publisher rejects results using a different tie policy.
