# KGFM evaluation protocol

The [README results](../README.md#kgfm-link-prediction) report full-test entity
prediction with released checkpoints and **no fine-tuning**:

- Inference graph: training triples plus inverse edges.
- Candidates: all entities in the combined train/validation/test vocabulary.
- Filtering: other known positives from all three splits, retaining the target.
- Queries: both head and tail prediction for every test triple.
- Ties: **pessimistic**, the worst rank among exactly equal scores after filtering.

**Bold** marks the best metric per dataset, including ties at displayed precision.
**Yes** marks a target graph used in pretraining. ULTRA-3g, TRIX, and Flock were
pretrained on FB15k-237, WN18RR, and CoDEx Medium, as specified by the official
[ULTRA](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/config/transductive/pretrain_3g.yaml),
[TRIX](https://github.com/yuchengz99/TRIX/blob/7596e14eefefe89e61396205a0550172cadeddb0/config/pretrain_entity.yaml),
and [Flock](https://github.com/jw9730/flock/blob/f35103d25a78bdf4075de5c673a51de4979aa4d7/src_entity/config/pretrain_3g.yaml)
configurations. ULTRA-4g additionally uses NELL995
([configuration](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/config/transductive/pretrain_4g.yaml));
NELL variants are marked **Related**, since exact split overlap is unverified.
ULTRA-50g's [model card](https://huggingface.co/mgalkin/ultra_50g) reports 50 training
graphs without a complete manifest. Its target membership is **Unknown** here;
these runs must not be assumed zero-shot. **No** indicates transfer outside a
checkpoint's documented training mixture.

## Checkpoints and evaluation settings

| Model | Official checkpoint | Local path | DICE guide |
|---|---|---|---|
| ULTRA-3g | [ultra_3g.pth](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/ckpts/ultra_3g.pth) | `checkpoints/ultra_3g.pth` | [ULTRA](ultra.md) |
| ULTRA-4g | [ultra_4g.pth](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/ckpts/ultra_4g.pth) | `checkpoints/ultra_4g.pth` | [ULTRA](ultra.md) |
| ULTRA-50g | [ultra_50g.pth](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/ckpts/ultra_50g.pth) | `checkpoints/ultra_50g.pth` | [ULTRA](ultra.md) |
| TRIX | [entity_prediction.pth](https://github.com/yuchengz99/TRIX/blob/7596e14eefefe89e61396205a0550172cadeddb0/entity_prediction.pth) | `checkpoints/trix/entity_prediction.pth` | [TRIX](trix.md) |
| Flock | [flock_entity.pth](https://github.com/jw9730/flock/blob/f35103d25a78bdf4075de5c673a51de4979aa4d7/checkpoints/flock_entity.pth) | `checkpoints/flock/flock_entity.pth` | [Flock](flock.md) |

Runs use float32, TF32 disabled on GPU, four CPU threads, 128 test triples per
batch, and seed 42. ULTRA/TRIX on UMLS use batches of 8 and seed 1.
Flock uses 128 base walks of length 128, six refinements, and one prediction per
query: a single-seed result, not an ensemble. Its query order and batch boundaries
affect sampling; see the [sampling notes](flock.md#checkpoints-and-zero-shot-evaluation).

| Dataset | ULTRA device / query batch | TRIX device / query batch | Flock device / query batch |
|---|---|---|---|
| Countries-S1 / S2 / S3, UMLS | CPU / 8 | CPU / 8 | CUDA / 1 |
| WN18RR | CUDA / 8 | CUDA / 4 | CUDA / 1 |
| YAGO3-10 | CUDA / 2 | CUDA / 2 | CUDA / 1 |
| FB15k-237 | CUDA / 4 | CUDA / 2 | CUDA / 1 |
| KINSHIP | CUDA / 8 | CUDA / 2 | CUDA / 1 |
| NELL-995-h25 / h50 / h75 / h100 | CUDA / 4 | CUDA / 2 | CUDA / 1 |

The September 10 runs used PyTorch 2.5.1, Triton 3.1.0, an AMD Ryzen 7 7800X3D
CPU, and an RTX 4070 Ti SUPER GPU. KINSHIP, all four NELL variants, and the ULTRA-4g/50g runs (September 21) use
PyTorch 2.9.1+cu128, Triton 3.5.1, and an RTX 5070 Laptop GPU. These are accuracy
results; controlled timings are in the [inference speed comparison](kgfm_inference.md).

## KINSHIP and NELL variants

All datasets retain their original splits. Pretraining exposure depends on the checkpoint above.

| Dataset | Entities | Relations | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| KINSHIP | 104 | 25 | 8,544 | 1,068 | 1,074 |
| NELL-995-h25 | 70,145 | 172 | 122,618 | 9,194 | 9,187 |
| NELL-995-h50 | 34,667 | 86 | 72,767 | 5,440 | 5,393 |
| NELL-995-h75 | 28,085 | 57 | 59,135 | 4,441 | 4,389 |
| NELL-995-h100 | 22,411 | 43 | 50,314 | 3,763 | 3,746 |

**Leakage:** h25/h50/h75 contain respectively **52/33/9 distinct test facts in
training**, 50/30/9 validation facts in training, and one validation/test overlap
each. These runs explicitly use `--allow-test-overlap`, preserve all facts, and
are marked † in the README; their scores are not strictly held-out estimates.
The runner rejects train/test overlap by default and records overlap counts
when allowed. NELL-995-h100 has no split overlap. The suffix denotes the
percentage of hierarchical relations ([dataset comparison](https://cdn.aaai.org/ojs/17095/17095-13-20589-1-2-20210518.pdf));
these results should not be labeled generic NELL-995.

## Reproduction

Place the checkpoints at the paths above and datasets under `KGs/<dataset>/`
with `train.txt`, `valid.txt`, and `test.txt`. For example:

```bash
for dataset in KINSHIP NELL-995-h100; do
  for model in ULTRA TRIX Flock; do
    query_batch=1
    if [ "$model" = ULTRA ]; then
      query_batch=4
      if [ "$dataset" = KINSHIP ]; then query_batch=8; fi
    elif [ "$model" = TRIX ]; then
      query_batch=2
    fi
    python benchmarks/kgfm_zero_shot.py \
      --model "$model" --dataset "$dataset" --device cuda:0 \
      --query-batch-size "$query_batch" --batch-size 128 --threads 4 \
      --walk-num 128 --test-samples 1 --seed 42 --tie-policy pessimistic \
      --output "Experiments/kgfm-pessimistic-20260921/$dataset/$model" || exit $?
  done
done
```

Select ULTRA weights with `--ultra-checkpoint 3g|4g|50g` (default `3g`).
The 4g/50g runs cover every dataset in the table using its ULTRA settings above:

```bash
python benchmarks/kgfm_zero_shot.py --model ULTRA --ultra-checkpoint 4g \
  --dataset NELL-995-h50 --device cuda:0 --query-batch-size 4 \
  --batch-size 128 --threads 4 --seed 42 --tie-policy pessimistic \
  --allow-test-overlap \
  --output Experiments/kgfm-ultra-checkpoints-20260921/NELL-995-h50/ULTRA-4g
```

Use a separate output per checkpoint. For other datasets, use the settings above. The
[runner](../benchmarks/kgfm_zero_shot.py) verifies unchanged weights and saves
progress after each batch. An identical invocation resumes; changed settings or
source require a new output directory. Local reports include metrics, exact
configuration, checkpoint/split/source SHA-256 hashes, and indexed data.

For h25/h50/h75, use the same model settings with `--allow-test-overlap` and
`--output "Experiments/kgfm-nell-original-20260921/$dataset/$model"`.

Reports remain in ignored `Experiments/kgfm-pessimistic-20260910/`,
`Experiments/kgfm-pessimistic-20260921/`, and
`Experiments/kgfm-nell-original-20260921/`, and
`Experiments/kgfm-ultra-checkpoints-20260921/` directories. Add only completed results
to the README; do not commit generated reports.

## Other tie policies

The runner accepts `--tie-policy sort|optimistic|random|pessimistic` and
`--tie-seed` (defaults to `--seed`). `sort` uses the existing sorting position;
`optimistic` uses the best tied rank; `random` samples a uniform integer rank
within the tie using an independent random stream. The README uses `pessimistic`.
General DICE evaluation exposes these as `--eval_tie_policy` and `--eval_tie_seed`;
loaded models accept `kge.eval_lp_performance(triples, tie_policy="pessimistic")`.
