# KGFM inference speed comparison

Measured on 2026-09-10 against the authors’ pinned official implementations:
[ULTRA](https://github.com/DeepGraphLearning/ULTRA/tree/427966ad8ed60420eef034063d44f3153addff90),
[TRIX](https://github.com/yuchengz99/TRIX/tree/7596e14eefefe89e61396205a0550172cadeddb0), and
[Flock](https://github.com/jw9730/flock/tree/f35103d25a78bdf4075de5c673a51de4979aa4d7).
DICE model code: [c567fc5e](https://github.com/dice-group/dice-embeddings/tree/c567fc5e5273a1501e8af01db9ae26c9956e9eb0/dicee/models).

Warm inference times are **milliseconds per directional query**, scoring every entity.
Speedup = official time / DICE time; values below 1 mean DICE is slower.
Memory is peak allocated CUDA MiB, official → DICE.

| Dataset | Model | Query batch | Official ms/query | DICE ms/query | Speedup | CUDA MiB |
|---|---|---:|---:|---:|---:|---:|
| FB15k-237 | ULTRA-3g | 4 | 6.92 | 1.89 | 3.66× | 202 → 202 |
| FB15k-237 | TRIX | 2 | 121.88 | 3.84 | 31.75× | 421 → 427 |
| FB15k-237 | Flock* | 1 | 216.53 | 82.09 | 2.64× | 453 → 462 |
| WN18RR | ULTRA-3g | 8 | 4.28 | 3.72 | 1.15× | 869 → 541 |
| WN18RR | TRIX† | 4 | 76.35 | 4.08 | 18.71× | 289 → 244 |
| WN18RR | Flock* | 1 | 83.51 | 45.52 | 1.83× | 458 → 459 |
| YAGO3-10 | ULTRA-3g† | 2 | 128.53 | 17.28 | 7.44× | 856 → 631 |
| YAGO3-10 | TRIX† | 2 | 747.51 | 21.13 | 35.38× | 710 → 689 |
| YAGO3-10 | Flock* | 1 | 564.18 | 65.52 | 8.61× | 548 → 591 |

\* Flock uses independently sampled walks; see the identical-walk comparison below.
† Score validation required the untimed float64 cross-check described below.

## What was measured

Each pair uses the same [released checkpoint](kgfm_benchmarks.md#checkpoints-and-evaluation-settings), complete training graph plus
inverse edges, entity vocabulary, selected queries, and query batch size.
The workload contains 128 randomly selected test triples for ULTRA/TRIX and
32 for Flock (selection seed 42), with both head and tail prediction.
These are sampled inference workloads, not full-test evaluation timings or
a search for each implementation’s best batch size.

Hardware: RTX 4070 Ti SUPER (16 GiB), Ryzen 7 7800X3D. Both implementations
use PyTorch 2.5.1 with CUDA 12.4, float32, TF32 disabled, and four PyTorch
CPU threads. The official Flock walker retains its native thread pool.
Official ULTRA/TRIX use their compiled CUDA `rspmm` kernels.
Runs execute sequentially in fresh processes, with two warmups and five
timed repetitions on the same graph and queries; DICE’s inference caches
are warm. The table reports medians. CUDA is synchronized around
each timing. Other CUDA compute jobs are rejected; the desktop compositor
remains active and is recorded.

Timings include candidate construction, model inference, and Flock walk
sampling/transfers. Dataset loading, graph construction, and ranking are
excluded. Checkpoint/split/source hashes, timing samples, first-forward
latency, and memory measurements are saved locally by the runner.

ULTRA/TRIX checks use `atol=2e-4, rtol=2e-4` for all-entity scores. When
float32 scores differ beyond that tolerance, the runner additionally checks
the official model in float64. It accepts the comparison only if DICE is
within the original tolerance of float64 and has lower mean error. For all
six ULTRA/TRIX pairs, float32 pessimistic hit rates match and MRR differs
by at most `1e-6`. The float64 diagnostic is not timed; hardcoded float32
query factories are promoted alongside the model weights and boundaries.

## Flock on identical walks

Flock uses 128 base walks, length 128, six refinements, and one prediction
per query. Its native samplers use different random generators, so matching
seeds does not produce identical predictions. The main table measures each
complete implementation at the same budget; it does not establish equal
prediction quality.

For a direct neural comparison, both models replay the official sampler’s
exact records for eight of the selected test triples. Records are already
on the GPU, so these timings exclude sampling and transfers.

| Dataset | Official ms/query | DICE ms/query | Neural speedup |
|---|---:|---:|---:|
| FB15k-237 | 35.05 | 28.46 | 1.23× |
| WN18RR | 39.99 | 32.64 | 1.23× |
| YAGO3-10 | 105.48 | 52.60 | 2.01× |

Identical-walk scores pass the same tolerance (maximum absolute error
`1.62e-05`); pessimistic filtered MRR and hit rates pass the checks above.

YAGO3-10 Flock had higher timing variance: official native repeats ranged
394–583 ms/query (DICE 65–67), and replay repeats ranged 76–170 ms/query
(DICE 47–54). Treat its reported median speedups as approximate.

## Reproduce

Use the pinned official checkout with PyTorch 2.5.1/CUDA 12.4, PyG 2.4.0,
a matching GPU torch-scatter 2.1.2 wheel, easydict, ninja, and a CUDA compiler.
Flock also needs its checkout’s compiled `graph-walker` package. Generate indexed
splits with the [zero-shot runner](kgfm_benchmarks.md) first. Both workers
use the interpreter running the benchmark. For example:

```bash
python benchmarks/kgfm_upstream.py --model ULTRA \
  --upstream-root /path/to/ULTRA \
  --indexed-data Experiments/kgfm-pessimistic-20260910/FB15k-237/ULTRA \
  --queries 128 --query-batch-size 4 --repeats 5 --warmups 2 \
  --output Experiments/official-comparison/FB15k-237/ULTRA
```

Use the table’s batch sizes for the other pairs. For Flock, use `--queries 32
--query-batch-size 1 --walk-num 128 --replay-queries 8`. All generated JSON,
score tensors, and walk records land in the `Experiments/` directories.
