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

## KG-ICL

Measured on 2026-10-05 against the [pinned official KG-ICL](https://github.com/nju-websoft/KG-ICL/tree/6a3166e347ae468acdfb30a70a2cf3608b66b8f1)
with the [bug-fix patch](../tests/fixtures/kgicl/upstream-fixes.patch) applied (see the [KG-ICL guide](kgicl.md)),
on different hardware than the table above: an RTX 5070 Laptop GPU (8 GiB) and
an Intel Core i7-14700HX.

| Dataset | Query batch | Official ms/query | DICE ms/query | Speedup | CUDA MiB |
|---|---:|---:|---:|---:|---:|
| FB15k-237 | 8 | 68.25 | 1.37 | 49.76× | 3096 → 177 |
| WN18RR | 16 | 3.63 | 0.82 | 4.43× | 924 → 326 |
| YAGO3-10† | 2 | 277.71 | 7.80 | 35.58× | 3382 → 373 |

† Validated against the official model in float64, see below.

The workload matches the table above: 128 test triples (seed 42), head and
tail prediction, and the complete training graph plus inverse edges. Both
implementations read the same prompt graphs, DICE's seeded samples written in
the official case format, and apply the official answer-distance mask. The
official code runs in its own environment (Python 3.9, PyTorch 2.7.1 with CUDA
12.8, torch-scatter 2.1.2); DICE uses PyTorch 2.9.1 with CUDA 12.8 and Triton
3.5.1. Both use float32 with TF32 disabled and four CPU threads, with two
warmups and five timed repetitions (medians), one process at a time. The
driver's GPU process listing was unavailable on this machine
(`--allow-unchecked-gpu`); no other compute job ran. Query batches are the
largest the official code fits in 8 GiB; it ran out of GPU memory on YAGO3-10
with four queries.

Prompt graph extraction is preprocessing for both implementations. The
official code assembles and encodes the prompt graphs of every batch, while
DICE encodes each relation's prompts once and caches them, as ULTRA caches
relation representations. Without that cache, DICE takes 8.97 ms/query on
FB15k-237. DICE gains from larger batches: with 16 queries per pass it takes
1.19 ms/query on FB15k-237 and 4.82 ms/query on YAGO3-10, with scores
bitwise identical to the table's runs.

Scores agree within `atol=2e-4, rtol=2e-4` on FB15k-237 (maximum absolute error
`1.84e-4`) and WN18RR (`7.25e-5`), and filtered MRR and hit rates match (MRR
within `5e-8`). On YAGO3-10, float32 scores differ by up to `2.7e-3`: one entity
has 61,044 incoming edges, and long float32 sums depend on their order.
Against the official model in float64, run with one query per batch because two
do not fit in float64, DICE's maximum error is `2.65e-5` (mean `6.9e-7`) and
the official float32 code's is `2.73e-3` (mean `9.7e-5`). DICE's MRR equals the
float64 value; hit rates match.

### Query answering rows

Warm atomic rows in the query evaluator under deterministic algorithms, as in
complex query answering: batches of 8 random head–relation pairs scored against
all entities, on the same laptop GPU, with the KG-ICL recipes' 512 MiB cache of
per-layer relation tables (`projection_cache_mb`):

| Inference graph | Entities | Relations with inverses | Edges | ms/row | First pass over all relations |
|---|---:|---:|---:|---:|---:|
| FB15k237+H | 14,505 | 474 | 544,230 | 1.01 | 16.9 s |
| NELL995+H | 63,361 | 400 | 228,426 | 1.71 | 8.4 s |
| ICEWS18+H | 20,840 | 500 | 426,608 | 1.04 | 20.0 s |
| UltraQuery FB15k237 | 14,505 | 474 | 544,230 | 0.98 | 14.1 s |
| UltraQuery NELL995 | 63,361 | 400 | 228,426 | 1.68 | 7.9 s |
| UltraQuery inductive FB15k237 (550) | 13,438 | 312 | 136,560 | 0.54 | 5.7 s |
| UltraQuery WikiTopics art | 10,000 | 129 | 54,524 | 0.55 | 2.7 s |

The first pass samples and encodes the prompt graphs of every relation. With the
default 64 MiB table cache, graphs with hundreds of relations recompute tables:
FB15k237+H then takes 2.55 ms/row.

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

KG-ICL's official worker runs in its own environment, passed with
`--upstream-python`: Python 3.9 with PyTorch, a matching torch-scatter build,
NumPy, SciPy and NetworkX. The runner applies the bug-fix patch to a temporary
copy of the pinned checkout; `--upstream-variant unpatched` times the released
code. Where `nvidia-smi` cannot list GPU processes, `--allow-unchecked-gpu`
records that instead of aborting; make sure no other GPU work runs.

```bash
python benchmarks/kgfm_upstream.py --model KGICL \
  --upstream-root /path/to/KG-ICL --upstream-python /path/to/kgicl-env/bin/python \
  --indexed-data Experiments/kgfm-kgicl-20261005/FB15k-237/KGICL \
  --queries 128 --query-batch-size 8 --repeats 5 --warmups 2 \
  --output Experiments/official-comparison/FB15k-237/KGICL
```
