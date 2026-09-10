# KGFM inference performance

ULTRA and TRIX automatically use fused CUDA message passing when Triton is
available. Float32 remains the default; explicitly selected float16/bfloat16
inputs use float32 accumulation, including on the tiled atomic path. CPU,
float64, and autograd use the existing PyTorch message-passing path.
The pretrained checkpoint keys and training objectives are unchanged. Model
inference requires `.eval()` and either frozen parameters (as in loaded `KGE`
models), `torch.no_grad()`, or `torch.inference_mode()`. An unfrozen model's
`.eval()` alone intentionally permits autograd.

## Optimizations

- **Fused message passing.** CSR kernels gather, multiply, and sum directly into
  node updates, avoiding the `[queries, edges, features]` message tensor. Layers
  share one graph layout. Dense graphs use bounded edge tiles to balance hubs;
  `torch.use_deterministic_algorithms(True)` selects a row-owned reduction
  without floating-point atomics. The output-row/input-column convention and
  boundary message match the released checkpoints.
- **ULTRA relation cache.** Query-relation representations are cached lazily in
  an LRU, bounded to 64 MiB by default. Graph, parameter-version, device, dtype,
  and backend changes invalidate cached representations. Autograd and autocast
  bypass this cache. A second bounded LRU reuses the six entity layers' relation
  projections. It defaults to another 64 MiB. Caching every projection for
  FB15k-237 would require about 165 MiB; the default retains only recent entries.
- **TRIX initial-state cache.** The first relation step precedes entity feedback
  and depends only on the relation seed. Its outputs are cached in a bounded
  64 MiB LRU. On a miss, identical initial entity labels are projected once per
  query and broadcast after the MLP. Subsequent feedback stays query-specific.
- **Dense layer operations.** Float32 CUDA normalization, ReLU, and residual
  addition share a fused kernel. Query-conditioned scoring and TRIX feedback
  project the shared query once instead of once per candidate. All-entity
  scoring skips identity gathers; subset scoring gathers before concatenation.
  The relation-prediction TRIX model also uses the shared-query scoring path.
- **Whole-split query reuse.** Deterministic ULTRA/TRIX evaluation scores each
  directional query once for all its targets. Only ranks or tie counts are
  retained, requiring memory proportional to test triples. The resumable runner
  shares this plan across evaluation batches and rebuilds it on restart.
- **Flock sampling.** Stable relation-edge lists are built once per graph. A
  single CPU worker prefetches one walk microbatch while CUDA scores the current
  one, using pinned transfers. Query order, sampling counts, and CPU generator
  consumption are preserved. The CPU transition loop is scripted while keeping
  the same Torch generator and draw shapes/order. Internal transfer buffers use
  the smallest safe integer widths and are restored to int64 on the GPU.
  Flock queries are never deduplicated.
- **Flock neural inference.** CUDA kernels fuse the seven record embedding
  lookups/additions and RMS normalization. Sparse walk coverage uses compact
  visited-entity state plus a default row; unvisited candidates share a single
  computed score. Only scalar scores are expanded back to candidate order. A
  coverage heuristic skips compaction when the record count is at least four
  times the graph's entity count. Training retains dense state and autograd.
- **Ranking.** Explicit optimistic, pessimistic, and random policies use batched
  comparisons on the score device and transfer only rank bounds to CPU. Random
  ties consume the independent generator in original tail/head query order.
  Triton computes tiled counts and subtracts sparse exclusions without dense
  masks or duplicating score rows for queries with multiple evaluation targets.
  Legacy `sort` keeps per-vector CPU sorting, preserving its existing tie rule.
- **Head prediction.** Inference converts relation IDs directly, avoiding the
  expanded `[batch, all entities, 3]` tensor of candidate triples.
- **Optional compilation.** `graph_inference_compile=True` compiles the dense
  convolution update with `torch.compile(mode='reduce-overhead')`, permitting
  CUDA Graph replay. Graph layouts, caches, and input validation stay outside
  that region. Outputs are copied out of replay-owned storage to protect live
  states across layers and calls. This is opt-in because compilation has cold
  cost, retains device memory, and can be slower than the default fused kernels.

Float32 reduction order may differ between fused and unfused kernels. Check
score tolerances and filtered metrics when comparing implementations. The
optimizations do not enable TF32, mixed precision, candidate pruning, fewer
walks, or smaller ensembles.
Cache tokens include graph/parameter versions, backend, device, dtype,
deterministic mode, compilation settings, and matmul precision. Autocast bypasses
representation caches. Changing GEMM shapes can also change TF32 rounding;
the benchmark explicitly disables TF32 for its float32 comparisons.

## Controls

For the DICE CLI or corresponding model argument dictionary:

| Setting | Default | Purpose |
|---|---|---|
| `graph_inference_backend` | `auto` | `auto`, `torch`, or `triton`; selects the supported no-grad CUDA convolution path |
| `graph_relation_cache_mb` | `64` | ULTRA relation / TRIX initial-state cache limit; `0` disables caching |
| `graph_projection_cache_mb` | `64` | Additional ULTRA layer-projection cache limit; `0` disables it |
| `graph_inference_compile` | `False` | Compile dense ULTRA/TRIX convolution updates with CUDA Graph support |
| `flock_prefetch_walks` | `True` | Enable one-microbatch CPU sampling lookahead on CUDA |
| `flock_compact_state` | `True` | Compact visited-entity state when the coverage heuristic permits |
| `flock_compile_sampler` | `True` | Script the CPU transition loop without changing random draws |
| `flock_pack_walks` | `True` | Reduce integer widths in internal pinned transfer buffers |

The CLI accepts `--no-flock_prefetch_walks` to disable prefetch. Query microbatch
sizes remain controlled by `ultra_query_batch_size`, `trix_query_batch_size`, or
`flock_query_batch_size`. Increase them only after measuring throughput and
peak memory on the target graph. Flock batching changes its sampled walks.

The zero-shot runner has matching `--inference-backend`, `--relation-cache-mb`,
`--projection-cache-mb`, `--[no-]compile-inference`, `--[no-]compact-state`,
`--[no-]compile-sampler`, `--[no-]pack-walks`, `--[no-]prefetch-walks`, and
`--[no-]reuse-queries` options. Standalone
`evaluate_lp(..., reuse_graph_queries=False)` disables query reuse. Use a new
output directory after changing code or settings; existing benchmark records
remain records of their original implementation.

## Reproducible comparison

Run `benchmarks/kgfm_inference.py` against indexed splits saved by the zero-shot
runner. To compare with an earlier implementation, supply a checkout containing
its `dicee/` package:

```bash
python benchmarks/kgfm_inference.py \
  --indexed-data Experiments/kgfm-zero-shot-20260909/FB15k-237/ULTRA \
  --reference-root /path/to/reference-checkout \
  --model ULTRA --queries 128 --query-batch-size 4 \
  --repeats 3 --device cuda:0 \
  --output Experiments/inference-comparison/FB15k-237-ULTRA
```

Reference and optimized implementations run sequentially in separate processes.
The harness records checkpoint, split, and source hashes; float32/TF32 settings;
synchronized latency samples; warm throughput; peak allocated CUDA memory; cold
forward latency (including initial kernel/layout work); filtered metrics; and
all-candidate score differences. Scores are checked with
`atol=2e-4, rtol=2e-4`. If ULTRA/TRIX exceed that comparison against the old
float32 kernel, the harness checks the old implementation in float64. It accepts
that case only if optimized scores meet the original tolerance against float64,
have lower mean numerical error, retain the hit rates, and differ in MRR by at
most `1e-6`. Otherwise it raises an error. Both comparisons and all metric
differences remain explicit in the report. Change
`--tie-policy` to benchmark comparison-based ranking separately from legacy sort.

The comparison now requires both score parity and matching hit rates, with an
absolute MRR tolerance of `1e-6`, even when the float32 score comparison passes.
Use `--profile` to save a Chrome trace and operator summary alongside each
report. GPU process snapshots are recorded to expose concurrent workloads.
Explicit `--dtype float16` or `--dtype bfloat16` runs do not enable reduced
precision by default or establish parity against float32; validate task metrics
before adopting them.

For ULTRA/TRIX, sweep microbatches while checking scores and metrics against the
first size (the fastest passing size is reported, without changing defaults):

```bash
python benchmarks/kgfm_inference.py \
  --indexed-data Experiments/kgfm-zero-shot-20260909/FB15k-237/ULTRA \
  --model ULTRA --queries 128 --sweep-batch-sizes 1 2 4 8 16 \
  --repeats 5 --output Experiments/inference-batch-sweep
```

The sweep rejects Flock, whose sampler changes with microbatch size. Use fixed
walk records when comparing Flock neural batching. Legacy sort ties and the
explicit comparison policies are distinct evaluation conventions; switching
between them is not a score-preserving optimization.

Small-prefix measurements establish local improvements, not state-of-the-art
status. Such a claim requires controlled comparisons with official runtimes on
full workloads, including matched sampling budgets, hardware, precision, graph
construction, ranking rules, warmup, and software versions.

## Measured comparison: 2026-09-09

On an RTX 4070 Ti SUPER, compared with DICE commit `91244d3e`, with the original
microbatch sizes and float32 settings. These are medians of three repetitions
on **small test prefixes**, with all head/tail candidates and one warmup. They
are not full-test timings or comparisons with competing inference engines.
Local benchmark records include source hashes, precision validation, and timing
limitations; generated JSON reports are not committed. Early exploratory probes
with another GPU process present were excluded. Pre-lock comparisons cannot
exclude unobserved contention; some CPU validation overlapped the campaign.

| Dataset | Model | Test triples | Forward speedup | Evaluation speedup | Peak CUDA MiB, before → after |
|---|---|---:|---:|---:|---:|
| FB15k-237 | ULTRA | 32 | 8.60× | 8.57× | 1690 → 181 |
| FB15k-237 | TRIX | 32 | 13.86× | 13.62× | 1931 → 420 |
| FB15k-237 | Flock | 16 | 2.39× | 2.37× | 501 → 486 |
| WN18RR | ULTRA | 32 | 1.92× | 1.72× | 1277 → 750 |
| WN18RR | TRIX | 32 | 2.95× | 2.55× | 702 → 254 |
| WN18RR | Flock | 16 | 1.55× | 1.64× | 519 → 478 |
| YAGO3-10 | ULTRA | 16 | 4.55× | 4.04× | 3512 → 703 |
| YAGO3-10 | TRIX | 16 | 4.55× | 4.10× | 2085 → 642 |
| YAGO3-10 | Flock | 16 | 2.45× | 2.32× | 721 → 600 |

All sampled hit rates match. MRR matches exactly except TRIX on WN18RR,
where the absolute difference is below `9e-10`. TRIX on WN18RR and YAGO,
and ULTRA on YAGO, require the documented float64 validation: optimized
scores are closer to float64 than the original float32 scatter results.

Validation: 238 targeted tests passed, including CUDA kernels, frozen KGE
prediction, upstream fixtures, available official checkpoints, training
gradients, cache invalidation, identical Flock walks/RNG state, tied ranks,
and resumed evaluation. Two unavailable ULTRA checkpoints were deselected.
Ruff passes; mypy has the same 1,287 existing errors as the reference checkout
in the local environment.

## Additional inference pass: 2026-09-10

Compared with `aa91c26f`, using all candidates for small test prefixes on the
RTX 4070 Ti SUPER: float32, TF32 disabled, two warmups, and five timing repeats.
ULTRA/TRIX use query microbatches of four/two; Flock uses one, with unchanged
walk budgets and seed. Another GPU process was active throughout, so these
observed medians are provisional and do not establish isolated speedups.
The timing samples, process snapshots, scores, and hashes remain local under
`Experiments/kgfm-next-inference/validated/` and are excluded from Git.

| Dataset | Model | Test triples | Forward speedup | Evaluation speedup | Peak CUDA MiB, before → after |
|---|---|---:|---:|---:|---:|
| FB15k-237 | ULTRA | 16 | 1.37× | 1.45× | 178 → 149 |
| FB15k-237 | TRIX | 16 | 1.36× | 1.28× | 418 → 412 |
| FB15k-237 | Flock | 4 | 1.18× | 1.13× | 485 → 460 |
| WN18RR | ULTRA | 16 | 1.38× | 1.23× | 385 → 267 |
| WN18RR | TRIX | 16 | 1.45× | 1.26× | 159 → 139 |
| WN18RR | Flock | 4 | 1.23× | 1.27× | 474 → 449 |
| YAGO3-10 | ULTRA | 16 | 1.32× | 1.16× | 1243 → 884 |
| YAGO3-10 | TRIX | 16 | 1.46× | 1.33× | 642 → 582 |
| YAGO3-10 | Flock | 4 | 1.14× | 1.15× | 588 → 562 |

Every sampled MRR and hit rate matches. Maximum absolute score difference
across these nine comparisons is `2.20e-5`, within the existing tolerance.
Validation: 288 targeted tests passed, including all locally available official
checkpoints; two unavailable ULTRA checkpoints were deselected. The batch-sweep,
profiling, and compiled-inference CLI paths were also exercised. Ruff and
`git diff --check` pass. These tests do not establish full-test metric parity
for low-precision inference; FP16/BF16 remain explicit choices.
