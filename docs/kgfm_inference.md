# KGFM inference performance

ULTRA and TRIX automatically use fused float32 CUDA message passing when Triton
is available. CPU, other dtypes, and autograd use the existing PyTorch path.
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
  bypass this cache. TRIX's entity feedback remains query-specific.
- **Whole-split query reuse.** Deterministic ULTRA/TRIX evaluation scores each
  directional query once for all its targets. Only ranks or tie counts are
  retained, requiring memory proportional to test triples. The resumable runner
  shares this plan across evaluation batches and rebuilds it on restart.
- **Flock sampling.** Stable relation-edge lists are built once per graph. A
  single CPU worker prefetches one walk microbatch while CUDA scores the current
  one, using pinned transfers. Query order, sampling counts, and CPU generator
  consumption are preserved. Flock queries are never deduplicated.
- **Ranking.** Explicit optimistic, pessimistic, and random policies use batched
  comparisons on the score device and transfer only rank bounds to CPU. Random
  ties consume the independent generator in original tail/head query order.
  Legacy `sort` keeps per-vector CPU sorting, preserving its existing tie rule.
- **Head prediction.** Inference converts relation IDs directly, avoiding the
  expanded `[batch, all entities, 3]` tensor of candidate triples.

Float32 reduction order may differ between fused and unfused kernels. Check
score tolerances and filtered metrics when comparing implementations. The
optimizations do not enable TF32, mixed precision, candidate pruning, fewer
walks, or smaller ensembles.

## Controls

For the DICE CLI or corresponding model argument dictionary:

| Setting | Default | Purpose |
|---|---|---|
| `graph_inference_backend` | `auto` | `auto`, `torch`, or `triton`; selects the supported no-grad CUDA convolution path |
| `graph_relation_cache_mb` | `64` | ULTRA relation-cache limit; `0` disables caching |
| `flock_prefetch_walks` | `True` | Enable one-microbatch CPU sampling lookahead on CUDA |

The CLI accepts `--no-flock_prefetch_walks` to disable prefetch. Query microbatch
sizes remain controlled by `ultra_query_batch_size`, `trix_query_batch_size`, or
`flock_query_batch_size`. Increase them only after measuring throughput and
peak memory on the target graph. Flock batching changes its sampled walks.

The zero-shot runner has matching `--inference-backend`, `--relation-cache-mb`,
`--[no-]prefetch-walks`, and `--[no-]reuse-queries` options. Standalone
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

Small-prefix measurements establish local improvements, not state-of-the-art
status. Such a claim requires controlled comparisons with official runtimes on
full workloads, including matched sampling budgets, hardware, precision, graph
construction, ranking rules, warmup, and software versions.

## Measured comparison: 2026-09-09

On an RTX 4070 Ti SUPER, compared with DICE commit `91244d3e`, with the original
microbatch sizes and float32 settings. These are medians of three repetitions
on **small test prefixes**, with all head/tail candidates and one warmup. They
are not full-test timings or comparisons with competing inference engines.
[Exact records and source hashes](../benchmarks/results/kgfm-inference/2026-09-09.json)
include precision validation and timing limitations. Early exploratory probes
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
