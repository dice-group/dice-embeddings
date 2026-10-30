# ProofPFN golden fixtures

Small fixtures of the golden references that the ProofPFN port is tested against, recorded from the research code. Most
are byte-for-byte copies of `golden/small/` in the proofpfn-golden folder (its `golden/README.md` documents inputs,
commands and environment). Two files were recorded later from the research code by the drivers named below, which ran
on a private copy of the research work tree (`scratch-3a/tree` in the proofpfn-golden folder).

Large goldens (research exports, tensor files) stay outside the repository. Tests that need them read the environment
variable `PROOFPFN_GOLDEN` (the proofpfn-golden folder) and skip when it is unset or the file is missing
(`tests/proofpfn_fixtures.py`). Digest rules: golden README section 1, implemented in `dicee/synthetic_kg/digests.py`.

The goldens were recorded in one environment: Python 3.12.2, torch 2.5.1 (CUDA 11.8, cuDNN 9.1.0), numpy 1.26.4, an
AMD Ryzen 7 7800X3D with `torch.backends.cpu.get_cpu_capability() == 'AVX512'`, 2 torch threads, and an NVIDIA GeForce
RTX 4070 Ti SUPER with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The CPU training goldens (G4: `g4_*.json.gz`) are bitwise for
torch 2.5.1 with AVX-512 CPU kernels only; with the AVX2 or the default kernels the first update already differs in the
last bits, so `require_cpu_training_golden_environment` skips them elsewhere. The CUDA goldens (G10, the CUDA scores,
G6 on the GPU) need torch 2.5.1 with CUDA 11.8 on that GPU (`require_gpu_golden_environment`). The worlds, orders,
initial weights, conversions and CPU scores and ranks do not depend on the CPU kernels.

| File | Bytes | SHA-256 | Source | Used by |
|---|---:|---|---|---|
| `g1_orders.json.gz` | 13,352 | `a0b6a1670904a88024083093e767ce23a52f67a6a8b916832bfd8aa797fc210c` | `golden/small` (G1: world orders) | `test_synthetic_priors.py` |
| `g2_worlds.json.gz` | 211,690 | `175a414a9309d5609b4fab3b307baf4a7dbfa46b3a63078fb736e1c96e3f9b8e` | `golden/small` (G2: worlds) | `test_synthetic_priors.py`, and through `tests/synthetic_worlds.py` the world-generator tests `test_synthetic_{legacy,stage2,records,real,null}.py` |
| `recipe_research.json.gz` | 5,096 | `20093fb707acf525d49af5cc954c565edfb1cf7cec201f693f1c90d99760ee5d` | `scratch-3a/tools/dump_recipe.py` (the 12 levels of the research configuration chain, the record families' parameters and the prior constants, with the SHA-256 of the research files it read) | `test_synthetic_priors.py` |
| `g3_init.json.gz` | 27,886 | `8d7424c02e36111685a5087c23d397ea7daa1d57817f929080f6333d8e3c39c6` | `golden/small` (G3: initial weights) | `test_synthetic_arms.py` |
| `g4_s42.json.gz` | 136,899 | `78b5a9347b59125dc6a200e1e4a210676985f661568a383f474fd3885454a105` | `golden/small` (G4: CPU training, seed 42; S with K = 100, CU, UL, HT, N with K = 20) | `test_synthetic_pretraining.py` |
| `g4_s42rm.json.gz` | 49,075 | `2ac55f699b1d64847ce45d530306d45a38c909d7a0d1279976aa44214d3b202f` | `golden/small` (G4: CPU training, seed 42; R and M with K = 20) | `test_synthetic_pretraining.py` |
| `g4_s43.json.gz` | 120,286 | `9c76cfba1bb4a5d5949e4de5ccbe9f3069f7d037fad7499f1faf9354373bd1ed` | `golden/small` (G4: CPU training, seed 43; S, CU, UL, HT, N with K = 20) | `test_synthetic_pretraining.py` |
| `g4_s44.json.gz` | 120,383 | `851d13ec0678807dd21dd06374cfe5d03f906cbfbbea522486f124fc37e688d2` | `golden/small` (G4: CPU training, seed 44; S, CU, UL, HT, N with K = 20) | `test_synthetic_pretraining.py` |
| `g5_cpu.json.gz` | 135,140 | `2d6ce30e5344dc58106bd933a0480ccbf5e1928506eb1a78f088a0a123b56f21` | `golden/small` (G5: CPU exports) | `test_synthetic_checkpoint.py` |
| `g5_gpu.json.gz` | 43,854 | `a92f60fb6b6786dd45bd13cd3b5140c392b41239b4c3c1a4274cf7b5fe6ee7c3` | `golden/small` (G5: genuine GPU exports) | `test_synthetic_checkpoint.py` |
| `g10_gpu.json.gz` | 57,492 | `d4e1f5e2ea249230d6bf0be588027834c1c470eb1326492a415f35895264d1ce` | `golden/small` (G10: the genuine research trainer on the local GPU; S seed 42 with 20 updates, UL, CU and HT with 2) | `test_synthetic_pretraining.py` |
| `g6_ranks.json.gz` | 22,956 | `8d137562cab30fdbe648122b9cc3f1f76565ad9be9d36ef8feac290a4382f8e7` | `golden/small` (G6: per-key int64 digests, row identity, metrics, header and record of the research rows on the gate graphs) | `test_synthetic_checkpoint.py`; `benchmarks/proofpfn/tests/`: `test_manifests.py`, `test_data.py`, `test_systems.py`, `test_golden_ranks.py` |
| `umls_research_scores.json` | 8,312 | `0e3ca65eb8ab4072718513dce6cc08050a82cfd6b1522688701ea655e1299720` | `scratch-3a/tools/ranks_umls.py cpu` and `cuda` (the research models of the G4 exports gs, gcu, gul at update 20 on UMLS: digests of the scores and of the per-seed raw ranks; on CUDA the ranks equal G6) | `test_synthetic_checkpoint.py` (scores); `benchmarks/proofpfn/tests/test_golden_ranks.py` (CPU ranks) |

The research CPU ranks of `gul.seed44` differ from its G6 ranks, which were computed on the GPU; the benchmark's CPU
rank test compares with the research CPU reference, its CUDA tests with G6. The benchmark tests read the two files
above from this folder (`benchmarks/proofpfn/tests/golden.py`, `SHARED`).
