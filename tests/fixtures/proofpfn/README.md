# Regression pins of the ProofPFN tests

`pins.json` holds values that the package computes itself. They keep the worlds of the thesis runs fixed.

| Section | Content | Used by |
|---|---|---|
| `worlds` | 123 worlds by `<seed>/<prior>/<family>/<position>`: the slot, the content digest, the digest of the record (task order, metadata and template provenance) and the digest of the prepared tensors for PCUltra | `test_synthetic_{legacy,stage2,records,real,null,prepare}.py` through `tests/synthetic_worlds.py` |
| `orders`, `mixtures` | the digest of the first 400 slots of each world order (the presets pin the full orders), and the shares and cumulative bounds of each mixture | `test_synthetic_priors.py`, `test_synthetic_variants.py` |
| `initial_weights` | the digest of the initial weights per model (PCUltra, CountULTRA, ULTRA) and seed | `test_synthetic_variants.py` |
| `training` | the first two CPU updates of five variants at seed 42: world, learning rate, loss, gradient norm, digests of the raw and EMA weights | `test_synthetic_pretraining.py`, `test_synthetic_command.py` |
| `prior` | per-world values and statistics of the evaluation on worlds of the prior, and the digest of the plain held-out slots | `benchmarks/proofpfn/tests/test_prior.py` |
| `gate` | sizes, row order and Flock setting of the three graphs of `benchmarks/proofpfn/tests/fixtures/gate.json` | `benchmarks/proofpfn/tests/test_data.py`, `test_systems.py` |

- The pins were recorded at commit `ea33c28b`. At that commit the code passed the golden tests against the research
  code of the thesis runs, and each pin was compared with the golden value that it replaces.
- Digests are SHA-256. A tensor digest hashes, for each key in sorted order, the key, the shape and type, and the bytes
  (`dicee/synthetic_kg/digests.py`).
- The pinned training updates are bitwise only for torch 2.5.1 with AVX-512 CPU kernels and 2 threads. Elsewhere these
  tests skip, and a smoke test compares the losses of two updates within a tolerance.
- `python -m tests.proofpfn_fixtures --write-pins` computes the sections `worlds`, `orders`, `mixtures` and
  `initial_weights` again and writes the file (set `PROOFPFN_REAL_SOURCES` for the real worlds). Do this only for an
  intended change of the worlds, and say so in the commit.
