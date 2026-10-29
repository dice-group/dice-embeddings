# ProofPFN golden fixtures

Small fixtures of the golden references that the ProofPFN port is tested against. Most are byte-for-byte copies of
`golden/small/` in the proofpfn-golden folder (phase 2 of the cleanup; its `golden/README.md` documents inputs,
commands and environment). The files marked "3a" were recorded from the research code during phase 3a by the drivers
named below, which ran on a private copy of the research work tree (`scratch-3a/tree` in the proofpfn-golden folder).

Large goldens (research exports, tensor files) stay outside the repository. Tests that need them read the environment
variable `PROOFPFN_GOLDEN` (the proofpfn-golden folder) and skip when it is unset or the file is missing
(`tests/proofpfn_fixtures.py`). Digest rules: golden README section 1, implemented in `dicee/synthetic_kg/digests.py`.

| File | Bytes | SHA-256 | Source | Used by |
|---|---:|---|---|---|
| `g1_orders.json.gz` | 13,352 | `a0b6a1670904a88024083093e767ce23a52f67a6a8b916832bfd8aa797fc210c` | `golden/small` (G1: world orders) | `test_synthetic_priors.py` |
| `g2_worlds.json.gz` | 211,690 | `175a414a9309d5609b4fab3b307baf4a7dbfa46b3a63078fb736e1c96e3f9b8e` | `golden/small` (G2: worlds) | `test_synthetic_priors.py` |
| `recipe_research.json.gz` | 5,096 | `20093fb707acf525d49af5cc954c565edfb1cf7cec201f693f1c90d99760ee5d` | 3a: `scratch-3a/tools/dump_recipe.py` (the 12 levels of the research configuration chain, the record families' parameters and the prior constants, with the SHA-256 of the research files it read) | `test_synthetic_priors.py` |
