# Golden fixtures of the ProofPFN benchmark

Byte-for-byte copies of small golden files of the proofpfn-golden folder (its `golden/README.md` documents inputs,
commands and environment), and fixtures of the sample rule, recorded from the research code. The G6
ranks (`g6_ranks.json.gz`) and the UMLS scores (`umls_research_scores.json`) are in `tests/fixtures/proofpfn/`, which
both test suites read. Large goldens (the G4 research exports, the full G6 row files) stay outside the repository.
Tests that need them read `PROOFPFN_GOLDEN` and skip without it (`tests/proofpfn_fixtures.py`). Digest rules: golden
README section 1.

| File | Bytes | SHA-256 | Source | Used by |
|---|---:|---|---|---|
| `g9_gate.json` | 5,947 | `7ab51b2b7495431716255b3b0cb820f92291e423998bbbe7334d9f630362fe02` | `golden/small` (G9: the research N validity gate, `s5_manip.py`, on the G4 exports of S and N at update 20, 2 held-out worlds per family) | `test_gate.py` |
| `sample_positions.json` | 2,524 | `05f2b710f3bdf4e88a8c6a1e7a0d7a13c4b3e6c49fbb8f4bbfc8f4c4ef3d6a21` | the research `allgraph-v1/ag_eval.sample_positions` on the test split of the research loader `ib_eval.load_dataset` (unchanged research code) for FB15k-237, YAGO3-10 and FB15k: the positions as an int64 digest, with the file hashes of the graphs | `test_data.py` |
