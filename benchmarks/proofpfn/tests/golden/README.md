# Golden fixtures of the ProofPFN benchmark

Byte-for-byte copies of small golden files of the proofpfn-golden folder (its `golden/README.md` documents inputs,
commands and environment), and fixtures of the adapters and of the sample rule, recorded from the research code. The G6
ranks (`g6_ranks.json.gz`) and the UMLS scores (`umls_research_scores.json`) are in `tests/fixtures/proofpfn/`, which
both test suites read. Large goldens (the G4 research exports, the full G6 row files) stay outside the repository.
Tests that need them read `PROOFPFN_GOLDEN` and skip without it (`tests/proofpfn_fixtures.py`). Digest rules: golden
README section 1.

| File | Bytes | SHA-256 | Source | Used by |
|---|---:|---|---|---|
| `anyburl/NELLInductive_v1-preds-225.gz` | 4,897 | `5584038fe7b8c24b7f8d7e9a60b5b111f29639cd692f6c6bd438e0234b779304` | the ranking file of AnyBURL Apply with the rule file below on ultra-suite-v1/NELLInductive_v1 (`gzip -n -9`) | `test_adapter_anyburl.py` |
| `anyburl/NELLInductive_v1-research.json` | 1,716 | `0dc588d88fb421782c517b4f19ae48ba9be1366effdc6a33315dc1d657979a97` | the research `anyburl_run.py` (unchanged copy) on that file: `rebuild` rows and native ranks as JSON digests, counts, `write_inputs` hashes, and the input hashes of the kiowl row | `test_adapter_anyburl.py` |
| `anyburl/NELLInductive_v1-rules-30` | 3,254 | `bb52f8c70090c6af478b9e9a988852a4d8ad91a83bae1ebfb5cf6e738801a153` | the rule file of a 30-second AnyBURL Learn on ultra-suite-v1/NELLInductive_v1 (KR setting, 4 threads) | `test_adapter_anyburl.py` |
| `kgpfn_probe_ranks.json` | 3,882 | `973e2cdf4de055b25d56726389005c29c1001442a405a7ad65e9bd08405220c0` | KGPFN ranks of the three probe graphs per GPU as int64 digests: the research `kgpfn_eval.py` on the local RTX 4070 Ti SUPER, and the kiowl rows (H100 NVL) | `test_adapter_kgpfn.py` |
| `merit_check_ranks.json` | 3,397 | `3ca7f35cd262e76c2d3e0fc87f3082dfa7d5b3737ceb9b6f1d465d1a1e811ba1` | the raw ranks of `K/merit-ref-v1/check/*.json` (kiowl) as int64 digests, with the upstream and checkpoint pins | `test_adapter_merit.py` |
| `g7_cells/queries-NELLInductive_v1.csv` | 18,931 | `d164be5b1eb9521b154eaf26cff025a4347f47b6c66fd81385ae81fa46e31e47` | `golden/small/g7_cells` (G7) | `test_cells.py` |
| `g7_cells/queries-UMLS.csv` | 62,816 | `4d07aed85f6b3e2ca4b99b1077f1a9a950766b7cacff89f6524b5d4471f6bfd7` | `golden/small/g7_cells` (G7: `fv1_cells.py --set gate:UMLS`) | `test_cells.py` |
| `g7_cells/queries-grail-WN18RR_v1.csv` | 17,646 | `4fcb79175685f07e29cfacf506ab597d99dc08a7f32d7e17bada1291be21f396` | `golden/small/g7_cells` (G7) | `test_cells.py` |
| `g8/expected/block3.json` | 4,013 | `e62b03f407eac9c0fe87e7e8043f3f5029f6820383e46118094b17426f985284` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/dropped.json` | 608 | `3d56d54d05eb6ce66b26b0ac05d16567cd19557619d4f0d822bb606e5aaf9ba7` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/gates.json` | 46 | `9cd683863542286e78d3b3c5fd2890dd02a5f92849c217a4c92ae571536e6deb` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/manifest.json` | 17,324 | `da08eee8609b4ccdd6df5186a9cd03fbe37d24d312217955188e4bc7e243780f` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/models.json` | 3,158 | `74edf375903f661ec9a4cdcfe761b84c8f2f8e448c81df963ed2392dfdbce51d` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/result-b50000.json.gz` | 45,349 | `ab98a4ed62740025d2c94ff1cac3c68587ef85a82eb6d96e38fcd9d53ec51f15` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/result-b999.json.gz` | 44,260 | `a0c2dfe4d609411bdedc59ade4a2b43b81f53e8f1cbc476aba21eedb7a3bce61` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/tables/cost.csv.gz` | 7,043 | `574b6986ecf2a7becf8d54f7c8a9095e2be3f61cf52e11d2461ec7ca0c3b25c4` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/tables/mrr.csv.gz` | 54,841 | `fe6c5b4ecc0f08bd9578b1db407804a35b5b4f888b666f1fc0db4159e8d886e3` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/expected/tables/tables.meta.json.gz` | 22,348 | `9888cd670665f3dd896d4f1b39ea4a5aeffcbf8097a78d8524eda773e2a2317a` | `golden/small/g8/expected` (G8: a file of the chain on the expected design) | `test_registered.py` |
| `g8/g8_analysis.json.gz` | 23,284 | `ba09c982be231430ad93aea8b16621ad1ab219bcfa74a2d584552ee3486fa8ad` | `golden/small/g8` (G8: anchors, expected design, SHA-256 of every input and output) | `test_registered.py` |
| `g9_gate.json` | 5,947 | `7ab51b2b7495431716255b3b0cb820f92291e423998bbbe7334d9f630362fe02` | `golden/small` (G9: the research N validity gate, `s5_manip.py`, on the G4 exports of S and N at update 20, 2 held-out worlds per family) | `test_gate.py` |
| `sample_positions.json` | 2,524 | `05f2b710f3bdf4e88a8c6a1e7a0d7a13c4b3e6c49fbb8f4bbfc8f4c4ef3d6a21` | the research `allgraph-v1/ag_eval.sample_positions` on the test split of the research loader `ib_eval.load_dataset` (unchanged research code) for FB15k-237, YAGO3-10 and FB15k: the positions as an int64 digest, with the file hashes of the graphs | `test_data.py` |
