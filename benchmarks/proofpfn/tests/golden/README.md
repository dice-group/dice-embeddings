# Golden fixtures of the ProofPFN benchmark

Byte-for-byte copies of small golden files of the proofpfn-golden folder (phase 2 of the ProofPFN cleanup; its
`golden/README.md` documents inputs, commands and environment), and one fixture of phase 3a. Large goldens (the G4
research exports, the full G6 row files) stay outside the repository: tests that need them read `PROOFPFN_GOLDEN` and
skip without it (`../golden.py`). Digest rules: golden README section 1.

| File | Bytes | SHA-256 | Source | Used by |
|---|---:|---|---|---|
| `g6_ranks.json.gz` | 22,956 | `8d137562cab30fdbe648122b9cc3f1f76565ad9be9d36ef8feac290a4382f8e7` | `golden/small` (G6: per-key int64 digests, row identity, metrics, header and record of the research rows on the gate graphs) | `test_manifests.py`, `test_data.py`, `test_systems.py`, `test_golden_ranks.py` |
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
| `umls_research_scores.json` | 8,312 | `0e3ca65eb8ab4072718513dce6cc08050a82cfd6b1522688701ea655e1299720` | `tests/fixtures/proofpfn` (3a: the research models of the G4 exports gs, gcu, gul on UMLS: per-seed raw rank digests on the CPU and CUDA) | `test_golden_ranks.py` |
