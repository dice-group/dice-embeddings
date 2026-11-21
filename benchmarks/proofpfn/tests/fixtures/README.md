# Fixtures of the ProofPFN benchmark tests

| File | Bytes | SHA-256 | Content | Used by |
|---|---:|---|---|---|
| `gate.json` | 2,245 | `d1c80a85f984943879dc3914b69925a8c6e5cc00a8f76673f57567359c80e2ad` | a manifest of three small graphs (UMLS, NELLInductive_v1, grail-WN18RR_v1) with the SHA-256 of their files and of their content | `test_manifests.py`, `test_data.py`, `test_systems.py`, `test_cli.py` |
| `sample_positions.json` | 2,157 | `b4fc68ce3a042900e5fc6bdb31fc065ef83d4b84f193622f804543117ae5eee9` | the first-1000 sample on the test split of FB15k-237, YAGO3-10 and FB15k: the positions as an int64 digest, with the file hashes of the graphs | `test_data.py` |

The pinned sizes, row orders and Flock settings of the three graphs of `gate.json`, and the pins of the evaluation on
worlds of the prior, are in `tests/fixtures/proofpfn/pins.json` (sections `gate` and `prior`). The graph files are not
in the repository; the tests that read them skip without them.
