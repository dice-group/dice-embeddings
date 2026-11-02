# Upstream inference fixtures

Generated on CPU with PyTorch 2.9.1. No DICE imports occur in the generator.
Each file records its upstream git commit, graph, queries, checkpoint state,
options, and expected complete score vectors. `ultra.pt` also records hashes of
the public `ultraquery.pth` and `ultra_3g.pth` weights. The random seed is 20260926.

The repositories and immutable commits are listed in `REFERENCES` in
`dicee/query_answering/catalog.py`.
Clone each repository and check out that commit before regenerating a fixture.

```bash
PYTHONPATH=/path/to/upstream \
python tests/fixtures/query_baselines/generate_reference.py \
  cone /path/to/upstream tests/fixtures/query_baselines/cone.pt
```

Use `cqd`, `clmpt`, `ultra`, `gnnqe`, or `qto` for the other fixtures. For QTO,
put its `kbc` directory on `PYTHONPATH`. Use separate processes to avoid upstream
package-name collisions. Generating ULTRA/GNN-QE requires their reference-only
dependencies; these are not DICE runtime dependencies.

The verification environment used torch-geometric 2.6.1, torch-scatter 2.1.2,
torch-cluster 1.6.3, TorchDrug 0.2.1, and Ninja. GPU visibility was disabled and
`MAX_JOBS=1` bounded reference extension compilation. For current Python/RDKit,
the generator aliases `collections.Sequence` and supplies the unused drawing
base class removed from RDKit. TorchDrug requires these two build-only changes
with PyTorch 2.9:

1. In `layers/functional/extension/{spmm,rspmm}.h`, change
   `ATen/SparseTensorUtils.h` to `ATen/native/SparseTensorUtils.h`.
2. In `utils/torch.py`, pass `extra_ldflags`, `extra_include_paths`,
   `build_directory`, and `verbose` to `cpp_extension.load` by keyword.

The original sparse kernels and neural computations are unchanged. GNN-QE
fixtures execute those compiled kernels, including symbolic traversal.

CLMPT fixtures evaluate DNF branches through the released reasoner with the
original relation groundings. Both pre/post-norm variants are covered. The
metadata's earlier incorrect `up` repair note has been corrected by regenerating
the fixture; all tensors remain bit-identical.
CQD fixtures intentionally retain the released 3p/4p quirks.
An additional plain CQD case verifies the `ip`/`up` cap with beam 6 and max_k 1.
CLMPT also records paired-query batches to test the post-norm layout directly.

The fixtures cover all 16 shapes for ConE, QTO, CLMPT, UltraQuery and GNN-QE;
CQD/CQD-Hybrid cover 11 positive shapes with product and min in `cqd.pt`, plus
the five negated shapes in `cqd-negation.pt`.

`cqd-negation.pt` covers the five negated types.
Its executor is official CQD-A at commit
`642ce042708be6247087c09c780b9deb47e941d3`, with +H min-max atomic scoring.
Both operators are covered. Hybrid checks use a full-width beam and observed
fact overrides; its dynamic beam is covered separately by the positive fixtures.

```bash
python tests/fixtures/query_baselines/generate_negation_reference.py \
  /path/to/adaptive-cqd tests/fixtures/query_baselines/cqd-negation.pt
```

For ConE, CLMPT, GNN-QE, CQD or QTO, `--checkpoint /path/to/released-weights` generates
an additional reference using those learned parameters. Vocabulary tensors
are cropped to this seven-entity, four-relation graph; other learned parameters
remain unchanged. The output records the original checkpoint hash. Verify full
checkpoint loading against its original dataset separately. Store these large
references under `Experiments/`, outside the checked-in fixtures.
