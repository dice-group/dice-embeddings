# Inductive KG Link Prediction (Set Transformer)

PyTorch implementation per the spec. Treats each 2-hop subgraph as an unordered
set of triples; intra-triple position (S/R/O) is encoded, inter-triple is not.

## Layout


```
ilp/
  cli.py                       # python -m ilp {train,infer,score}
  __main__.py                  # dispatch
  vocab.py                     # schema / Z-pool vocabulary
  dataset.py                   # KG, two-hop subgraph, anonymized sample
  model.py                     # per-triple encoder + SAB stack
  train.py                     # AdamW + warmup→cosine + BCE (+ train_model())
  eval.py                      # filtered MRR/Hits@K + invariance checks
  debug_subgraph_sizes.py      # dev tool
  configs/default.yaml         # bundled YAML hyperparams
  configs/countries_s1.yaml    # example small-KG config
  configs/z1.yaml              # ablation: collapse Z pool to size 1
  tests/test_dataset.py        # vocab/anonymization sanity
  tests/test_invariance.py     # spec §7.2 regression tests
data/download.py               # DBpedia50 fetcher (kept outside the package)
```

## Setup

Assumes the `dicee` conda env (PyTorch ≥ 2.0, numpy, pyyaml, pytest).

```bash
conda activate dicee
pip install pyyaml pytest  # if missing
```

## End-to-end

```bash
# 1. Fetch data
python data/download.py
#   If automatic mirrors fail, the script prints manual links. Drop
#   train.txt / valid.txt / test.txt into data/dbpedia50/.

# 2. Run regression tests (the permutation invariance test MUST pass at init)
pytest ilp/tests/ -v

# 3. Train  (writes everything to checkpoints/{run_name}/)
python -m ilp.train --config ilp/configs/default.yaml

# 4. Evaluate  (loads config, vocab, and ckpt from one folder)
python -m ilp.eval --run checkpoints/default
```

## CLI (single-file model bundle)


```bash
# Train: writes {model, vocab, fixed_values, cfg} into one file
python -m ilp train --kg-dir KGs/Countries-S1/ --epochs 1000 --save model.pt

# Infer: top-k tails for (head, relation, ?)
python -m ilp infer --model model.pt --train-file KGs/Countries-S1/train.txt \
    --head slovakia --relation neighbor --k 5

# Score: a specific (head, relation, tail) triple
python -m ilp score --model model.pt --data KGs/Countries-S1/train.txt \
    --triple slovakia neighbor austria
```

`infer` and `score` need a triples file (`--train-file` / `--data`) because
the model is inductive — it scores by extracting a 2-hop subgraph around the
head, which has to come from some KG context.

The CLI assumes `triple_format: head_relation_tail` by default (standard
KG-completion layout). Use the YAML workflow above if you need other formats
or fine-grained hyperparameter control. The two flows share the same core
`train_model()` and are interchangeable for inference — a bundle produced
either way loads via `ilp.cli._load_bundle`.

## Notes

- `[X]` is a CLS-style token prepended to the triple-set; permutation
  invariance comes from `nn.TransformerEncoderLayer` not adding any
  positional encoding (only the intra-triple S/R/O positions are added,
  inside the per-triple encoder).
- Z-randomization at every sample draw is what trains all `[Z_i]`
  embeddings into a shared "anonymous variable" role. Disabling it
  silently breaks inductive generalization.
- **Inverse relations.** Every relation `r` has a paired `r__inv` token.
  KGs are augmented with `(t, r__inv, h)` for each `(h, r, t)`, and head
  queries `(?, r, t)` are evaluated as tail queries `(t, r__inv, ?)`.
  Without this, head MRR collapses (~0.02 on DBpedia50) — the model
  can't disambiguate direction from subgraph orientation alone.
- Filtered ranking uses `train ∪ valid ∪ test` as the known-triple set,
  consistent with the standard KG-completion convention.
- **DBpedia50 caveat.** ~5,700 entities appear in both train and test;
  the runtime prints a warning. For *this* model it's harmless — every
  entity is anonymized to `[Z_*]`, so "seen-before entity strings" leak
  nothing. Inductiveness here is a property of the architecture, not
  the split.
- **Triple file format is configurable.** Set `triple_format` in the
  config to match the source layout. Currently supported:
  `head_tail_relation` (DBpedia50: `h <TAB> t <TAB> r`) and
  `head_relation_tail` (standard KG-completion order: `h <TAB> r <TAB> t`,
  e.g. Countries-S1, FB15k-237, WN18RR). New layouts go in
  `TRIPLE_FORMATS` in `ilp/dataset.py` — one tuple of column
  indices, no other code changes needed.
- **Run-scoped artifacts.** Each config has a `run_name` field; all of
  that run's outputs (`model_step*.pt`, `model_final.pt`, `vocab.json`,
  and a snapshot of the config) land in `checkpoints/{run_name}/`. Eval
  takes `--run checkpoints/{run_name}` and pulls everything from one
  place — nothing loose under `data/`, no stale-vocab footgun, two
  configs never clobber each other.
- **`cardinality_cutoff` (default 100) splits schema from instances.**
  For each relation, if its training-set object range has *fewer* than
  `cardinality_cutoff` distinct values, every value gets its own
  `[VAL_*]` embedding (treated as closed schema vocabulary). Otherwise
  the values are anonymized to `[Z_i]` at sample time (treated as
  open-ended instances). On DBpedia50 with cutoff=100, this puts
  small enumerations like `kingdom` (1 value), `musicFusionGenre` (6),
  `distributingLabel` (9) on the schema side, and open sets like
  `birthPlace` (547), `team` (581), `starring` (630) on the instance
  side — the model is forced to handle the latter inductively via
  subgraph context rather than lookup. The threshold is a heuristic:
  raise it to fold borderline taxonomies (e.g. `family`, `location`,
  both ~100 in DBpedia50) into the schema side; lower it to push
  rarer values toward inductive treatment.

## Acceptance

- `test_permutation_invariance` passes at init.
- `test_z_relabeling_invariance` should pass within a loose tolerance
  after training; record the max diff at each val checkpoint.
