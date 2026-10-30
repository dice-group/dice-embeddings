# Regularization sweep

`benchmarks/regularization_sweep.py` trains and evaluates every combination of datasets, models and
nine experiment types, and appends one CSV row per run.

| Experiment type | Loss | Weight averaging |
|---|---|---|
| `BCE`, `LS`, `ACLS` | BCE, BCE + label smoothing (0.1), ACLS | none |
| `BCE_ASWA`, `LS_ASWA`, `ACLS_ASWA` | as above | ASWA (`adaptive_swa`) |
| `BCE_AMWA`, `LS_AMWA`, `ACLS_AMWA` | as above | AMWA (starts at epoch 20, averaged every 10) |

## Setup

```bash
git clone https://github.com/dice-group/dice-embeddings && cd dice-embeddings
pip install -e .
wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip && unzip KGs.zip   # creates ./KGs
```

## Run

```bash
# Smoke test: all 9 types on UMLS, about a minute on CPU.
python benchmarks/regularization_sweep.py --datasets UMLS --models DistMult --num_epochs 25 --trainer torchCPUTrainer

# FB15k-237, DistMult, 128 dimensions (the setup used when validating 1p query answering).
python benchmarks/regularization_sweep.py --datasets FB15k-237 --models DistMult \
    --embedding_dim 128 --lr 0.01 --num_epochs 50 --gpu 0 --out fb15k237_distmult.csv
```

Options: `--datasets`, `--models`, `--exp_types` (a subset of the nine), `--kg_root`, `--out`,
`--experiments_dir`, `--gpu`, `--seed`, `--num_epochs`, `--embedding_dim`, `--lr`, `--batch_size`,
`--trainer`. The defaults (32 dimensions, learning rate 0.1, 100 epochs) are the original sweep
settings. Run `--help` for the full list.

## Notes

- **Output.** `--out` is appended to, so rerunning a sweep with the same file adds duplicate rows.
  Use one file per job.
- **AMWA needs more than 20 epochs**, since averaging starts at epoch 20.
- **ASWA can equal the final model.** It replaces its ensemble whenever validation MRR improves, so
  while validation MRR is still rising every epoch, `*_ASWA` results are identical to the
  no-averaging ones. This is expected, not a bug.
- **MRR is the mean of tail and head prediction.** Models trained with `KvsAll` use reciprocal
  triples, and the built-in test MRR averages `(h, r, ?)` and `(?, r, t)` queries. A 1-hop query
  corresponds to the tail direction only, which scores higher (on FB15k-237 DistMult: tail 0.318,
  head 0.136, reported 0.227).
- `Seed` in the CSV is the value of `--seed`, which is also used as `random_seed`.
