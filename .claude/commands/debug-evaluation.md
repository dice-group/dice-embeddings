---
description: Diagnose and improve poor KGE link prediction metrics in dicee (low MRR/HITS@k, overfitting, NaN scores)
argument-hint: "[paste eval_report.json / configuration.json / describe the problem]"
---

# Diagnose Poor Link Prediction Performance

Systematically diagnose and improve the link prediction metrics (MRR, HITS@k) for the user's dicee model. If `$ARGUMENTS` doesn't already include an eval report or config, ask for them before proceeding.

## What to Gather First (if not already provided)

1. **The eval report** — `eval_report.json` or terminal output (Train/Val/Test MRR and HITS@k values)
2. **The configuration** — `configuration.json` or key settings (model, scoring_technique, embedding_dim, num_epochs, lr, batch_size)
3. **The dataset** — name or size (approximate number of entities, relations, triples)
4. **What's already been tried** — any hyperparameter changes already attempted

---

## Diagnostic Checklist (work through in order)

### Layer 1 — Data
- **Train/Val/Test split ratio** — is it standard (80/10/10)? Low test MRR with high train MRR → data leakage or overfitting.
- **Reciprocal relations** — are inverse relations (`rel_inverse`) present? Models like Keci benefit significantly from reciprocal training.
- **KG noise** — duplicate or contradictory triples? `--add_noise_rate` should be `None` unless intentional.
- **Separator mismatch** — does `--separator` match the file format? Wrong separator parses entities incorrectly.

### Layer 2 — Scoring Technique
- `AllvsAll` on a large KG causes extreme memory pressure → training degrades. Large KGs: `NegSample` with `neg_ratio 10–20`. Medium KGs: `KvsAll` (default, recommended).
- `NegSample` with `--neg_ratio 0` → zero negatives → model learns nothing. Must be ≥ 1.
- `--label_smoothing_rate > 0.3` can prevent the model from fitting signal.

### Layer 3 — Model and Embedding Dimension
- `embedding_dim 32` may underfit complex KGs. Try 64 → 128 → 256.
- Clifford constraint (Keci/DeCaL): `embedding_dim / (p + q + 1)` must be integer. E.g. `embedding_dim=64, p=0, q=1` → `32` ✓; `embedding_dim=65` → not integer → AssertionError.
- Wrong model for relation type: symmetric relations → ComplEx/QMult; hierarchical → TransE.

### Layer 4 — Training Dynamics
- Default `lr=0.1` with Adam. Oscillating loss → try `lr=0.01`. Barely decreasing → try more `num_epochs`.
- Is train MRR still rising at the last epoch? Use `--eval_every_n_epochs 20` to plot learning curves instead of guessing.
- Large `batch_size` with `KvsAll` can exhaust memory silently → OOM or silent NaN. Reduce `batch_size` or switch to `KvsSample`.
- Adam (default) is robust; SGD needs careful lr scheduling; try `--optim ADOPT` for adaptive gradient clipping.

### Layer 5 — Regularisation
- No dropout → add `--input_dropout_rate 0.1 --hidden_dropout_rate 0.1`.
- No weight decay → try `--weight_decay 1e-5`.
- No normalisation → try `--normalization LayerNorm` or `BatchNorm1d`.
- Overfitting (train MRR >> test MRR) → increase dropout, add weight decay, reduce embedding_dim, or use weight averaging (`--swa`, `--adaptive_swa`).

### Layer 6 — Evaluation Settings
- If there is no `valid.txt`, `--eval_model train_val_test` may silently skip val — use `--eval_model train_test` instead.
- Confirm evaluation actually ran after the expected number of epochs; use `--eval_every_n_epochs` to verify.

---

## Recommended Baseline Configurations

Medium-sized KG (10K–500K triples):
```bash
dicee --dataset_dir "KGs/YOUR_KG" \
      --model Keci --embedding_dim 128 \
      --scoring_technique KvsAll --num_epochs 200 --lr 0.1 --batch_size 1024 \
      --input_dropout_rate 0.1 --normalization LayerNorm \
      --eval_model train_val_test --eval_every_n_epochs 50
```

Large KG (>500K triples):
```bash
dicee --dataset_dir "KGs/YOUR_KG" \
      --model Keci --embedding_dim 256 \
      --scoring_technique NegSample --neg_ratio 10 --num_core 10 \
      --num_epochs 500 --lr 0.1 --batch_size 2048 --trainer PL \
      --eval_model train_val_test --eval_every_n_epochs 100
```

---

## Reading eval_report.json

```json
{
  "Train": { "MRR": 0.95, "MR": 3,  "HITS@1": 0.91, "HITS@3": 0.97, "HITS@10": 0.99 },
  "Val":   { "MRR": 0.72, "MR": 18, "HITS@1": 0.62, "HITS@3": 0.80, "HITS@10": 0.90 },
  "Test":  { "MRR": 0.70, "MR": 20, "HITS@1": 0.60, "HITS@3": 0.78, "HITS@10": 0.88 }
}
```

- **Train MRR >> Val MRR** → overfitting. Add dropout/weight decay/weight averaging.
- **All MRRs low** → underfitting. Increase embedding_dim, epochs, or change scoring technique.
- **Val MRR >> Test MRR** → possible train/test leakage or distribution mismatch.

Work through the checklist above against what the user provides, and give a prioritised, concrete list of changes with expected impact.
