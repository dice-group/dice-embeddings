---
name: debug-evaluation
description: "Diagnose and improve poor KGE link prediction metrics in dicee. Use when: MRR / HITS@k are unexpectedly low, model overfits or underfits, evaluation produces NaN or zero scores, need a hyperparameter tuning checklist."
mode: ask
---

# Diagnose Poor Link Prediction Performance

I will help you systematically diagnose and improve the link prediction metrics (MRR, HITS@k) for your dicee model.

## What to Provide

To start the diagnosis, share:

1. **The eval report** — paste contents of `eval_report.json` or the terminal output (Train/Val/Test MRR and HITS@k values)
2. **The configuration** — paste `configuration.json` or describe key settings (model, scoring_technique, embedding_dim, num_epochs, lr, batch_size)
3. **The dataset** — name or size (approximate number of entities, relations, triples)
4. **What you have tried** — any hyperparameter changes already attempted

---

## Diagnostic Checklist (I Will Work Through These)

### Layer 1 — Data

- [ ] **Train/Val/Test split ratio** — Is the split standard (80/10/10)?  
  Low test MRR with high train MRR → data leakage or overfitting.
- [ ] **Reciprocal relations** — Are inverse relations (`rel_inverse`) present?  
  Models like Keci benefit significantly from reciprocal training.
- [ ] **KG noise** — Are there duplicate or contradictory triples?  
  `--add_noise_rate` introduces random triples — check it is `None`.
- [ ] **Separator mismatch** — Does `--separator` match the file format?  
  Wrong separator → entities parsed as `"entity\ttail"` instead of two columns.

### Layer 2 — Scoring Technique

- [ ] **Scoring technique vs KG size mismatch**  
  Using `AllvsAll` on a large KG causes extreme memory pressure → training degrades.  
  For large KGs: `NegSample` with `neg_ratio 10–20`.  
  For medium KGs: `KvsAll` (default, recommended).
- [ ] **NegSample with neg_ratio too low**  
  `--neg_ratio 0` means 0 negatives → model learns nothing. Must be ≥ 1 for `NegSample`.
- [ ] **Label smoothing too high**  
  `--label_smoothing_rate > 0.3` can prevent the model from fitting signal.

### Layer 3 — Model and Embedding Dimension

- [ ] **Embedding dimension too small**  
  `embedding_dim 32` may underfit complex KGs. Try 64 → 128 → 256.
- [ ] **Clifford constraint violated**  
  For Keci/DeCaL: `embedding_dim / (p + q + 1)` must be integer.  
  E.g., `embedding_dim=64, p=0, q=1` → `64/2=32` ✓  
  `embedding_dim=65, p=0, q=1` → not integer → AssertionError.
- [ ] **Wrong model for relation type**  
  Symmetric relations → ComplEx or QMult performs better.  
  Hierarchical relations → TransE often works well.

### Layer 4 — Training Dynamics

- [ ] **Learning rate too high or too low**  
  Default `lr=0.1` with Adam. If loss oscillates → try `lr=0.01`.  
  If loss barely decreases → try `lr=0.1` or increase `num_epochs`.
- [ ] **Too few epochs**  
  Check: is train MRR still rising at the last epoch?  
  Use `--eval_every_n_epochs 20` to plot learning curves.
- [ ] **Batch size interaction with KvsAll**  
  Large `batch_size` with `KvsAll` can exhaust memory silently → OOM crash or silent NaN.  
  Reduce `batch_size` or switch to `KvsSample`.
- [ ] **Optimizer choice**  
  Adam (default) is robust. SGD needs careful lr scheduling.  
  Try `--optim ADOPT` for adaptive gradient clipping.

### Layer 5 — Regularisation

- [ ] **No dropout**  
  Add `--input_dropout_rate 0.1 --hidden_dropout_rate 0.1`.
- [ ] **No weight decay**  
  Try `--weight_decay 1e-5`.
- [ ] **No normalisation**  
  Try `--normalization LayerNorm` or `--normalization BatchNorm1d`.
- [ ] **Overfitting** (train MRR >> test MRR)  
  Increase dropout, add weight decay, reduce embedding_dim, or use weight averaging (`--swa`).

### Layer 6 — Evaluation Settings

- [ ] **`eval_model` includes splits that don't exist**  
  If there is no `valid.txt`, setting `eval_model train_val_test` may silently skip val.  
  Use `--eval_model train_test` if no validation set exists.
- [ ] **Evaluation after too few epochs**  
  Default final eval happens after `num_epochs`. Use `--eval_every_n_epochs` to verify.

---

## Recommended Baseline Configuration

For a medium-sized KG (10K–500K triples):

```bash
dicee --dataset_dir "KGs/YOUR_KG" \
      --model Keci \
      --embedding_dim 128 \
      --scoring_technique KvsAll \
      --num_epochs 200 \
      --lr 0.1 \
      --batch_size 1024 \
      --input_dropout_rate 0.1 \
      --normalization LayerNorm \
      --eval_model train_val_test \
      --eval_every_n_epochs 50
```

For large KG (>500K triples):

```bash
dicee --dataset_dir "KGs/YOUR_KG" \
      --model Keci \
      --embedding_dim 256 \
      --scoring_technique NegSample \
      --neg_ratio 10 \
      --num_core 10 \
      --num_epochs 500 \
      --lr 0.1 \
      --batch_size 2048 \
      --trainer PL \
      --eval_model train_val_test \
      --eval_every_n_epochs 100
```

---

## Reading the eval_report.json

```json
{
  "Train": { "MRR": 0.95, "MR": 3,  "HITS@1": 0.91, "HITS@3": 0.97, "HITS@10": 0.99 },
  "Val":   { "MRR": 0.72, "MR": 18, "HITS@1": 0.62, "HITS@3": 0.80, "HITS@10": 0.90 },
  "Test":  { "MRR": 0.70, "MR": 20, "HITS@1": 0.60, "HITS@3": 0.78, "HITS@10": 0.88 }
}
```

- **Train MRR >> Val MRR** → overfitting. Add dropout/weight decay/weight averaging.
- **All MRRs low** → model underfit. Increase embedding_dim, epochs, or change scoring technique.
- **Val MRR >> Test MRR** → possible train/test leakage or distribution mismatch.

---

Paste your eval report and config and I will walk through the checklist above to find the issue.
