import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dicee import KGE


def load_triples(path):
    with open(path, "r") as f:
        triples = [tuple(line.strip().split()[:3]) for line in f if line.strip()]
    return triples


def triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx, device=None):
    """
    Returns:
      idx: LongTensor [N,3] on device
      kept_triples: list[(h,r,t)] aligned with idx rows
      dropped: int
    """
    idx_rows = []
    kept = []
    dropped = 0
    for h, r, t in triples:
        if h in entity_to_idx and r in relation_to_idx and t in entity_to_idx:
            idx_rows.append([entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]])
            kept.append((h, r, t))
        else:
            dropped += 1

    if len(idx_rows) == 0:
        raise ValueError("No triples could be mapped to indices. Check your maps/files.")

    idx = torch.tensor(idx_rows, dtype=torch.long)
    if device is not None:
        idx = idx.to(device)
    return idx, kept, dropped


# ----------------------------
# 2) Negative sampling
# ----------------------------
def corrupt_triples(pos_idx: torch.Tensor, num_entities: int, num_negs: int):
    """
    pos_idx: [N,3] (h,r,t)
    returns neg_idx: [N, num_negs, 3]
    """
    device = pos_idx.device
    N = pos_idx.size(0)

    neg = pos_idx.unsqueeze(1).repeat(1, num_negs, 1)  # [N,k,3]
    rand_entities = torch.randint(0, num_entities, (N, num_negs), device=device)

    # 0 -> corrupt head, 1 -> corrupt tail
    corrupt_side = torch.randint(0, 2, (N, num_negs), device=device)

    neg[..., 0] = torch.where(corrupt_side == 0, rand_entities, neg[..., 0])
    neg[..., 2] = torch.where(corrupt_side == 1, rand_entities, neg[..., 2])

    return neg


# ----------------------------
# 3) Training dynamics tracker
# ----------------------------
class TripleDynamicsTracker:
    """
    Tracks per-triple stats across epochs:
      - mean/var of loss
      - mean/var of confidence
      - forgetting events (correct -> incorrect transitions)
    """
    def __init__(self, n_triples: int):
        self.n = n_triples
        self.epoch_count = 0

        # store on CPU to avoid pinning huge arrays on GPU
        self.mean_loss = torch.zeros(n_triples, dtype=torch.float32)
        self.M2_loss = torch.zeros(n_triples, dtype=torch.float32)

        self.mean_conf = torch.zeros(n_triples, dtype=torch.float32)
        self.M2_conf = torch.zeros(n_triples, dtype=torch.float32)

        self.forgetting = torch.zeros(n_triples, dtype=torch.int32)
        self.prev_correct = None

    @torch.no_grad()
    def update_epoch(self, loss_per_triple_cpu, conf_cpu, correct_cpu):
        """
        All args are CPU tensors [N]
        """
        if self.prev_correct is None:
            self.prev_correct = correct_cpu.clone()
        else:
            self.forgetting += (self.prev_correct & ~correct_cpu).to(torch.int32)
            self.prev_correct = correct_cpu.clone()

        self.epoch_count += 1
        k = float(self.epoch_count)

        # Welford updates for loss
        delta = loss_per_triple_cpu - self.mean_loss
        self.mean_loss += delta / k
        self.M2_loss += delta * (loss_per_triple_cpu - self.mean_loss)

        # Welford updates for confidence
        delta = conf_cpu - self.mean_conf
        self.mean_conf += delta / k
        self.M2_conf += delta * (conf_cpu - self.mean_conf)

    def finalize(self):
        if self.epoch_count < 2:
            var_loss = torch.zeros_like(self.mean_loss)
            var_conf = torch.zeros_like(self.mean_conf)
        else:
            var_loss = self.M2_loss / (self.epoch_count - 1)
            var_conf = self.M2_conf / (self.epoch_count - 1)

        return (
            self.mean_loss.numpy(),
            var_loss.numpy(),
            self.mean_conf.numpy(),
            var_conf.numpy(),
            self.forgetting.numpy(),
        )


# ----------------------------
# 4) Full-batch training + logging
# ----------------------------
def train_and_collect_dynamics(
    model,
    pos_idx: torch.Tensor,          # [N,3] on device
    num_entities: int,
    epochs: int = 50,
    num_negs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    device = pos_idx.device
    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tracker = TripleDynamicsTracker(n_triples=pos_idx.size(0))

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)

        # --- YOUR "no batch" forward for positive logits ---
        s_pos = model.forward_triples(pos_idx).reshape(-1)  # [N]

        # negatives: [N,k,3] -> score -> [N,k]
        neg_idx = corrupt_triples(pos_idx, num_entities=num_entities, num_negs=num_negs)
        s_neg = model.forward_triples(neg_idx.reshape(-1, 3)).reshape(pos_idx.size(0), num_negs)

        # Pairwise diff: positive should beat (mean) negatives
        diff = s_pos - s_neg.mean(dim=1)  # [N]

        # Per-triple logistic loss: softplus(-diff) = log(1 + exp(-diff))
        loss_per = F.softplus(-diff)      # [N]
        loss = loss_per.mean()

        loss.backward()
        opt.step()

        # Training dynamics signals (on CPU, per triple)
        with torch.no_grad():
            conf = torch.sigmoid(diff)          # [N] pseudo-confidence: pos > negs
            correct = diff > 0                 # [N] did pos beat mean neg?
            tracker.update_epoch(
                loss_per.detach().cpu(),
                conf.detach().cpu(),
                correct.detach().cpu(),
            )

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"epoch={ep:03d}  loss={loss.item():.6f}  conf_mean={conf.mean().item():.4f}")

    return tracker.finalize()


# ----------------------------
# 5) Labeling: helpful / neutral / harmful
# ----------------------------
def label_triples(mean_loss, var_loss, mean_conf, var_conf, forgetting):
    """
    Pure heuristic, but works well enough to start:
      harmful: low confidence + high loss + (unstable or lots of forgetting)
      helpful: high confidence + moderate loss (not trivial) + low forgetting
      neutral: everything else
    """
    N = len(mean_loss)
    labels = np.array(["neutral"] * N, dtype=object)

    q_loss_lo, q_loss_hi = np.quantile(mean_loss, [0.33, 0.66])
    q_var_hi = np.quantile(var_loss, 0.66)
    q_conf_lo, q_conf_hi = np.quantile(mean_conf, [0.33, 0.66])
    q_forget_hi = np.quantile(forgetting, 0.66)

    harmful = (mean_conf <= q_conf_lo) & (mean_loss >= q_loss_hi) & (
        (var_loss >= q_var_hi) | (forgetting >= q_forget_hi)
    )

    helpful = (mean_conf >= q_conf_hi) & (mean_loss > q_loss_lo) & (mean_loss < q_loss_hi) & (
        forgetting <= q_forget_hi
    )

    labels[harmful] = "harmful"
    labels[helpful & ~harmful] = "helpful"
    return labels


# ----------------------------
# 6) End-to-end runner
# ----------------------------
def run_pipeline(
    train_triples_path: str,
    dicee_model_path: str,  # path to experiment folder OR whatever you already use
    device: str = "cuda",
    epochs: int = 50,
    num_negs: int = 20,
    lr: float = 1e-3,
    out_csv: str = "triple_dynamics_labeled.csv",
):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Load triples
    triples = load_triples(train_triples_path)

    # Load oracle + maps
    oracle = KGE(path=str(dicee_model_path))
    entity_to_idx = oracle.entity_to_idx
    relation_to_idx = oracle.relation_to_idx

    # Get actual torch model (try common patterns)
    model = getattr(oracle, "model", None)
    if model is None:
        # If your code already has `model`, pass it instead of using oracle.model
        raise AttributeError("Could not find oracle.model. Use your existing `model` and call train_and_collect_dynamics(...) directly.")

    # Map triples -> idx tensor
    pos_idx, kept_triples, dropped = triples_to_idx_with_maps(
        triples, entity_to_idx, relation_to_idx, device=device
    )
    print(f"mapped_triples={len(kept_triples)}  dropped={dropped}")

    num_entities = len(entity_to_idx)

    # Train + collect dynamics
    mean_loss, var_loss, mean_conf, var_conf, forgetting = train_and_collect_dynamics(
        model=model,
        pos_idx=pos_idx,
        num_entities=num_entities,
        epochs=epochs,
        num_negs=num_negs,
        lr=lr,
    )

    # Label
    labels = label_triples(mean_loss, var_loss, mean_conf, var_conf, forgetting)

    # Save results
    df = pd.DataFrame({
        "h": [x[0] for x in kept_triples],
        "r": [x[1] for x in kept_triples],
        "t": [x[2] for x in kept_triples],
        "mean_loss": mean_loss,
        "var_loss": var_loss,
        "mean_conf": mean_conf,
        "var_conf": var_conf,
        "forgetting": forgetting,
        "label": labels,
    })
    df.to_csv(out_csv, index=False)

    # quick breakdown
    counts = df["label"].value_counts().to_dict()
    print("label_counts:", counts)
    print(f"saved: {out_csv}")

    return df


# ----------------------------
# 7) If you already have `model`
# ----------------------------
def compute_logits_no_batch(model, idx_tensor):
    """
    This is your original snippet, but without batching.
    """
    with torch.no_grad():
        logits = model.forward_triples(idx_tensor).reshape(-1)
    return logits
