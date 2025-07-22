from random import sample
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error

def get_oracle_score(triple, oracle):
    h, r, t = triple
    idxs = torch.LongTensor([
        oracle.entity_to_idx[h],
        oracle.relation_to_idx[r],
        oracle.entity_to_idx[t]
    ]).unsqueeze(0)
    with torch.no_grad():
        logit = oracle.model.forward_triples(idxs)
        return torch.sigmoid(logit).item()

class ProxyDataset(Dataset):
    def __init__(self, triples, entity_emb, relation_emb, oracle):
        self.xs, self.ys = [], []
        for h, r, t in triples:
            h_vec = entity_emb[h]
            r_vec = relation_emb[r]
            t_vec = entity_emb[t]
            y = get_oracle_score((h, r, t), oracle)
            self.xs.append(torch.cat([h_vec, r_vec, t_vec]))
            self.ys.append(torch.tensor([y], dtype=torch.float32))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

"""
class ProxyModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate=0.2):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
"""
class ProxyModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        layers = []
        prev_dim = 3 * input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)

def uncertainty_sampling(model, unlabeled, ee, re, top_k):
    model.eval()
    scored = []
    with torch.no_grad():
        for triple in unlabeled:
            x = torch.cat([ee[triple[0]], re[triple[1]], ee[triple[2]]]).unsqueeze(0)
            p = model(x).item()
            scored.append((abs(p - 0.5), triple))

    return [triple for _, triple in sorted(scored, key=lambda x: x[0])[:top_k]] #, reverse=descending

def train_proxy(model, dl, epochs, lr):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    for e in range(epochs):
        tot = 0
        for x, y in dl:
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            tot += loss.item() * x.size(0)

def evaluate_proxy(model, triples, ee, re, oracle):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for h, r, t in triples:
            x = torch.cat([ee[h], re[r], ee[t]]).unsqueeze(0)
            preds.append(model(x).item())
            targets.append(get_oracle_score((h, r, t), oracle))
    mse = mean_squared_error(targets, preds)
    print(f"  → MSE = {mse:.14f}")
    return mse


def active_learning_loop(all_triples,
    ee,
    re,
    oracle,
    initial_k,
    query_k,
    max_rounds,
    patience,
    min_delta,
    train_proxy_lr,
    hidden_dims,
    dropout):

    labeled = sample(all_triples, initial_k)
    unlabeled = list(set(all_triples) - set(labeled))
    #input_dim = ee[next(iter(ee))].shape[0]*2 + re[next(iter(re))].shape[0]

    sample_ent = next(iter(ee.values()))
    sample_rel = next(iter(re.values()))
    input_dim = sample_ent.shape[0]
    print("*********************", input_dim)


    model = ProxyModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)

    eval_log = []
    best_mse = float('inf')
    no_improve_count = 0


    for rd in range(1, max_rounds + 1):
        print(f"\n=== Round {rd}: labeled {len(labeled)}, unlabeled {len(unlabeled)} ===")

        ds = ProxyDataset(labeled, ee, re, oracle)
        dl = DataLoader(ds, batch_size=512, shuffle=True)

        train_proxy(model, dl, epochs=100, lr=train_proxy_lr)
        mse = evaluate_proxy(model, labeled, ee, re, oracle)

        eval_log.append({"round": rd, "labeled_size": len(labeled), "mse": mse})

        if best_mse - mse < min_delta:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping after {rd} rounds (no improvement for {patience} rounds).")

                print("--------------------------------------------------------------------------")
                break
        else:
            best_mse = mse
            no_improve_count = 0

        to_query = uncertainty_sampling(model, unlabeled, ee, re, top_k=query_k)
        labeled += to_query
        unlabeled = list(set(unlabeled) - set(to_query))

    return model, eval_log
