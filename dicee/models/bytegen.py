import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import os
import random
import math
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm
import numpy as np

# Configuration
@dataclass
class ByteGenConfig:
    block_size: int = 256
    vocab_size: int = 259  # 256 bytes + PAD(256) + SEP_HR(257) + SEP_RT(258)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    lr: float = 1e-4
    batch_size: int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class SpecialTokens:
    PAD = 256
    SEP_HR = 257
    SEP_RT = 258

# Data Loading
class ByteGenDataset(Dataset):
    """
    Standard Random Walk Dataset.
    """
    def __init__(self, folder_path: str, split: str = 'train', block_size: int = 128, inverse: bool = True):
        self.block_size = block_size
        self.triples: List[Tuple[bytes, bytes, bytes]] = []
        self.adj: Dict[bytes, List[Tuple[bytes, bytes]]] = {}
        
        file_path = os.path.join(folder_path, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return

        print(f"Loading {split} from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                h, r, t = parts[0], parts[1], parts[2]
                self._add_triple(h, r, t)
                
                # Add inverse relations for training to double graph density
                if inverse and split == 'train':
                    self._add_triple(t, "INV_" + r, h)

    def _add_triple(self, h_str, r_str, t_str):
        h, r, t = h_str.encode('utf-8'), r_str.encode('utf-8'), t_str.encode('utf-8')
        self.triples.append((h, r, t))
        if h not in self.adj: self.adj[h] = []
        self.adj[h].append((r, t))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Start specific triple
        h, r, t = self.triples[idx]
        
        # Build sequence: [H] [SEP_HR] [R] [SEP_RT] [T]
        seq = list(h) + [SpecialTokens.SEP_HR] + list(r) + [SpecialTokens.SEP_RT] + list(t)
        
        # Random Walk
        curr = t
        while len(seq) < self.block_size:
            if curr not in self.adj: break
            r_next, t_next = random.choice(self.adj[curr])
            
            # Append: <SEP_HR> R <SEP_RT> T <SEP_HR> R <SEP_RT> T ...
            extension = [SpecialTokens.SEP_HR] + list(r_next) + [SpecialTokens.SEP_RT] + list(t_next)
            seq.extend(extension)
            curr = t_next

        # Truncate/Pad
        if len(seq) > self.block_size:
            seq = seq[:self.block_size]
        else:
            seq.extend([SpecialTokens.PAD] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)

# Model Components

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ByteGenConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate q, k, v
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config: ByteGenConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ GPT-Style Block """
    def __init__(self, config: ByteGenConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ByteGenModel(nn.Module):
    def __init__(self, config: ByteGenConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# Evaluation System 

class Evaluator:
    def __init__(self, model: ByteGenModel, train_dataset: ByteGenDataset, test_dataset: ByteGenDataset):
        self.model = model
        self.device = model.config.device
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        
        # Build Entity Catalog
        self.entities = sorted(list(set([t[0] for t in train_dataset.triples] + [t[2] for t in train_dataset.triples] +
                                        [t[0] for t in test_dataset.triples] + [t[2] for t in test_dataset.triples])))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        
        # Build Known Facts for Filtering
        self.known_facts = set()
        for h, r, t in (train_dataset.triples + test_dataset.triples):
            self.known_facts.add((h, r, t))
            
    def evaluate(self, limit: int = None, batch_size: int = 64):
        self.model.eval()
        ranks = []
        hits1, hits3, hits10 = 0, 0, 0
        
        # Pre-calculate candidate bytes 
        candidates = self.entities
        
        triples = self.test_ds.triples[:limit] if limit else self.test_ds.triples
        print(f"Evaluating {len(triples)} triples...")

        for h, r, t in tqdm(triples):
            # We want to score: h + r + ?
            scores = self._score_candidates(h, r, candidates, batch_size)
            
            # Identify target rank
            if t not in self.entity_to_idx: continue
            target_idx = self.entity_to_idx[t]
            target_score = scores[target_idx]
            
            # Filter Loop: Penalize known facts that are NOT the target
            for i, cand in enumerate(candidates):
                if cand == t: continue # Don't filter the ground truth
                if (h, r, cand) in self.known_facts:
                    scores[i] = -float('inf')

            # Calculate Rank
            # Rank = (count of scores > target_score) + 1
            rank = np.sum(scores > target_score) + 1
            
            ranks.append(rank)
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 10: hits10 += 1
            
        mrr = np.mean(1.0 / np.array(ranks))
        print(f"MRR: {mrr:.4f} | H@1: {hits1/len(ranks):.4f} | H@3: {hits3/len(ranks):.4f} | H@10: {hits10/len(ranks):.4f}")

    def _score_candidates(self, h: bytes, r: bytes, candidates: List[bytes], batch_size: int):
        prefix = list(h) + [SpecialTokens.SEP_HR] + list(r) + [SpecialTokens.SEP_RT]
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i+batch_size]
                
                # Dynamic Batching
                max_cand_len = max([len(c) for c in batch])
                input_ids = []
                slices = [] # (start_idx, end_idx) used for gathering probabilities
                
                for cand in batch:
                    seq = prefix + list(cand)
                    # We predict the candidate bytes.
                    # The prediction for seq[k] comes from seq[k-1]
                    # Start of candidate in seq is len(prefix). 
                    # So we look at logits at indices: len(prefix)-1 ... to ... end-1
                    start_logit_idx = len(prefix) - 1
                    end_logit_idx = len(seq) - 1
                    
                    input_ids.append(seq + [SpecialTokens.PAD] * (len(prefix) + max_cand_len - len(seq)))
                    slices.append((start_logit_idx, end_logit_idx, len(cand)))
                
                x = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                logits = self.model(x)
                log_probs = F.log_softmax(logits, dim=-1)
                
                for b, (s, e, l) in enumerate(slices):
                    # Gather scores for the specific bytes of the candidate
                    # Target indices are x[b, s+1 : e+1]
                    relevant_logits = log_probs[b, s:e, :]
                    targets = x[b, s+1 : e+1]
                    
                    # Sum log_probs for the candidate string
                    token_scores = relevant_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                    score = token_scores.sum().item() / l # Normalize by length
                    all_scores.append(score)
                    
        return np.array(all_scores)



if __name__ == "__main__":
    # Setup
    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    conf = ByteGenConfig(
        block_size=256, 
        n_layer=6, 
        n_head=4, 
        n_embd=256, 
        dropout=0.3, # Maintained original dropout
        lr=1e-4
    )
    
    # Dataset
    train_ds = ByteGenDataset(dataset_path, split='train', block_size=conf.block_size)
    test_ds = ByteGenDataset(dataset_path, split='test', block_size=conf.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = ByteGenModel(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    
    # Training Loop
    EPOCHS = 50
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            batch = batch.to(conf.device)
            
            # Input: x[:-1], Target: x[1:]
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            
            loss = F.cross_entropy(
                logits.reshape(-1, conf.vocab_size), 
                targets.reshape(-1), 
                ignore_index=SpecialTokens.PAD
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
    # Evaluate
    print("Training complete. Evaluating...")
    evaluator = Evaluator(model, train_ds, test_ds)
    evaluator.evaluate()