import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
import os
from typing import List
from tqdm import tqdm
import numpy as np
from dicee.bytegen.tokenizer import ByteTokenizer
from dicee.bytegen.dataset import ByteGenDataset
from dicee.bytegen.transformer import Block

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

    def generate(self, idx, tokenizer, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            with torch.no_grad():
                logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next.item() == tokenizer.pad_token_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Evaluation System 

class Evaluator:
    def __init__(self, model: ByteGenModel, train_dataset: ByteGenDataset, test_dataset: ByteGenDataset, tokenizer: ByteTokenizer):
        self.model = model
        self.device = model.config.device
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        self.tokenizer = tokenizer
        
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

    def _score_candidates(self, h: tuple, r: tuple, candidates: List[tuple], batch_size: int):
        prefix = list(h) + [self.tokenizer.sep_hr_token_id] + list(r) + [self.tokenizer.sep_rt_token_id]
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
                    
                    input_ids.append(seq + [self.tokenizer.pad_token_id] * (len(prefix) + max_cand_len - len(seq)))
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

# Training System

class Trainer:
    def __init__(self, model: ByteGenModel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, optimizer: torch.optim.Optimizer = None):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.device = config.device
        self.tokenizer = tokenizer

    def train(self, epochs: int = 500):
        print(f"Starting training for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Input: x[:-1], Target: x[1:]
                logits = self.model(batch[:, :-1])
                targets = batch[:, 1:]
                
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size), 
                    targets.reshape(-1), 
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

if __name__ == "__main__":
    # Setup
    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    
    # Initialize Tokenizer
    tokenizer = ByteTokenizer()
    
    conf = ByteGenConfig(
        block_size=128, 
        n_layer=4, 
        n_head=4, 
        n_embd=256, 
        dropout=0.1, 
        batch_size=512,
        lr=0.001,
        vocab_size=tokenizer.vocab_size
    )
    
    # Dataset
    train_ds = ByteGenDataset(dataset_path, tokenizer, split='train', block_size=conf.block_size, inverse=True)
    test_ds = ByteGenDataset(dataset_path, tokenizer, split='test', block_size=conf.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = ByteGenModel(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    
    # Trainer
    EPOCHS = 300
    trainer = Trainer(model, train_loader, conf, tokenizer, optimizer)
    trainer.train(EPOCHS)
            
    # Evaluate
    print("Training complete. Evaluating...")
    evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
    evaluator.evaluate()

    while True:
        print("\n--- Interactive Generation ---")
        head = input("Enter head entity (or 'q' to quit): ").strip()
        if head.lower() in ['q', 'quit']:
            break
            
        try:
            context = tokenizer.encode(head)
            x = torch.tensor([context], dtype=torch.long, device=conf.device)
            y = model.generate(x, tokenizer, max_new_tokens=64, temperature=0.8, top_k=10)
            
            out = y[0].tolist()
            decoded = tokenizer.decode(out)
                
            print(f"Generated: {decoded}")
            
        except Exception as e:
            print(f"Error: {e}")
    