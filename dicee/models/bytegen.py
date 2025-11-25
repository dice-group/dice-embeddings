import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dicee.models.transformers import Block
from dataclasses import dataclass
import os
import random
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class ByteGenConfig:
    block_size: int = 128
    vocab_size: int = 259  # 256 bytes + PAD(256) + SEP_HR(257) + SEP_RT(258)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    causal: bool = True

class ByteGenDataset(Dataset):
    def __init__(self, folder_path: str, split: str = 'train', block_size: int = 128):
        self.block_size = block_size
        self.PAD = 256
        self.SEP_HR = 257
        self.SEP_RT = 258
        
        file_path = os.path.join(folder_path, f"{split}.txt")
        self.adj: Dict[bytes, List[Tuple[bytes, bytes]]] = {}
        self.triples: List[Tuple[bytes, bytes, bytes]] = []
        
        if os.path.exists(file_path):
            # Reading the file once to build the graph structure
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    h, r, t = parts[0], parts[1], parts[2]
                    hb = h.encode('utf-8')
                    rb = r.encode('utf-8')
                    tb = t.encode('utf-8')
                    
                    self.triples.append((hb, rb, tb))

                    if hb not in self.adj:
                        self.adj[hb] = []
                    self.adj[hb].append((rb, tb))
        else:
            print(f"Warning: {file_path} not found. Dataset will be empty.")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        if not self.triples:
            return torch.full((self.block_size,), self.PAD, dtype=torch.long)
            
        # Start with the specific triple at idx
        h, r, t = self.triples[idx]
        
        # Construct initial sequence: Head <SEP_HR> Relation <SEP_RT> Tail
        seq = list(h)
        seq.append(self.SEP_HR)
        seq.extend(list(r))
        seq.append(self.SEP_RT)
        seq.extend(list(t))
        
        # Continue random walk from Tail (which becomes the new Head)
        curr = t
        
        while len(seq) < self.block_size:
            if curr not in self.adj:
                break
            
            # Randomly choose next step
            r_next, t_next = random.choice(self.adj[curr])
            
            # Append sequence: <SEP_HR> Relation <SEP_RT> Tail
            seq.append(self.SEP_HR)
            seq.extend(list(r_next))
            seq.append(self.SEP_RT)
            seq.extend(list(t_next))
            
            curr = t_next
            
        # Truncate if too long
        if len(seq) > self.block_size:
            seq = seq[:self.block_size]
        
        # Pad if too short
        if len(seq) < self.block_size:
            seq.extend([self.PAD] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)

class ByteGenKGE(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.name = 'ByteGenKGE'
        self.config = ByteGenConfig()
        
        # Allow overriding config from args
        if hasattr(args, 'block_size'): self.config.block_size = args.block_size
        if hasattr(args, 'n_layer'): self.config.n_layer = args.n_layer
        if hasattr(args, 'n_head'): self.config.n_head = args.n_head
        if hasattr(args, 'n_embd'): self.config.n_embd = args.n_embd

        self.pad_idx = 256
        self.sep_hr_idx = 257
        self.sep_rt_idx = 258
        
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        # Weight Tying
        self.token_embedding.weight = self.lm_head.weight
        
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, x):
        device = x.device
        b, t = x.size()
        if t > self.config.block_size:
             raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def training_step(self, batch, batch_idx=None):
        input_ids = batch
        # Predict next token
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        
        logits = self.forward(x)
        loss_batch = self.loss(logits.reshape(-1, self.config.vocab_size), y.reshape(-1))
        
        return loss_batch

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens.
        idx: (B, T) tensor of indices
        """
        for _ in range(max_new_tokens):
            # crop to block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] 
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

def train_bytegen(model: ByteGenKGE, folder_path: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-4, device: str = 'cuda'):
    """
    Simple training loop for ByteGenKGE.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to cpu")
        device = 'cpu'
        
    train_dataset = ByteGenDataset(folder_path, split='train', block_size=model.config.block_size)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()
    
    print(f"Starting training on {device} with {len(train_dataset)} samples per epoch...")
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Path to UMLS dataset
    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    
    print(f"Testing ByteGenDataset with path: {dataset_path}")
    
    # Initialize dataset
    dataset = ByteGenDataset(dataset_path, split='train', block_size=128)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Fetch one batch
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        print("First sample in batch (raw indices):")
        print(batch[0])
        
        # Decode and print to verify structure
        sample = batch[0].tolist()
        decoded_parts = []
        current_bytes = []
        
        for idx in sample:
            if idx == 256:
                if current_bytes:
                    decoded_parts.append(bytes(current_bytes).decode('utf-8', errors='replace'))
                    current_bytes = []
                decoded_parts.append("<PAD>")
                break # Stop at padding
            elif idx == 257:
                if current_bytes:
                    decoded_parts.append(bytes(current_bytes).decode('utf-8', errors='replace'))
                    current_bytes = []
                decoded_parts.append("<SEP_HR>")
            elif idx == 258:
                if current_bytes:
                    decoded_parts.append(bytes(current_bytes).decode('utf-8', errors='replace'))
                    current_bytes = []
                decoded_parts.append("<SEP_RT>")
            else:
                current_bytes.append(idx)
        
        if current_bytes:
             decoded_parts.append(bytes(current_bytes).decode('utf-8', errors='replace'))
             
        print("Decoded structure:")
        print(" ".join(decoded_parts))
        break

    print("\nInitializing Model and running 1 training step...")
    class MockArgs:
        block_size = 128
        n_layer = 2
        n_head = 2
        n_embd = 128
    
    args = MockArgs()
    model = ByteGenKGE(args)
    # Test training loop for 1 epoch with small batch
    train_bytegen(model, dataset_path, epochs=100, batch_size=128, device='cuda')
