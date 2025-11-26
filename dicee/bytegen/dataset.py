from typing import List, Tuple,Dict
import torch
from torch.utils.data import Dataset
from dicee.bytegen.tokenizer import ByteTokenizer
import os
import random

# Data Loading
class ByteGenDataset(Dataset):
    """
    Standard Random Walk Dataset.
    """
    def __init__(self, folder_path: str, tokenizer: ByteTokenizer, split: str = 'train', block_size: int = 128, inverse: bool = True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.triples: List[Tuple[tuple, tuple, tuple]] = []
        self.adj: Dict[tuple, List[Tuple[tuple, tuple]]] = {}
        
        file_path = os.path.join(folder_path, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return

        print(f"Loading {split} from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    parts = line.strip().split()
                if len(parts) < 3: continue
                h, r, t = parts[0], parts[1], parts[2]
                self._add_triple(h, r, t)
                
                # Add inverse relations for training to double graph density
                if inverse and split == 'train':
                    self._add_triple(t, "INV_" + r, h)

    def _add_triple(self, h_str, r_str, t_str):
        h = tuple(self.tokenizer.encode(h_str))
        r = tuple(self.tokenizer.encode(r_str))
        t = tuple(self.tokenizer.encode(t_str))
        self.triples.append((h, r, t))
        if h not in self.adj: self.adj[h] = []
        self.adj[h].append((r, t))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Start specific triple
        h, r, t = self.triples[idx]
        
        # Build sequence: [H] [SEP_HR] [R] [SEP_RT] [T]
        seq = list(h) + [self.tokenizer.sep_hr_token_id] + list(r) + [self.tokenizer.sep_rt_token_id] + list(t)
        if len(seq) >= self.block_size:
            print(f"Sequence for triple {h} {r} {t} is longer than block size.")
            print(f"Increase block size to at least {len(seq)}")
            exit(1)
        # random Walk
        curr = t
        while len(seq) < self.block_size:
            if curr not in self.adj: break
            r_next, t_next = random.choice(self.adj[curr])
            
            # Append: <SEP_HR> R <SEP_RT> T <SEP_HR> R <SEP_RT> T ...
            extension = [self.tokenizer.sep_hr_token_id] + list(r_next) + [self.tokenizer.sep_rt_token_id] + list(t_next)
            seq.extend(extension)
            curr = t_next

        # Truncate/Pad
        if len(seq) > self.block_size:
            seq = seq[:self.block_size]
        elif len(seq) < self.block_size:
            print("dead end tail + using padding")
            # TODO: LF we have to think how to deal with this :) invese relations prevent this, e.g. A -> B inv_-> A -> ...
            seq.extend([self.tokenizer.pad_token_id] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)