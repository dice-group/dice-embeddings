from typing import List, Tuple,Dict, Union
import torch
from torch.utils.data import Dataset
from dicee.bytegen.tokenizer import ByteTokenizer, BPETokenizer
import os
import random
from collections import deque


# Data Loading
class ByteGenDataset(Dataset):
    """
    Standard Random Walk Dataset.
    """
    def __init__(self, folder_path: str, tokenizer: Union[ByteTokenizer, BPETokenizer], split: str = 'train', block_size: int = 128, inverse: bool = True):
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
        seq = [self.tokenizer.eos_token_id] + list(h) + \
              [self.tokenizer.sep_hr_token_id] + list(r) + \
              [self.tokenizer.sep_rt_token_id] + list(t)
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
            # TODO: LF we have to think how to deal with this :) invese relations prevent this, e.g. A -> B inv_-> A -> ...
            seq.extend([self.tokenizer.pad_token_id] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)


class IsolatedTripleDataset(Dataset):
    """
    Isolated Triple Dataset - trains on single triples only.
    This aligns training distribution with evaluation (single triple scoring).
    
    Block size can be:
    - Explicitly provided: validates that it's sufficient
    - None: auto-calculated to fit the longest triple in this split
    
    For evaluation, use compute_required_block_size() with both train and test datasets
    to get a block_size that works for all entity combinations.
    """
    def __init__(self, folder_path: str, tokenizer: Union[ByteTokenizer, BPETokenizer], split: str = 'train', block_size: int = None, inverse: bool = False):
        self.tokenizer = tokenizer
        self.triples: List[Tuple[tuple, tuple, tuple]] = []
        self.adj: Dict[tuple, List[Tuple[tuple, tuple]]] = {}
        self.entities: set = set()  # Track all entities for eval block size calculation
        self.max_seq_len = 0  # Track maximum sequence length for training
        self.max_prefix_len = 0  # Track max prefix length: [EOS] + H + [SEP_HR] + R + [SEP_RT]
        self.max_entity_len = 0  # Track max entity length (for eval: any entity can be candidate tail)
        
        file_path = os.path.join(folder_path, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            self.block_size = block_size or 128
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
                
                if inverse and split == 'train':
                    self._add_triple(t, "INV_" + r, h)
        
        # Set block_size: auto-calculate if None, otherwise validate
        if block_size is None:
            self.block_size = self.max_seq_len
            print(f"  [IsolatedTripleDataset] Auto-calculated block_size: {self.block_size}")
        else:
            self.block_size = block_size
            self._validate_block_size()
        
        print(f"  [IsolatedTripleDataset] Loaded {len(self.triples)} triples, {len(self.entities)} unique entities")
        print(f"  [IsolatedTripleDataset] Max seq len: {self.max_seq_len}, Max prefix: {self.max_prefix_len}, Max entity: {self.max_entity_len}")

    def _add_triple(self, h_str, r_str, t_str):
        h = tuple(self.tokenizer.encode(h_str))
        r = tuple(self.tokenizer.encode(r_str))
        t = tuple(self.tokenizer.encode(t_str))
        self.triples.append((h, r, t))
        if h not in self.adj: self.adj[h] = []
        self.adj[h].append((r, t))
        
        # Track entities
        self.entities.add(h)
        self.entities.add(t)
        
        # Calculate sequence length: [EOS] + H + [SEP_HR] + R + [SEP_RT] + T + [EOS]
        seq_len = 1 + len(h) + 1 + len(r) + 1 + len(t) + 1
        self.max_seq_len = max(self.max_seq_len, seq_len)
        
        # Track prefix length: [EOS] + H + [SEP_HR] + R + [SEP_RT]
        prefix_len = 1 + len(h) + 1 + len(r) + 1
        self.max_prefix_len = max(self.max_prefix_len, prefix_len)
        
        # Track entity lengths (both h and t can be candidate tails during eval)
        self.max_entity_len = max(self.max_entity_len, len(h), len(t))

    def _validate_block_size(self):
        """Ensure block_size is sufficient for all triples."""
        if self.max_seq_len > self.block_size:
            raise ValueError(
                f"Block size {self.block_size} is too small! "
                f"Minimum required: {self.max_seq_len}. "
                f"Some triples would be truncated. Please increase block_size."
            )
        print(f"  [IsolatedTripleDataset] Block size {self.block_size} validated (min needed: {self.max_seq_len})")

    @staticmethod
    def compute_required_block_size(*datasets: 'IsolatedTripleDataset') -> int:
        """
        Compute the minimum block_size needed for evaluation across multiple datasets.
        
        During evaluation, we score ALL entities as candidate tails for each (h, r, ?) query.
        This means we need: max_prefix_len + max_entity_len + 1 (for suffix EOS)
        
        This is a STATIC method that should be called AFTER loading all datasets to get
        a block_size that works for both training AND evaluation.
        
        Args:
            *datasets: One or more IsolatedTripleDataset instances (e.g., train_ds, test_ds)
            
        Returns:
            Minimum block_size required for evaluation (use this for model config)
            
        Example:
            train_ds = IsolatedTripleDataset(path, tokenizer, 'train', block_size=None)
            test_ds = IsolatedTripleDataset(path, tokenizer, 'test', block_size=None)
            eval_block_size = IsolatedTripleDataset.compute_required_block_size(train_ds, test_ds)
            # Now update both datasets and use eval_block_size for the model
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        # Collect all entities and find max prefix across all datasets
        all_entities = set()
        max_prefix_len = 0
        
        for ds in datasets:
            all_entities.update(ds.entities)
            max_prefix_len = max(max_prefix_len, ds.max_prefix_len)
        
        # Find longest entity (any entity could be a candidate tail)
        max_entity_len = max(len(e) for e in all_entities) + 1  # +1 for suffix EOS
        
        required_block_size = max_prefix_len + max_entity_len
        
        print(f"  [Eval Block Size Calculation]")
        print(f"    Total unique entities: {len(all_entities)}")
        print(f"    Max prefix length: {max_prefix_len}")
        print(f"    Max entity length (with EOS): {max_entity_len}")
        print(f"    Required block_size for eval: {required_block_size}")
        
        return required_block_size
    
    def get_eval_block_size(self, other_dataset: 'IsolatedTripleDataset' = None) -> int:
        """
        Calculate the minimum block_size needed for evaluation.
        DEPRECATED: Use compute_required_block_size() static method instead.
        """
        datasets = [self] + ([other_dataset] if other_dataset else [])
        return IsolatedTripleDataset.compute_required_block_size(*datasets)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Build isolated triple sequence: [EOS] [H] [SEP_HR] [R] [SEP_RT] [T] [EOS]
        seq = [self.tokenizer.eos_token_id] + list(h) + \
              [self.tokenizer.sep_hr_token_id] + list(r) + \
              [self.tokenizer.sep_rt_token_id] + list(t) + \
              [self.tokenizer.eos_token_id]
        
        # Pad to block_size (no truncation needed - validated in __init__)
        if len(seq) < self.block_size:
            seq.extend([self.tokenizer.pad_token_id] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)


class ByteGenBFSDataset(Dataset):
    """
    BFS Dataset.
    """
    def __init__(self, folder_path: str, tokenizer: Union[ByteTokenizer, BPETokenizer], split: str = 'train', block_size: int = 128, inverse: bool = False):
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
        h_start, r_start, t_start = self.triples[idx]
        
        # Build sequence: [H] [SEP_HR] [R] [SEP_RT] [T] [EOS]
        seq = seq = [self.tokenizer.eos_token_id] + list(h_start) + \
          [self.tokenizer.sep_hr_token_id] + list(r_start) + \
          [self.tokenizer.sep_rt_token_id] + list(t_start) + \
          [self.tokenizer.eos_token_id]
        
        if len(seq) >= self.block_size:
             seq = seq[:self.block_size]
             return torch.tensor(seq, dtype=torch.long)
             
        # BFS
        queue = deque([h_start])
        visited_nodes = {h_start}
        emitted_triples = set([(h_start, r_start, t_start)])
        
        if t_start not in visited_nodes:
            queue.append(t_start)
            visited_nodes.add(t_start)
            
        while len(seq) < self.block_size and queue:
            curr = queue.popleft()
            neighbors = self.adj.get(curr, [])
            neighbors = list(neighbors)
            random.shuffle(neighbors)
            
            for r, t in neighbors:
                if (curr, r, t) in emitted_triples:
                    continue
                
                # Triple: curr r t EOS
                triple_seq = list(curr) + [self.tokenizer.sep_hr_token_id] + list(r) + [self.tokenizer.sep_rt_token_id] + list(t) + [self.tokenizer.eos_token_id]
                
                if len(seq) + len(triple_seq) > self.block_size:
                    remaining = self.block_size - len(seq)
                    seq.extend(triple_seq[:remaining])
                    break 
                
                seq.extend(triple_seq)
                emitted_triples.add((curr, r, t))
                
                if t not in visited_nodes:
                    visited_nodes.add(t)
                    queue.append(t)
                    
                if len(seq) >= self.block_size:
                    break
            
            if len(seq) >= self.block_size:
                break

        # Padding
        if len(seq) < self.block_size:
            seq.extend([self.tokenizer.pad_token_id] * (self.block_size - len(seq)))
        elif len(seq) > self.block_size:
             seq = seq[:self.block_size]
             
        return torch.tensor(seq, dtype=torch.long)
