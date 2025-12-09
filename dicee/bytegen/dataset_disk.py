from collections import deque
import os
import pickle
import numpy as np
from tqdm import tqdm
from typing import Union, Tuple, List
from dicee.bytegen.tokenizer import train_bpe_tokenizer
from torch.utils.data import Dataset
import torch
import os
from numpy.lib.format import open_memmap 
from dicee.bytegen.dataset import ByteGenDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define dtypes for scale
# int64 is mandatory for indices if you have > 2 Billion entities/edges
# int16 is sufficient for token IDs (vocab size < 32,767)
IDX_DTYPE = np.int64 
TOK_DTYPE = np.int16

def preprocess_dataset(
    folder_path: str, 
    split: str, 
    tokenizer, 
    inverse: bool = True
):
    input_file = os.path.join(folder_path, f"{split}.txt")
    output_dir = os.path.join(folder_path, "processed_mmap")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Step 1: Scanning {split} to build ID maps ---")
    entity_to_id = {}
    relation_to_id = {}
    num_edges = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning"):
            parts = line.strip().split('\t')
            if len(parts) < 3: parts = line.strip().split()
            if len(parts) < 3: continue
            
            h, r, t = parts[0], parts[1], parts[2]
            
            if h not in entity_to_id: entity_to_id[h] = len(entity_to_id)
            if t not in entity_to_id: entity_to_id[t] = len(entity_to_id)
            if r not in relation_to_id: relation_to_id[r] = len(relation_to_id)
            
            num_edges += 1
            if inverse:
                inv_r = "INV_" + r
                if inv_r not in relation_to_id: relation_to_id[inv_r] = len(relation_to_id)
                num_edges += 1

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    print(f"Found: {num_entities:,} entities, {num_relations:,} relations, {num_edges:,} edges")

    # --- HELPER: WRITE TOKENS TO DISK AND GET MAX LEN ---
    def write_token_index(item_map, prefix):
        print(f"--- Tokenizing {prefix} to disk ---")
        sorted_items = sorted(item_map.items(), key=lambda x: x[1])
        
        temp_tokenized = []
        total_tokens = 0
        max_len = 0 # Track max length
        
        for string, _ in tqdm(sorted_items, desc="Tokenizing"):
            toks = tokenizer.encode(string)
            temp_tokenized.append(toks)
            total_tokens += len(toks)
            if len(toks) > max_len:
                max_len = len(toks)
            
        indptr_path = os.path.join(output_dir, f'{prefix}_tok_indptr.npy')
        values_path = os.path.join(output_dir, f'{prefix}_tok_values.npy')
        
        fp_indptr = open_memmap(indptr_path, mode='w+', dtype=IDX_DTYPE, shape=(len(item_map) + 1,))
        fp_values = open_memmap(values_path, mode='w+', dtype=TOK_DTYPE, shape=(total_tokens,))
        
        current_offset = 0
        for i, tokens in enumerate(tqdm(temp_tokenized, desc="Writing Tokens")):
            fp_indptr[i] = current_offset
            l = len(tokens)
            fp_values[current_offset : current_offset + l] = tokens
            current_offset += l
            
        fp_indptr[-1] = current_offset
        fp_indptr.flush()
        fp_values.flush()
        del temp_tokenized
        return max_len

    max_ent_len = write_token_index(entity_to_id, 'entity')
    max_rel_len = write_token_index(relation_to_id, 'relation')

    # Save metadata including Max Lengths
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({
            'num_entities': num_entities, 
            'num_relations': num_relations, 
            'num_edges': num_edges,
            'max_entity_len': max_ent_len,
            'max_relation_len': max_rel_len
        }, f)

    print("--- Step 3: Writing Graph Structure (CSR) ---")
    all_edges = np.zeros((num_edges, 3), dtype=IDX_DTYPE)
    cursor = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Edges"):
            parts = line.strip().split('\t')
            if len(parts) < 3: parts = line.strip().split()
            if len(parts) < 3: continue
            
            h, r, t = parts[0], parts[1], parts[2]
            h_id, r_id, t_id = entity_to_id[h], relation_to_id[r], entity_to_id[t]
            
            all_edges[cursor] = [h_id, r_id, t_id]
            cursor += 1
            if inverse:
                all_edges[cursor] = [t_id, relation_to_id["INV_"+r], h_id]
                cursor += 1
    
    print("Sorting edges by Head ID...")
    all_edges = all_edges[all_edges[:, 0].argsort()] 
    
    print("Building Index Arrays...")
    adj_indptr = open_memmap(os.path.join(output_dir, 'adj_indptr.npy'), mode='w+', dtype=IDX_DTYPE, shape=(num_entities + 1,))
    adj_indices = open_memmap(os.path.join(output_dir, 'adj_indices.npy'), mode='w+', dtype=IDX_DTYPE, shape=(num_edges,))
    adj_data = open_memmap(os.path.join(output_dir, 'adj_data.npy'), mode='w+', dtype=IDX_DTYPE, shape=(num_edges,))
    
    adj_indices[:] = all_edges[:, 2] # T
    adj_data[:] = all_edges[:, 1]    # R
    
    heads = all_edges[:, 0]
    unique_heads, return_index = np.unique(heads, return_index=True)
    
    adj_indptr[:] = num_edges 
    adj_indptr[unique_heads] = return_index
    
    # Fill gaps in indptr
    # Efficient fill: We just need to make sure empty rows point to the start of the next valid row
    for i in range(num_entities - 1, -1, -1):
        if adj_indptr[i] == num_edges and i < num_entities - 1:
             adj_indptr[i] = adj_indptr[i+1]
             
    adj_indptr[-1] = num_edges 

    adj_indptr.flush()
    adj_indices.flush()
    adj_data.flush()
    
    print("Preprocessing Complete.")

class DiskByteGenDataset(Dataset):
    def __init__(self, processed_dir: str, tokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load Metadata
        with open(os.path.join(processed_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            self.num_entities = meta['num_entities']
            self.num_edges = meta['num_edges']
        
        # --- MAP FILES ---
        # 1. Graph Topology (CSR Format)
        # indptr: points to start of neighbors for each head
        self.adj_indptr = np.load(os.path.join(processed_dir, 'adj_indptr.npy'), mmap_mode='r')
        # indices: contains the Tail IDs for every edge
        self.adj_indices = np.load(os.path.join(processed_dir, 'adj_indices.npy'), mmap_mode='r')
        # data: contains the Relation IDs for every edge
        self.adj_data = np.load(os.path.join(processed_dir, 'adj_data.npy'), mmap_mode='r')
        
        # 2. Node Content (Tokens)
        self.ent_tok_indptr = np.load(os.path.join(processed_dir, 'entity_tok_indptr.npy'), mmap_mode='r')
        self.ent_tok_values = np.load(os.path.join(processed_dir, 'entity_tok_values.npy'), mmap_mode='r')
        
        # 3. Edge Content (Relation Tokens)
        self.rel_tok_indptr = np.load(os.path.join(processed_dir, 'relation_tok_indptr.npy'), mmap_mode='r')
        self.rel_tok_values = np.load(os.path.join(processed_dir, 'relation_tok_values.npy'), mmap_mode='r')

        # Cache special tokens
        self.eos = [tokenizer.eos_token_id]
        self.sep_hr = [tokenizer.sep_hr_token_id]
        self.sep_rt = [tokenizer.sep_rt_token_id]
        self.pad = tokenizer.pad_token_id

    def _get_entity_tokens(self, ent_id):
        start = self.ent_tok_indptr[ent_id]
        end = self.ent_tok_indptr[ent_id + 1]
        # Copy to list to ensure it's standard python list for concat
        return self.ent_tok_values[start:end].tolist()

    def _get_relation_tokens(self, rel_id):
        start = self.rel_tok_indptr[rel_id]
        end = self.rel_tok_indptr[rel_id + 1]
        return self.rel_tok_values[start:end].tolist()

    def __len__(self):
        # iterate over edges
        return self.num_edges

    def __getitem__(self, idx):
        # Treat idx as an Edge Index, not an Entity Index.
        # We need to find (h, r, t) corresponding to the idx-th edge in the flattened arrays.
        
        # 1. Recover Head ID (h)
        # The indptr array is sorted. We find where 'idx' fits.
        # This tells us which node's neighbor list contains this specific edge index.
        # side='right' finds the first index > idx. Subtract 1 to get the actual head.
        h_id = np.searchsorted(self.adj_indptr, idx, side='right') - 1
        
        # 2. Recover Relation (r) and Tail (t) directly from flattened arrays
        r_id = self.adj_data[idx]
        t_id = self.adj_indices[idx]
        
        # 3. Decode Tokens
        h_toks = self._get_entity_tokens(h_id)
        r_toks = self._get_relation_tokens(r_id)
        t_toks = self._get_entity_tokens(t_id)

        # 4. Construct Start Sequence: [EOS] H [SEP] R [SEP] T
        seq = self.eos + h_toks + self.sep_hr + r_toks + self.sep_rt + t_toks

        # Length Check (matches original code behavior)
        if len(seq) >= self.block_size:
            # Truncate if too long immediately
            seq = seq[:self.block_size]
            return torch.tensor(seq, dtype=torch.long)

        # 5. Random Walk Extension
        # Current node is now T
        curr = t_id
        
        while len(seq) < self.block_size:
            # Look up neighbors of 'curr'
            start = self.adj_indptr[curr]
            end = self.adj_indptr[curr + 1]
            
            # Dead end check
            if start == end:
                break
            
            # Randomly select NEXT edge
            # Note: We use random offset relative to the start of this node's neighbors
            rand_offset = np.random.randint(0, end - start)
            next_edge_idx = start + rand_offset
            
            r_next_id = self.adj_data[next_edge_idx]
            t_next_id = self.adj_indices[next_edge_idx]
            
            r_next_toks = self._get_relation_tokens(r_next_id)
            t_next_toks = self._get_entity_tokens(t_next_id)
            
            # Extension: [SEP_HR] R [SEP_RT] T
            extension = self.sep_hr + r_next_toks + self.sep_rt + t_next_toks
            
            if len(seq) + len(extension) > self.block_size:
                seq.extend(extension[:self.block_size - len(seq)])
                break
            
            seq.extend(extension)
            curr = t_next_id

        # 6. Padding
        if len(seq) < self.block_size:
            seq.extend([self.pad] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)

class DiskIsolatedTripleDataset(Dataset):
    def __init__(self, processed_dir: str, tokenizer, block_size: int = None):
        self.tokenizer = tokenizer
        
        # Load Metadata
        meta_path = os.path.join(processed_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
             raise FileNotFoundError(f"Metadata not found at {meta_path}. Run preprocess_dataset_v2 first.")
             
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
            self.num_edges = self.meta['num_edges']
            self.num_entities = self.meta['num_entities']
            self.max_ent_len = self.meta['max_entity_len']
            self.max_rel_len = self.meta['max_relation_len']
        
        # Load Data Maps
        self.adj_indptr = np.load(os.path.join(processed_dir, 'adj_indptr.npy'), mmap_mode='r')
        self.adj_indices = np.load(os.path.join(processed_dir, 'adj_indices.npy'), mmap_mode='r') # T
        self.adj_data = np.load(os.path.join(processed_dir, 'adj_data.npy'), mmap_mode='r')    # R
        
        self.ent_tok_indptr = np.load(os.path.join(processed_dir, 'entity_tok_indptr.npy'), mmap_mode='r')
        self.ent_tok_values = np.load(os.path.join(processed_dir, 'entity_tok_values.npy'), mmap_mode='r')
        
        self.rel_tok_indptr = np.load(os.path.join(processed_dir, 'relation_tok_indptr.npy'), mmap_mode='r')
        self.rel_tok_values = np.load(os.path.join(processed_dir, 'relation_tok_values.npy'), mmap_mode='r')

        # Cache special tokens
        self.eos = [tokenizer.eos_token_id]
        self.sep_hr = [tokenizer.sep_hr_token_id]
        self.sep_rt = [tokenizer.sep_rt_token_id]
        self.pad = tokenizer.pad_token_id

        # Calculate max possible sequence length for a single triple
        # [EOS] H [SEP] R [SEP] T [EOS]
        self.max_seq_len = 1 + self.max_ent_len + 1 + self.max_rel_len + 1 + self.max_ent_len + 1
        
        # Determine Block Size
        if block_size is None:
            self.block_size = self.max_seq_len
            print(f"  [DiskIsolated] Auto-calculated block_size: {self.block_size}")
        else:
            self.block_size = block_size
            if self.block_size < self.max_seq_len:
                print(f"  [DiskIsolated] Warning: block_size {self.block_size} < max triple len {self.max_seq_len}. Truncation may occur.")

    def _get_entity_tokens(self, ent_id):
        return self.ent_tok_values[self.ent_tok_indptr[ent_id] : self.ent_tok_indptr[ent_id+1]].tolist()

    def _get_relation_tokens(self, rel_id):
        return self.rel_tok_values[self.rel_tok_indptr[rel_id] : self.rel_tok_indptr[rel_id+1]].tolist()

    @staticmethod
    def compute_required_block_size(*dataset_dirs: str) -> int:
        """
        Reads metadata from multiple processed directories to find the global
        max length required for evaluation (where any entity can be a tail).
        """
        global_max_prefix = 0
        global_max_ent = 0
        
        for d in dataset_dirs:
            meta_path = os.path.join(d, 'meta.pkl')
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                
            # Prefix: [EOS] + H + [SEP] + R + [SEP]
            # Max prefix occurs with max_ent_len and max_rel_len
            prefix_len = 1 + meta['max_entity_len'] + 1 + meta['max_relation_len'] + 1
            global_max_prefix = max(global_max_prefix, prefix_len)
            global_max_ent = max(global_max_ent, meta['max_entity_len'])

        # Eval Requirement: Prefix + Any Candidate Tail + EOS
        required = global_max_prefix + global_max_ent + 1
        
        print(f"  [Disk Eval Block Size] Max Prefix: {global_max_prefix}, Max Entity: {global_max_ent}")
        print(f"  [Disk Eval Block Size] Required: {required}")
        return required

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        # 1. Recover IDs from CSR
        # H is found by searching indptr. 
        h_id = np.searchsorted(self.adj_indptr, idx, side='right') - 1
        r_id = self.adj_data[idx]
        t_id = self.adj_indices[idx]
        
        # 2. Get Tokens
        h_toks = self._get_entity_tokens(h_id)
        r_toks = self._get_relation_tokens(r_id)
        t_toks = self._get_entity_tokens(t_id)
        
        # 3. Build Sequence
        seq = self.eos + h_toks + self.sep_hr + r_toks + self.sep_rt + t_toks + self.eos
        
        # 4. Pad / Truncate
        if len(seq) > self.block_size:
            seq = seq[:self.block_size]
        elif len(seq) < self.block_size:
            seq.extend([self.pad] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)

class DiskByteGenBFSDataset(Dataset):
    def __init__(self, processed_dir: str, tokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load Metadata and Maps (Same as above)
        with open(os.path.join(processed_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            self.num_edges = meta['num_edges']
            
        self.adj_indptr = np.load(os.path.join(processed_dir, 'adj_indptr.npy'), mmap_mode='r')
        self.adj_indices = np.load(os.path.join(processed_dir, 'adj_indices.npy'), mmap_mode='r')
        self.adj_data = np.load(os.path.join(processed_dir, 'adj_data.npy'), mmap_mode='r')
        
        self.ent_tok_indptr = np.load(os.path.join(processed_dir, 'entity_tok_indptr.npy'), mmap_mode='r')
        self.ent_tok_values = np.load(os.path.join(processed_dir, 'entity_tok_values.npy'), mmap_mode='r')
        self.rel_tok_indptr = np.load(os.path.join(processed_dir, 'relation_tok_indptr.npy'), mmap_mode='r')
        self.rel_tok_values = np.load(os.path.join(processed_dir, 'relation_tok_values.npy'), mmap_mode='r')

        self.eos = [tokenizer.eos_token_id]
        self.sep_hr = [tokenizer.sep_hr_token_id]
        self.sep_rt = [tokenizer.sep_rt_token_id]
        self.pad = tokenizer.pad_token_id

    def _get_entity_tokens(self, ent_id):
        return self.ent_tok_values[self.ent_tok_indptr[ent_id] : self.ent_tok_indptr[ent_id+1]].tolist()

    def _get_relation_tokens(self, rel_id):
        return self.rel_tok_values[self.rel_tok_indptr[rel_id] : self.rel_tok_indptr[rel_id+1]].tolist()

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        # 1. Start with specific triple (h, r, t)
        h_start_id = np.searchsorted(self.adj_indptr, idx, side='right') - 1
        r_start_id = self.adj_data[idx]
        t_start_id = self.adj_indices[idx]
        
        # Build initial sequence: [EOS] H [SEP] R [SEP] T [EOS]
        h_toks = self._get_entity_tokens(h_start_id)
        r_toks = self._get_relation_tokens(r_start_id)
        t_toks = self._get_entity_tokens(t_start_id)
        
        seq = self.eos + h_toks + self.sep_hr + r_toks + self.sep_rt + t_toks + self.eos
        
        if len(seq) >= self.block_size:
            return torch.tensor(seq[:self.block_size], dtype=torch.long)
            
        # 2. BFS
        # Queue stores Entity IDs (integers)
        queue = deque([h_start_id])
        visited_nodes = {h_start_id}
        # Emitted stores hashes of (h_id, r_id, t_id) to avoid duplicates
        emitted_triples = set([(h_start_id, r_start_id, t_start_id)])
        
        if t_start_id not in visited_nodes:
            queue.append(t_start_id)
            visited_nodes.add(t_start_id)
            
        while len(seq) < self.block_size and queue:
            curr_id = queue.popleft()
            
            # Get neighbors from CSR
            start = self.adj_indptr[curr_id]
            end = self.adj_indptr[curr_id + 1]
            if start == end: continue
            
            # Get range of edges for this node
            # We copy to memory to shuffle
            # Note: For extremely high-degree nodes, this copy might be slightly heavy, 
            # but usually fine for block_size constraints.
            neighbor_indices = np.arange(start, end)
            np.random.shuffle(neighbor_indices)
            
            curr_toks = self._get_entity_tokens(curr_id)
            
            for edge_idx in neighbor_indices:
                r_id = self.adj_data[edge_idx]
                t_id = self.adj_indices[edge_idx]
                
                if (curr_id, r_id, t_id) in emitted_triples:
                    continue
                
                # Retrieve tokens for R and T
                r_toks = self._get_relation_tokens(r_id)
                t_toks = self._get_entity_tokens(t_id)
                
                # Triple: H [SEP] R [SEP] T [EOS]
                # Note: 'curr_toks' is H here
                triple_seq = curr_toks + self.sep_hr + r_toks + self.sep_rt + t_toks + self.eos
                
                if len(seq) + len(triple_seq) > self.block_size:
                    remaining = self.block_size - len(seq)
                    seq.extend(triple_seq[:remaining])
                    break
                
                seq.extend(triple_seq)
                emitted_triples.add((curr_id, r_id, t_id))
                
                if t_id not in visited_nodes:
                    visited_nodes.add(t_id)
                    queue.append(t_id)
                
                if len(seq) >= self.block_size:
                    break
            
            if len(seq) >= self.block_size:
                break
                
        # Padding
        if len(seq) < self.block_size:
            seq.extend([self.pad] * (self.block_size - len(seq)))
            
        return torch.tensor(seq, dtype=torch.long)

if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), "KGs/WN18RR")
    tokenizer_path = "tokenizer.json"
    tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=1024, inverse=True)
    preprocess_dataset(dataset_path, "train", tokenizer)

    import random
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_ds = DiskByteGenDataset(os.path.join(dataset_path, "processed_mmap"), tokenizer, block_size=256)
    train_ds_not_disk = ByteGenDataset(dataset_path, tokenizer, split='train', block_size=256, inverse=True)
    print(len(train_ds), len(train_ds_not_disk))
    print(len(train_ds[0]), len(train_ds_not_disk[0]))
    exit()
    # 3. Create DataLoader
    # IMPORTANT: Use num_workers > 0 for disk-based datasets!
    # This allows the CPU to fetch disk pages while GPU computes.
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8,  
        pin_memory=True
    )

    # Test
    for batch in train_loader:
        print(batch.shape) # [32, 256]
        break