from typing import List, Dict, Tuple
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from dicee.bytegen.bytegen import ByteGenModel
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset, IsolatedTripleDataset
from dicee.bytegen.tokenizer import ByteTokenizer

class Evaluator:
    def __init__(self, model: ByteGenModel, train_dataset: ByteGenDataset, test_dataset: ByteGenDataset, tokenizer: ByteTokenizer, compile_model: bool = True):
        self.model = model
        self.device = model.config.device
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        self.tokenizer = tokenizer
        self._compiled = False
        self._compile_model = compile_model
        
        # 1. Determine suffix
        if isinstance(train_dataset, (ByteGenBFSDataset, IsolatedTripleDataset)):
            self.suffix_token_id = tokenizer.eos_token_id
        else:
            self.suffix_token_id = tokenizer.sep_hr_token_id
        
        # 2. Build Entity Catalog (SORTED BY LENGTH for batch efficiency)
        raw_entities = set([t[0] for t in train_dataset.triples] + [t[2] for t in train_dataset.triples] +
                           [t[0] for t in test_dataset.triples] + [t[2] for t in test_dataset.triples])
        self.entities = sorted(list(raw_entities), key=lambda x: (len(x), x))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        
        # 3. Build Known Facts Dictionary 
        # Using Pandas is ~20x faster than Python loops for large graphs
        print("Indexing known facts...")
        all_triples = train_dataset.triples + test_dataset.triples
        
        # Create DataFrame
        df = pd.DataFrame(all_triples, columns=['h', 'r', 't'])
        
        # Filter only entities present in our catalog 
        valid_entities = set(self.entity_to_idx.keys())
        df = df[df['t'].isin(valid_entities)]
        
        # Map targets to indices efficiently
        df['t_idx'] = df['t'].apply(lambda x: self.entity_to_idx[x])
        
        # Group by (h, r) -> list of t_idx
        # This creates the Dict[Tuple[str, str], List[int]]
        self.known_facts_dict = df.groupby(['h', 'r'])['t_idx'].apply(list).to_dict()
        
        # 4. Create Entity Tensor
        max_entity_len = max(len(e) for e in self.entities) + 1 
        print("Pre-loading entities to GPU...")
        entity_tensor = torch.full((len(self.entities), max_entity_len), 
                                   self.tokenizer.pad_token_id, 
                                   dtype=torch.long)
        entity_lengths = torch.zeros(len(self.entities), dtype=torch.long)
        
        for i, entity in enumerate(self.entities):
            e_seq = list(entity) + [self.suffix_token_id]
            l = len(e_seq)
            entity_tensor[i, :l] = torch.tensor(e_seq, dtype=torch.long)
            entity_lengths[i] = l
            
        self.entity_tensor = entity_tensor.to(self.device)
        self.entity_lengths = entity_lengths.to(self.device)    
    
    def _ensure_compiled(self):
        if self._compiled or not self._compile_model:
            return
        if str(self.device).startswith('cuda'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self._compiled = True
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")
            
    def evaluate(self, split: str = 'test', limit: int = None, batch_size: int = 2048):
        self._ensure_compiled()
        self.model.eval()
        
        # OPTIMIZATION: List to hold GPU tensors. 
        # We do NOT move to CPU inside the loop.
        gpu_ranks_list = []
        
        # Select Split
        if split == 'train':
            triples = self.train_ds.triples
        elif split == 'test':
            triples = self.test_ds.triples
        else:
            raise ValueError(f"Invalid split: {split}")
            
        triples = triples[:limit] if limit else triples
        
        # Group triples for evaluation
        hr_to_targets: Dict[Tuple[tuple, tuple], List[tuple]] = defaultdict(list)
        for h, r, t in triples:
            hr_to_targets[(h, r)].append(t)
        
        num_unique_hr = len(hr_to_targets)
        print(f"Evaluating on {split} split: {len(triples)} triples grouped into {num_unique_hr} (h, r) groups...")
        
        # --- MAIN EVALUATION LOOP ---
        # OPTIMIZATION: Inference Mode prevents graph creation for rank tensors
        with torch.inference_mode():
            for (h, r), target_entities in tqdm(hr_to_targets.items(), total=num_unique_hr):
                
                # 1. Get Indices of targets
                target_indices = [self.entity_to_idx[t] for t in target_entities if t in self.entity_to_idx]
                if not target_indices:
                    continue
                    
                target_indices_tensor = torch.tensor(target_indices, device=self.device, dtype=torch.long)
                
                # 2. Score Candidates (Returns Tensor on GPU)
                # scores shape: [Num_Candidates]
                scores = self._score_candidates(h, r, batch_size)
                
                # 3. Retrieve Target Scores BEFORE Filtering
                target_scores = scores[target_indices_tensor]
                
                # 4. Filter Known Facts on GPU
                known_indices = self.known_facts_dict.get((h, r), [])
                if known_indices:
                    mask_indices = torch.tensor(known_indices, device=self.device, dtype=torch.long)
                    # In-place fill with -inf
                    scores.index_fill_(0, mask_indices, -float('inf'))
                
                # 5. Vectorized Ranking (Broadcasting)
                # Compare [Num_Targets, 1] vs [1, Num_Candidates]
                better_than_target = scores.unsqueeze(0) > target_scores.unsqueeze(1)
                
                # Summing booleans gives the rank (0-based count of items better than target)
                # Add 1 for 1-based ranking
                ranks = better_than_target.sum(dim=1) + 1 
                
                # OPTIMIZATION: Keep 'ranks' on GPU. Do NOT call .tolist() or .cpu() here.
                # This allows the CPU to immediately schedule the next batch.
                gpu_ranks_list.append(ranks)
            
        # 6. Final Aggregation (Sync happens ONCE here)
        if not gpu_ranks_list:
            return {}
            
        # Concatenate all GPU tensors into one large tensor
        all_ranks_tensor = torch.cat(gpu_ranks_list)
        
        # Move to CPU once
        all_ranks_cpu = all_ranks_tensor.float().cpu().numpy()
        
        # Vectorized Metrics Calculation via NumPy
        mrr = (1.0 / all_ranks_cpu).mean()
        hits1 = (all_ranks_cpu <= 1).mean()
        hits3 = (all_ranks_cpu <= 3).mean()
        hits10 = (all_ranks_cpu <= 10).mean()
        
        results = {
            "mrr": mrr,
            "h1": hits1,
            "h3": hits3,
            "h10": hits10
        }
        print(f"MRR: {results['mrr']:.4f} | H@1: {results['h1']:.4f} | H@3: {results['h3']:.4f} | H@10: {results['h10']:.4f}")
        return results

    def _score_candidates(self, h: tuple, r: tuple, batch_size: int) -> torch.Tensor:
        """
        Computes scores using KV-caching and manual LogProb calculation.
        """
        # 1. Prepare Prefix
        prefix_seq = [self.tokenizer.eos_token_id] + list(h) + \
                    [self.tokenizer.sep_hr_token_id] + list(r) + \
                    [self.tokenizer.sep_rt_token_id]
        
        prefix_tensor = torch.tensor(prefix_seq, device=self.device, dtype=torch.long).unsqueeze(0)
        
        num_candidates = self.entity_tensor.size(0)
        
        # Pre-allocate GPU tensor
        # OPTIMIZATION: Keep as float32 only if needed, or keep same dtype as model output to save conversion
        gpu_scores = torch.zeros(num_candidates, device=self.device, dtype=torch.float32)
        
        # AMP Setup
        use_amp = str(self.device).startswith('cuda')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 2. Compute KV cache for prefix (ONCE per (h, r) pair)
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            prefix_logits, prefix_kvs = self.model(prefix_tensor)
            
            # Clone KV cache
            prefix_kvs = [(k.clone(), v.clone()) for k, v in prefix_kvs]
            # We only need the very last logit of the prefix to predict the first char of candidates
            # Clone to prevent CUDA graph overwrite in subsequent forward passes
            last_prefix_logit = prefix_logits[:, -1:, :].clone() # [1, 1, vocab]

        for i in range(0, num_candidates, batch_size):
            cand_batch = self.entity_tensor[i : i + batch_size]
            cand_lens = self.entity_lengths[i : i + batch_size]
            
            # Trim to max length in this batch
            max_len_in_batch = cand_lens.max().item()
            cand_batch = cand_batch[:, :max_len_in_batch]
            
            current_batch_size = cand_batch.size(0)
            
            # 3. Expand KV cache
            batch_kvs = [
                (k.expand(current_batch_size, -1, -1, -1), 
                 v.expand(current_batch_size, -1, -1, -1))
                for k, v in prefix_kvs
            ]
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                # 4. Forward pass
                logits, _ = self.model(cand_batch, past_kvs=batch_kvs)
                
                # 5. Optimized Score Calculation (Manual LogProb)
                # Avoids instantiating CrossEntropyLoss object
                
                # Expand prefix logit: [Batch, 1, Vocab]
                batch_prefix_logit = last_prefix_logit.expand(current_batch_size, -1, -1)
                
                # Combine: [Batch, L+1, Vocab]
                full_logits = torch.cat([batch_prefix_logit, logits], dim=1)
                
                # Shift for autoregressive prediction
                # logits at t predict token at t+1
                shift_logits = full_logits[:, :-1, :] # [Batch, L, Vocab]
                shift_labels = cand_batch             # [Batch, L]
                
                # Calculate LogSoftmax manually
                # log_softmax = logits - logsumexp(logits)
                # target_score = log_softmax[target_index]
                #              = logits[target_index] - logsumexp(logits)
                
                # A. LogSumExp (Denominator)
                lse = shift_logits.logsumexp(dim=-1) # [Batch, L]
                
                # B. Gather Target Logits (Numerator)
                # gather requires indices to be [Batch, L, 1]
                target_logits = shift_logits.gather(2, shift_labels.unsqueeze(2)).squeeze(2) # [Batch, L]
                
                # C. Token Scores
                token_scores = target_logits - lse
                
                # 6. Masking and Summation
                # Create mask for valid tokens (ignoring padding)
                pos_indices = torch.arange(max_len_in_batch, device=self.device).unsqueeze(0)
                mask = pos_indices < cand_lens.unsqueeze(1)
                
                summed_scores = (token_scores * mask).sum(dim=1)
                
                # Normalize and store
                gpu_scores[i : i + current_batch_size] = summed_scores / cand_lens

        return gpu_scores