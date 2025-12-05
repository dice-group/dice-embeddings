from typing import List, Dict, Tuple
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
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
        
        # Determine suffix token for evaluation based on dataset type
        if isinstance(train_dataset, (ByteGenBFSDataset, IsolatedTripleDataset)):
            # BFS and Isolated datasets end entities with EOS
            self.suffix_token_id = tokenizer.eos_token_id
        else:
            # Random Walk Dataset ends entities with SEP_HR (start of next relation)
            # This prevents the model from predicting a prefix of the entity, Cat vs Category
            self.suffix_token_id = tokenizer.sep_hr_token_id
        
        # Build Entity Catalog
        self.entities = sorted(list(set([t[0] for t in train_dataset.triples] + [t[2] for t in train_dataset.triples] +
                                        [t[0] for t in test_dataset.triples] + [t[2] for t in test_dataset.triples])))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        
        # Build Known Facts for Filtering
        self.known_facts = set()
        for h, r, t in (train_dataset.triples + test_dataset.triples):
            self.known_facts.add((h, r, t))

        max_entity_len = max(len(e) for e in self.entities) + 1 # +1 for suffix
        
        # Create padded tensor: Shape (Num_Entities, Max_Len)
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

        # Move to GPU once and store
        self.entity_tensor = entity_tensor.to(self.device)
        self.entity_lengths = entity_lengths.to(self.device)    
    
    def _ensure_compiled(self):
        """Compile model with torch.compile for faster inference"""
        if self._compiled or not self._compile_model:
            return
            
        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type != 'cuda':
            return
            
        try:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            self._compiled = True
            print("Model compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")
            
    def evaluate(self, split: str = 'test', limit: int = None, batch_size: int = 2048):
        self._ensure_compiled()
        self.model.eval()
        ranks = []
        hits1, hits3, hits10 = 0, 0, 0
        
        # Pre-calculate candidate bytes 
        candidates = self.entities
        
        if split == 'train':
            triples = self.train_ds.triples
        elif split == 'test':
            triples = self.test_ds.triples
        else:
            raise ValueError(f"Invalid split: {split}")
            
        triples = triples[:limit] if limit else triples
        
        # Group triples by (h, r) for algorithmic speedup
        # This avoids redundant scoring for one-to-many relations
        # e.g., (Spielberg, director_of, Jaws) and (Spielberg, director_of, E.T.)
        # share the same prefix and only need one scoring pass
        hr_to_targets: Dict[Tuple[tuple, tuple], List[tuple]] = defaultdict(list)
        for h, r, t in triples:
            hr_to_targets[(h, r)].append(t)
        
        num_unique_hr = len(hr_to_targets)
        print(f"Evaluating on {split} split: {len(triples)} triples grouped into {num_unique_hr} unique (h, r) pairs...")

        for (h, r), targets in tqdm(hr_to_targets.items(), total=num_unique_hr):
            # Score all candidates once for this (h, r) pair
            scores = self._score_candidates(h, r, candidates, batch_size)
            
            # Pre-compute filtered scores for this (h, r) pair
            # We filter out all known facts except we'll restore target scores individually
            filtered_scores = scores.copy()
            for i, cand in enumerate(candidates):
                if (h, r, cand) in self.known_facts:
                    filtered_scores[i] = -float('inf')
            
            # Process each target in the group
            for t in targets:
                if t not in self.entity_to_idx: 
                    continue
                    
                target_idx = self.entity_to_idx[t]
                target_score = scores[target_idx]  # Use original unfiltered score
                
                # For ranking, we use filtered_scores but restore the target's score
                # This implements: filter known facts EXCEPT the ground truth
                ranking_scores = filtered_scores.copy()
                ranking_scores[target_idx] = target_score
                
                # Calculate Rank
                # Rank = (count of scores > target_score) + 1
                rank = np.sum(ranking_scores > target_score) + 1
                
                ranks.append(rank)
                if rank <= 1: hits1 += 1
                if rank <= 3: hits3 += 1
                if rank <= 10: hits10 += 1
            
        mrr = np.mean(1.0 / np.array(ranks))
        results = {
            "mrr": mrr,
            "h1": hits1/len(ranks),
            "h3": hits3/len(ranks),
            "h10": hits10/len(ranks)
        }
        print(f"MRR: {results['mrr']:.4f} | H@1: {results['h1']:.4f} | H@3: {results['h3']:.4f} | H@10: {results['h10']:.4f}")
        return results

    def _score_candidates(self, h: tuple, r: tuple, candidates: List[tuple], batch_size: int):
        # 1. Prepare Prefix on GPU (Done once per query)
        prefix_seq = [self.tokenizer.eos_token_id] + list(h) + \
                     [self.tokenizer.sep_hr_token_id] + list(r) + \
                     [self.tokenizer.sep_rt_token_id]
        
        # Create prefix tensor: Shape (1, Prefix_Len)
        prefix_tensor = torch.tensor(prefix_seq, device=self.device, dtype=torch.long).unsqueeze(0)
        prefix_len = prefix_tensor.size(1)
        
        # We start evaluating logits at the end of the prefix
        # (In the shifted target space, this is index prefix_len - 1)
        start_logit_idx = prefix_len - 1
        
        num_candidates = self.entity_tensor.size(0)
        all_scores = []
        
        # AMP Setup
        device_type = self.device if isinstance(self.device, str) else self.device.type
        use_amp = device_type == 'cuda'
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.inference_mode():
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                
                # Iterate through the pre-computed GPU tensor directly
                for i in range(0, num_candidates, batch_size):
                    # Slice the pre-computed tensors 
                    # This is instant (GPU-to-GPU view)
                    cand_batch = self.entity_tensor[i : i + batch_size]
                    cand_lens = self.entity_lengths[i : i + batch_size]
                    
                    # Optimization: Trim batch to the longest entity in *this* batch
                    # (Avoids processing excessive padding from the global max length)
                    max_len_in_batch = cand_lens.max().item()
                    cand_batch = cand_batch[:, :max_len_in_batch]
                    
                    current_batch_size = cand_batch.size(0)
                    
                    # Expand prefix to match batch size
                    batch_prefix = prefix_tensor.expand(current_batch_size, -1)
                    
                    # Concatenate Prefix + Candidates on GPU
                    # This replaces the slow Python list concatenation
                    x = torch.cat([batch_prefix, cand_batch], dim=1)
                    
                    # --- Forward Pass ---
                    logits, _ = self.model(x)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # --- Vectorized Scoring ---
                    
                    # Shift to align logits with next-token targets
                    shifted_logits = log_probs[:, :-1, :] # Shape: (B, T-1, Vocab)
                    shifted_targets = x[:, 1:]            # Shape: (B, T-1)
                    
                    # Gather scores
                    token_scores = shifted_logits.gather(2, shifted_targets.unsqueeze(2)).squeeze(2)
                    
                    # Create Mask
                    # Valid indices are [start_logit_idx, start_logit_idx + cand_len)
                    seq_len = token_scores.shape[1]
                    pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)
                    
                    # Calculate end indices for every item in batch
                    end_indices = start_logit_idx + cand_lens.unsqueeze(1)
                    
                    # Boolean Mask
                    mask = (pos_indices >= start_logit_idx) & (pos_indices < end_indices)
                    
                    # Apply Mask, Sum and Normalize
                    token_scores = token_scores * mask
                    summed_scores = token_scores.sum(dim=1)
                    
                    # Normalize by length
                    normalized_scores = summed_scores / cand_lens
                    
                    # Move to CPU list once per batch
                    all_scores.extend(normalized_scores.tolist())

        return np.array(all_scores)