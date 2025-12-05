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
        
        # 1. Determine suffix
        if isinstance(train_dataset, (ByteGenBFSDataset, IsolatedTripleDataset)):
            self.suffix_token_id = tokenizer.eos_token_id
        else:
            self.suffix_token_id = tokenizer.sep_hr_token_id
        
        # 2. Build Entity Catalog (SORTED BY LENGTH for batch efficiency)
        # Optimization: Sorting by length drastically reduces padding overhead in _score_candidates
        raw_entities = set([t[0] for t in train_dataset.triples] + [t[2] for t in train_dataset.triples] +
                           [t[0] for t in test_dataset.triples] + [t[2] for t in test_dataset.triples])
        self.entities = sorted(list(raw_entities), key=lambda x: (len(x), x))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        
        # 3. Build Known Facts Dictionary for Fast Lookup
        # Map (h, r) -> List of target indices (integers)
        # This replaces the Set[(h,r,t)] which is slow to query inside a GPU loop
        self.known_facts_dict = defaultdict(list)
        all_triples = train_dataset.triples + test_dataset.triples
        
        print("Indexing known facts...")
        for h, r, t in all_triples:
            if t in self.entity_to_idx:
                t_idx = self.entity_to_idx[t]
                self.known_facts_dict[(h, r)].append(t_idx)

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
        
        # Use a list to collect ranks to avoid growing numpy arrays
        all_ranks = []
        
        # Select Split
        if split == 'train':
            triples = self.train_ds.triples
        elif split == 'test':
            triples = self.test_ds.triples
        else:
            raise ValueError(f"Invalid split: {split}")
            
        triples = triples[:limit] if limit else triples
        
        # Group triples
        hr_to_targets: Dict[Tuple[tuple, tuple], List[tuple]] = defaultdict(list)
        for h, r, t in triples:
            hr_to_targets[(h, r)].append(t)
        
        num_unique_hr = len(hr_to_targets)
        print(f"Evaluating on {split} split: {len(triples)} triples grouped into {num_unique_hr} (h, r) groups...")

        # Metrics
        hits1 = hits3 = hits10 = 0
        total_predictions = 0

        # --- MAIN EVALUATION LOOP ---
        for (h, r), target_entities in tqdm(hr_to_targets.items(), total=num_unique_hr):
            
            # 1. Get Indices of targets
            # Filter out any targets not in our known entity list (edge case)
            target_indices = [self.entity_to_idx[t] for t in target_entities if t in self.entity_to_idx]
            if not target_indices:
                continue
                
            # Move target indices to GPU [Num_Targets]
            target_indices_tensor = torch.tensor(target_indices, device=self.device, dtype=torch.long)

            # 2. Score Candidates (Returns Tensor on GPU)
            # scores shape: [Num_Candidates]
            scores = self._score_candidates(h, r, batch_size)
            
            # 3. Retrieve Target Scores BEFORE Filtering
            # We need the 'real' score of the ground truth to compare against others.
            # shape: [Num_Targets]
            target_scores = scores[target_indices_tensor]
            
            # 4. Filter Known Facts on GPU
            # Get all known 't' indices for this (h, r)
            known_indices = self.known_facts_dict.get((h, r), [])
            if known_indices:
                # Move indices to GPU. This is small data transfer (e.g. 50 integers), very fast.
                mask_indices = torch.tensor(known_indices, device=self.device, dtype=torch.long)
                
                # In-place fill with -inf
                # This masks ALL known facts, including the ground truths we are currently testing.
                # This is correct because we saved 'target_scores' in step 3.
                scores.index_fill_(0, mask_indices, -float('inf'))
            
            # 5. Vectorized Ranking (Broadcasting)
            # We compare every target score against every candidate score simultaneously.
            # target_scores: [Num_Targets, 1]
            # scores:        [1, Num_Candidates] (masked)
            # Result:        [Num_Targets, Num_Candidates] boolean matrix
            
            # Logic: Count how many candidates have a score strictly greater than the target
            # Note: Since 'scores' has -inf at the target's own index, it won't be > target_score.
            # This effectively filters the target out of its own ranking comparison.
            
            better_than_target = scores.unsqueeze(0) > target_scores.unsqueeze(1)
            ranks = better_than_target.sum(dim=1) + 1  # [Num_Targets]
            
            # 6. Aggregate Results (Move only the ranks to CPU)
            ranks_cpu = ranks.tolist()
            all_ranks.extend(ranks_cpu)
            
            # Update metrics
            for r in ranks_cpu:
                total_predictions += 1
                if r <= 1: hits1 += 1
                if r <= 3: hits3 += 1
                if r <= 10: hits10 += 1
            
        # Final Metrics
        if total_predictions == 0:
            return {}
            
        mrr = np.mean([1.0 / r for r in all_ranks])
        results = {
            "mrr": mrr,
            "h1": hits1 / total_predictions,
            "h3": hits3 / total_predictions,
            "h10": hits10 / total_predictions
        }
        print(f"MRR: {results['mrr']:.4f} | H@1: {results['h1']:.4f} | H@3: {results['h3']:.4f} | H@10: {results['h10']:.4f}")
        return results

    def _score_candidates(self, h: tuple, r: tuple, batch_size: int) -> torch.Tensor:
        """
        Computes scores for all entities given (h, r).
        Uses KV-caching to avoid redundant computation of the prefix.
        Returns a GPU Tensor of shape [Num_Entities].
        """
        # 1. Prepare Prefix
        prefix_seq = [self.tokenizer.eos_token_id] + list(h) + \
                    [self.tokenizer.sep_hr_token_id] + list(r) + \
                    [self.tokenizer.sep_rt_token_id]
        
        prefix_tensor = torch.tensor(prefix_seq, device=self.device, dtype=torch.long).unsqueeze(0)
        prefix_len = prefix_tensor.size(1)
        
        # Loss function
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        num_candidates = self.entity_tensor.size(0)
        
        # Pre-allocate GPU tensor
        gpu_scores = torch.zeros(num_candidates, device=self.device, dtype=torch.float32)

        # AMP Setup
        use_amp = str(self.device).startswith('cuda')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.inference_mode():
            # 2. Compute KV cache for prefix (ONCE per (h, r) pair)
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                prefix_logits, prefix_kvs = self.model(prefix_tensor)
                # prefix_logits shape: [1, prefix_len, vocab_size]
                # prefix_kvs: list of (k, v) tuples for each layer
                
                # Clone KV cache to avoid CUDA graph memory reuse issues with torch.compile
                prefix_kvs = [(k.clone(), v.clone()) for k, v in prefix_kvs]
                prefix_logits = prefix_logits.clone()
            
            for i in range(0, num_candidates, batch_size):
                cand_batch = self.entity_tensor[i : i + batch_size]
                cand_lens = self.entity_lengths[i : i + batch_size]
                
                # Trim to max length in this batch (Efficient due to sorted entities)
                max_len_in_batch = cand_lens.max().item()
                cand_batch = cand_batch[:, :max_len_in_batch]
                
                current_batch_size = cand_batch.size(0)
                
                # 3. Expand KV cache for batch
                # Each layer's KV has shape (1, n_head, prefix_len, head_dim)
                # Expand to (batch_size, n_head, prefix_len, head_dim)
                batch_kvs = [
                    (k.expand(current_batch_size, -1, -1, -1), 
                     v.expand(current_batch_size, -1, -1, -1))
                    for k, v in prefix_kvs
                ]
                
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    # 4. Forward pass with KV cache - only process candidate tokens
                    logits, _ = self.model(cand_batch, past_kvs=batch_kvs)
                    # logits shape: [batch_size, cand_len, vocab_size]
                    
                    # 5. Score computation
                    # We need to score each token in the candidate sequence
                    # The first token is predicted by the last prefix logit
                    # Subsequent tokens are predicted by the candidate logits
                    
                    # Get the last prefix logit (predicts first candidate token)
                    # Shape: [1, vocab_size] -> expand to [batch_size, 1, vocab_size]
                    last_prefix_logit = prefix_logits[:, -1:, :].expand(current_batch_size, -1, -1)
                    
                    # Concat: [batch_size, 1 + cand_len, vocab_size]
                    # This gives us logits for predicting all candidate tokens
                    full_logits = torch.cat([last_prefix_logit, logits], dim=1)
                    
                    # Shift: predict cand_batch from full_logits[:-1]
                    # full_logits[:, :-1, :] predicts cand_batch
                    shift_logits = full_logits[:, :-1, :].contiguous()
                    shift_labels = cand_batch.contiguous()
                    
                    # Fused Loss Calculation
                    token_losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                    token_losses = token_losses.view(current_batch_size, -1)
                    
                    # Log Probs
                    token_scores = -token_losses

                    # Masking (Zero out padding - no prefix in this tensor)
                    seq_len = token_scores.shape[1]
                    pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)
                    
                    # Mask: only score up to the actual entity length
                    mask = pos_indices < cand_lens.unsqueeze(1)
                    
                    summed_scores = (token_scores * mask).sum(dim=1)
                    
                    # Normalize and store on GPU
                    gpu_scores[i : i + current_batch_size] = summed_scores / cand_lens

        return gpu_scores