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
        if not device_type.startswith('cuda'):
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
        # 1. Prepare Prefix
        prefix_seq = [self.tokenizer.eos_token_id] + list(h) + \
                    [self.tokenizer.sep_hr_token_id] + list(r) + \
                    [self.tokenizer.sep_rt_token_id]
        
        prefix_tensor = torch.tensor(prefix_seq, device=self.device, dtype=torch.long).unsqueeze(0)
        prefix_len = prefix_tensor.size(1)
        
        # Pre-calculate Loss function (More efficient than manual log_softmax + gather)
        # We want the log_prob, which is negative loss.
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        num_candidates = self.entity_tensor.size(0)
        
        # OPTIMIZATION 1: Pre-allocate GPU tensor for results to avoid CPU Sync inside loop
        # We collect all scores here and move to CPU only once at the end.
        gpu_scores = torch.zeros(num_candidates, device=self.device, dtype=torch.float32)

        # AMP Setup
        device_str = self.device if isinstance(self.device, str) else self.device.type
        use_amp = device_str.startswith('cuda')
        device_type = 'cuda' if use_amp else 'cpu'  # autocast expects 'cuda' not 'cuda:0'
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.inference_mode():
            # Optional: Compute KV Cache for prefix here if model supports it
            # prefix_outputs = self.model(prefix_tensor, use_cache=True)
            # past_key_values = prefix_outputs.past_key_values
            
            for i in range(0, num_candidates, batch_size):
                cand_batch = self.entity_tensor[i : i + batch_size]
                cand_lens = self.entity_lengths[i : i + batch_size]
                
                # Trim to max length in this batch
                max_len_in_batch = cand_lens.max().item()
                cand_batch = cand_batch[:, :max_len_in_batch]
                
                current_batch_size = cand_batch.size(0)
                
                # --- KV CACHE PATH (Conceptual) ---
                # If using KV Cache, you would ONLY feed cand_batch here, 
                # passing `past_key_values` and `position_ids` starting at prefix_len.
                
                # --- STANDARD PATH (Current Logic) ---
                batch_prefix = prefix_tensor.expand(current_batch_size, -1)
                x = torch.cat([batch_prefix, cand_batch], dim=1)
                
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    logits, _ = self.model(x)
                    
                    # Shift for next-token prediction
                    # logits: [B, Seq_Len, Vocab] -> Preds for pos 0 to N-1
                    # targets: [B, Seq_Len] -> Actual tokens at pos 1 to N
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = x[..., 1:].contiguous()
                    
                    # OPTIMIZATION 3: Fused CrossEntropy
                    # This calculates loss (negative log prob) for the targets.
                    # Shape: [B, Seq_Len-1]
                    # We flatten the input to utilize the fused kernel efficiently
                    token_losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                    
                    # Reshape back to [B, Seq_Len-1]
                    token_losses = token_losses.view(current_batch_size, -1)
                    
                    # We want Log Probability, which is -Loss
                    token_scores = -token_losses

                    # --- Masking ---
                    # We only care about scores in the candidate region
                    # The candidate starts at index: prefix_len
                    # (Note: In shifted space, the prediction for the first candidate token 
                    # comes from the last prefix token, which is at index prefix_len - 1)
                    
                    start_idx = prefix_len - 1
                    seq_len = token_scores.shape[1]
                    
                    # Vectorized Mask Creation
                    pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)
                    end_indices = start_idx + cand_lens.unsqueeze(1)
                    
                    mask = (pos_indices >= start_idx) & (pos_indices < end_indices)
                    
                    # Apply mask and sum
                    # Zeros out prefix scores and padding scores
                    summed_scores = (token_scores * mask).sum(dim=1)
                    
                    # Normalize
                    batch_scores = summed_scores / cand_lens
                    
                    # Store directly in pre-allocated GPU tensor
                    gpu_scores[i : i + current_batch_size] = batch_scores

        # Final Move to CPU (Done ONCE)
        return gpu_scores.cpu().numpy() 