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
            
    def evaluate(self, split: str = 'test', limit: int = None, batch_size: int = 64):
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
        prefix = [self.tokenizer.eos_token_id] + list(h) + \
             [self.tokenizer.sep_hr_token_id] + list(r) + \
             [self.tokenizer.sep_rt_token_id] 
        all_scores = []
        
        # Determine device type and AMP dtype
        device_type = self.device if isinstance(self.device, str) else self.device.type
        use_amp = device_type == 'cuda'
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.inference_mode():
            # Use AMP for faster inference on CUDA
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i:i+batch_size]
                    
                    # Dynamic Batching
                    max_cand_len = max([len(c) for c in batch]) + 1
                    input_ids = []
                    slices = [] # (start_idx, end_idx) used for gathering probabilities
                    
                    for cand in batch:
                        seq = prefix + list(cand) + [self.suffix_token_id]
                        # We predict the candidate bytes.
                        # The prediction for seq[k] comes from seq[k-1]
                        # Start of candidate in seq is len(prefix). 
                        # So we look at logits at indices: len(prefix)-1 ... to ... end-1
                        start_logit_idx = len(prefix) - 1
                        end_logit_idx = len(seq) - 1
                        
                        input_ids.append(seq + [self.tokenizer.pad_token_id] * (len(prefix) + max_cand_len - len(seq)))
                        slices.append((start_logit_idx, end_logit_idx, len(cand) + 1))
                    
                    x = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                    logits, _ = self.model(x)
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
