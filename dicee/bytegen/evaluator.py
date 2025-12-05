from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dicee.bytegen.bytegen import ByteGenModel
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset, IsolatedTripleDataset
from dicee.bytegen.tokenizer import ByteTokenizer

class Evaluator:
    def __init__(self, model: ByteGenModel, train_dataset: ByteGenDataset, test_dataset: ByteGenDataset, tokenizer: ByteTokenizer):
        self.model = model
        self.device = model.config.device
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        self.tokenizer = tokenizer
        
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
            
    def evaluate(self, split: str = 'test', limit: int = None, batch_size: int = 64):
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
        print(f"Evaluating on {split} split with {len(triples)} triples...")

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
        results = {
            "mrr": mrr,
            "h1": hits1/len(ranks),
            "h3": hits3/len(ranks),
            "h10": hits10/len(ranks)
        }
        print(f"MRR: {results['mrr']:.4f} | H@1: {results['h1']:.4f} | H@3: {results['h3']:.4f} | H@10: {results['h10']:.4f}")
        return results

    def _score_candidates(self, h: tuple, r: tuple, candidates: List[tuple], batch_size: int):
        """
        Score candidates using KV caching for efficiency.
        
        The prefix (h + r) is processed once and cached, then each candidate batch
        only processes the candidate tokens while attending to the cached prefix.
        """
        prefix = [self.tokenizer.eos_token_id] + list(h) + \
             [self.tokenizer.sep_hr_token_id] + list(r) + \
             [self.tokenizer.sep_rt_token_id] 
        all_scores = []
        
        with torch.no_grad():
            # Step 1: Process prefix once and cache KV
            prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)
            prefix_logits, prefix_kv = self.model(prefix_tensor)
            
            # The last position's logits predict the first candidate token
            prefix_last_logprobs = F.log_softmax(prefix_logits[:, -1, :], dim=-1)  # (1, vocab_size)
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i+batch_size]
                curr_batch_size = len(batch)
                
                # Step 2: Expand cached KV for the batch
                # Each layer's KV is (k, v) with shape (1, n_head, seq_len, head_dim)
                # Expand to (batch_size, n_head, seq_len, head_dim)
                batch_kv = []
                for layer_kv in prefix_kv:
                    k, v = layer_kv
                    batch_kv.append((
                        k.expand(curr_batch_size, -1, -1, -1),
                        v.expand(curr_batch_size, -1, -1, -1)
                    ))
                
                # Step 3: Build candidate sequences (without prefix)
                max_cand_len = max(len(c) for c in batch) + 1  # +1 for suffix
                cand_seqs = []
                cand_lengths = []
                
                for cand in batch:
                    seq = list(cand) + [self.suffix_token_id]
                    cand_lengths.append(len(seq))
                    padded = seq + [self.tokenizer.pad_token_id] * (max_cand_len - len(seq))
                    cand_seqs.append(padded)
                
                cand_tensor = torch.tensor(cand_seqs, dtype=torch.long, device=self.device)
                
                # Step 4: Forward pass with cached KV - only processes candidate tokens
                # but attention sees the full prefix via the cache
                cand_logits, _ = self.model(cand_tensor, past_kvs=batch_kv)
                cand_logprobs = F.log_softmax(cand_logits, dim=-1)
                
                # Step 5: Score each candidate
                for b in range(curr_batch_size):
                    length = cand_lengths[b]
                    
                    # First candidate token is predicted by the prefix's last position
                    first_token_score = prefix_last_logprobs[0, cand_tensor[b, 0]].item()
                    
                    # Remaining tokens: cand_logprobs[b, j] predicts cand_tensor[b, j+1]
                    if length > 1:
                        remaining_logprobs = cand_logprobs[b, :length-1, :]
                        remaining_targets = cand_tensor[b, 1:length]
                        remaining_scores = remaining_logprobs.gather(1, remaining_targets.unsqueeze(1)).squeeze(1)
                        total_score = first_token_score + remaining_scores.sum().item()
                    else:
                        total_score = first_token_score
                    
                    all_scores.append(total_score / length)
                    
        return np.array(all_scores)
