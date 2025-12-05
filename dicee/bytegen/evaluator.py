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
        prefix = [self.tokenizer.eos_token_id] + list(h) + \
             [self.tokenizer.sep_hr_token_id] + list(r) + \
             [self.tokenizer.sep_rt_token_id] 
        all_scores = []
        
        with torch.no_grad():
            # Compute KV cache for prefix (excluding last token) ONCE
            # We exclude the last token (SEP_RT) so we can include it with candidates
            # This way, the first logit position will predict the first candidate token
            prefix_for_cache = prefix[:-1]
            prefix_tensor = torch.tensor([prefix_for_cache], dtype=torch.long, device=self.device)
            _, past_kv = self.model(prefix_tensor, use_cache=True)
            
            # The last prefix token will be prepended to each candidate
            last_prefix_token = prefix[-1]
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i+batch_size]
                current_batch_size = len(batch)
                
                # Expand cache for batch (memory-efficient, no copy)
                batch_past_kv = [
                    (k.expand(current_batch_size, -1, -1, -1),
                     v.expand(current_batch_size, -1, -1, -1))
                    for k, v in past_kv
                ]
                
                # Prepend last_prefix_token to candidates so logits align correctly
                # logits[i] predicts token[i+1], so:
                # - logits[0] (from last_prefix_token) predicts candidate[0]
                # - logits[1] (from candidate[0]) predicts candidate[1]
                # - etc.
                max_cand_len = max(len(c) for c in batch) + 1  # +1 for suffix
                input_ids = []
                lengths = []
                
                for cand in batch:
                    # [last_prefix_token, cand_bytes..., suffix_token]
                    seq = [last_prefix_token] + list(cand) + [self.suffix_token_id]
                    padded = seq + [self.tokenizer.pad_token_id] * (1 + max_cand_len - len(seq))
                    input_ids.append(padded)
                    lengths.append(len(cand) + 1)  # candidate length + suffix
                
                x = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                logits, _ = self.model(x, past_kv=batch_past_kv, use_cache=False)
                log_probs = F.log_softmax(logits, dim=-1)
                
                for b, length in enumerate(lengths):
                    # logits[b, 0:length, :] predict tokens x[b, 1:length+1]
                    # which are the candidate bytes + suffix token
                    relevant_logits = log_probs[b, :length, :]
                    targets = x[b, 1:length+1]
                    
                    # Sum log_probs for the candidate string
                    token_scores = relevant_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                    score = token_scores.sum().item() / length  # Normalize by length
                    all_scores.append(score)
                    
        return np.array(all_scores)
