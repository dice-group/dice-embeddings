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
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        
        # 1. Determine suffix
        if isinstance(train_dataset, (ByteGenBFSDataset, IsolatedTripleDataset)):
            self.suffix_token_id = tokenizer.eos_token_id
        else:
            self.suffix_token_id = tokenizer.sep_hr_token_id
        
        # 2. Build Entity Catalog (SORTED BY LENGTH)
        raw_entities = set([t[0] for t in train_dataset.triples] + [t[2] for t in train_dataset.triples] +
                           [t[0] for t in test_dataset.triples] + [t[2] for t in test_dataset.triples])
        self.entities = sorted(list(raw_entities), key=lambda x: (len(x), x))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        
        # 3. Build Known Facts Dictionary 
        print("Indexing known facts...")
        all_triples = train_dataset.triples + test_dataset.triples
        df = pd.DataFrame(all_triples, columns=['h', 'r', 't'])
        valid_entities = set(self.entity_to_idx.keys())
        df = df[df['t'].isin(valid_entities)]
        df['t_idx'] = df['t'].apply(lambda x: self.entity_to_idx[x])
        self.known_facts_dict = df.groupby(['h', 'r'])['t_idx'].apply(list).to_dict()
        
        # 4. Create Entity Tensor
        print("Pre-loading entities to GPU...")
        max_entity_len = max(len(e) for e in self.entities) + 1 
        
        # Optimize: Create directly on target device if memory allows, or pin memory
        entity_tensor = torch.full((len(self.entities), max_entity_len), 
                                   self.pad_token_id, 
                                   dtype=torch.long)
        entity_lengths = torch.zeros(len(self.entities), dtype=torch.long)
        
        for i, entity in enumerate(self.entities):
            e_seq = list(entity) + [self.suffix_token_id]
            l = len(e_seq)
            entity_tensor[i, :l] = torch.tensor(e_seq, dtype=torch.long)
            entity_lengths[i] = l
            
        self.entity_tensor = entity_tensor.to(self.device)
        self.entity_lengths = entity_lengths.to(self.device)
        self.train_ds = train_dataset
        self.test_ds = test_dataset

        # Compile setup
        self._compiled = False
        self._compile_model = compile_model

    def _ensure_compiled(self):
        if self._compiled or not self._compile_model:
            return
        if str(self.device).startswith('cuda'):
            try:
                # 'reduce-overhead' is great for small batches but can be unstable. 
                # If you crash, switch mode to 'default'
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self._compiled = True
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")

    def evaluate(self, split: str = 'test', limit: int = None, batch_size: int = 2048):
        self._ensure_compiled()
        self.model.eval()
        
        gpu_ranks_list = []
        
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
        
        with torch.inference_mode():
            for (h, r), target_entities in tqdm(hr_to_targets.items(), total=num_unique_hr):
                
                # Check targets existence early
                target_indices = [self.entity_to_idx[t] for t in target_entities if t in self.entity_to_idx]
                if not target_indices:
                    continue
                
                # OPTIMIZATION: Prepare prefix tensor immediately
                prefix_seq = [self.tokenizer.eos_token_id] + list(h) + \
                             [self.tokenizer.sep_hr_token_id] + list(r) + \
                             [self.tokenizer.sep_rt_token_id]
                # Non-blocking transfer if possible (though blocking usually fine here due to compute intensity)
                prefix_tensor = torch.tensor(prefix_seq, device=self.device, dtype=torch.long).unsqueeze(0)

                # 2. Score Candidates
                scores = self._score_candidates(prefix_tensor, batch_size)
                
                # 3. Metrics Calculation
                target_indices_tensor = torch.tensor(target_indices, device=self.device, dtype=torch.long)
                target_scores = scores[target_indices_tensor]
                
                # 4. Filter Known Facts
                known_indices = self.known_facts_dict.get((h, r), [])
                if known_indices:
                    mask_indices = torch.tensor(known_indices, device=self.device, dtype=torch.long)
                    scores.index_fill_(0, mask_indices, -float('inf'))
                
                # 5. Ranking
                # scores: [Num_Candidates], target_scores: [Num_Targets]
                # Broadcasting: [1, Num_Candidates] vs [Num_Targets, 1]
                better_than_target = scores.unsqueeze(0) > target_scores.unsqueeze(1)
                ranks = better_than_target.sum(dim=1) + 1 
                
                gpu_ranks_list.append(ranks)
                
                # Cleanup intermediate tensors to prevent fragmentation
                del scores, target_scores, better_than_target
        
        # Clear fragmented memory after evaluation loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if not gpu_ranks_list:
            return {}
            
        all_ranks_tensor = torch.cat(gpu_ranks_list)
        all_ranks_cpu = all_ranks_tensor.float().cpu().numpy()
        
        mrr = (1.0 / all_ranks_cpu).mean()
        hits1 = (all_ranks_cpu <= 1).mean()
        hits3 = (all_ranks_cpu <= 3).mean()
        hits10 = (all_ranks_cpu <= 10).mean()
        
        results = {"mrr": mrr, "h1": hits1, "h3": hits3, "h10": hits10}
        print(f"MRR: {results['mrr']:.4f} | H@1: {results['h1']:.4f} | H@3: {results['h3']:.4f} | H@10: {results['h10']:.4f}")
        return results

    def _score_candidates(self, prefix_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Computes scores using F.cross_entropy for fused, optimized LogProb calculation.
        """
        num_candidates = self.entity_tensor.size(0)
        
        # Use float16/bfloat16 for storage if possible to save memory bandwidth, 
        gpu_scores = torch.zeros(num_candidates, device=self.device, dtype=torch.float32)
        
        use_amp = str(self.device).startswith('cuda')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 1. Compute KV cache for prefix
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            prefix_logits, prefix_kvs = self.model(prefix_tensor)
            
            # Prepare Prefix KV
            prefix_kvs = [(k.clone(), v.clone()) for k, v in prefix_kvs]
            # We need the logit from the LAST token of the prefix to predict the FIRST token of the entity
            last_prefix_logit = prefix_logits[:, -1:, :].clone() # [1, 1, Vocab]

        # 2. Batch Loop
        for i in range(0, num_candidates, batch_size):
            cand_batch = self.entity_tensor[i : i + batch_size]
            cand_lens = self.entity_lengths[i : i + batch_size]
            
            # Dynamic trimming to max length in this batch
            max_len_in_batch = cand_lens.max().item()
            cand_batch = cand_batch[:, :max_len_in_batch] # [Batch, L]
            
            current_batch_size = cand_batch.size(0)
            
            # Expand KV cache
            batch_kvs = [
                (k.expand(current_batch_size, -1, -1, -1), 
                 v.expand(current_batch_size, -1, -1, -1))
                for k, v in prefix_kvs
            ]
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                # Forward Pass
                # logits: [Batch, L, Vocab] - Predictions for tokens 2..L+1 of entity
                logits, _ = self.model(cand_batch, past_kvs=batch_kvs)
                
                # Expand prefix logit: [Batch, 1, Vocab]
                batch_prefix_logit = last_prefix_logit.expand(current_batch_size, -1, -1)
                
                # Concatenate: [Batch, L+1, Vocab]
                # Index 0 is prefix output (predicts cand_batch[0])
                # Index 1 is cand_batch[0] output (predicts cand_batch[1])
                full_logits = torch.cat([batch_prefix_logit, logits], dim=1)
                
                
                shift_logits = full_logits[:, :-1, :] # [Batch, L, Vocab]
                shift_labels = cand_batch             # [Batch, L]
                
                # Permute for F.cross_entropy: [Batch, Vocab, L]
                shift_logits = shift_logits.permute(0, 2, 1)
                
                # Calculate Negative Log Likelihood per token
                # reduction='none' returns [Batch, L]
                token_nll = F.cross_entropy(
                    shift_logits, 
                    shift_labels, 
                    reduction='none', 
                    ignore_index=self.pad_token_id
                )
                
                # Sum NLL to get Sequence NLL, then Negate to get LogProb Score
                summed_scores = -token_nll.sum(dim=1)
                
                # Normalize by length
                gpu_scores[i : i + current_batch_size] = summed_scores / cand_lens
        
        # Explicit cleanup to prevent memory fragmentation
        del prefix_kvs, prefix_logits, last_prefix_logit

        return gpu_scores