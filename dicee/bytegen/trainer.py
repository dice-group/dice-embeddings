import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import torch.nn.functional as F
# DDP training
from torch.distributed import init_process_group, destroy_process_group
# model compoenents
from dicee.bytegen.bytegen import ByteGenModel
from dicee.bytegen.bytegen import ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer
from dicee.bytegen.dataset import ByteGenBFSDataset, IsolatedTripleDataset
import os
import wandb
import numpy as np
from typing import Dict
from collections import defaultdict


class Trainer:
    def __init__(self, model: ByteGenModel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, 
                 optimizer: torch.optim.Optimizer = None, save_path: str = "checkpoints",
                 warmup_epochs: int = 5, label_smoothing: float = 0.0, grad_clip: float = 1.0,
                 train_dataset=None, eval_batch_size: int = 128):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.device = config.device
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        self.train_dataset = train_dataset
        self.eval_batch_size = eval_batch_size
        os.makedirs(self.save_path, exist_ok=True)
        
        # Build entity list for H@1 computation
        if train_dataset is not None:
            self.entities = sorted(list(set(
                [t[0] for t in train_dataset.triples] + 
                [t[2] for t in train_dataset.triples]
            )))
            self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
            # Build known facts for filtering
            self.known_facts = set((h, r, t) for h, r, t in train_dataset.triples)
            self.hr_to_t = defaultdict(set)
            for h, r, t in train_dataset.triples:
                self.hr_to_t[(h, r)].add(t)
        else:
            self.entities = None
            self.known_facts = None
            self.hr_to_t = None

    def _create_scheduler(self, epochs: int):
        """Create LR scheduler with linear warmup followed by cosine annealing."""
        warmup_steps = self.warmup_epochs
        main_steps = max(epochs - warmup_steps, 1)
        
        # NEW: If no warmup requested, use constant LR (no scheduling)
        if warmup_steps == 0:
            return None  # Signal to skip scheduler
        
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=main_steps,
            eta_min=self.config.lr * 0.01  # Minimum LR is 1% of base LR
        )
        
        scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )
        return scheduler

    def _compute_tail_loss(self, logits, targets):
        """Compute loss only on tail tokens (after SEP_RT, until EOS/PAD) """
        sep_rt_id = self.tokenizer.sep_rt_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # 1. Create a coordinate grid [0, 1, ..., seq_len-1]
        seq_len = targets.size(1)
        range_tensor = torch.arange(seq_len, device=targets.device).unsqueeze(0) # [1, L]
        
        # 2. Find start positions (first occurrence of SEP_RT)
        # (targets == sep_rt_id) gives a boolean mask. 
        is_sep = (targets == sep_rt_id).float()
        sep_indices = is_sep.argmax(dim=1) # [B]
        
        # 3. Find end positions (first EOS or PAD)
        is_end = ((targets == eos_id) | (targets == pad_id)).float()
        end_indices = is_end.argmax(dim=1)
        
        # Handle case where no end token is found (set to end of sequence)
        # If argmax returns 0 and it wasn't actually at 0 (checked via is_end[:, 0]), it means not found
        # But typically argmax on all-zeros returns 0. 
        # Let's be robust: if sum(is_end) == 0 for a row, set end to seq_len
        no_end_found = (is_end.sum(dim=1) == 0)
        end_indices[no_end_found] = seq_len
        
        # 4. Create Mask
        # active if index > sep_index AND index <= end_index
        mask = (range_tensor > sep_indices.unsqueeze(1)) & \
               (range_tensor <= end_indices.unsqueeze(1))
               
        # Compute Loss
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction='none'
        ).reshape(targets.shape)
        
        masked_loss = (loss_per_token * mask.float()).sum()
        num_active = mask.sum()
        
        if num_active > 0:
            return (masked_loss / num_active).item()
        return 0.0

    def _compute_sampled_metrics(self, sample_size: int = 200, batch_size: int = 64) -> Dict[str, float]:
        """Compute MRR and H@1 on a sample of training triples using KV caching and optimized scoring."""
        if self.train_dataset is None or self.entities is None:
            return {"mrr": 0.0, "h1": 0.0}
        
        import random
        
        self.model.eval()
        
        # Sample triples
        triples = self.train_dataset.triples
        sample_indices = random.sample(range(len(triples)), min(sample_size, len(triples)))
        
        mrr_sum = 0.0
        hits1 = 0
        
        # Determine suffix based on dataset type
        if isinstance(self.train_dataset, (ByteGenBFSDataset, IsolatedTripleDataset)):
            suffix_token_id = self.tokenizer.eos_token_id
        else:
            suffix_token_id = self.tokenizer.sep_hr_token_id
            
        # AMP setup
        use_amp = str(self.device).startswith('cuda')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.no_grad():
            for idx in sample_indices:
                h, r, t = triples[idx]
                
                # Build prefix: [EOS] H [SEP_HR] R [SEP_RT]
                prefix = [self.tokenizer.eos_token_id] + list(h) + \
                         [self.tokenizer.sep_hr_token_id] + list(r) + \
                         [self.tokenizer.sep_rt_token_id]
                
                # Cache prefix once
                prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)
                
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    prefix_logits, prefix_kv = self.model(prefix_tensor)
                    # Use raw logits for cross_entropy
                    last_prefix_logit = prefix_logits[:, -1:, :] # [1, 1, V]
                
                all_scores_list = []
                
                for i in range(0, len(self.entities), batch_size):
                    batch_entities = self.entities[i:i+batch_size]
                    curr_batch_size = len(batch_entities)
                    
                    # Build candidate sequences (without prefix)
                    max_cand_len = max(len(e) for e in batch_entities) + 1
                    cand_seqs = []
                    cand_lengths = []
                    
                    for cand in batch_entities:
                        seq = list(cand) + [suffix_token_id]
                        cand_lengths.append(len(seq))
                        padded = seq + [self.tokenizer.pad_token_id] * (max_cand_len - len(seq))
                        cand_seqs.append(padded)
                    
                    cand_tensor = torch.tensor(cand_seqs, dtype=torch.long, device=self.device)
                    cand_lens_tensor = torch.tensor(cand_lengths, device=self.device)
                    
                    # Expand KV cache
                    batch_kv = []
                    for layer_kv in prefix_kv:
                        k, v = layer_kv
                        batch_kv.append((
                            k.expand(curr_batch_size, -1, -1, -1),
                            v.expand(curr_batch_size, -1, -1, -1)
                        ))
                    
                    with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                        # Forward with cached KV
                        cand_logits, _ = self.model(cand_tensor, past_kvs=batch_kv)
                        
                        # Expand prefix logit: [B, 1, V]
                        batch_prefix_logit = last_prefix_logit.expand(curr_batch_size, -1, -1)
                        
                        # Concatenate to get predictions for all tokens including first
                        # full_logits[:, 0] comes from prefix and predicts cand_tensor[:, 0]
                        full_logits = torch.cat([batch_prefix_logit, cand_logits], dim=1)
                        
                        # Shift for Cross Entropy
                        # We predict cand_tensor. Input logits are full_logits[:, :-1]
                        shift_logits = full_logits[:, :-1, :] # [B, L, V]
                        shift_labels = cand_tensor            # [B, L]
                        
                        # Permute for F.cross_entropy: [B, V, L]
                        shift_logits = shift_logits.permute(0, 2, 1)
                        
                        # Calculate NLL
                        token_nll = F.cross_entropy(
                            shift_logits,
                            shift_labels,
                            reduction='none',
                            ignore_index=self.tokenizer.pad_token_id
                        )
                        
                        # Sum and Negate to get LogProb
                        summed_scores = -token_nll.sum(dim=1)
                        
                        # Normalize
                        final_scores = summed_scores / cand_lens_tensor
                        all_scores_list.append(final_scores)
                
                # Concatenate all scores on GPU
                scores = torch.cat(all_scores_list)
                
                # Find target index
                if t in self.entity_to_idx:
                    target_idx = self.entity_to_idx[t]
                else:
                    continue
                    
                target_score = scores[target_idx]
                
                # Filter known facts (Vectorized on GPU)
                valid_tails = self.hr_to_t.get((h, r), set())
                if len(valid_tails) > 1:
                    valid_indices = [self.entity_to_idx[vt] for vt in valid_tails if vt in self.entity_to_idx and vt != t]
                    if valid_indices:
                        mask_indices = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
                        scores.index_fill_(0, mask_indices, -float('inf'))
                
                # Rank on GPU
                rank = (scores > target_score).sum().item() + 1
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hits1 += 1
        
        self.model.train()
        return {
            "mrr": mrr_sum / len(sample_indices),
            "h1": hits1 / len(sample_indices)
        }

    def train(self, epochs: int = 500):
        print(f"Starting training for {epochs} epochs...")
        print(f"  - Warmup epochs: {self.warmup_epochs}")
        print(f"  - Label smoothing: {self.label_smoothing}")
        print(f"  - Gradient clipping: {self.grad_clip}")
        print(f"  - Base LR: {self.config.lr}")
        
        self.model.train()
        scheduler = self._create_scheduler(epochs)
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Input: x[:-1], Target: x[1:]
                logits, _ = self.model(batch[:, :-1])
                targets = batch[:, 1:]
                
                # Cross-entropy with label smoothing
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size), 
                    targets.reshape(-1), 
                    ignore_index=self.tokenizer.pad_token_id,
                    label_smoothing=self.label_smoothing
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Compute tail-only loss for diagnostics
                with torch.no_grad():
                    tail_loss = self._compute_tail_loss(logits, targets)

                # Log to wandb if available
                if wandb.run is not None:
                    global_step = epoch * len(self.train_loader) + num_batches
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/tail_loss": tail_loss,
                        "train/lr": scheduler.get_last_lr()[0] if scheduler else self.config.lr,
                        "epoch": epoch,
                    }
                    
                    # Periodically compute sampled MRR
                    if global_step % 500 == 0:
                        metrics = self._compute_sampled_metrics(sample_size=128, batch_size=self.eval_batch_size)
                        log_dict.update({
                            "train/sampled_mrr": metrics["mrr"],
                            "train/sampled_h1": metrics["h1"]
                        })
                    
                    wandb.log(log_dict, step=global_step)
                
                avg_loss = total_loss / num_batches
                if scheduler is not None:
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.4f}", 
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                else:
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                    })
            
            # Step scheduler after each epoch (only if it exists)
            if scheduler is not None:
                scheduler.step()
            
            
        self.save_model(epochs, os.path.join(self.save_path, f"model_epoch_{epochs}.pt"))

    def save_model(self, epoch, path):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}.")


class DDPTrainer(Trainer):
    def __init__(self, model: DistributedDataParallel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, 
                 optimizer: torch.optim.Optimizer = None, save_path: str = "checkpoints", gradient_acc_steps: int = 1,
                 warmup_epochs: int = 10, label_smoothing: float = 0.1, grad_clip: float = 1.0):
        super().__init__(model, train_loader, config, tokenizer, optimizer, save_path, warmup_epochs, label_smoothing, grad_clip)
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.gradient_acc_steps = gradient_acc_steps

    def train(self, epochs: int = 500):
        assert torch.cuda.is_available(), "Training using DDPTrainer only works on gpu(s)"
        print(f"Starting training for {epochs} epochs...")
        print(f"  - Warmup epochs: {self.warmup_epochs}")
        print(f"  - Label smoothing: {self.label_smoothing}")
        print(f"  - Gradient clipping: {self.grad_clip}")
        print(f"  - Base LR: {self.config.lr}")
        
        self.model.train()
        scheduler = self._create_scheduler(epochs)
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
            for step_number, batch in enumerate(pbar):
                batch = batch.to(self.device)
                targets = batch[:, 1:]
                last_step = step_number == len(self.train_loader) - 1

                if (step_number + 1) % self.gradient_acc_steps != 0 and not last_step:
                    with self.model.no_sync():
                        logits, _ = self.model.module(batch[:, :-1]) 
                        loss = F.cross_entropy(
                            logits.reshape(-1, self.config.vocab_size), 
                            targets.reshape(-1), 
                            ignore_index=self.tokenizer.pad_token_id,
                            label_smoothing=self.label_smoothing
                        )
                        loss.backward()
                else:
                    # Input: x[:-1], Target: x[1:]
                    logits, _ = self.model.module(batch[:, :-1])
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, self.config.vocab_size), 
                        targets.reshape(-1), 
                        ignore_index=self.tokenizer.pad_token_id,
                        label_smoothing=self.label_smoothing
                    )
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.grad_clip)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f"{total_loss / (pbar.n + 1):.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
            
            # Step scheduler after each epoch
            scheduler.step()

        if self.global_rank == 0:
            self.save_model(epochs, os.path.join(self.save_path, f"model_epoch_{epochs}.pt"))

        destroy_process_group()