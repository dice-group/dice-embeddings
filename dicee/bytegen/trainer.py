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
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
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
            # Sort by length for efficient batching
            self.entities = sorted(list(set(
                [t[0] for t in train_dataset.triples] + 
                [t[2] for t in train_dataset.triples]
            )), key=lambda x: (len(x), x))
            self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
            
            # Pre-compute entity tensor for fast evaluation
            if isinstance(self.train_dataset, (ByteGenBFSDataset, IsolatedTripleDataset)):
                suffix_token_id = self.tokenizer.eos_token_id
            else:
                suffix_token_id = self.tokenizer.sep_hr_token_id

            max_entity_len = max(len(e) for e in self.entities) + 1
            # Create on CPU first to avoid OOM if very large, then move to device
            self.entity_tensor = torch.full((len(self.entities), max_entity_len), 
                                          self.tokenizer.pad_token_id, 
                                          dtype=torch.long)
            self.entity_lengths = torch.zeros(len(self.entities), dtype=torch.long)
            
            for i, entity in enumerate(self.entities):
                e_seq = list(entity) + [suffix_token_id]
                l = len(e_seq)
                self.entity_tensor[i, :l] = torch.tensor(e_seq, dtype=torch.long)
                self.entity_lengths[i] = l
            
            self.entity_tensor = self.entity_tensor.to(self.device)
            self.entity_lengths = self.entity_lengths.to(self.device)

            # Build known facts for filtering
            self.known_facts = set((h, r, t) for h, r, t in train_dataset.triples)
            self.hr_to_t = defaultdict(set)
            for h, r, t in train_dataset.triples:
                self.hr_to_t[(h, r)].add(t)
        else:
            self.entities = None
            self.entity_tensor = None
            self.entity_lengths = None
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

    def _compute_sampled_metrics(self, sample_size: int = 200, batch_size: int = 512) -> Dict[str, float]:
        """
        Compute MRR and H@1 
        """
        if self.train_dataset is None or self.entities is None:
            return {"mrr": 0.0, "h1": 0.0}
        
        import random
        
        self.model.eval()
        
        # Sample triples
        triples = self.train_dataset.triples
        sample_indices = random.sample(range(len(triples)), min(sample_size, len(triples)))
        
        mrr_sum = 0.0
        hits1 = 0
        
        # AMP setup
        use_amp = str(self.device).startswith('cuda')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        num_candidates = self.entity_tensor.size(0)
        
        # Pre-allocate scores tensor on GPU to avoid allocation overhead
        gpu_scores = torch.zeros(num_candidates, device=self.device, dtype=torch.float32)
        
        # Pre-calculate candidate lengths on CPU to avoid .item() syncs if entity_lengths is on GPU
        # If entity_lengths is already CPU, this is just a reference.
        cpu_cand_lens = self.entity_lengths.cpu() if hasattr(self.entity_lengths, "cpu") else self.entity_lengths

        with torch.inference_mode():
            for idx in sample_indices:
                h, r, t = triples[idx]
                
                # Build prefix: [EOS] H [SEP_HR] R [SEP_RT]
                prefix = [self.tokenizer.eos_token_id] + list(h) + \
                         [self.tokenizer.sep_hr_token_id] + list(r) + \
                         [self.tokenizer.sep_rt_token_id]
                
                prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)
                
                # --- Step 1: Cache Prefix KV ---
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    prefix_logits, prefix_kv = self.model(prefix_tensor)
                    # Clone to decouple from graph (though inference_mode handles this, it's safer for memory)
                    prefix_kv = [(k.clone(), v.clone()) for k, v in prefix_kv]
                    last_prefix_logit = prefix_logits[:, -1:, :] # [1, 1, V]
                
                # --- Step 2: Batched Candidate Evaluation ---
                for i in range(0, num_candidates, batch_size):
                    # Slicing
                    end_i = min(i + batch_size, num_candidates)
                    curr_batch_size = end_i - i
                    
                    cand_batch = self.entity_tensor[i : end_i]
                    
                    # Optimization: Get max len from CPU tensor to avoid GPU sync
                    # (Assuming cpu_cand_lens corresponds to entity_tensor indices)
                    max_len_in_batch = cpu_cand_lens[i : end_i].max().item()
                    
                    cand_batch = cand_batch[:, :max_len_in_batch]
                    
                    # Move batch to GPU if it isn't already
                    if cand_batch.device != self.device:
                        cand_batch = cand_batch.to(self.device, non_blocking=True)
                    
                    # Expand KV cache
                    # We use .expand() which is a view (zero memory copy), very fast
                    batch_kv = [
                        (k.expand(curr_batch_size, -1, -1, -1),
                         v.expand(curr_batch_size, -1, -1, -1))
                        for k, v in prefix_kv
                    ]
                    
                    with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                        # Forward pass
                        cand_logits, _ = self.model(cand_batch, past_kvs=batch_kv)
                        
                        # Expand prefix logit
                        batch_prefix_logit = last_prefix_logit.expand(curr_batch_size, -1, -1)
                        
                        # Concat
                        full_logits = torch.cat([batch_prefix_logit, cand_logits], dim=1)
                        
                        # Shift for Cross Entropy
                        # Logits: [B, L, V], Labels: [B, L]
                        shift_logits = full_logits[:, :-1, :].contiguous() # Contiguous often needed for view/reshape
                        shift_labels = cand_batch
                        
                        # Optimization: Use ignore_index in CrossEntropy directly
                        # We calculate loss per token, then sum
                        
                        # Flattening is usually faster than permuting for CE in PyTorch
                        loss_per_token = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.reshape(-1),
                            reduction='none',
                            ignore_index=self.tokenizer.pad_token_id
                        )
                        
                        # Reshape back to [B, L] to sum per entity
                        loss_per_token = loss_per_token.view(curr_batch_size, -1)
                        
                        # Sum and Negate
                        summed_scores = -loss_per_token.sum(dim=1)
                        
                        # Normalize by length (Length must be on GPU for division)
                        batch_lens = cpu_cand_lens[i:end_i].to(self.device, non_blocking=True)
                        gpu_scores[i : end_i] = summed_scores / batch_lens
                
                # --- Step 3: Filtering & Ranking ---
                if t not in self.entity_to_idx:
                    continue
                    
                target_idx = self.entity_to_idx[t]
                target_score = gpu_scores[target_idx]
                
                # Filter known facts
                # Optimization: Check if filtering is actually needed (len > 1) to avoid overhead
                valid_tails = self.hr_to_t.get((h, r), set())
                
                if len(valid_tails) > 1:
                    # Pre-filter in CPU to minimize data transfer size
                    valid_indices = [
                        self.entity_to_idx[vt] 
                        for vt in valid_tails 
                        if vt in self.entity_to_idx and vt != t
                    ]
                    
                    if valid_indices:
                        # Transfer only the small list of indices
                        mask_indices = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
                        gpu_scores.index_fill_(0, mask_indices, -float('inf'))
                
                # Rank
                # Note: .item() here is unavoidable for logic flow, but happens only once per triple
                rank = (gpu_scores > target_score).sum().item() + 1
                
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hits1 += 1
                
                # Explicit memory cleanup to prevent accumulation and fragmentation
                del prefix_kv, batch_kv, prefix_logits, cand_logits
                gpu_scores.zero_()  # Reuse instead of reallocating
        
        # Clear fragmented memory before returning to training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model.train()
        return {
            "mrr": mrr_sum / len(sample_indices),
            "h1": hits1 / len(sample_indices)
        }

    def train(self, epochs: int = 500, checkpoint_interval: int = None):
        """
        Train the model for a given number of epochs.
        
        Args:
            epochs: Number of epochs to train for
            checkpoint_interval: Save checkpoint every N epochs (None = only save final model)
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"  - Warmup epochs: {self.warmup_epochs}")
        print(f"  - Label smoothing: {self.label_smoothing}")
        print(f"  - Gradient clipping: {self.grad_clip}")
        print(f"  - Base LR: {self.config.lr}")
        print(f"  - Weight decay: {self.config.weight_decay}")
        if checkpoint_interval:
            print(f"  - Checkpoint interval: every {checkpoint_interval} epochs")
        
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
            
            # Compute sampled metrics at the end of each epoch
            if wandb.run is not None:
                metrics = self._compute_sampled_metrics(sample_size=64, batch_size=self.eval_batch_size)
                # Use the step count at the end of the epoch
                current_step = (epoch + 1) * len(self.train_loader)
                wandb.log({
                    "train/sampled_mrr": metrics["mrr"],
                    "train/sampled_h1": metrics["h1"],
                    "epoch": epoch + 1
                }, step=current_step)

            # Step scheduler after each epoch (only if it exists)
            if scheduler is not None:
                scheduler.step()
            
            # Save checkpoint at specified intervals
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.save_path, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_model(epoch + 1, checkpoint_path)
            
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