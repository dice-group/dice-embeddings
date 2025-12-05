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
import os


class Trainer:
    def __init__(self, model: ByteGenModel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, 
                 optimizer: torch.optim.Optimizer = None, save_path: str = "checkpoints",
                 warmup_epochs: int = 5, label_smoothing: float = 0.0, grad_clip: float = 1.0,
                 train_dataset=None):
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
        os.makedirs(self.save_path, exist_ok=True)
        
        # Build entity list for H@1 computation
        if train_dataset is not None:
            self.entities = sorted(list(set(
                [t[0] for t in train_dataset.triples] + 
                [t[2] for t in train_dataset.triples]
            )))
            # Build known facts for filtering
            self.known_facts = set((h, r, t) for h, r, t in train_dataset.triples)
        else:
            self.entities = None
            self.known_facts = None

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
        """Compute loss only on tail tokens (after SEP_RT, until EOS/PAD)."""
        sep_rt_id = self.tokenizer.sep_rt_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # Compute per-token loss
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction='none'
        ).reshape(targets.shape)
        
        # Create mask for tail tokens only
        mask = torch.zeros_like(targets, dtype=torch.float)
        for b in range(targets.shape[0]):
            # Find SEP_RT position in targets
            sep_rt_positions = (targets[b] == sep_rt_id).nonzero(as_tuple=True)[0]
            if len(sep_rt_positions) == 0:
                continue
            start = sep_rt_positions[0].item() + 1  # Start after SEP_RT
            
            # Find end (EOS or PAD)
            end = targets.shape[1]
            for pos in range(start, targets.shape[1]):
                if targets[b, pos] == eos_id or targets[b, pos] == pad_id:
                    end = pos + 1 if targets[b, pos] == eos_id else pos  # Include EOS
                    break
            mask[b, start:end] = 1.0
        
        # Compute masked loss
        if mask.sum() > 0:
            tail_loss = (loss_per_token * mask).sum() / mask.sum()
            return tail_loss.item()
        return 0.0

    def _compute_hits_at_1(self, sample_size: int = 200, batch_size: int = 64) -> float:
        """Compute H@1 on a sample of training triples."""
        if self.train_dataset is None or self.entities is None:
            return 0.0
        
        import random
        import numpy as np
        
        self.model.eval()
        
        # Sample triples
        triples = self.train_dataset.triples
        sample_indices = random.sample(range(len(triples)), min(sample_size, len(triples)))
        
        hits1 = 0
        suffix_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for idx in sample_indices:
                h, r, t = triples[idx]
                
                # Build prefix: [EOS] H [SEP_HR] R [SEP_RT]
                prefix = [self.tokenizer.eos_token_id] + list(h) + \
                         [self.tokenizer.sep_hr_token_id] + list(r) + \
                         [self.tokenizer.sep_rt_token_id]
                
                # Score all candidate entities
                scores = []
                for i in range(0, len(self.entities), batch_size):
                    batch_entities = self.entities[i:i+batch_size]
                    
                    # Build sequences for batch
                    max_cand_len = max(len(e) for e in batch_entities) + 1
                    input_ids = []
                    slices = []
                    
                    for cand in batch_entities:
                        seq = prefix + list(cand) + [suffix_token_id]
                        start_idx = len(prefix) - 1
                        end_idx = len(seq) - 1
                        padded = seq + [self.tokenizer.pad_token_id] * (len(prefix) + max_cand_len - len(seq))
                        input_ids.append(padded)
                        slices.append((start_idx, end_idx, len(cand) + 1))
                    
                    x = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                    logits, _ = self.model(x)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    for b, (s, e, l) in enumerate(slices):
                        relevant_logits = log_probs[b, s:e, :]
                        targets = x[b, s+1:e+1]
                        token_scores = relevant_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                        score = token_scores.sum().item() / l
                        scores.append(score)
                
                scores = np.array(scores)
                
                # Find target index and apply filtering
                target_idx = self.entities.index(t) if t in self.entities else -1
                if target_idx < 0:
                    continue
                    
                target_score = scores[target_idx]
                
                # Filter known facts (other valid tails for same h,r)
                for i, cand in enumerate(self.entities):
                    if cand == t:
                        continue
                    if (h, r, cand) in self.known_facts:
                        scores[i] = -float('inf')
                
                # Check if target is rank 1
                rank = np.sum(scores > target_score) + 1
                if rank == 1:
                    hits1 += 1
        
        self.model.train()
        return hits1 / len(sample_indices)

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
                
                # Compute tail-only loss for diagnostics (every 10 batches)
                # if num_batches % 10 == 0:
                #     with torch.no_grad():
                #         tail_loss = self._compute_tail_loss(logits, targets)
                #         total_tail_loss += tail_loss
                #         tail_loss_count += 1
                
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
                        logits = self.model.module(batch[:, :-1]) 
                        loss = F.cross_entropy(
                            logits.reshape(-1, self.config.vocab_size), 
                            targets.reshape(-1), 
                            ignore_index=self.tokenizer.pad_token_id,
                            label_smoothing=self.label_smoothing
                        )
                        loss.backward()
                else:
                    # Input: x[:-1], Target: x[1:]
                    logits = self.model.module(batch[:, :-1])
                    
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