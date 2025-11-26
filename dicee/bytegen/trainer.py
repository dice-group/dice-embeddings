import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
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
    def __init__(self, model: ByteGenModel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, optimizer: torch.optim.Optimizer = None, save_path: str = "checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.device = config.device
        self.tokenizer = tokenizer
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, epochs: int = 500):
        print(f"Starting training for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Input: x[:-1], Target: x[1:]
                logits = self.model(batch[:, :-1])
                targets = batch[:, 1:]
                
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size), 
                    targets.reshape(-1), 
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
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
    def __init__(self, model: DistributedDataParallel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, optimizer: torch.optim.Optimizer = None, save_path: str = "checkpoints", gradient_acc_steps = 1):
        super().__init__(model, train_loader, config, tokenizer, optimizer, save_path)
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.gradient_acc_steps = gradient_acc_steps
        init_process_group(backend="nccl") 
        torch.cuda.set_device(self.local_rank)

    def train(self, epochs: int = 500):
        assert torch.cuda.is_available(), "Training using DDPTrainer only works on gpu(s)"
        print(f"Starting training for {epochs} epochs...")
        self.model.train()
        
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
                            ignore_index=self.tokenizer.pad_token_id
                        )
                        loss.backward()
                else:
                    # Input: x[:-1], Target: x[1:]
                    logits = self.model.module(batch[:, :-1])
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, self.config.vocab_size), 
                        targets.reshape(-1), 
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        if self.global_rank == 0:
            self.save_model(epochs, os.path.join(self.save_path, f"model_epoch_{epochs}.pt"))

        destroy_process_group()