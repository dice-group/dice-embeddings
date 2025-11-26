import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dicee.bytegen.bytegen import ByteGenModel
from dicee.bytegen.bytegen import ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer


class Trainer:
    def __init__(self, model: ByteGenModel, train_loader: DataLoader, config: ByteGenConfig, tokenizer: ByteTokenizer, optimizer: torch.optim.Optimizer = None):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.device = config.device
        self.tokenizer = tokenizer

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