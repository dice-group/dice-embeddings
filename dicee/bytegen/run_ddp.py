import os
from torch.utils.data import DataLoader
import torch
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer
from dicee.bytegen.dataset import ByteGenDataset
from dicee.bytegen.trainer import DDPTrainer, Trainer
from dicee.bytegen.evaluator import Evaluator
#DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

if __name__ == "__main__":
    # Setup
    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Initialize Tokenizer
    tokenizer = ByteTokenizer()
    
    conf = ByteGenConfig(
        block_size=128, 
        n_layer=4, 
        n_head=4, 
        n_embd=256, 
        dropout=0.1, 
        batch_size=512,
        lr=0.001,
        vocab_size=tokenizer.vocab_size
    )
    
    # Dataset TODO: LF: Here we have to use other Dataset that does not load whole graph into memory (for bigger graphs)
    train_ds = ByteGenDataset(dataset_path, tokenizer, split='train', block_size=conf.block_size, inverse=True)
    test_ds = ByteGenDataset(dataset_path, tokenizer, split='test', block_size=conf.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=False, sampler=DistributedSampler(train_ds, shuffle=True), num_workers=4)
    
    # Model
    model = ByteGenModel(conf).to(conf.device)
    # DDP
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    
    # Trainer
    EPOCHS = 300
    trainer = DDPTrainer(model, train_loader, conf, tokenizer, optimizer, gradient_acc_steps=100)
    trainer.train(EPOCHS)
            
    # Evaluate
    if global_rank == 0:
        print("Training complete. Evaluating on rank 0...")
        evaluator = Evaluator(model.module, train_ds, test_ds, tokenizer)
        evaluator.evaluate()

