import os
from sympy import Inverse
from torch.utils.data import DataLoader
import torch
import wandb
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer, train_bpe_tokenizer
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset, IsolatedTripleDataset
from dicee.bytegen.trainer import Trainer
from dicee.bytegen.evaluator import Evaluator


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wandb.init(project="bytegen-experiment")
    # Setup
    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    
    # Initialize Tokenizer (with inverse=True to match dataset)
    USE_INVERSE = True
    # tokenizer_path = "tokenizer.json"
    # tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=1024, inverse=USE_INVERSE)
    tokenizer = ByteTokenizer()

    BFS_BLOCK_SIZE = 256
    
    train_ds = ByteGenDataset(dataset_path, tokenizer, split='train', block_size=BFS_BLOCK_SIZE, inverse=USE_INVERSE)
    test_ds = ByteGenDataset(dataset_path, tokenizer, split='test', block_size=BFS_BLOCK_SIZE, inverse=USE_INVERSE)
    
    conf = ByteGenConfig(
        block_size=BFS_BLOCK_SIZE,  # Larger for BFS sequences
        n_layer=6, 
        n_head=2, 
        n_embd=16, 
        dropout=0.0,  # Add dropout for generalization
        batch_size=128,
        lr=3e-4,
        vocab_size=tokenizer.vocab_size
    )


    
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = ByteGenModel(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    
    # Trainer - with H@1 diagnostic every 10 epochs
    EPOCHS = 100
    trainer = Trainer(model, train_loader, conf, tokenizer, optimizer, 
                      label_smoothing=0.1, warmup_epochs=5,
                      train_dataset=train_ds)
    trainer.train(EPOCHS)
            
    # Evaluate
    print("Training complete. Evaluating...")
    evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
    evaluator.evaluate(split='train')
    evaluator.evaluate(split='test')

    while True:
        print("\n--- Interactive Generation ---")
        head = input("Enter head entity (or 'q' to quit): ").strip()
        if head.lower() in ['q', 'quit']:
            break
            
        try:
            context = tokenizer.encode(head)
            x = torch.tensor([context], dtype=torch.long, device=conf.device)
            y = model.generate(x, tokenizer, max_new_tokens=64, temperature=0.8, top_k=10)
            
            out = y[0].tolist()
            decoded = tokenizer.decode(out)
                
            print(f"Generated: {decoded}")
            
        except Exception as e:
            print(f"Error: {e}")
    