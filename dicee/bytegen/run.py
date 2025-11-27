import os
from torch.utils.data import DataLoader
import torch
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer, train_bpe_tokenizer
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset
from dicee.bytegen.trainer import Trainer
from dicee.bytegen.evaluator import Evaluator


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Setup
    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    
    # Initialize Tokenizer
    tokenizer = ByteTokenizer()
    # tokenizer_path = "tokenizer.json"
    # tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=512)
    
    conf = ByteGenConfig(
        block_size=512, 
        n_layer=8, 
        n_head=8, 
        n_embd=512, 
        dropout=0.1, 
        batch_size=64,
        lr=6e-4,
        vocab_size=tokenizer.vocab_size
    )
    
    # Dataset
    train_ds = ByteGenBFSDataset(dataset_path, tokenizer, split='train', block_size=conf.block_size)
    test_ds = ByteGenBFSDataset(dataset_path, tokenizer, split='test', block_size=conf.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=4)
    # for batch in train_loader:
    #     decoded_batch = [tokenizer.decode(x.tolist()) for x in batch]
    #     print(decoded_batch)
    #     exit()
    # Model
    model = ByteGenModel(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    
    # Trainer
    EPOCHS = 30
    trainer = Trainer(model, train_loader, conf, tokenizer, optimizer)
    trainer.train(EPOCHS)
            
    # Evaluate
    print("Training complete. Evaluating...")
    evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
    evaluator.evaluate()

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
    