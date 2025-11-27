import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer, train_bpe_tokenizer, BPETokenizer
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset
from dicee.bytegen.trainer import Trainer
from dicee.bytegen.evaluator import Evaluator

def run_experiment(tokenizer_type, vocab_size_arg, dataset_type, inverse, epochs=300):
    start_time = time.time()

    dataset_path = os.path.join(os.getcwd(), "KGs/UMLS")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")

    # Initialize Tokenizer
    if tokenizer_type == 'Byte':
        tokenizer = ByteTokenizer()
        actual_vocab_size = tokenizer.vocab_size
        tokenizer_name = "ByteTokenizer"
    else: # BPE
        tokenizer_path = f"tokenizer_bpe_{vocab_size_arg}.json"
        print(f"Training BPE tokenizer with vocab_size={vocab_size_arg}...")
        try:
            tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=vocab_size_arg)
            actual_vocab_size = tokenizer.vocab_size
        except Exception as e:
            print(f"Failed to train BPE tokenizer with vocab_size={vocab_size_arg}: {e}")
            raise e
        tokenizer_name = f"BPE-{vocab_size_arg}"

    print(f"--- Running experiment with {tokenizer_name} (vocab size: {actual_vocab_size}) on {dataset_type} dataset (Inverse={inverse}) ---")

    # Config
    conf = ByteGenConfig(
        block_size=512, 
        n_layer=4, 
        n_head=4, 
        n_embd=256, 
        dropout=0.1, 
        batch_size=64,
        lr=0.001,
        vocab_size=actual_vocab_size
    )
    
    # Dataset
    print(f"Loading datasets ({dataset_type})...")
    DatasetClass = ByteGenBFSDataset if dataset_type == 'BFS' else ByteGenDataset
    train_ds = DatasetClass(dataset_path, tokenizer, split='train', block_size=conf.block_size, inverse=inverse)
    test_ds = DatasetClass(dataset_path, tokenizer, split='test', block_size=conf.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = ByteGenModel(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")

    # Trainer
    print(f"Training for {epochs} epochs...")
    trainer = Trainer(model, train_loader, conf, tokenizer, optimizer)
    trainer.train(epochs)
            
    # Evaluate
    print("Evaluating...")
    evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
    metrics = evaluator.evaluate()
    
    end_time = time.time()
    duration = end_time - start_time

    return {
        "Dataset": dataset_type,
        "Inverse": inverse,
        "Tokenizer": tokenizer_name,
        "Vocab Size": actual_vocab_size,
        "Params": num_params,
        "Epochs": epochs,
        "Time (s)": round(duration, 2),
        "MRR": metrics["mrr"],
        "H@1": metrics["h1"],
        "H@3": metrics["h3"],
        "H@10": metrics["h10"]
    }

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    dataset_types = ['RandomWalk', 'BFS']
    inverse_settings = [True, False]
    experiments = [
        ('Byte', None),
        ('BPE', 260),
        ('BPE', 512),
        ('BPE', 728),
    ]
    
    epochs = 30
    results = []
    for dataset_type in dataset_types:
        for inverse in inverse_settings:
            for tokenizer_type, vocab_size in experiments:
                try:
                    res = run_experiment(tokenizer_type, vocab_size, dataset_type, inverse, epochs)
                    results.append(res)
                except Exception as e:
                    print(f"Experiment failed for {tokenizer_type} {vocab_size} on {dataset_type} (Inv={inverse}): {e}")
                    # Log failure
                    results.append({
                        "Dataset": dataset_type,
                        "Inverse": inverse,
                        "Tokenizer": f"{tokenizer_type}-{vocab_size}" if vocab_size else tokenizer_type,
                        "Vocab Size": vocab_size,
                        "Params": 0,
                        "Epochs": epochs,
                        "Time (s)": 0.0,
                        "MRR": 0.0,
                        "H@1": 0.0,
                        "H@3": 0.0,
                        "H@10": 0.0
                    })
            
    df = pd.DataFrame(results)
    
    print("\n=== Final Results ===")
    
    # Group by Dataset and Inverse for display
    for dataset_type in dataset_types:
        for inverse in inverse_settings:
            subset = df[(df['Dataset'] == dataset_type) & (df['Inverse'] == inverse)].copy()
            
            if subset.empty:
                continue
                
            print(f"\n--- Dataset: {dataset_type} | Inverse: {inverse} ---")
            
            # Find best performers
            metrics = ['MRR', 'H@1', 'H@3', 'H@10']
            best_performers = {}
            for metric in metrics:
                if not subset[metric].empty:
                    idx = subset[metric].idxmax()
                    best_row = subset.loc[idx]
                    best_performers[metric] = (best_row['Tokenizer'], best_row[metric])
            
            # Print Table
            display_cols = [c for c in subset.columns if c not in ['Dataset', 'Inverse']]
            print(subset[display_cols].to_string(index=False))
            
            # Print Highlights
            print("\nBest Performers:")
            for metric, (tok, score) in best_performers.items():
                print(f"  {metric:<5}: {score:.4f} ({tok})")
    
    # Save to CSV
    df.to_csv(f"grid_search_results_{dataset_type}_{inverse}.csv", index=False)
    print(f"\nFull results saved to grid_search_results_{dataset_type}_{inverse}.csv")

if __name__ == "__main__":
    main()

