import os
import time
import torch
import pandas as pd
from multiprocessing import Pool, set_start_method
from torch.utils.data import DataLoader
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer, train_bpe_tokenizer, BPETokenizer
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset
from dicee.bytegen.trainer import Trainer
from dicee.bytegen.evaluator import Evaluator

def run_experiment(args):
    """Worker function for parallel execution."""
    tokenizer_type, vocab_size_arg, dataset_type, inverse, epochs, gpu_id = args
    
    try:
        # Set GPU for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
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
            # Unique path per process to avoid conflicts
            tokenizer_path = f"tokenizer_bpe_{vocab_size_arg}_gpu{gpu_id}.json"
            tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=vocab_size_arg)
            actual_vocab_size = tokenizer.vocab_size
            tokenizer_name = f"BPE-{vocab_size_arg}"

        print(f"[GPU {gpu_id}] Running: {tokenizer_name} (vocab={actual_vocab_size}) on {dataset_type} (Inverse={inverse})")

        # Config
        conf = ByteGenConfig(
            block_size=512, 
            n_layer=4, 
            n_head=4, 
            n_embd=256, 
            dropout=0.1, 
            batch_size=64,
            lr=0.001,
            vocab_size=actual_vocab_size,
            device=device  # Use assigned GPU
        )
        
        # Dataset
        DatasetClass = ByteGenBFSDataset if dataset_type == 'BFS' else ByteGenDataset
        train_ds = DatasetClass(dataset_path, tokenizer, split='train', block_size=conf.block_size, inverse=inverse)
        test_ds = DatasetClass(dataset_path, tokenizer, split='test', block_size=conf.block_size)
        
        train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=0)
        
        # Model
        model = ByteGenModel(conf).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[GPU {gpu_id}] Model parameters: {num_params}")

        # Trainer
        trainer = Trainer(model, train_loader, conf, tokenizer, optimizer)
        trainer.train(epochs)
                
        # Evaluate
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
    except Exception as e:
        print(f"[GPU {gpu_id}] FAILED: {tokenizer_type}-{vocab_size_arg} on {dataset_type} (Inv={inverse}): {e}")
        return {
            "Dataset": dataset_type,
            "Inverse": inverse,
            "Tokenizer": f"{tokenizer_type}-{vocab_size_arg}" if vocab_size_arg else tokenizer_type,
            "Vocab Size": vocab_size_arg or 0,
            "Params": 0,
            "Epochs": epochs,
            "Time (s)": 0.0,
            "MRR": 0.0,
            "H@1": 0.0,
            "H@3": 0.0,
            "H@10": 0.0
        }

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    print(f"Found {num_gpus} GPUs")
    
    dataset_types = ['RandomWalk', 'BFS']
    inverse_settings = [True, False]
    experiments = [
        ('Byte', None),
        ('BPE', 260),
        ('BPE', 512),
        ('BPE', 728),
    ]
    
    epochs = 300
    
    # Build all experiment configs with round-robin GPU assignment
    all_configs = []
    for i, (dataset_type, inverse, (tokenizer_type, vocab_size)) in enumerate(
        [(d, inv, exp) for d in dataset_types for inv in inverse_settings for exp in experiments]
    ):
        gpu_id = i % num_gpus
        all_configs.append((tokenizer_type, vocab_size, dataset_type, inverse, epochs, gpu_id))
    
    print(f"Running {len(all_configs)} experiments across {num_gpus} GPUs...")
    
    # Run in parallel with one process per GPU
    with Pool(processes=num_gpus) as pool:
        results = pool.map(run_experiment, all_configs)
            
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
    df.to_csv("grid_search_results.csv", index=False)
    print("\nFull results saved to grid_search_results.csv")

if __name__ == "__main__":
    set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()

