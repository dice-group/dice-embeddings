import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, set_start_method
from torch.utils.data import DataLoader
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer, train_bpe_tokenizer, BPETokenizer
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset, IsolatedTripleDataset
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

        # Initialize Tokenizer (with inverse parameter to match dataset)
        if tokenizer_type == 'Byte':
            tokenizer = ByteTokenizer()
            actual_vocab_size = tokenizer.vocab_size
            tokenizer_name = "ByteTokenizer"
        else:  # BPE
            # Unique path per process to avoid conflicts
            tokenizer_path = f"tokenizer_bpe_{vocab_size_arg}_gpu{gpu_id}.json"
            tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=vocab_size_arg, inverse=inverse)
            actual_vocab_size = tokenizer.vocab_size
            tokenizer_name = f"BPE-{vocab_size_arg}"

        print(f"[GPU {gpu_id}] Running: {tokenizer_name} (vocab={actual_vocab_size}) on {dataset_type} (Inverse={inverse})")

        # Dataset selection with block_size handling
        if dataset_type == 'Isolated':
            # For Isolated: auto-calculate block_size from data to ensure it works for eval
            # First load with block_size=None to auto-calculate per-split minimums
            train_ds = IsolatedTripleDataset(dataset_path, tokenizer, split='train', block_size=None, inverse=inverse)
            test_ds = IsolatedTripleDataset(dataset_path, tokenizer, split='test', block_size=None)
            
            # Compute eval-safe block_size (considers ALL entities as potential candidates)
            block_size = IsolatedTripleDataset.compute_required_block_size(train_ds, test_ds)
            
            # Update datasets to use the eval-safe block_size
            train_ds.block_size = block_size
            test_ds.block_size = block_size
            print(f"[GPU {gpu_id}] Isolated block_size set to {block_size} (eval-safe)")
        elif dataset_type == 'BFS':
            block_size = 256
            train_ds = ByteGenBFSDataset(dataset_path, tokenizer, split='train', block_size=block_size, inverse=inverse)
            test_ds = ByteGenBFSDataset(dataset_path, tokenizer, split='test', block_size=block_size)
        else:  # RandomWalk
            block_size = 256
            train_ds = ByteGenDataset(dataset_path, tokenizer, split='train', block_size=block_size, inverse=inverse)
            test_ds = ByteGenDataset(dataset_path, tokenizer, split='test', block_size=block_size)
        
        # Config with optimized parameters from run.py
        conf = ByteGenConfig(
            block_size=block_size, 
            n_layer=8, 
            n_head=8, 
            n_embd=512, 
            dropout=0.1,  # Add dropout for generalization
            batch_size=32,
            lr=3e-4,
            vocab_size=actual_vocab_size,
            device=device
        )
        
        train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=0)
        
        # Model
        model = ByteGenModel(conf).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[GPU {gpu_id}] Model parameters: {num_params}")

        # Trainer with updated parameters (matching run.py)
        trainer = Trainer(
            model, train_loader, conf, tokenizer, optimizer,
            label_smoothing=0.1,
            warmup_epochs=5,
            train_dataset=train_ds,
            eval_every=301
        )
        trainer.train(epochs)
                
        # Evaluate on BOTH train and test sets
        evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
        train_metrics = evaluator.evaluate(split='train')
        test_metrics = evaluator.evaluate(split='test')
        
        end_time = time.time()
        duration = end_time - start_time

        return {
            "Dataset": dataset_type,
            "Inverse": inverse,
            "Tokenizer": tokenizer_name,
            "Vocab Size": actual_vocab_size,
            "Block Size": block_size,
            "Params": num_params,
            "Epochs": epochs,
            "Time (s)": round(duration, 2),
            # Train metrics
            "Train_MRR": train_metrics["mrr"],
            "Train_H@1": train_metrics["h1"],
            "Train_H@3": train_metrics["h3"],
            "Train_H@10": train_metrics["h10"],
            # Test metrics
            "Test_MRR": test_metrics["mrr"],
            "Test_H@1": test_metrics["h1"],
            "Test_H@3": test_metrics["h3"],
            "Test_H@10": test_metrics["h10"]
        }
    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] FAILED: {tokenizer_type}-{vocab_size_arg} on {dataset_type} (Inv={inverse}): {e}")
        traceback.print_exc()
        return {
            "Dataset": dataset_type,
            "Inverse": inverse,
            "Tokenizer": f"{tokenizer_type}-{vocab_size_arg}" if vocab_size_arg else tokenizer_type,
            "Vocab Size": vocab_size_arg or 0,
            "Block Size": 128 if dataset_type == 'Isolated' else 256,
            "Params": 0,
            "Epochs": epochs,
            "Time (s)": 0.0,
            "Train_MRR": 0.0, "Train_H@1": 0.0, "Train_H@3": 0.0, "Train_H@10": 0.0,
            "Test_MRR": 0.0, "Test_H@1": 0.0, "Test_H@3": 0.0, "Test_H@10": 0.0
        }


def create_plots(df: pd.DataFrame, output_dir: str = "comparison_results"):
    """Generate visualization plots for the grid search results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    # --- Plot 1: Comparison by Dataset Type (Test Set) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test Set Performance by Dataset Type', fontsize=14, fontweight='bold')
    
    metrics = ['Test_MRR', 'Test_H@1', 'Test_H@3', 'Test_H@10']
    metric_names = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    for ax, metric, name in zip(axes.flatten(), metrics, metric_names):
        pivot = df.pivot_table(values=metric, index='Dataset', columns='Tokenizer', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, color=colors[:len(pivot.columns)], edgecolor='black', linewidth=0.5)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(name)
        ax.legend(title='Tokenizer', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_by_dataset.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Train vs Test Performance ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Train vs Test Performance Comparison', fontsize=14, fontweight='bold')
    
    train_metrics = ['Train_MRR', 'Train_H@1', 'Train_H@3', 'Train_H@10']
    test_metrics = ['Test_MRR', 'Test_H@1', 'Test_H@3', 'Test_H@10']
    
    for ax, (train_m, test_m, name) in zip(axes.flatten(), zip(train_metrics, test_metrics, metric_names)):
        x = np.arange(len(df))
        width = 0.35
        
        # Group by config name
        df_sorted = df.sort_values(['Dataset', 'Tokenizer'])
        config_names = [f"{row['Dataset'][:3]}-{row['Tokenizer'][:5]}" for _, row in df_sorted.iterrows()]
        
        bars1 = ax.bar(x - width/2, df_sorted[train_m], width, label='Train', color=colors[0], edgecolor='black')
        bars2 = ax.bar(x + width/2, df_sorted[test_m], width, label='Test', color=colors[1], edgecolor='black')
        
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel(name)
        ax.set_xticks(x[::2])  # Show every other label to avoid clutter
        ax.set_xticklabels([config_names[i] for i in range(0, len(config_names), 2)], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_vs_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Effect of Inverse Relations ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Effect of Inverse Relations on Test Performance', fontsize=14, fontweight='bold')
    
    for ax, metric, name in zip(axes, ['Test_MRR', 'Test_H@1'], ['MRR', 'Hits@1']):
        pivot = df.pivot_table(values=metric, index='Dataset', columns='Inverse', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, color=[colors[2], colors[3]], edgecolor='black', linewidth=0.5)
        ax.set_title(f'{name} by Inverse Setting', fontweight='bold')
        ax.set_xlabel('Dataset Type')
        ax.set_ylabel(name)
        ax.legend(title='Inverse', labels=['False', 'True'])
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inverse_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 4: Heatmap of Test MRR ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot_table(values='Test_MRR', index=['Dataset', 'Inverse'], columns='Tokenizer', aggfunc='mean')
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels([f"{d} (Inv={i})" for d, i in pivot.index])
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontweight='bold')
    
    ax.set_title('Test MRR Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='MRR')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_mrr_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 5: Training Time vs Performance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df['Time (s)'], df['Test_MRR'], 
                         c=df['Dataset'].astype('category').cat.codes, 
                         s=df['Params'] / 10000,  # Size by parameters
                         alpha=0.7, cmap='Set1', edgecolor='black')
    
    ax.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax.set_ylabel('Test MRR', fontweight='bold')
    ax.set_title('Training Time vs Test MRR\n(size = model parameters)', fontsize=14, fontweight='bold')
    
    # Add legend for dataset types
    handles = []
    for i, dataset in enumerate(df['Dataset'].unique()):
        handles.append(plt.scatter([], [], c=plt.cm.Set1(i/3), label=dataset, s=100))
    ax.legend(handles=handles, title='Dataset')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def log_best_approaches(df: pd.DataFrame):
    """Log the overall best performing approaches for train and test sets."""
    print("\n" + "="*80)
    print("OVERALL BEST PERFORMING APPROACHES")
    print("="*80)
    
    metrics_train = ['Train_MRR', 'Train_H@1', 'Train_H@3', 'Train_H@10']
    metrics_test = ['Test_MRR', 'Test_H@1', 'Test_H@3', 'Test_H@10']
    metric_names = ['MRR', 'H@1', 'H@3', 'H@10']
    
    def format_config(row):
        return f"{row['Dataset']} + {row['Tokenizer']} (Inverse={row['Inverse']})"
    
    # --- Best on Train Set ---
    print("\n📊 BEST ON TRAIN SET:")
    print("-" * 60)
    for train_m, name in zip(metrics_train, metric_names):
        idx = df[train_m].idxmax()
        best_row = df.loc[idx]
        config = format_config(best_row)
        print(f"  {name:>6}: {best_row[train_m]:.4f}  →  {config}")
    
    # --- Best on Test Set ---
    print("\n📊 BEST ON TEST SET:")
    print("-" * 60)
    for test_m, name in zip(metrics_test, metric_names):
        idx = df[test_m].idxmax()
        best_row = df.loc[idx]
        config = format_config(best_row)
        print(f"  {name:>6}: {best_row[test_m]:.4f}  →  {config}")
    
    # --- Overall Best Configuration (average of key metrics) ---
    print("\n🏆 OVERALL BEST CONFIGURATION:")
    print("-" * 60)
    
    # Compute combined score (average of MRR, H@1, H@3, H@10 on test)
    df['Combined_Test_Score'] = df[metrics_test].mean(axis=1)
    df['Combined_Train_Score'] = df[metrics_train].mean(axis=1)
    
    best_test_idx = df['Combined_Test_Score'].idxmax()
    best_test_row = df.loc[best_test_idx]
    
    best_train_idx = df['Combined_Train_Score'].idxmax()
    best_train_row = df.loc[best_train_idx]
    
    print(f"\n  Best Test Generalization:")
    print(f"    Config: {format_config(best_test_row)}")
    print(f"    Test Metrics:  MRR={best_test_row['Test_MRR']:.4f}, H@1={best_test_row['Test_H@1']:.4f}, "
          f"H@3={best_test_row['Test_H@3']:.4f}, H@10={best_test_row['Test_H@10']:.4f}")
    print(f"    Train Metrics: MRR={best_test_row['Train_MRR']:.4f}, H@1={best_test_row['Train_H@1']:.4f}, "
          f"H@3={best_test_row['Train_H@3']:.4f}, H@10={best_test_row['Train_H@10']:.4f}")
    
    print(f"\n  Best Train Performance:")
    print(f"    Config: {format_config(best_train_row)}")
    print(f"    Train Metrics: MRR={best_train_row['Train_MRR']:.4f}, H@1={best_train_row['Train_H@1']:.4f}, "
          f"H@3={best_train_row['Train_H@3']:.4f}, H@10={best_train_row['Train_H@10']:.4f}")
    print(f"    Test Metrics:  MRR={best_train_row['Test_MRR']:.4f}, H@1={best_train_row['Test_H@1']:.4f}, "
          f"H@3={best_train_row['Test_H@3']:.4f}, H@10={best_train_row['Test_H@10']:.4f}")
    
    # --- Generalization Gap Analysis ---
    print("\n📉 GENERALIZATION GAP (Train - Test):")
    print("-" * 60)
    df['Gap_MRR'] = df['Train_MRR'] - df['Test_MRR']
    df['Gap_H@1'] = df['Train_H@1'] - df['Test_H@1']
    
    best_generalization_idx = df['Gap_MRR'].abs().idxmin()
    best_gen_row = df.loc[best_generalization_idx]
    
    print(f"  Smallest MRR Gap: {best_gen_row['Gap_MRR']:.4f}")
    print(f"    Config: {format_config(best_gen_row)}")
    print(f"    Train MRR: {best_gen_row['Train_MRR']:.4f}, Test MRR: {best_gen_row['Test_MRR']:.4f}")
    
    print("\n" + "="*80)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    print(f"Found {num_gpus} GPUs")
    
    dataset_types = ['RandomWalk', 'BFS', 'Isolated']
    inverse_settings = [True, False]
    experiments = [
        ('Byte', None),
        ('BPE', 260),
        ('BPE', 512),
        ('BPE', 733),
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
            
            # Find best performers for test set
            test_metrics = ['Test_MRR', 'Test_H@1', 'Test_H@3', 'Test_H@10']
            train_metrics = ['Train_MRR', 'Train_H@1', 'Train_H@3', 'Train_H@10']
            
            # Print Table
            display_cols = [c for c in subset.columns if c not in ['Dataset', 'Inverse']]
            print(subset[display_cols].to_string(index=False))
            
            # Print Best Performers
            print("\nBest Performers (Test):")
            for metric in test_metrics:
                if not subset[metric].empty:
                    idx = subset[metric].idxmax()
                    best_row = subset.loc[idx]
                    print(f"  {metric:<10}: {best_row[metric]:.4f} ({best_row['Tokenizer']})")
            
            print("\nBest Performers (Train):")
            for metric in train_metrics:
                if not subset[metric].empty:
                    idx = subset[metric].idxmax()
                    best_row = subset.loc[idx]
                    print(f"  {metric:<10}: {best_row[metric]:.4f} ({best_row['Tokenizer']})")
    
    # Log overall best approaches
    log_best_approaches(df)
    
    # Create visualization plots
    create_plots(df)
    
    # Save to CSV
    df.to_csv("grid_search_results.csv", index=False)
    print("\nFull results saved to grid_search_results.csv")


if __name__ == "__main__":
    set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()
