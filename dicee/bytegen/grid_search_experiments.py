import os
import time
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, set_start_method, Manager
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import wandb
import sys
from dicee.bytegen.bytegen import ByteGenModel, ByteGenConfig
from dicee.bytegen.tokenizer import ByteTokenizer, train_bpe_tokenizer, BPETokenizer
from dicee.bytegen.dataset import ByteGenDataset, ByteGenBFSDataset, IsolatedTripleDataset
from dicee.bytegen.trainer import Trainer
from dicee.bytegen.evaluator import Evaluator
import gc

# Load environment variables from .env file
load_dotenv()

# Global queue for workers
gpu_queue = None

def init_worker(q):
    global gpu_queue
    gpu_queue = q

def run_experiment(args):
    """Worker function for parallel execution."""
    (tokenizer_type, vocab_size_arg, dataset_type, inverse, epochs, dataset_path, output_dir,
     n_layer, n_head, n_embd, dropout, batch_size, lr, label_smoothing, eval_batch_size, wandb_config) = args
    
    # Get a free GPU ID from the queue
    gpu_id = gpu_queue.get()
    
    # Setup logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    tok_label = "Byte" if tokenizer_type == 'Byte' else f"BPE-{vocab_size_arg}"
    log_filename = f"gpu{gpu_id}_{dataset_type}_{tok_label}_Inv{inverse}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    print(f"[GPU {gpu_id}] 🚀 Starting {dataset_type} {tok_label} Inv={inverse}. Logs: {log_path}")
    
    # Redirect stdout/stderr to file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, 'w', buffering=1) # Line buffered
    sys.stdout = log_file
    sys.stderr = log_file
    
    # Initialize wandb run for this worker
    if wandb_config:
        try:
            # Set start method for wandb to work in multiprocess
            # wandb.require("service") # Uncomment if issues arise
            if wandb_config.get('api_key'):
                wandb.login(key=wandb_config['api_key'])
        except:
            pass
            
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        gc.collect() 
 

        # Set GPU for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        start_time = time.time()

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could not find dataset at {dataset_path}")

        # Initialize Tokenizer (with inverse parameter to match dataset)
        if tokenizer_type == 'Byte':
            tokenizer = ByteTokenizer()
            actual_vocab_size = tokenizer.vocab_size
            tokenizer_name = "ByteTokenizer"
        else:  # BPE
            # Unique path per process to avoid conflicts (save in output_dir)
            tokenizer_path = os.path.join(output_dir, f"tokenizer_bpe_{vocab_size_arg}_gpu{gpu_id}.json")
            tokenizer = train_bpe_tokenizer(dataset_path, tokenizer_path, vocab_size=vocab_size_arg, inverse=inverse)
            actual_vocab_size = tokenizer.vocab_size
            tokenizer_name = f"BPE-{vocab_size_arg}"

        print(f"[GPU {gpu_id}] Running: {tokenizer_name} (vocab={actual_vocab_size}) on {dataset_type} (Inverse={inverse})")

        # Initialize wandb run
        if wandb_config:
            run_name = f"{dataset_type}-{tokenizer_name}-Inv{inverse}"
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                group=wandb_config['group'],
                name=run_name,
                config={
                    "dataset": dataset_type,
                    "tokenizer": tokenizer_name,
                    "inverse": inverse,
                    "vocab_size": actual_vocab_size,
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                    "lr": lr,
                    "batch_size": batch_size
                },
                reinit=True
            )

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
            n_layer=n_layer, 
            n_head=n_head, 
            n_embd=n_embd, 
            dropout=dropout,
            batch_size=batch_size,
            lr=lr,
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
            label_smoothing=label_smoothing,
            warmup_epochs=5,
            train_dataset=train_ds,
        )
        trainer.train(epochs)
                
        # Evaluate on BOTH train and test sets
        evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
        train_metrics = evaluator.evaluate(split='train', batch_size=eval_batch_size)
        test_metrics = evaluator.evaluate(split='test', batch_size=eval_batch_size)
        
        end_time = time.time()
        duration = end_time - start_time

        del model, optimizer, trainer, train_loader, train_ds, test_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
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
    finally:
        # Finish wandb run if it was started
        if wandb_config and wandb.run is not None:
            wandb.finish()
            
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
            
        # Always return the GPU to the queue
        gpu_queue.put(gpu_id)


def create_plots(df: pd.DataFrame, output_dir: str = "comparison_results", log_to_wandb: bool = False):
    """Generate visualization plots for the grid search results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    wandb_images = {}  # Collect images for wandb logging
    
    # --- Plot 1a: Comparison by Dataset Type (Test Set) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test Set Performance by Dataset Type', fontsize=14, fontweight='bold')
    
    test_metrics = ['Test_MRR', 'Test_H@1', 'Test_H@3', 'Test_H@10']
    metric_names = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    for ax, metric, name in zip(axes.flatten(), test_metrics, metric_names):
        pivot = df.pivot_table(values=metric, index='Dataset', columns='Tokenizer', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, color=colors[:len(pivot.columns)], edgecolor='black', linewidth=0.5)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(name)
        ax.legend(title='Tokenizer', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'test_by_dataset.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['test_by_dataset'] = wandb.Image(plot_path, caption="Test Set Performance by Dataset Type")
    plt.close()
    
    # --- Plot 1b: Comparison by Dataset Type (Train Set) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Train Set Performance by Dataset Type', fontsize=14, fontweight='bold')
    
    train_metrics_plot = ['Train_MRR', 'Train_H@1', 'Train_H@3', 'Train_H@10']
    
    for ax, metric, name in zip(axes.flatten(), train_metrics_plot, metric_names):
        pivot = df.pivot_table(values=metric, index='Dataset', columns='Tokenizer', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, color=colors[:len(pivot.columns)], edgecolor='black', linewidth=0.5)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(name)
        ax.legend(title='Tokenizer', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'train_by_dataset.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['train_by_dataset'] = wandb.Image(plot_path, caption="Train Set Performance by Dataset Type")
    plt.close()
    
    # --- Plot 2: Train vs Test Performance ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Train vs Test Performance Comparison', fontsize=14, fontweight='bold')
    
    train_metrics = ['Train_MRR', 'Train_H@1', 'Train_H@3', 'Train_H@10']
    test_metrics_list = ['Test_MRR', 'Test_H@1', 'Test_H@3', 'Test_H@10']
    
    # Define colors for 4 bar types
    train_color_normal = colors[0]
    test_color_normal = colors[1]
    train_color_inverse = colors[2]
    test_color_inverse = colors[3]
    
    # Get unique (Dataset, Tokenizer) combinations - consistent order for all subplots
    unique_configs = df[['Dataset', 'Tokenizer']].drop_duplicates()
    config_names = [f"{row['Dataset'][:3]}-{row['Tokenizer'][:5]}" for _, row in unique_configs.iterrows()]
    n_configs = len(unique_configs)
    x = np.arange(n_configs)
    width = 0.2  # 4 bars per group
    
    for ax, (train_m, test_m, name) in zip(axes.flatten(), zip(train_metrics, test_metrics_list, metric_names)):
        train_normal_vals = []
        test_normal_vals = []
        train_inverse_vals = []
        test_inverse_vals = []
        
        for _, config_row in unique_configs.iterrows():
            dataset, tokenizer = config_row['Dataset'], config_row['Tokenizer']
            
            # Get normal (Inverse=False) row
            normal_row = df[(df['Dataset'] == dataset) & (df['Tokenizer'] == tokenizer) & (df['Inverse'] == False)]
            if not normal_row.empty:
                train_normal_vals.append(normal_row[train_m].values[0])
                test_normal_vals.append(normal_row[test_m].values[0])
            else:
                train_normal_vals.append(0)
                test_normal_vals.append(0)
            
            # Get inverse (Inverse=True) row
            inverse_row = df[(df['Dataset'] == dataset) & (df['Tokenizer'] == tokenizer) & (df['Inverse'] == True)]
            if not inverse_row.empty:
                train_inverse_vals.append(inverse_row[train_m].values[0])
                test_inverse_vals.append(inverse_row[test_m].values[0])
            else:
                train_inverse_vals.append(0)
                test_inverse_vals.append(0)
        
        # Plot 4 bars per config: Train-Normal, Test-Normal, Train-Inverse, Test-Inverse
        ax.bar(x - 1.5*width, train_normal_vals, width, color=train_color_normal, edgecolor='black', 
               linewidth=0.5, label='Train')
        ax.bar(x - 0.5*width, test_normal_vals, width, color=test_color_normal, edgecolor='black', 
               linewidth=0.5, label='Test')
        ax.bar(x + 0.5*width, train_inverse_vals, width, color=train_color_inverse, edgecolor='black', 
               linewidth=0.5, hatch='///', label='Train (Inv)')
        ax.bar(x + 1.5*width, test_inverse_vals, width, color=test_color_inverse, edgecolor='black', 
               linewidth=0.5, hatch='///', label='Test (Inv)')
        
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel(name)
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=60, ha='right', fontsize=7)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'train_vs_test.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['train_vs_test'] = wandb.Image(plot_path, caption="Train vs Test Performance Comparison")
    plt.close()
    
    # --- Plot 3a: Effect of Inverse Relations (Test) ---
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
    plot_path = os.path.join(output_dir, 'test_inverse_effect.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['test_inverse_effect'] = wandb.Image(plot_path, caption="Effect of Inverse Relations on Test Performance")
    plt.close()
    
    # --- Plot 3b: Effect of Inverse Relations (Train) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Effect of Inverse Relations on Train Performance', fontsize=14, fontweight='bold')
    
    for ax, metric, name in zip(axes, ['Train_MRR', 'Train_H@1'], ['MRR', 'Hits@1']):
        pivot = df.pivot_table(values=metric, index='Dataset', columns='Inverse', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, color=[colors[2], colors[3]], edgecolor='black', linewidth=0.5)
        ax.set_title(f'{name} by Inverse Setting', fontweight='bold')
        ax.set_xlabel('Dataset Type')
        ax.set_ylabel(name)
        ax.legend(title='Inverse', labels=['False', 'True'])
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'train_inverse_effect.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['train_inverse_effect'] = wandb.Image(plot_path, caption="Effect of Inverse Relations on Train Performance")
    plt.close()
    
    # --- Plot 4a: Heatmap of Test MRR ---
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
    plot_path = os.path.join(output_dir, 'test_mrr_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['test_mrr_heatmap'] = wandb.Image(plot_path, caption="Test MRR Heatmap")
    plt.close()
    
    # --- Plot 4b: Heatmap of Train MRR ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot_table(values='Train_MRR', index=['Dataset', 'Inverse'], columns='Tokenizer', aggfunc='mean')
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
    
    ax.set_title('Train MRR Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='MRR')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'train_mrr_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['train_mrr_heatmap'] = wandb.Image(plot_path, caption="Train MRR Heatmap")
    plt.close()
    
    # --- Plot 5: Training Time vs Performance ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create consistent color mapping for datasets
    datasets = df['Dataset'].unique()
    dataset_colors = {ds: colors[i] for i, ds in enumerate(datasets)}
    
    # Marker shapes for tokenizers
    tokenizers = df['Tokenizer'].unique()
    tokenizer_markers = {tok: m for tok, m in zip(tokenizers, ['o', 's', '^', 'D', 'v', 'p', 'h', '*'])}
    
    # Plot each combination of dataset, tokenizer, and inverse
    for _, row in df.iterrows():
        dataset = row['Dataset']
        tokenizer = row['Tokenizer']
        inverse = row['Inverse']
        
        # Filled for normal, hollow for inverse
        edgecolor = dataset_colors[dataset]
        linewidth = 1.5 if inverse else 0.5
        
        if inverse:
            ax.scatter(row['Time (s)'], row['Test_MRR'], 
                       marker=tokenizer_markers[tokenizer],
                       facecolors='none',
                       edgecolors=[edgecolor],
                       s=120, alpha=0.8, linewidth=linewidth)
        else:
            ax.scatter(row['Time (s)'], row['Test_MRR'], 
                       marker=tokenizer_markers[tokenizer],
                       c=[dataset_colors[dataset]],
                       edgecolors=[edgecolor],
                       s=120, alpha=0.8, linewidth=linewidth)
    
    ax.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax.set_ylabel('Test MRR', fontweight='bold')
    ax.set_title('Training Time vs Test MRR', fontsize=14, fontweight='bold')
    
    # Legend for datasets (colors)
    from matplotlib.lines import Line2D
    dataset_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=dataset_colors[ds], 
                              markersize=10, label=ds, markeredgecolor='black', markeredgewidth=0.5) 
                       for ds in datasets]
    legend1 = ax.legend(handles=dataset_handles, title='Dataset', loc='upper left')
    ax.add_artist(legend1)
    
    # Legend for tokenizers (shapes)
    tokenizer_handles = [Line2D([0], [0], marker=tokenizer_markers[tok], color='w', 
                                markerfacecolor='gray', markersize=10, label=tok,
                                markeredgecolor='black', markeredgewidth=0.5) 
                         for tok in tokenizers]
    legend2 = ax.legend(handles=tokenizer_handles, title='Tokenizer', loc='lower left')
    ax.add_artist(legend2)
    
    # Legend for inverse (filled vs hollow)
    inverse_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Normal', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markersize=10, label='Inverse', markeredgecolor='black', markeredgewidth=1.5),
    ]
    legend3 = ax.legend(handles=inverse_handles, title='Inverse', loc='lower right')
    ax.add_artist(legend3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'time_vs_performance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        wandb_images['time_vs_performance'] = wandb.Image(plot_path, caption="Training Time vs Test MRR")
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")
    
    # Log all images to wandb at once
    if log_to_wandb and wandb_images:
        wandb.log({"plots": wandb_images})
        print("Plots logged to wandb")


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
    parser = argparse.ArgumentParser(description='Grid search experiments for ByteGen')
    parser.add_argument('--data_path', type=str, default='KGs/UMLS',
                        help='Path to the knowledge graph dataset (default: KGs/UMLS)')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Output directory for results, plots, and CSV (default: comparison_results)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs (default: 300)')
    parser.add_argument('--wandb_project', type=str, default='bytegen-grid-search',
                        help='Wandb project name (default: bytegen-grid-search)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/team name (default: None, uses default entity)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    # Model architecture arguments
    parser.add_argument('--n_layer', type=int, default=8,
                        help='Number of transformer layers (default: 8)')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--n_embd', type=int, default=512,
                        help='Embedding dimension (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')
    parser.add_argument('--eval_batch_size', type=int, default=8192*2,
                        help='Batch size for evaluation (default: 8192)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    dataset_types = ['RandomWalk', 'BFS', 'Isolated']
    inverse_settings = [True, False]
    experiments = [
        ('Byte', None),
        ('BPE', 260),
        ('BPE', 512),
        ('BPE', 1024),
    ]

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        # Check if API key is available (from env or .env file)
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        # Extract dataset name from path for run naming
        dataset_name = os.path.basename(args.data_path.rstrip('/'))
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"grid-search-{dataset_name}-{args.epochs}ep",
            config={
                "data_path": args.data_path,
                "epochs": args.epochs,
                "output_dir": args.output_dir,
                "dataset_types": dataset_types,
                "inverse_settings": inverse_settings,
                "tokenizers": [f"{t}-{v}" if v else t for t, v in experiments],
                # Model architecture
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "label_smoothing": args.label_smoothing,
                "eval_batch_size": args.eval_batch_size,
            }
        )
        print(f"Wandb initialized: project={args.wandb_project}, entity={args.wandb_entity or 'default'}")
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    print(f"Found {num_gpus} GPUs")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    epochs = args.epochs
    
    # Build all experiment configs - GPU ID is now managed by queue
    all_configs = []
    for i, (dataset_type, inverse, (tokenizer_type, vocab_size)) in enumerate(
        [(d, inv, exp) for d in dataset_types for inv in inverse_settings for exp in experiments]
    ):
        all_configs.append((tokenizer_type, vocab_size, dataset_type, inverse, epochs, 
                           args.data_path, args.output_dir,
                           args.n_layer, args.n_head, args.n_embd, args.dropout, 
                           args.batch_size, args.lr, args.label_smoothing, args.eval_batch_size))
    
    print(f"Running {len(all_configs)} experiments across {num_gpus} GPUs...")

    # Create manager for queue to handle GPU assignment
    manager = Manager()
    queue = manager.Queue()
    for i in range(num_gpus):
        queue.put(i)
    
    # Run in parallel with one process per GPU
    # maxtasksperchild=1 ensures fresh process for each task to release memory completely
    with Pool(processes=num_gpus, initializer=init_worker, initargs=(queue,), maxtasksperchild=1) as pool:
        results = pool.map(run_experiment, all_configs, chunksize=1)
            
    df = pd.DataFrame(results)
    
    # Log individual experiment results to wandb
    if use_wandb:
        # Log each experiment as a row in a wandb table
        experiment_table = wandb.Table(dataframe=df)
        wandb.log({"experiment_results": experiment_table})
        
        # Log summary metrics
        for idx, row in df.iterrows():
            config_name = f"{row['Dataset']}/{row['Tokenizer']}/inv={row['Inverse']}"
            wandb.log({
                f"experiments/{config_name}/test_mrr": row['Test_MRR'],
                f"experiments/{config_name}/test_h1": row['Test_H@1'],
                f"experiments/{config_name}/test_h3": row['Test_H@3'],
                f"experiments/{config_name}/test_h10": row['Test_H@10'],
                f"experiments/{config_name}/train_mrr": row['Train_MRR'],
                f"experiments/{config_name}/train_h1": row['Train_H@1'],
                f"experiments/{config_name}/train_h3": row['Train_H@3'],
                f"experiments/{config_name}/train_h10": row['Train_H@10'],
                f"experiments/{config_name}/time_s": row['Time (s)'],
                f"experiments/{config_name}/params": row['Params'],
                f"experiments/{config_name}/vocab_size": row['Vocab Size'],
                f"experiments/{config_name}/block_size": row['Block Size'],
            })
    
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
    
    # Create visualization plots (with optional wandb logging)
    create_plots(df, output_dir=args.output_dir, log_to_wandb=use_wandb)
    
    # Save to CSV in output directory
    csv_path = os.path.join(args.output_dir, "grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")
    
    # Log summary statistics to wandb
    if use_wandb:
        # Best test metrics
        wandb.run.summary["best_test_mrr"] = df['Test_MRR'].max()
        wandb.run.summary["best_test_h1"] = df['Test_H@1'].max()
        wandb.run.summary["best_test_h3"] = df['Test_H@3'].max()
        wandb.run.summary["best_test_h10"] = df['Test_H@10'].max()
        
        # Best train metrics
        wandb.run.summary["best_train_mrr"] = df['Train_MRR'].max()
        wandb.run.summary["best_train_h1"] = df['Train_H@1'].max()
        wandb.run.summary["best_train_h3"] = df['Train_H@3'].max()
        wandb.run.summary["best_train_h10"] = df['Train_H@10'].max()
        
        # Best configuration info
        best_test_idx = df['Test_MRR'].idxmax()
        best_test_row = df.loc[best_test_idx]
        wandb.run.summary["best_config_dataset"] = best_test_row['Dataset']
        wandb.run.summary["best_config_tokenizer"] = best_test_row['Tokenizer']
        wandb.run.summary["best_config_inverse"] = best_test_row['Inverse']
        
        # Upload CSV as artifact
        artifact = wandb.Artifact('grid_search_results', type='results')
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)
        
        # Finish wandb run
        wandb.finish()
        print("Wandb run finished and results logged.")


if __name__ == "__main__":
    set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()
