from pathlib import Path
import json
import random
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dicee.executer import run_dicee_eval

# Add the robust-kge directory to the path for config import
robust_kge_dir = Path(__file__).parent
sys.path.insert(0, str(robust_kge_dir))

# Get project root (parent of robust-kge directory)
project_root = robust_kge_dir.parent

# Add project root to Python path so dicee can be imported
sys.path.insert(0, str(project_root))



from config import (DBS,
                    MODELS,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    NUM_EPOCHS,
                    EMB_DIM,
                    LOSS_FN,
                    SCORING_TECH,
                    OPTIM,
                    EVAL_MODEL
                    )

def create_results_table(results_dict):
    """Create a pandas DataFrame table from results dictionary"""
    # Structure: results_dict[dataset][model] = mrr_value
    rows = []
    for dataset, models_dict in results_dict.items():
        for model, mrr in models_dict.items():
            rows.append({
                'Dataset': dataset,
                'Model': model,
                'Test_MRR': mrr
            })
    
    df = pd.DataFrame(rows)
    
    # Create pivot table: models as rows, datasets as columns
    pivot_df = df.pivot_table(
        index='Model',
        columns='Dataset',
        values='Test_MRR',
        aggfunc='first'
    )
    
    return df, pivot_df

def create_visualization(pivot_df, output_dir, loss_fn=None):
    """Create heatmap visualization of MRR results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (max(12, len(pivot_df.columns) * 1.5), max(8, len(pivot_df.index) * 0.8))
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(max(14, len(pivot_df.columns) * 1.8), max(10, len(pivot_df.index) * 1.0)))
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn', 
        cbar_kws={'label': 'Test MRR'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1.0
    )
    
    # Create title with loss function name
    loss_name = loss_fn if loss_fn else "Default"
    title = f'MRR Comparison: Models vs Datasets (Test Set) - Loss: {loss_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    loss_suffix = f"_{loss_fn}" if loss_fn else ""
    image_path = output_dir / f"mrr_comparison_heatmap{loss_suffix}_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap visualization saved to: {image_path}")
    
    plt.close()
    
    return image_path

def save_results_table(df, pivot_df, output_dir, loss_fn=None):
    """Save results to CSV files and create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Append loss function name to filenames if provided
    loss_suffix = f"_{loss_fn}" if loss_fn else ""
    
    # Save detailed results
    detailed_path = output_dir / f"detailed_results{loss_suffix}_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")
    
    # Save pivot table
    pivot_path = output_dir / f"comparison_table{loss_suffix}_{timestamp}.csv"
    pivot_df.to_csv(pivot_path)
    print(f"Comparison table saved to: {pivot_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("MRR Comparison Table (Test Set)")
    print("="*80)
    print(pivot_df.to_string())
    print("="*80)
    
    # Create heatmap visualization
    print("\nGenerating heatmap visualization...")
    create_visualization(pivot_df, output_dir, loss_fn)
    
    return detailed_path, pivot_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run KGE experiments')
    parser.add_argument('--loss_fn', type=str, default=None,
                        help='Loss function to use (e.g., BCELoss). If not provided, uses value from config.py')
    parser.add_argument('--lr', type=str, default=None,
    help = 'Learning rate to use (e.g., 0.1). If not provided, uses value from config.py')
    parser.add_argument('--batch_size', type=str, default=None,
    help = 'Batch size to use (e.g., 1024). If not provided, uses value from config.py')
    parser.add_argument('--num_epochs', type=str, default=None,
    help = 'Number of epochs to use (e.g., 100). If not provided, uses value from config.py')
    parser.add_argument('--emb_dim', type=str, default=None,
    help = 'Embedding dimension to use (e.g., 32). If not provided, uses value from config.py')
    parser.add_argument('--scoring_technique', type=str, default=None,
    help = 'Scoring technique to use (e.g., KvsAll). If not provided, uses value from config.py')
    parser.add_argument('--optim', type=str, default=None,
    help = 'Optimizer to use (e.g., Adam). If not provided, uses value from config.py')
    parser.add_argument('--eval_model', type=str, default=None,
    help = 'Evaluation model to use (e.g., train_val_test). If not provided, uses value from config.py')
    args = parser.parse_args()
    
    # Use command-line argument if provided, otherwise use config value
    loss_function = args.loss_fn if args.loss_fn else LOSS_FN
    learning_rate = args.lr if args.lr else LEARNING_RATE
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    num_epochs = args.num_epochs if args.num_epochs else NUM_EPOCHS
    embedding_dim = args.emb_dim if args.emb_dim else EMB_DIM
    scoring_technique = args.scoring_technique if args.scoring_technique else SCORING_TECH
    optim = args.optim if args.optim else OPTIM
    eval_model = args.eval_model if args.eval_model else EVAL_MODEL
    
    # Dictionary to store all results: {dataset: {model: mrr}}
    all_results = {}
    
    # Default to saved_models in robust-kge directory
    results_dir = robust_kge_dir / "saved_models"
    
    # Run new experiments
    for DB in DBS:
        # Get all subdirectories in UNITEKG/{DB} (relative to project root)
        db_path = project_root / "UNITEKG" / DB
        if db_path.exists():
            subdirs = sorted([d.name for d in db_path.iterdir() if d.is_dir()])
        else:
            print(f"Warning: {db_path} does not exist, skipping {DB}")
            continue
        
        for subdir in subdirs:
            dataset_name = f"{DB}/{subdir}"
            all_results[dataset_name] = {}
            
            for MODEL in MODELS:
                # if not HAS_DICEE:
                #     print(f"Error: Cannot run experiments - dicee module not found.")
                #     continue
                
                print(f"Running experiment: {MODEL} on {dataset_name}")
                try:
                    result = run_dicee_eval(
                        dataset_folder=str(project_root / "UNITEKG" / DB / subdir),
                        model=MODEL,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        embedding_dim=EMB_DIM,
                        loss_function=loss_function,
                        path_to_store_single_run=str(results_dir / DB / subdir / MODEL / ""),
                        scoring_technique=SCORING_TECH,
                        optim=OPTIM,
                        eval_model=EVAL_MODEL
                    )
                    test_mrr = result.get('Test', {}).get('MRR', None)
                    all_results[dataset_name][MODEL] = test_mrr
                    print(f"Completed: {MODEL} on {dataset_name} - Test MRR: {test_mrr}")
                except Exception as e:
                    print(f"Error running {MODEL} on {dataset_name}: {e}")
                    all_results[dataset_name][MODEL] = None
    
    # Create and save results table
    if all_results:
        df, pivot_df = create_results_table(all_results)
        save_results_table(df, pivot_df, project_root / "robust-kge" / "results", loss_function)
    else:
        print("No results to save.")