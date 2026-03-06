from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dicee.executer import run_dicee_eval

robust_kge_dir = Path(__file__).parent
sys.path.insert(0, str(robust_kge_dir))

project_root = robust_kge_dir.parent

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
                    EVAL_MODEL_TEST
                    )

def create_results_table(results_dict):
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
    
    pivot_df = df.pivot_table(
        index='Model',
        columns='Dataset',
        values='Test_MRR',
        aggfunc='first'
    )
    
    return df, pivot_df

def create_visualization(pivot_df, output_dir, loss_fn=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (max(12, len(pivot_df.columns) * 1.5), max(8, len(pivot_df.index) * 0.8))
    
    fig, ax = plt.subplots(figsize=(max(14, len(pivot_df.columns) * 1.8), max(10, len(pivot_df.index) * 1.0)))
    
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
    
    loss_name = loss_fn if loss_fn else "Default"
    title = f'MRR Comparison: Models vs Datasets (Test Set) - Loss: {loss_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    loss_suffix = f"_{loss_fn}" if loss_fn else ""
    image_path = output_dir / f"mrr_comparison_heatmap{loss_suffix}_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap visualization saved to: {image_path}")
    
    plt.close()
    
    return image_path

def save_results_table(df, pivot_df, output_dir, loss_fn=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    loss_suffix = f"_{loss_fn}" if loss_fn else ""
    
    detailed_path = output_dir / f"detailed_results{loss_suffix}_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")
    
    pivot_path = output_dir / f"comparison_table{loss_suffix}_{timestamp}.csv"
    pivot_df.to_csv(pivot_path)
    print(f"Comparison table saved to: {pivot_path}")
    
    print("\n" + "="*80)
    print("MRR Comparison Table (Test Set)")
    print("="*80)
    print(pivot_df.to_string())
    print("="*80)
    
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
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--random_seed", type=str, default=None)
    parser.add_argument("--neg_ratio", type=str, default=None)
    parser.add_argument("--num_of_output_channels", type=str, default=None)
    parser.add_argument("--block_size", type=str, default=None)
    args = parser.parse_args()
    
    loss_function = args.loss_fn if args.loss_fn else LOSS_FN
    learning_rate = args.lr if args.lr else LEARNING_RATE
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    num_epochs = args.num_epochs if args.num_epochs else NUM_EPOCHS
    embedding_dim = args.emb_dim if args.emb_dim else EMB_DIM
    scoring_technique = args.scoring_technique if args.scoring_technique else SCORING_TECH
    optim = args.optim if args.optim else OPTIM
    eval_model = args.eval_model if args.eval_model else EVAL_MODEL_TEST
    trainer = args.trainer if args.trainer else None
    accelerator = args.accelerator if args.accelerator else None
    devices = args.devices if args.devices else None
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    precision = args.precision if args.precision else None
    random_seed = args.random_seed if args.random_seed else None
    neg_ratio = args.neg_ratio if args.neg_ratio else None
    num_of_output_channels = args.num_of_output_channels if args.num_of_output_channels else None
    block_size = args.block_size if args.block_size else None

    all_results = {}
    
    results_dir = robust_kge_dir / "saved_models"
    
    for DB in DBS:
        
        db_path = project_root / "Datasets_Perturbed" / DB
        if db_path.exists():
            subdirs = sorted([d.name for d in db_path.iterdir() if d.is_dir()])
        else:
            print(f"Warning: {db_path} does not exist, skipping {DB}")
            continue
        
        for subdir in subdirs:
            dataset_name = f"{DB}/{subdir}"
            all_results[dataset_name] = {}
            
            for MODEL in MODELS:
                
                print(f"Running experiment: {MODEL} on {dataset_name}")
                try:
                    result = run_dicee_eval(
                        dataset_folder=str(project_root / "Datasets_Perturbed" / DB / subdir),
                        model=MODEL,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        embedding_dim=embedding_dim,
                        loss_function=loss_function,
                        path_to_store_single_run=str(results_dir / DB / subdir / MODEL / ""),
                        scoring_technique=scoring_technique,
                        optim=optim,
                        eval_model=eval_model,
                        trainer=trainer,
                        accelerator=accelerator,
                        devices=devices,
                        precision=precision,
                        random_seed=random_seed,
                        neg_ratio=neg_ratio,
                        num_of_output_channels=num_of_output_channels,
                        block_size=block_size
                    )
                    test_mrr = result.get('Test', {}).get('MRR', None)
                    all_results[dataset_name][MODEL] = test_mrr
                    print(f"Completed: {MODEL} on {dataset_name} - Test MRR: {test_mrr}")
                except Exception as e:
                    print(f"Error running {MODEL} on {dataset_name}: {e}")
                    all_results[dataset_name][MODEL] = None
    
    if all_results:
        df, pivot_df = create_results_table(all_results)
        save_results_table(df, pivot_df, project_root / "robust-kge" / "results", loss_function)
    else:
        print("No results to save.")
