#!/usr/bin/env python3
"""
Standalone script to generate plots from grid search results (CSV or wandb JSON).

Usage:
    python -m dicee.bytegen.plot_results results_UMLS_300ep/grid_search_results.csv
    python -m dicee.bytegen.plot_results wandb_table.json --output_dir my_plots --dpi 300
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_results(file_path: str) -> pd.DataFrame:
    """Load results from CSV or wandb JSON table format.
    
    Args:
        file_path: Path to CSV or JSON file
        
    Returns:
        DataFrame with experiment results
    """
    if file_path.endswith('.json'):
        # Load wandb table JSON format
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # wandb table format: {"columns": [...], "data": [[...], ...]}
        if 'columns' in data and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data['columns'])
        else:
            # Try to load as regular JSON (list of dicts)
            df = pd.DataFrame(data)
    else:
        # Load as CSV
        df = pd.read_csv(file_path)
    
    return df


def create_plots(df: pd.DataFrame, output_dir: str = "comparison_results", 
                 log_to_wandb: bool = False, dpi: int = 300):
    """Generate visualization plots for the grid search results.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
        log_to_wandb: Whether to log plots to wandb
        dpi: Resolution of saved plots (default: 300)
    """
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
        
        # Plot bars based on available data
        has_normal = any(v > 0 for v in train_normal_vals)
        
        if not has_normal:
             # Assume only Inverse data is relevant/present - center and remove special styling
             ax.bar(x - 0.5*width, train_inverse_vals, width, color=train_color_inverse, edgecolor='black', 
                   linewidth=0.5, label='Train')
             ax.bar(x + 0.5*width, test_inverse_vals, width, color=test_color_inverse, edgecolor='black', 
                   linewidth=0.5, label='Test')
        else:
             # Plot 4 bars per config: Train-Normal, Test-Normal, Train-Inverse, Test-Inverse
             ax.bar(x - 1.5*width, train_normal_vals, width, color=train_color_normal, edgecolor='black', 
                    linewidth=0.5, label='Train')
             ax.bar(x - 0.5*width, test_normal_vals, width, color=test_color_normal, edgecolor='black', 
                    linewidth=0.5, label='Test')
             ax.bar(x + 0.5*width, train_inverse_vals, width, color=train_color_inverse, edgecolor='black', 
                    linewidth=0.5, label='Train (Inv)')
             ax.bar(x + 1.5*width, test_inverse_vals, width, color=test_color_inverse, edgecolor='black', 
                    linewidth=0.5, label='Test (Inv)')
        
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel(name)
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=60, ha='right', fontsize=7)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'train_vs_test.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
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
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    if log_to_wandb and WANDB_AVAILABLE:
        wandb_images['time_vs_performance'] = wandb.Image(plot_path, caption="Training Time vs Test MRR")
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")
    
    # Log all images to wandb at once
    if log_to_wandb and WANDB_AVAILABLE and wandb_images:
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
    df = df.copy()  # Avoid modifying original
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
    parser = argparse.ArgumentParser(
        description='Generate plots from grid search results (CSV or wandb JSON)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m dicee.bytegen.plot_results results_UMLS_300ep/grid_search_results.csv
    python -m dicee.bytegen.plot_results wandb_table.json --output_dir my_plots
    python -m dicee.bytegen.plot_results results.csv --dpi 600 --show_summary
        """
    )
    parser.add_argument('results_path', type=str,
                        help='Path to the grid search results file (CSV or wandb JSON)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same directory as input file)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Resolution of saved plots (default: 300)')
    parser.add_argument('--show_summary', action='store_true',
                        help='Print summary of best performing approaches')
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.results_path):
        raise FileNotFoundError(f"Results file not found: {args.results_path}")
    
    # Load results (CSV or JSON)
    print(f"Loading results from: {args.results_path}")
    df = load_results(args.results_path)
    print(f"Loaded {len(df)} experiment results")
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_path) or "."
    
    # Generate plots
    create_plots(df, output_dir=args.output_dir, dpi=args.dpi)
    
    # Optionally show summary
    if args.show_summary:
        log_best_approaches(df)
    
    print(f"\nDone! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
