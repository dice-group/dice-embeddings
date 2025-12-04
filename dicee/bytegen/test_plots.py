"""
Test script for plot functions in grid_search_experiments.py
Generates dummy data and tests create_plots() and log_best_approaches()
"""

import os
import pandas as pd
import numpy as np
import tempfile
import shutil

# Import the functions to test
from grid_search_experiments import create_plots, log_best_approaches


def generate_dummy_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate dummy experiment results DataFrame matching the expected structure.
    """
    np.random.seed(seed)
    
    dataset_types = ['RandomWalk', 'BFS', 'Isolated']
    inverse_settings = [True, False]
    tokenizers = [
        ('ByteTokenizer', 259),
        ('BPE-260', 260),
        ('BPE-512', 512),
        ('BPE-733', 733),
    ]
    
    results = []
    
    for dataset_type in dataset_types:
        for inverse in inverse_settings:
            for tokenizer_name, vocab_size in tokenizers:
                # Generate realistic-looking metrics
                # Train metrics are generally higher than test metrics
                base_train = np.random.uniform(0.4, 0.9)
                base_test = base_train * np.random.uniform(0.7, 0.95)  # Test is typically lower
                
                # Inverse typically improves performance slightly
                if inverse:
                    base_train += np.random.uniform(0.02, 0.08)
                    base_test += np.random.uniform(0.01, 0.06)
                
                # Clamp to [0, 1]
                base_train = min(base_train, 1.0)
                base_test = min(base_test, 1.0)
                
                # H@k metrics increase with k
                train_mrr = base_train
                train_h1 = base_train * np.random.uniform(0.7, 0.9)
                train_h3 = base_train * np.random.uniform(0.85, 0.95)
                train_h10 = base_train * np.random.uniform(0.92, 1.0)
                
                test_mrr = base_test
                test_h1 = base_test * np.random.uniform(0.7, 0.9)
                test_h3 = base_test * np.random.uniform(0.85, 0.95)
                test_h10 = base_test * np.random.uniform(0.92, 1.0)
                
                # Block size depends on dataset type
                if dataset_type == 'Isolated':
                    block_size = np.random.randint(100, 200)
                else:
                    block_size = 256
                
                # Model parameters vary slightly with vocab size
                base_params = 5_000_000
                params = base_params + vocab_size * 512  # Embedding layer contribution
                
                # Training time varies
                time_s = np.random.uniform(100, 600)
                
                results.append({
                    "Dataset": dataset_type,
                    "Inverse": inverse,
                    "Tokenizer": tokenizer_name,
                    "Vocab Size": vocab_size,
                    "Block Size": block_size,
                    "Params": params,
                    "Epochs": 300,
                    "Time (s)": round(time_s, 2),
                    "Train_MRR": round(train_mrr, 4),
                    "Train_H@1": round(train_h1, 4),
                    "Train_H@3": round(train_h3, 4),
                    "Train_H@10": round(train_h10, 4),
                    "Test_MRR": round(test_mrr, 4),
                    "Test_H@1": round(test_h1, 4),
                    "Test_H@3": round(test_h3, 4),
                    "Test_H@10": round(test_h10, 4),
                })
    
    return pd.DataFrame(results)


def test_create_plots(df: pd.DataFrame, output_dir: str):
    """Test the create_plots function."""
    print("\n" + "="*60)
    print("Testing create_plots()")
    print("="*60)
    
    # Test without wandb logging
    create_plots(df, output_dir=output_dir, log_to_wandb=False)
    
    # Verify all expected plots were created
    expected_plots = [
        'test_by_dataset.png',
        'train_by_dataset.png',
        'train_vs_test.png',
        'test_inverse_effect.png',
        'train_inverse_effect.png',
        'test_mrr_heatmap.png',
        'train_mrr_heatmap.png',
        'time_vs_performance.png',
    ]
    
    print(f"\nChecking for generated plots in {output_dir}:")
    all_found = True
    for plot_name in expected_plots:
        plot_path = os.path.join(output_dir, plot_name)
        exists = os.path.exists(plot_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {plot_name}")
        if not exists:
            all_found = False
    
    if all_found:
        print("\n✓ All plots generated successfully!")
    else:
        print("\n✗ Some plots are missing!")
    
    return all_found


def test_log_best_approaches(df: pd.DataFrame):
    """Test the log_best_approaches function."""
    print("\n" + "="*60)
    print("Testing log_best_approaches()")
    print("="*60)
    
    # This function just prints, so we call it and check it doesn't raise
    try:
        log_best_approaches(df)
        print("\n✓ log_best_approaches() completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ log_best_approaches() failed with error: {e}")
        return False


def main():
    print("="*60)
    print("Grid Search Plot Functions Test Suite")
    print("="*60)
    
    # Generate dummy data
    print("\nGenerating dummy experiment data...")
    df = generate_dummy_data(seed=42)
    print(f"Generated {len(df)} experiment results")
    print(f"\nDataFrame columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(4).to_string(index=False))
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_plots_")
    print(f"\nUsing temporary output directory: {output_dir}")
    
    try:
        # Run tests
        plots_ok = test_create_plots(df, output_dir)
        log_ok = test_log_best_approaches(df)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  create_plots():       {'PASSED' if plots_ok else 'FAILED'}")
        print(f"  log_best_approaches(): {'PASSED' if log_ok else 'FAILED'}")
        
        if plots_ok and log_ok:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
        
        # Keep plots for inspection
        print(f"\nPlots saved to: {output_dir}")
        print("(Temporary directory will be preserved for inspection)")
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on failure
        shutil.rmtree(output_dir, ignore_errors=True)
        return 1
    
    return 0 if (plots_ok and log_ok) else 1


if __name__ == "__main__":
    exit(main())

