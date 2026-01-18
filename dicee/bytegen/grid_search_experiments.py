import os
import time
import argparse
import torch
import pandas as pd
import json
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
from dicee.bytegen.plot_results import create_plots, log_best_approaches
import gc

# Load environment variables from .env file
load_dotenv()

# Global queue for workers
gpu_queue = None

def init_worker(q):
    global gpu_queue
    gpu_queue = q


def get_experiment_key(dataset_type: str, tokenizer_type: str, vocab_size: int, inverse: bool) -> str:
    """Generate a unique key for an experiment configuration."""
    tok_label = "Byte" if tokenizer_type == 'Byte' else f"BPE-{vocab_size}"
    return f"{dataset_type}_{tok_label}_Inv{inverse}"


def load_completed_experiments(output_dir: str) -> set:
    """Load completed experiments from existing results CSV and saved models."""
    completed = set()
    
    # Method 1: Check CSV results
    csv_path = os.path.join(output_dir, "grid_search_results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # Check if the experiment has valid results (non-zero metrics)
                if row.get('Test_MRR', 0) > 0 or row.get('Train_MRR', 0) > 0:
                    key = get_experiment_key(
                        row['Dataset'], 
                        'Byte' if row['Tokenizer'] == 'ByteTokenizer' else 'BPE',
                        row.get('Vocab Size', 0),
                        row['Inverse']
                    )
                    completed.add(key)
            print(f"Found {len(completed)} completed experiments in CSV")
        except Exception as e:
            print(f"Warning: Could not load previous results from CSV: {e}")
    
    # Method 2: Check for saved model files (for runs before resume logic was added)
    models_dir = os.path.join(output_dir, "models")
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        for model_file in model_files:
            # Parse model filename: {dataset_type}_{tokenizer_name}_Inv{inverse}.pt
            # Examples: RandomWalk_ByteTokenizer_InvTrue.pt, BFS_BPE-512_InvFalse.pt
            base_name = model_file.replace('.pt', '')
            parts = base_name.rsplit('_Inv', 1)
            if len(parts) == 2:
                dataset_tokenizer = parts[0]
                inverse_str = parts[1]
                
                # Parse dataset and tokenizer
                for ds in ['RandomWalk', 'BFS', 'Isolated']:
                    if dataset_tokenizer.startswith(ds + '_'):
                        dataset_type = ds
                        tokenizer_name = dataset_tokenizer[len(ds) + 1:]
                        inverse = inverse_str == 'True'
                        
                        # Convert tokenizer name to key format
                        if tokenizer_name == 'ByteTokenizer':
                            key = get_experiment_key(dataset_type, 'Byte', None, inverse)
                        elif tokenizer_name.startswith('BPE-'):
                            vocab_size = int(tokenizer_name.split('-')[1])
                            key = get_experiment_key(dataset_type, 'BPE', vocab_size, inverse)
                        else:
                            continue
                        
                        if key not in completed:
                            completed.add(key)
                        break
        
        model_count = len([f for f in model_files])
        if model_count > 0:
            print(f"Found {model_count} saved model files in {models_dir}")
    
    if completed:
        print(f"Total: {len(completed)} completed experiments detected")
    
    return completed


def save_run_config(output_dir: str, config: dict):
    """Save run configuration including wandb run ID."""
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to {config_path}")


def load_run_config(output_dir: str) -> dict:
    """Load run configuration from previous run."""
    config_path = os.path.join(output_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def run_experiment(args):
    """Worker function for parallel execution."""
    (tokenizer_type, vocab_size_arg, dataset_type, inverse, epochs, dataset_path, output_dir,
     n_layer, n_head, n_embd, dropout, batch_size, lr, label_smoothing, eval_batch_size, 
     wandb_config, checkpoint_interval) = args
    
    # Get a free GPU ID from the queue
    gpu_id = gpu_queue.get()
    
    # Generate experiment key for this run
    experiment_key = get_experiment_key(dataset_type, tokenizer_type, vocab_size_arg, inverse)
    
    # Setup logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    tok_label = "Byte" if tokenizer_type == 'Byte' else f"BPE-{vocab_size_arg}"
    log_filename = f"gpu{gpu_id}_{dataset_type}_{tok_label}_Inv{inverse}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Setup checkpoint directory for this experiment
    checkpoint_dir = os.path.join(output_dir, "checkpoints", experiment_key)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
                job_type=wandb_config.get('job_type'),
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
            test_ds = IsolatedTripleDataset(dataset_path, tokenizer, split='test', block_size=None, inverse=inverse)
            
            # Compute eval-safe block_size (considers ALL entities as potential candidates)
            block_size = IsolatedTripleDataset.compute_required_block_size(train_ds, test_ds)
            
            # Update datasets to use the eval-safe block_size
            train_ds.block_size = block_size
            test_ds.block_size = block_size
            print(f"[GPU {gpu_id}] Isolated block_size set to {block_size} (eval-safe)")
        elif dataset_type == 'BFS':
            block_size = 256
            train_ds = ByteGenBFSDataset(dataset_path, tokenizer, split='train', block_size=block_size, inverse=inverse)
            test_ds = ByteGenBFSDataset(dataset_path, tokenizer, split='test', block_size=block_size, inverse=inverse)
        else:  # RandomWalk
            block_size = 256
            train_ds = ByteGenDataset(dataset_path, tokenizer, split='train', block_size=block_size, inverse=inverse)
            test_ds = ByteGenDataset(dataset_path, tokenizer, split='test', block_size=block_size, inverse=inverse)
        
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
            save_path=checkpoint_dir,
            label_smoothing=label_smoothing,
            warmup_epochs=5,
            train_dataset=train_ds,
            eval_batch_size=eval_batch_size
        )
        trainer.train(epochs, checkpoint_interval=checkpoint_interval)
        
        # Clear training memory before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
                
        # Evaluate on BOTH train and test sets
        evaluator = Evaluator(model, train_ds, test_ds, tokenizer)
        train_metrics = evaluator.evaluate(split='train', batch_size=eval_batch_size)
        test_metrics = evaluator.evaluate(split='test', batch_size=eval_batch_size)
        
        end_time = time.time()
        duration = end_time - start_time

        # Save model with config name
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_filename = f"{dataset_type}_{tokenizer_name}_Inv{inverse}.pt"
        model_path = os.path.join(models_dir, model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'block_size': block_size,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'dropout': dropout,
                'vocab_size': actual_vocab_size,
                'batch_size': batch_size,
                'lr': lr,
            },
            'tokenizer_type': tokenizer_type,
            'vocab_size_arg': vocab_size_arg,
            'dataset_type': dataset_type,
            'inverse': inverse,
            'epochs': epochs,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }, model_path)
        print(f"[GPU {gpu_id}] Model saved to {model_path}")

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


def main():
    parser = argparse.ArgumentParser(description='Grid search experiments for ByteGen')
    parser.add_argument('--data_path', type=str, default='KGs/UMLS',
                        help='Path to the knowledge graph dataset (default: KGs/UMLS)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: auto-generated from dataset and epochs)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs (default: 300)')
    parser.add_argument('--wandb_project', type=str, default='bytegen-grid-search',
                        help='Wandb project name (default: bytegen-grid-search)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/team name (default: None, uses default entity)')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='Wandb group name (default: auto-generated)')
    parser.add_argument('--wandb_job_type', type=str, default='train',
                        help='Wandb job type (default: train)')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous run by providing the output directory path')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50, set to 0 to disable)')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='Manually specify wandb run ID to resume (useful when resuming on a different machine)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt and run immediately')
    args = parser.parse_args()
    
    # Handle resume mode
    is_resume = args.resume is not None
    if is_resume:
        args.output_dir = args.resume
        if not os.path.exists(args.output_dir):
            raise FileNotFoundError(f"Resume directory not found: {args.output_dir}")
        print(f"=== RESUME MODE: Resuming from {args.output_dir} ===")
        
        # Load previous run config and restore all parameters
        prev_config = load_run_config(args.output_dir)
        if prev_config:
            # Restore all parameters from previous config (CLI args override if explicitly set)
            # We check against default values to detect if user explicitly set them
            defaults = {
                'data_path': 'KGs/UMLS',
                'epochs': 300,
                'n_layer': 8,
                'n_head': 8,
                'n_embd': 512,
                'dropout': 0.0,
                'batch_size': 32,
                'lr': 3e-4,
                'label_smoothing': 0.0,
                'eval_batch_size': 8192*2,
                'checkpoint_interval': 50,
                'wandb_project': 'bytegen-grid-search',
                'wandb_entity': None,
            }
            
            restored = []
            for key, default_val in defaults.items():
                if key in prev_config:
                    current_val = getattr(args, key, default_val)
                    # If current value is the default, use the saved config value
                    if current_val == default_val:
                        setattr(args, key, prev_config[key])
                        if prev_config[key] != default_val:
                            restored.append(f"{key}={prev_config[key]}")
            
            if restored:
                print(f"Restored from config: {', '.join(restored)}")
            print(f"Using: data_path={args.data_path}, epochs={args.epochs}, batch_size={args.batch_size}")
        else:
            print("Warning: No run_config.json found. Using CLI arguments or defaults.")
    else:
        # Auto-generate output directory if not provided
        if args.output_dir is None:
            dataset_name = os.path.basename(args.data_path.rstrip('/'))
            args.output_dir = f"results_{dataset_name}_{args.epochs}ep"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load completed experiments if resuming
    completed_experiments = set()
    existing_results = []
    if is_resume:
        completed_experiments = load_completed_experiments(args.output_dir)
        # Load existing results to merge later
        csv_path = os.path.join(args.output_dir, "grid_search_results.csv")
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            existing_results = existing_df.to_dict('records')
            print(f"Loaded {len(existing_results)} existing results")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    dataset_types = ['RandomWalk', 'BFS', 'Isolated']
    inverse_settings = [True]
    experiments = [
        ('Byte', None),
        ('BPE', 260),
        ('BPE', 512),
        ('BPE', 1024),
    ]

    # Initialize wandb
    use_wandb = not args.no_wandb
    wandb_config = None
    wandb_run_id = None
    
    if use_wandb:
        # Check if API key is available (from env or .env file)
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        # Extract dataset name from path for run naming
        dataset_name = os.path.basename(args.data_path.rstrip('/'))
        
        # Determine group name
        group_name = args.wandb_group if args.wandb_group else f"grid-search-{dataset_name}-{args.epochs}ep"
        
        # Check for existing wandb run ID when resuming
        prev_config = load_run_config(args.output_dir) if is_resume else {}
        # Priority: CLI arg > config file
        wandb_run_id = args.wandb_run_id or prev_config.get('wandb_run_id')
        
        if is_resume and wandb_run_id:
            print(f"Attempting to resume wandb run: {wandb_run_id}")
            try:
                wandb.init(
                    project=prev_config.get('wandb_project', args.wandb_project),
                    entity=prev_config.get('wandb_entity', args.wandb_entity),
                    id=wandb_run_id,
                    resume="must"
                )
                print(f"Successfully resumed wandb run: {wandb_run_id}")
            except Exception as e:
                print(f"Warning: Could not resume wandb run {wandb_run_id}: {e}")
                print("Starting a new wandb run instead...")
                wandb_run_id = None  # Reset to create new run
        
        if not wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=group_name,
                job_type="summary",
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
                    "wandb_group": group_name,
                    "wandb_job_type": args.wandb_job_type
                }
            )
            wandb_run_id = wandb.run.id
        
        wandb_config = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": group_name,
            "job_type": args.wandb_job_type,
            "api_key": wandb_api_key
        }
        print(f"Wandb initialized: project={args.wandb_project}, entity={args.wandb_entity or 'default'}, group={group_name}, run_id={wandb_run_id}")
    
    # Save run config (including wandb run ID for resume)
    run_config = {
        "data_path": args.data_path,
        "epochs": args.epochs,
        "output_dir": args.output_dir,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_id": wandb_run_id,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "eval_batch_size": args.eval_batch_size,
        "checkpoint_interval": args.checkpoint_interval,
    }
    save_run_config(args.output_dir, run_config)
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    print(f"Found {num_gpus} GPUs")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    epochs = args.epochs
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval > 0 else None
    
    # Build all experiment configs - GPU ID is now managed by queue
    all_configs = []
    pending_experiments = []
    completed_list = []
    
    for i, (dataset_type, inverse, (tokenizer_type, vocab_size)) in enumerate(
        [(d, inv, exp) for d in dataset_types for inv in inverse_settings for exp in experiments]
    ):
        # Check if experiment is already completed
        exp_key = get_experiment_key(dataset_type, tokenizer_type, vocab_size, inverse)
        if exp_key in completed_experiments:
            completed_list.append(exp_key)
            continue
        
        pending_experiments.append(exp_key)
        all_configs.append((tokenizer_type, vocab_size, dataset_type, inverse, epochs, 
                           args.data_path, args.output_dir,
                           args.n_layer, args.n_head, args.n_embd, args.dropout, 
                           args.batch_size, args.lr, args.label_smoothing, args.eval_batch_size, 
                           wandb_config, checkpoint_interval))
    
    # Display experiment plan
    print("\n" + "="*60)
    print("EXPERIMENT PLAN")
    print("="*60)
    
    if completed_list:
        print(f"\n✅ COMPLETED ({len(completed_list)} experiments - will be skipped):")
        for exp in completed_list:
            print(f"   • {exp}")
    
    if pending_experiments:
        print(f"\n🔄 PENDING ({len(pending_experiments)} experiments - will be run):")
        for exp in pending_experiments:
            print(f"   • {exp}")
    else:
        print("\n✨ All experiments already completed! Nothing to run.")
        if use_wandb:
            wandb.finish()
        return
    
    print(f"\n📊 Configuration:")
    print(f"   • Data path: {args.data_path}")
    print(f"   • Epochs: {epochs}")
    print(f"   • GPUs: {num_gpus}")
    print(f"   • Checkpoint interval: {checkpoint_interval or 'disabled'}")
    print(f"   • Output: {args.output_dir}")
    print("="*60)
    
    # Ask for confirmation unless --yes flag is provided
    if not args.yes:
        try:
            response = input(f"\nProceed with {len(pending_experiments)} experiments? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("Aborted by user.")
                if use_wandb:
                    wandb.finish()
                return
        except EOFError:
            # Non-interactive mode (e.g., running in background)
            print("Non-interactive mode detected. Use --yes to skip confirmation.")
            if use_wandb:
                wandb.finish()
            return
    
    print(f"\n🚀 Starting {len(all_configs)} experiments across {num_gpus} GPUs...")

    # Create manager for queue to handle GPU assignment
    manager = Manager()
    queue = manager.Queue()
    for i in range(num_gpus):
        queue.put(i)
    
    # Run in parallel with one process per GPU
    # maxtasksperchild=1 ensures fresh process for each task to release memory completely
    with Pool(processes=num_gpus, initializer=init_worker, initargs=(queue,), maxtasksperchild=1) as pool:
        new_results = pool.map(run_experiment, all_configs, chunksize=1)
    
    # Merge new results with existing results
    if existing_results:
        # Create a set of new result keys for deduplication
        new_keys = set()
        for r in new_results:
            key = get_experiment_key(
                r['Dataset'], 
                'Byte' if r['Tokenizer'] == 'ByteTokenizer' else 'BPE',
                r.get('Vocab Size', 0),
                r['Inverse']
            )
            new_keys.add(key)
        
        # Keep existing results that weren't re-run
        merged_results = []
        for r in existing_results:
            key = get_experiment_key(
                r['Dataset'], 
                'Byte' if r['Tokenizer'] == 'ByteTokenizer' else 'BPE',
                r.get('Vocab Size', 0),
                r['Inverse']
            )
            if key not in new_keys:
                merged_results.append(r)
        
        # Add new results
        merged_results.extend(new_results)
        df = pd.DataFrame(merged_results)
        print(f"Merged {len(existing_results)} existing + {len(new_results)} new = {len(merged_results)} total results")
    else:
        df = pd.DataFrame(new_results)
    
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
