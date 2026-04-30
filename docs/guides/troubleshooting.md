# Troubleshooting Guide

This guide covers common issues and their solutions when using DICE Embeddings.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Data Loading Issues](#data-loading-issues)
- [Evaluation Issues](#evaluation-issues)
- [Multi-GPU Issues](#multi-gpu-issues)

---

## Installation Issues

### Problem: CUDA version mismatch

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall matching your CUDA version
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: Import errors after installation

**Error:**
```
ModuleNotFoundError: No module named 'dicee'
```

**Solution:**
```bash
# Ensure you're in the correct environment
conda activate dice  # or your environment name

# Reinstall in development mode
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Training Issues

### Problem: NaN loss during training

**Error:**
```
Epoch:10: loss_step=nan, loss_epoch=nan
```

**Causes & Solutions:**

1. **Learning rate too high**
   ```bash
   # Reduce learning rate
   dicee --dataset_dir KGs/UMLS --model Keci --lr 0.01  # instead of 0.1
   ```

2. **Numerical instability**
   ```bash
   # Use mixed precision training
   dicee --trainer PL --pl_trainer_kwargs '{"precision":"16-mixed"}'
   ```

3. **Bad initialization**
   ```bash
   # Try different random seed
   dicee --dataset_dir KGs/UMLS --model Keci --random_seed 42
   ```

4. **Gradient explosion**
   ```bash
   # Enable gradient clipping
   dicee --dataset_dir KGs/UMLS --model Keci --gradient_clip_value 1.0
   ```

### Problem: Loss not decreasing

**Symptoms:** Loss stays constant or decreases very slowly

**Solutions:**

1. **Check learning rate**
   ```bash
   # Try different learning rates
   dicee --dataset_dir KGs/UMLS --model Keci --lr 0.1  # default
   dicee --dataset_dir KGs/UMLS --model Keci --lr 0.01  # lower
   ```

2. **Check batch size**
   ```bash
   # Smaller batches for better gradients
   dicee --dataset_dir KGs/UMLS --model Keci --batch_size 256  # instead of 1024
   ```

3. **Try different scoring technique**
   ```bash
   # KvsAll instead of NegSample
   dicee --dataset_dir KGs/UMLS --model Keci --scoring_technique KvsAll
   ```

4. **Verify data is loaded correctly**
   ```python
   from dicee.executer import Execute
   from dicee.config import Namespace
   
   args = Namespace()
   args.dataset_dir = "KGs/UMLS"
   args.num_epochs = 0  # No training, just load data
   Execute(args).start()
   # Check if no errors occur
   ```

### Problem: Model overfitting

**Symptoms:** Train MRR → 1.0, but Val/Test MRR is low

**Solutions:**

1. **Add regularization**
   ```bash
   dicee --dataset_dir KGs/UMLS --model Keci --weight_decay 0.01
   ```

2. **Add dropout**
   ```bash
   dicee --dataset_dir KGs/UMLS --model Keci --input_dropout_rate 0.1 --hidden_dropout_rate 0.2
   ```

3. **Use weight averaging**
   ```bash
   # Adaptive SWA
   dicee --dataset_dir KGs/UMLS --model Keci --aswa
   ```

4. **Reduce model capacity**
   ```bash
   dicee --dataset_dir KGs/UMLS --model Keci --embedding_dim 64  # instead of 256
   ```

5. **Early stopping**
   ```bash
   dicee --dataset_dir KGs/UMLS --model Keci --eval_every_n_epochs 10 --n_epochs_eval_model val
   # Manually stop when validation performance plateaus
   ```

---

## Memory Issues

### Problem: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**Solutions (in order of effectiveness):**

1. **Reduce batch size**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --model Keci --batch_size 512  # instead of 1024
   ```

2. **Reduce embedding dimension**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --model Keci --embedding_dim 128  # instead of 256
   ```

3. **Use negative sampling instead of KvsAll**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --model Keci --scoring_technique NegSample --neg_ratio 10
   ```

4. **Enable gradient accumulation**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --model Keci --batch_size 256 --accumulate_grad_batches 4
   # Effective batch size = 256 * 4 = 1024
   ```

5. **Use mixed precision**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --trainer PL --pl_trainer_kwargs '{"precision":"16-mixed"}'
   ```

6. **Train on CPU**
   ```bash
   CUDA_VISIBLE_DEVICES="" dicee --dataset_dir KGs/YAGO3-10 --model Keci --trainer torchCPUTrainer
   ```

### Problem: RAM out of memory during data loading

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Use polars backend for large files**
   ```bash
   dicee --path_single_kg large_kg.nt --backend polars --separator " "
   ```

2. **Use memory-mapped arrays**
   ```python
   # Already done automatically with memory_map_train_set.npy
   # Ensure you have enough disk space
   ```

3. **Reduce number of workers**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --num_core 2  # instead of 10
   ```

---

## Performance Issues

### Problem: Training is very slow

**Solutions:**

1. **Use GPU instead of CPU**
   ```bash
   # Check CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Use PL trainer for automatic GPU utilization
   dicee --dataset_dir KGs/UMLS --trainer PL
   ```

2. **Optimize number of workers**
   ```bash
   # Try different values (usually 4-8 is optimal)
   dicee --dataset_dir KGs/YAGO3-10 --num_core 4
   ```

3. **Use NegSample for large KGs**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --scoring_technique NegSample --neg_ratio 10
   ```

4. **Enable mixed precision**
   ```bash
   dicee --trainer PL --pl_trainer_kwargs '{"precision":"16-mixed"}'
   ```

5. **Use compiled model (PyTorch 2.x)**
   ```python
   # In your training script
   import torch
   model = torch.compile(model)  # Currently not exposed in CLI
   ```

### Problem: Data loading is slow

**Solutions:**

1. **Increase workers**
   ```bash
   dicee --dataset_dir KGs/YAGO3-10 --num_core 8
   ```

2. **Use faster backend**
   ```bash
   # Use polars instead of pandas for large files
   dicee --path_single_kg data.nt --backend polars
   ```

3. **Use persistent workers**
   ```bash
   # Currently not exposed in CLI, but used internally when num_core > 0
   ```

---

## Data Loading Issues

### Problem: Cannot find dataset

**Error:**
```
FileNotFoundError: Dataset directory not found: KGs/UMLS
```

**Solution:**
```bash
# Download datasets
wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate
unzip KGs.zip

# Or specify absolute path
dicee --dataset_dir /absolute/path/to/KGs/UMLS
```

### Problem: Parsing errors in data files

**Error:**
```
ParserError: Error tokenizing data
```

**Solutions:**

1. **Check separator**
   ```bash
   # Default is whitespace
   dicee --dataset_dir KGs/UMLS --separator "\s+"
   
   # For tab-separated
   dicee --dataset_dir KGs/Custom --separator "\t"
   
   # For comma-separated
   dicee --dataset_dir KGs/Custom --separator ","
   ```

2. **Try different backend**
   ```bash
   # rdflib for RDF formats
   dicee --path_single_kg data.owl --backend rdflib
   
   # polars for large n-triples
   dicee --path_single_kg data.nt --backend polars --separator " "
   ```

3. **Validate data format**
   ```bash
   # Check first few lines
   head -5 KGs/UMLS/train.txt
   
   # Should be: subject relation object (whitespace separated)
   # Example:
   # entity1 relation1 entity2
   # entity2 relation2 entity3
   ```

### Problem: Blank nodes in RDF data

**Error:**
```
ValueError: Blank nodes are not supported
```

**Solution:**
```bash
# Remove blank nodes from your RDF file before loading
# Or use a triplestore and SPARQL endpoint instead
dicee --sparql_endpoint "http://localhost:3030/mydata/" --model Keci
```

---

## Evaluation Issues

### Problem: Zero MRR/Hits@k scores

**Possible Causes:**

1. **Model not trained**
   ```bash
   # Ensure num_epochs > 0
   dicee --dataset_dir KGs/UMLS --num_epochs 100  # not 0
   ```

2. **Wrong evaluation mode**
   ```bash
   # Enable evaluation
   dicee --dataset_dir KGs/UMLS --eval_model "train_val_test"  # not "None"
   ```

3. **Data mismatch**
   ```python
   # Verify entities in test set exist in training
   import pandas as pd
   train = pd.read_csv("KGs/UMLS/train.txt", sep="\s+", header=None)
   test = pd.read_csv("KGs/UMLS/test.txt", sep="\s+", header=None)
   
   train_entities = set(train[0]).union(set(train[2]))
   test_entities = set(test[0]).union(set(test[2]))
   
   unseen = test_entities - train_entities
   print(f"Unseen entities in test: {len(unseen)}")  # Should be 0 for transductive
   ```

### Problem: Inconsistent results across runs

**Cause:** Different random seeds

**Solution:**
```bash
# Fix random seed
dicee --dataset_dir KGs/UMLS --model Keci --random_seed 42

# Run multiple times with different seeds for statistical significance
dicee --dataset_dir KGs/UMLS --model Keci --random_seed 1
dicee --dataset_dir KGs/UMLS --model Keci --random_seed 2
dicee --dataset_dir KGs/UMLS --model Keci --random_seed 3
```

---

## Multi-GPU Issues

### Problem: Multi-GPU training not working

**Error:**
```
RuntimeError: Default process group has not been initialized
```

**Solutions:**

1. **Use PL trainer for automatic multi-GPU**
   ```bash
   # Automatically uses all visible GPUs
   dicee --dataset_dir KGs/UMLS --trainer PL
   ```

2. **For torchDDP, use torchrun**
   ```bash
   torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
     dicee --trainer torchDDP --dataset_dir KGs/UMLS --path_to_store_single_run output
   ```

3. **Specify visible GPUs**
   ```bash
   # Use only GPU 0 and 1
   CUDA_VISIBLE_DEVICES=0,1 dicee --trainer PL --dataset_dir KGs/UMLS
   ```

### Problem: Uneven GPU utilization

**Cause:** Model parallelism not balanced

**Solutions:**

1. **Increase batch size**
   ```bash
   dicee --trainer PL --dataset_dir KGs/UMLS --batch_size 2048
   ```

2. **Use DDP strategy explicitly**
   ```bash
   dicee --trainer PL --pl_trainer_kwargs '{"strategy":"ddp"}'
   ```

---

## Quick Diagnosis Commands

### Check Installation
```bash
python -c "import dicee; print(dicee.__version__)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Check GPU
```bash
nvidia-smi  # Check GPU availability and memory
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### Test Data Loading
```bash
# Load data without training
dicee --dataset_dir KGs/UMLS --num_epochs 0
```

### Check Saved Model
```python
from dicee import KGE
model = KGE(path="Experiments/2024-01-15...")
print(model)
print(f"Entities: {len(model.entity_to_idx)}")
print(f"Relations: {len(model.relation_to_idx)}")
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/dice-group/dice-embeddings/issues)
2. **Run tests**: `python -m pytest -p no:warnings -x` to verify installation
3. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **Report bugs**: Include minimal reproducible example, error message, and environment details

### Useful Information to Include in Bug Reports

```bash
# System info
python --version
pip list | grep -E "torch|dicee|lightning"

# GPU info
nvidia-smi

# Configuration
cat path/to/experiment/configuration.json
```

---

## Common Pitfalls

❌ **Don't:**
- Mix different PyTorch/CUDA versions
- Use `--eval_model None` and expect evaluation results
- Forget `--path_to_store_single_run` with torchDDP
- Use very high learning rates (> 1.0) without tuning
- Load huge KGs with pandas (use polars)

✅ **Do:**
- Start with small datasets for testing
- Use `--eval_every_n_epochs` for monitoring
- Save checkpoints regularly
- Use version control for configurations
- Profile before optimizing
