# PFN Module

This repository includes a `pfn` package that provides all GraphPFN functionality through a unified module interface.

## Structure

All PFN functionality is now contained within the `pfn/` package:

- `pfn/__init__.py`: convenience exports for Python API
- `pfn/__main__.py`: unified CLI (`python -m pfn ...`)
- `pfn/train.py`: training implementation
- `pfn/inference.py`: inference and scoring implementation
- `pfn/evaluate.py`: ranking and BCE evaluation implementation
- `pfn/model.py`: TriplePFN model architecture
- `pfn/dataset.py`: dataset loading and episode generation

## Usage

### Option 1: Use unified module CLI

```bash
python -m pfn train --kg-dir KGs/Countries-S1/ --epochs 1000 --save model.pt
python -m pfn infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5
python -m pfn score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria
python -m pfn eval rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
python -m pfn eval bce --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
```

### Option 2: Use as Python API

```python
from pfn import TriplePFN, train, infer, evaluate, evaluate_bce

model = train(num_epochs=10, kg_dir="KGs/Countries-S1/", save_path="model.pt")
```

## DDP notes

For multi-GPU training, use `torchrun` with `python -m pfn train`.
Make sure `--nproc_per_node` does not exceed the number of visible GPUs.

```bash
torchrun --standalone --nproc_per_node=2 -m pfn train --ddp --kg-dir KGs --epochs 10 --save model.pt
```
