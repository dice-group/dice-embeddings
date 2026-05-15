# PFN Module Layout

This repository now includes a lightweight `pfn` package to group the GraphPFN components while keeping the existing top-level scripts available.

## Why this structure

Grouping the PFN files under a package makes it easier to:

- import PFN functionality from code (`import pfn`)
- expose a single module entrypoint (`python -m pfn ...`)
- keep script-based workflows unchanged (`python pfn_train.py ...`)
- evolve internals without constantly changing user-facing commands

## Canonical scripts

These are the primary entrypoints:

- `pfn_train.py` for training
- `pfn_inference.py` for inference and scoring
- `pfn_evaluate.py` for ranking and BCE evaluation

## Package wrappers

The `pfn/` directory provides wrappers and a package CLI:

- `pfn/__init__.py`: convenience exports
- `pfn/__main__.py`: unified CLI (`python -m pfn ...`)
- `pfn/train.py`: wraps `pfn_train.py`
- `pfn/inference.py`: wraps `pfn_inference.py`
- `pfn/evaluate.py`: wraps `pfn_evaluate.py`
- `pfn/model.py`: wraps `pfn_model.py`
- `pfn/dataset.py`: wraps `pfn_dataset.py`

## Usage

### Option 1: Keep script commands

```bash
python pfn_train.py --kg-dir KGs/Countries-S1/ --epochs 1000 --save model.pt
python pfn_inference.py infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5
python pfn_evaluate.py rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
```

### Option 2: Use unified module CLI

```bash
python -m pfn train --kg-dir KGs/Countries-S1/ --epochs 1000 --save model.pt
python -m pfn infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5
python -m pfn score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria
python -m pfn eval rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
python -m pfn eval bce --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
```

### Option 3: Use as Python API

```python
from pfn import TriplePFN, train, infer, evaluate, evaluate_bce

model = train(num_epochs=10, kg_dir="KGs/Countries-S1/", save_path="model.pt")
```

## DDP notes

For multi-GPU training, use `torchrun` with `pfn_train.py` (or `python -m pfn train`).
Make sure `--nproc_per_node` does not exceed the number of visible GPUs.

```bash
torchrun --standalone --nproc_per_node=2 pfn_train.py --ddp --kg-dir KGs --epochs 10 --save model.pt
```

## Recommended next cleanup (optional)

If desired, a later refactor can move the core implementations into `pfn/` directly (not just wrappers), and keep top-level scripts as thin compatibility launchers.
