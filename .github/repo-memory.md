# DICE Embeddings Repository Memory

## Environment Setup

**Conda Installation Path:** `/home/cdemir/anaconda3/condabin/conda`

**Complete Installation:**
```bash
git clone https://github.com/dice-group/dice-embeddings.git
cd dice-embeddings && conda create -n dice python=3.11.14 --no-default-packages && conda activate dice && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
# or for development with all dependencies
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
```

**Conda Environment:** `dice` (Python 3.11.14)

## Pre-Commit Checks

Always run before committing:
```bash
ruff check dicee/ --line-length=200
```
