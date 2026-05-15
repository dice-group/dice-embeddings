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

---

## Architecture Design

### Frozen SentenceTransformer Embeddings

GraphPFN uses **frozen** `all-MiniLM-L6-v2` embeddings for entities and relations. The SentenceTransformer is **not** updated during training—only the lightweight projection layers (`entity_proj`, `relation_proj`) are learned.

#### Why Keep It Frozen?

**Advantages:**
- ✅ **Universal semantic space**: Pretrained embeddings provide consistent representations across different KG domains (medical, geographic, social, etc.)
- ✅ **Zero-shot generalization**: Can score triples with completely novel entities at test time (just encode the string)
- ✅ **Meta-learning alignment**: Frozen features enable the transformer to learn **relational reasoning** rather than memorizing entity-specific patterns
- ✅ **Cross-domain transfer**: Same embedding space allows knowledge transfer from medical KGs to geographic KGs during episodic training
- ✅ **Prevents catastrophic forgetting**: Retains pretrained semantic knowledge (e.g., "czechoslovakia" → country) that helps with unseen entities
- ✅ **Smaller model**: Only ~10-20M trainable parameters vs ~42M if fine-tuning the encoder
- ✅ **Faster training**: ~3× faster and lower GPU memory usage

#### What If We Fine-Tuned?

**Potential benefits:**
- ⚠️ Task-specific embeddings optimized for triple scoring
- ⚠️ End-to-end optimization of the embedding space

**Significant drawbacks:**
- ❌ **3× parameter increase**: ~22M more parameters from MiniLM
- ❌ **3× slower training**: Backprop through the entire encoder
- ❌ **Higher GPU memory**: Requires gradient checkpointing
- ❌ **Overfitting risk**: Limited unique entities per episode
- ❌ **Loss of zero-shot capability**: Bias toward seen entities
- ❌ **Breaks meta-learning paradigm**: Couples embeddings to specific KG domains
- ❌ **Catastrophic forgetting**: Loses pretrained semantic knowledge

#### When Would Fine-Tuning Make Sense?

- **Single-domain deployment**: Only targeting one KG domain (e.g., only biomedical) without cross-domain transfer
- **Large-scale data**: Millions of triples from a single domain to prevent overfitting
- **Entity IDs**: Using discrete entity IDs instead of natural language strings

#### Current Design Rationale

The frozen SentenceTransformer is **critical for meta-learning**:

```python
# Episode 1: Medical KG
("aspirin", "treats", "headache")  # SentenceTransformer provides semantic grounding

# Episode 2: Geographic KG  
("slovakia", "neighbor", "poland")  # Same semantic space!

# → Transformer learns RELATIONAL REASONING across domains
# → Embeddings stay consistent = better generalization
```

This is analogous to how GPT-3 in-context learning works: **frozen token embeddings** + **meta-learned reasoning**.

**Recommendation**: Keep it frozen. The current bottleneck is the **transformer's reasoning capability**, not the embeddings. If you need more task-specific representations, consider:
- Using a KG-pretrained encoder (but keep it frozen during meta-training)
- Adding **adapter layers** - small trainable modules inserted into the frozen encoder (best of both worlds)
- Experimenting with different frozen encoders (e.g., `sentence-transformers/all-mpnet-base-v2`)
