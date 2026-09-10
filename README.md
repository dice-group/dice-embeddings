<div align="center">

![dicee_logo](docs/_static/images/dicee_logo.png)

# DICE Embeddings

**Hardware-agnostic Framework for Large-scale Knowledge Graph Embeddings**

[![Downloads](https://static.pepy.tech/badge/dicee)](https://pepy.tech/project/dicee)
[![Downloads](https://img.shields.io/pypi/dm/dicee)](https://pypi.org/project/dicee/)
[![Coverage](https://img.shields.io/badge/coverage-54%25-green)](https://dice-group.github.io/dice-embeddings/usage/main.html#coverage-report)
[![Pypi](https://img.shields.io/badge/pypi-0.3.2-blue)](https://pypi.org/project/dicee/0.3.2/)
[![Docs](https://img.shields.io/badge/documentation-0.3.2-yellow)](https://dice-group.github.io/dice-embeddings/index.html)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dice-group/dice-embeddings)

</div>

Knowledge graph embedding research has mainly focused on learning continuous representations of knowledge graphs towards the link prediction problem. Recently developed frameworks can be effectively applied in a wide range of research-related applications, yet using them in real-world settings becomes more challenging as the knowledge graph grows.

**dicee** computes embeddings for knowledge graphs of any size — from a few hundred triples to knowledge graphs with hundreds of millions of entities — running on a single CPU or scaled across GPUs and nodes, without changing a line of model code.

- 🧩 **30+ models** — real, complex, quaternion, octonion, and Clifford-algebra scoring functions, plus any [PyKEEN](https://github.com/pykeen/pykeen) model
- ⚙️ **Any hardware** — CPU, single/multi-GPU, native DDP, FSDP, and Tensor Parallelism
- 📈 **Scales up** — row-wise sharded entity tables and distributed optimizer states for very large knowledge graphs
- 🔌 **Two entry points** — the `dicee` CLI for quick runs, and a Python API (`Execute`, `KGE`) for programmatic control

## Quick Reference

| Task | Command |
|------|---------|
| 📦 **Install (CPU)** | `pip install dicee --extra-index-url https://download.pytorch.org/whl/cpu` |
| 📦 **Install (GPU)** | `pip install dicee` |
| 🚀 **Train a model** | `dicee --dataset_dir "KGs/UMLS" --model Keci` |
| 📂 **Load a pretrained model** | `from dicee import KGE; model = KGE(path='...')` |
| 🔮 **Predict links** | `model.predict_topk(h=["entity"], r=["relation"], topk=10)` |

📖 For more, visit the [dicee documentation](https://dice-group.github.io/dice-embeddings/)!

## Installation
<details><summary> Click me! </summary>

### Installation from PyPI

**CPU-only installation (recommended for most users):**
```bash
pip install dicee --extra-index-url https://download.pytorch.org/whl/cpu
```

**GPU/CUDA installation (for NVIDIA GPU users):**
```bash
pip install dicee
```

> **Note:** Installing without `--extra-index-url https://download.pytorch.org/whl/cpu` will include ~2GB of NVIDIA CUDA dependencies. For CPU-only usage, always include this flag.

### Installation from Source
``` bash
git clone https://github.com/dice-group/dice-embeddings.git
cd dice-embeddings && conda create -n dice python=3.11.14 --no-default-packages && conda activate dice && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
# or for development with all dependencies
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
```

## Download Knowledge Graphs
```bash
wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate && unzip KGs.zip
```
To test the Installation
```bash
python -m pytest -p no:warnings -x # Runs >119 tests leading to > 15 mins
python -m pytest -p no:warnings --lf # run only the last failed test
python -m pytest -p no:warnings --ff # to run the failures first and then the rest of the tests.
```

</details>

## Knowledge Graph Embedding Models
<details> <summary> To see available Models</summary>

* ```--model Decal | Keci | DualE | ComplEx | QMult | OMult | ConvQ | ConvO | ConEx | TransE | TransH | DistMult | Shallom | MuRE | RotatE```
* ```--model Pykeen_QuatE | Pykeen_Mure ``` all embedding models available in https://github.com/pykeen/pykeen#models can be selected. **📖 [PyKEEN integration →](docs/guides/pykeen_integration.md)** | **📖 [Examples →](tests/test_pykeen.py)**

Training and scoring techniques
* ```--trainer torchCPUTrainer | PL | MP | torchDDP | torchFSDP ```
* ```--scoring_technique 1vsAll | KvsAll  | AllvsAll | KvsSample | NegSample | FixedNegSample | FSDP1vsSample```

</details>

## How to Train
<details> <summary> To see a code snippet </summary>

#### Training Techniques

A KGE model can be trained with a state-of-the-art training technique ```--trainer "torchCPUTrainer" | "PL" | "MP" | "torchDDP" | "torchFSDP" ```
```bash
# CPU training
dicee --dataset_dir "KGs/UMLS" --trainer "torchCPUTrainer" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test"
# Distributed Data Parallelism
dicee --dataset_dir "KGs/UMLS" --trainer "PL" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test"
# Tensor Parallelism (implements Multiple Run Ensemble Learning with Low-Dimensional Knowledge Graph Embeddings)
dicee --dataset_dir "KGs/UMLS" --trainer "TP" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test"
# Distributed Data Parallelism in native torch
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test" --trainer "torchDDP" --scoring_technique KvsAll --path_to_store_single_run "UMLS_torchDDP"

```

#### Logging
<details><summary> Click me! </summary>

Progress, timing, and checkpoint messages (dataset info, epoch loss, "Saving model...", etc.) are emitted through Python's standard `logging` module rather than `print()`. By default `--log_level` is `INFO`, so these messages are shown, matching the classic CLI output.

```bash
# Default: INFO messages (dataset stats, timings, checkpoints, ...) are printed to stderr
dicee --dataset_dir "KGs/UMLS" --model Keci

# Quieter: only show warnings and errors (e.g. on noisy multi-node/multi-GPU logs)
dicee --dataset_dir "KGs/UMLS" --model Keci --log_level WARNING

# Silent: only errors are shown
dicee --dataset_dir "KGs/UMLS" --model Keci --log_level ERROR

# Verbose: include DEBUG-level messages too
dicee --dataset_dir "KGs/UMLS" --model Keci --log_level DEBUG
```

`--log_level` accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. The same setting is available when training from Python via `dicee.config.Namespace`:

```python
from dicee.executer import Execute
from dicee.config import Namespace
args = Namespace()
args.dataset_dir = "KGs/UMLS"
args.log_level = "WARNING"  # silence INFO-level progress messages
Execute(args).start()
```
Since every rank in a distributed run (`torchDDP`/`torchFSDP`/`PL` with multiple GPUs or nodes) configures its own logger, each process prints its own messages; lower the level to `WARNING` or `ERROR` to cut down on duplicate output across ranks.

A KGE model model can also be trained in multi-node multi-gpu DDP setting. 
```bash
torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 0 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula  dicee --trainer "torchDDP" --dataset_dir "KGs/YAGO3-10" --path_to_store_single_run "YAGO3_torchDDP"
torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 1 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula  dicee --trainer "torchDDP" --dataset_dir "KGs/YAGO3-10" --path_to_store_single_run "YAGO3_torchDDP"
```
Multi-node training is also possible with the `PL` trainer 
```bash
torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 0 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula  dicee --trainer "PL" --dataset_dir "KGs/YAGO3-10" --path_to_store_single_run "YAGO3_PL"
torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 1 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula  dicee --trainer "PL" --dataset_dir "KGs/YAGO3-10" --path_to_store_single_run "YAGO3_PL"
```
</details>

#### FSDP Training (large entity tables)

`torchFSDP` is designed for knowledge graphs where the entity embedding table is too large to fit on a single GPU. Entity embeddings are sharded row-wise across all GPUs — each rank owns a contiguous slice of entity rows and serves lookup requests from other ranks via `all_to_all`. Dense model parameters (relation embeddings, scoring layers) are wrapped with PyTorch FSDP.

```bash
# Single-node multi-GPU FSDP (replace --nproc_per_node with your GPU count)
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  dicee --dataset_dir "KGs/YAGO3-10" --model Keci \
  --trainer "torchFSDP" --scoring_technique "NegSample" \
  --path_to_store_single_run "YAGO_fsdp" --num_epochs 100 --batch_size 200000

# With FSDP1vsSample (GPU-efficient 1-vs-sample with true-negative sampling)
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  dicee --dataset_dir "KGs/YAGO3-10" --model Keci \
  --trainer "torchFSDP" --scoring_technique "FSDP1vsSample" --neg_ratio 10 \
  --path_to_store_single_run "YAGO_fsdp" --num_epochs 100 --batch_size 200000
```

FSDP-specific options are passed via `--fsdp_trainer_kwargs` (JSON dict):

| Key | Default | Description |
|---|---|---|
| `precision` | `float32` | `float32 \| bfloat16 \| float16` |
| `fsdp_optim_device` | `cpu` | Entity Adam state device — `cpu` saves GPU RAM, `gpu` is faster |
| `sharding_strategy` | `FULL_SHARD` | FSDP sharding strategy |
| `gradient_clip_val` | `null` | Optional gradient norm clipping |
| `num_workers` | `num_core` | DataLoader worker count |
| `prefetch_factor` | `4` | DataLoader prefetch depth |
| `checkpoint_every_n_epochs` | `null` | Save a resumable sharded checkpoint every N epochs (see below) |

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  dicee --dataset_dir "KGs/YAGO3-10" --model Keci \
  --trainer "torchFSDP" --scoring_technique "NegSample" \
  --path_to_store_single_run "YAGO_fsdp" \
  --fsdp_trainer_kwargs '{"precision": "bfloat16", "fsdp_optim_device": "gpu"}'
```

Compatible scoring techniques: `NegSample`, `FixedNegSample`, `KvsSample`, `FSDP1vsSample`. Mid-epoch evaluation (`--eval_every_n_epochs`, `--eval_at_epochs`) is not supported — entity embeddings are gathered on rank 0 only after training completes.

##### Resuming an interrupted FSDP run

The entity table is sharded across ranks, so a plain `model.pt` (only written once
training finishes) can't be used to recover a run that crashed or was preempted
partway through — for a large enough table, even reconstructing that single file
can exceed one node's RAM (see [issue #422](https://github.com/dice-group/dice-embeddings/issues/422)).
Set `checkpoint_every_n_epochs` to write a *sharded* checkpoint instead: each rank
saves only its own slice of the entity table plus its own optimizer state, so
checkpointing cost stays proportional to `num_entities / world_size`, not
`num_entities`.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  dicee --dataset_dir "KGs/YAGO3-10" --model Keci \
  --trainer "torchFSDP" --scoring_technique "NegSample" \
  --path_to_store_single_run "YAGO_fsdp" --num_epochs 500 --reuse_existing_run_dir true \
  --fsdp_trainer_kwargs '{"checkpoint_every_n_epochs": 10}'

# If this run is killed (OOM, preemption, node failure) and restarted with the
# SAME --path_to_store_single_run and the SAME number of ranks, it picks up
# from the last checkpoint automatically — no --continual_learning needed.
# --reuse_existing_run_dir true is REQUIRED for this: without it, Execute
# deletes path_to_store_single_run before training starts if it already
# exists (see executer.py:_setup_single_run_directory), wiping the checkpoint
# before the trainer ever gets a chance to look for it.
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  dicee --dataset_dir "KGs/YAGO3-10" --model Keci \
  --trainer "torchFSDP" --scoring_technique "NegSample" \
  --path_to_store_single_run "YAGO_fsdp" --num_epochs 500 --reuse_existing_run_dir true \
  --fsdp_trainer_kwargs '{"checkpoint_every_n_epochs": 10}'

# To resume into a NEW output directory instead, point --continual_learning at
# the old one (it must contain a fsdp_shard_checkpoint/ dir, not just model.pt):
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  dicee --dataset_dir "KGs/YAGO3-10" --model Keci \
  --trainer "torchFSDP" --scoring_technique "NegSample" \
  --continual_learning "YAGO_fsdp" --num_epochs 500
```

Notes:

- **`--reuse_existing_run_dir true` is required for same-directory auto-resume.** Without it, a fresh (non-`--continual_learning`) run always deletes `path_to_store_single_run` first if it already exists, before the trainer gets a chance to check for a checkpoint there.
- Resuming requires launching with the same world_size (rank count) used to save the checkpoint — shard boundaries are a deterministic function of `(num_entities, world_size)`, so a different rank count means a different, incompatible partition. Re-sharding across a different world_size isn't supported yet.
- Resuming from a fully-materialized `model.pt` (the classic continual-learning path used by other trainers) is not supported for `torchFSDP` — only from a checkpoint a `torchFSDP` run wrote itself via `checkpoint_every_n_epochs`.

On large knowledge graphs, this configurations should be used.
Note: When training with multi-GPU or Distributed Data Parallel (DDP) settings, you must provide the `--path_to_store_single_run` argument to specify where to store the results of a single training run. This ensures that all processes write to the correct directory and prevents conflicts.

Here is an example of an iterative training of a a KGE model can be resumed.
```bash
# No training.
torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --dataset_dir "KGs/UMLS" --model Keci --scoring_technique "FixedNegSample" --trainer "torchDDP" --scoring_technique FixedNegSample --path_to_store_single_run "UMLS_torchDDP" --num_epochs 0
# Train 10 epochs on fixed negative samples.
torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --dataset_dir "KGs/UMLS" --model Keci --scoring_technique "FixedNegSample" --trainer "torchDDP" --scoring_technique FixedNegSample --num_epochs 10 --continual_learning "UMLS_torchDDP" --random_seed 1
# Train 10 epochs on fixed negative samples.
torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --dataset_dir "KGs/UMLS" --model Keci --scoring_technique "FixedNegSample" --trainer "torchDDP" --scoring_technique FixedNegSample --num_epochs 10 --continual_learning "UMLS_torchDDP" --random_seed 2
# Train 10 epochs on fixed negative samples.
torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --dataset_dir "KGs/UMLS" --model Keci --scoring_technique "FixedNegSample" --trainer "torchDDP" --scoring_technique FixedNegSample --num_epochs 10 --continual_learning "UMLS_torchDDP" --random_seed 3
```
When using a multi-GPU setup, `PL` Trainer  automatically utilizes all available CUDA devices. To perform training on a single device, set the environment variable `CUDA_VISIBLE_DEVICES=0` before running your command. For example:

```bash
CUDA_VISIBLE_DEVICES=0 dicee --dataset_dir "KGs/UMLS" --trainer "PL" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test" --num_epochs 100
``` 
The `CUDA_VISIBLE_DEVICES=0` setting limits the program to access only the specified GPU(s), making all others invisible.  
Multiple GPUs can be selected by providing a comma-separated list, for example: `CUDA_VISIBLE_DEVICES=0,1`.
Additional PyTorch Lightning trainer options can be passed with `--pl_trainer_kwargs`, e.g. `--pl_trainer_kwargs '{"precision":"16-mixed","strategy":"ddp"}'`. PyTorch Lightning Trainer has many optional parameters; see: https://lightning.ai/docs/pytorch/stable/common/trainer.html

**📖 [See trainer examples →](tests/test_trainers.py)** | **📖 [Custom trainer →](tests/test_custom_trainer.py)**


The data is in the following form
```bash
$ head -3 KGs/UMLS/train.txt 
acquired_abnormality    location_of     experimental_model_of_disease
anatomical_abnormality  manifestation_of        physiologic_function
alga    isa     entity

$ head -3 KGs/YAGO3-10/valid.txt 
Mikheil_Khutsishvili    playsFor        FC_Merani_Tbilisi
Ebbw_Vale       isLocatedIn     Blaenau_Gwent
Valenciennes    isLocatedIn     Nord-Pas-de-Calais
```
By default, ```--backend "pandas" --separator "\s+" ``` is used in ```pandas.read_csv(sep=args.separator)``` to obtain triples.
You can choose a suitable backend for your knowledge graph ```--backend pandas | polars  | rdflib ```.
On large knowledge graphs n-triples, ```--backend "polars" --separator " " ``` is a good option.
**Apart from n-triples or standard link prediction dataset formats, we support ["owl", "nt", "turtle", "rdf/xml", "n3"]***.
On other RDF knowledge graphs,  ```--backend "rdflib" ``` can be used. Note that knowledge graphs must not contain blank nodes or literals.
Moreover, a KGE model can be also trained  by providing **an endpoint of a triple store**. 
```bash
dicee --sparql_endpoint "http://localhost:3030/mutagenesis/" --model Keci
```

**📖 [See dataset format details →](docs/guides/datasets.md)** | **📖 [Different backend examples →](tests/test_different_backends.py)**

#### Scoring Techniques

We have implemented state-of-the-art scoring techniques to train a KGE model ```--scoring_technique 1vsAll | KvsAll  | AllvsAll | KvsSample | NegSample | FixedNegSample```.
```bash
dicee --dataset_dir "KGs/YAGO3-10" --model Keci --trainer "torchCPUTrainer" --scoring_technique "NegSample" --neg_ratio 10 --num_epochs 10 --batch_size 10_000 --num_core 0 --eval_model None
# Epoch:10: 100%|███████████| 10/10 [01:31<00:00,  9.11s/it, loss_step=0.09423, loss_epoch=0.07897]
# Training Runtime: 1.520 minutes.
dicee --dataset_dir "KGs/YAGO3-10" --model Keci --trainer "torchCPUTrainer" --scoring_technique "NegSample" --neg_ratio 10 --num_epochs 10 --batch_size 10_000 --num_core 10 --eval_model None
# Epoch:10: 100%|███████████| 10/10 [00:58<00:00,  5.80s/it, loss_step=0.11909, loss_epoch=0.07991]
# Training Runtime: 58.106 seconds.
dicee --dataset_dir "KGs/YAGO3-10" --model Keci --trainer "torchCPUTrainer" --scoring_technique "NegSample" --neg_ratio 10 --num_epochs 10 --batch_size 10_000 --num_core 20 --eval_model None
# Epoch:10: 100%|███████████| 10/10 [01:01<00:00,  6.16s/it, loss_step=0.10751, loss_epoch=0.06962]
# Training Runtime: 1.029 minutes.
dicee --dataset_dir "KGs/YAGO3-10" --model Keci --trainer "torchCPUTrainer" --scoring_technique "NegSample" --neg_ratio 10 --num_epochs 10 --batch_size 10_000 --num_core 50 --eval_model None
# Epoch:10: 100%|███████████| 10/10 [01:08<00:00,  6.83s/it, loss_step=0.05347, loss_epoch=0.07003]
# Training Runtime: 1.140 minutes.
```
Increasing the number of cores often (but not always) helps to decrease the runtimes on large knowledge graphs ```--num_core 4 --scoring_technique KvsSample | NegSample --neg_ratio 1``` 

**📖 [See scoring technique examples →](tests/test_k_fold_cv_*.py)** | **📖 [KvsSample →](tests/test_onevssample.py)** 

A KGE model can be also trained in a python script
```python
from dicee.executer import Execute
from dicee.config import Namespace
args = Namespace()
args.model = 'Keci'
args.scoring_technique = "KvsAll"  # 1vsAll, or AllvsAll, or NegSample
args.dataset_dir = "KGs/UMLS"
args.path_to_store_single_run = "Keci_UMLS"
args.num_epochs = 100
args.embedding_dim = 32
args.batch_size = 1024
reports = Execute(args).start()
print(reports["Train"]["MRR"]) # => 0.9912
print(reports["Test"]["MRR"]) # => 0.8155
```

**📖 [See more training examples →](tests/test_execute_start.py)** | **📖 [Model-specific tests →](tests/test_regression_*.py)**
args.path_to_store_single_run = "Keci_UMLS"
args.num_epochs = 100
args.embedding_dim = 32
args.batch_size = 1024
reports = Execute(args).start()
print(reports["Train"]["MRR"]) # => 0.9912
print(reports["Test"]["MRR"]) # => 0.8155
# See the Keci_UMLS folder embeddings and all other files
```

#### Continual Learning

Train a KGE model by providing the path of a single file and store all parameters under newly created directory
called `KeciFamilyRun`.
```bash
dicee --path_single_kg "KGs/Family/family-benchmark_rich_background.owl" --model Keci --path_to_store_single_run KeciFamilyRun --backend rdflib --eval_model None
```
where the data is in the following form
```bash
$ head -3 KGs/Family/train.txt 
_:1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#Ontology> .
<http://www.benchmark.org/family#hasChild> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#ObjectProperty> .
<http://www.benchmark.org/family#hasParent> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#ObjectProperty> .
```

**Continual Training:** the training phase of a pretrained model can be resumed.
The run reuses configuration and serialized artifacts from the existing experiment folder and stores updated outputs in the same directory using `--continual_learning "KeciFamilyRun"`.
```bash
dicee --continual_learning "KeciFamilyRun" --path_single_kg "KGs/Family/family-benchmark_rich_background.owl" --model Keci --backend rdflib --eval_model None
```
The continual directory should contain the stored configuration and serialized training data (for example `configuration.json`, `memory_map_train_set.npy`, and mapping files `entity_to_idx`/`relation_to_idx` in `.csv` or legacy `.p` format).
If `--eval_model` is set, evaluation runs after training using stored indexed artifacts. If `--eval_model None`, no evaluation is executed.
Periodic evaluation and weight-averaging callbacks are also supported in continual training.

**📖 [See continual learning examples →](tests/test_continual_training.py)** | **📖 [Online learning →](tests/test_online_learning.py)**

#### Ensemble Learning with Knowledge Graph Embeddings

The KGE models in our **dice-embedding** framework now support a range of state-of-the-art weight averaging techniques, including:

- **Stochastic Weight Averaging (SWA)**
- **Adaptive Stochastic Weight Averaging (ASWA)**
- **Stochastic Weight Averaging–Gaussian (SWAG)**
- **Exponential Moving Average (EMA)**
- **Trainable Weight Averaging (TWA)**

To enable any of these methods, use the corresponding command-line options as shown below.

**SWA**
```bash
dicee --dataset_dir "KGs/UMLS" --trainer "PL" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test" --num_epochs 100 --swa
``` 
**ASWA**
```bash
dicee --dataset_dir "KGs/UMLS" --trainer "PL" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test" --num_epochs 100 --aswa
``` 
**Weight Averaging Start Epoch**

Weight averaging begins at **epoch 0** by default. To start averaging from a later epoch, set `--swa_start_epoch`. This applies to all methods **except ASWA**.

**EMA**
```bash
dicee --dataset_dir "KGs/UMLS" --trainer "PL" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test" \
 --num_epochs 100 --ema --swa_start_epoch 50
```  
**Interval Averaging and Multi-device training**

Weight Averaging can also be performed by aggregating weights of running model at certain interval. Use the command *--swa_c_epochs* to do so. For example,  to average the weights at every 2 epochs along with SWA starting at 50 epochs, use the command: 

```bash
dicee --dataset_dir "KGs/UMLS" --trainer "PL" --scoring_technique KvsAll --model "Keci" --eval_model "train_val_test" \
 --num_epochs 100 --swa --swa_start_epoch 50 --swa_c_epochs 2
``` 
The weight averaging methods can also be used in multi-device settings using the `PL` trainer. However, some of the approaches are not currently supported for TP and torchDDP trainers.

The weight averaging methods can also be evaluated at certain epochs during training or at certain intervals.
```bash
dicee  --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_every_n_epochs 50 --save_every_n_epochs --n_epochs_eval_model val_test --swa
```
For more details on periodic evaluations, please refer to the periodic evaluation section below in this file.

**📖 [See SWA examples →](tests/test_swa.py)** | **📖 [Adaptive SWA →](tests/test_adaptive_swa.py)** | **📖 [Ensemble construction →](tests/test_ensemble_construction.py)**

#### Periodic Evaluation during training

The Periodic evaluation method automates periodic model evaluation and checkpointing during training. It allows evaluations at fixed intervals or specific epochs. Results and model states are stored systematically for efficient hyperparameter search.

Configure automatic evaluation by setting `eval_every_n_epochs` to run evaluations every N epochs, or `eval_at_epochs` for specific epochs—these options can be combined. Use `save_model_every_n_epoch` to save a checkpoint at each evaluation, and specify evaluation splits (`val`, `test`, or `val_test`) with `n_epochs_eval_model`. If the last training epoch matches a scheduled evaluation and the default trainer evaluates all specified splits, evaluation with `n_epochs` is skipped to prevent duplicate results.

``` bash
# Evaluate every 50 epochs on validation and test sets, saving a model checkpoint at each evaluation
dicee  --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_every_n_epochs 50 --save_every_n_epochs --n_epochs_eval_model val_test

# Evaluate only at epochs 128 and 256 on the validation set, saving the model each time
dicee  --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_at_epochs 128 256 --save_every_n_epochs --n_epochs_eval_model val

# Evaluate every 100 epochs on the test set only; model checkpoints are not saved
dicee  --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_every_n_epochs 100 --n_epochs_eval_model test

# Evaluate only at epochs 50 and 150 on both validation and test sets; no checkpoint is saved
dicee  --dataset_dir "KGs/UMLS" --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_at_epochs 50 150 --n_epochs_eval_model val_test

# Evaluate every 100 epochs and additionally at epochs 45 and 275 on validation; models are saved at each evaluation point
dicee --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_every_n_epochs 100 --eval_at_epochs 45 275 --save_every_n_epochs --n_epochs_eval_model val
```
#### Periodic Evaluation during training with Weight Averaging (Ensemble) Methods
The periodic evaluation function also allows evaluating the underlying ensemble model at particular epochs when using ensemble learning. The features can be paired simply by combining the arguments.
```bash
# Evaluate SWA ensemble model at epochs 128 and 256 on the validation set, saving the model each time
dicee  --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_at_epochs 128 256 --save_every_n_epochs --n_epochs_eval_model val --swa

# Evaluate ASWA checkpoints at every 100 epochs on the test set only; save model checkpoints
dicee  --dataset_dir "KGs/UMLS" --model Keci --scoring_technique KvsAll --num_epochs 300 --lr 0.1 \
      --eval_every_n_epochs 100 --n_epochs_eval_model test --save_every_n_epochs --adaptive_swa
```

**📖 [See periodic evaluation examples →](tests/test_periodic_eval_callback.py)** | **📖 [Periodic eval with weight averaging →](tests/test_periodic_eval_weight_averaging.py)**

Currently, Periodic Evaluations as well as Ensemble Models can only be used in combination with `torchCPUTrainer` or `PL` trainer with a single CUDA-capable device. They are not supported with `torchDDP`, `torchFSDP`, or `TP` trainers.
</details>

## Link Prediction & Inference
<details> <summary> Using Pre-trained Models </summary>

### Download & Use Pretrained Models

```python
from dicee import KGE

# Download from URL
model = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/KINSHIP-Keci-dim128-epoch256-KvsAll")

# Or load local model
model = KGE(path="Experiments/2024-01-15...")

# Make a prediction
model.predict(h="person49", r="term12", t="person39", logits=False)
```

**📖 [See download & evaluation examples →](tests/test_download_and_eval.py)**

### Predict Missing Links

```python
from dicee import KGE

model = KGE(path="...")

# Predict missing tail entities
predictions = model.predict_topk(h=["Mongolia"], r=["isLocatedIn"], topk=3)
# [('Asia', 0.65), ('Airport', 0.37), ...]

# Predict missing head entities  
predictions = model.predict_topk(r=["isLocatedIn"], t=["Asia"], topk=10)

# Predict missing relations
predictions = model.predict_topk(h=["Mongolia"], t=["Asia"], topk=5)
```

**📖 [See complete link prediction examples →](tests/test_predict_kge.py)**

</details>

## Multi-Hop Query Answering
<details> <summary> EPFO Queries (1p, 2p, 3p, 2i, 3i, ip, pi, 2u, up) </summary>

```python
from dicee import KGE

# Load pre-trained model
model = KGE(path="...")

# 1-hop: Who are the siblings of F9M167?
# Query: ?E : ∃E.hasSibling(E, F9M167)
predictions = model.answer_multi_hop_query(
    query_type="1p",
    query=('http://www.benchmark.org/family#F9M167',
           ('http://www.benchmark.org/family#hasSibling',)),
    tnorm="min", k=3
)
# => [('F9F141', 0.99), ('F9M157', 0.98), ...]

# 2-hop: To whom is a sibling of F9M167 married?
# Query: ?D : ∃E.Married(D,E) ∧ hasSibling(E, F9M167)
predictions = model.answer_multi_hop_query(
    query_type="2p",
    query=("http://www.benchmark.org/family#F9M167",
           ("http://www.benchmark.org/family#hasSibling",
            "http://www.benchmark.org/family#married")),
    tnorm="min", k=3
)
# => [('F9F158', 0.95), ('F9M142', 0.93), ...]

# 3-hop: What type of people are married to a sibling of F9M167?
# Query: ?T : ∃D.type(D,T) ∧ Married(D,E) ∧ hasSibling(E, F9M167)
predictions = model.answer_multi_hop_query(
    query_type="3p",
    query=("http://www.benchmark.org/family#F9M167",
           ("http://www.benchmark.org/family#hasSibling",
            "http://www.benchmark.org/family#married",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),
    tnorm="min", k=5
)
# => [('Person', 0.99), ('Male', 0.99), ('Father', 0.98), ...]
```

**📖 [See multi-hop query examples →](tests/test_answer_multi_hop_query.py)**  
**Supported query types:** `1p` (1-hop projection), `2p` (2-hop), `3p` (3-hop), `2i` (2-way intersection), `3i` (3-way intersection), `ip` (intersection-projection), `pi` (projection-intersection), `2u` (2-way union), `up` (union-projection)

</details>

## Literal Prediction
<details> <summary> Predicting Numeric/Literal Values </summary>

```python
from dicee import KGE

# Load pre-trained model
model = KGE(path="...")

# Train literal prediction module on top of KGE
model.train_literals(train_file_path="literals_train.csv")

# Predict literal values
predictions = model.predict_literals(entity=["Person1"], attribute=["hasAge"])
# => [(42.5, 0.89), (43.2, 0.85), ...]
```

**📖 [See literal prediction examples →](tests/test_predict_kge_literals.py)**

</details>

## Pre-trained Models

We provide pre-trained knowledge graph embedding models at [dice-research.org/projects/DiceEmbeddings/](https://files.dice-research.org/projects/DiceEmbeddings/).
<details> <summary> Download & Compare Models </summary>

```python
from dicee import KGE

# Download different models
mure = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/YAGO3-10-Pykeen_MuRE-dim128-epoch256-KvsAll")
quate = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/YAGO3-10-Pykeen_QuatE-dim128-epoch256-KvsAll")
keci = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/YAGO3-10-Keci-dim128-epoch256-KvsAll")

# Compare predictions
mure.predict_topk(h=["Mongolia"], r=["isLocatedIn"], topk=3)
# [('Asia', 0.9997), ('Ulan_Bator', 0.0010), ('Philippines', 0.0003)]

quate.predict_topk(h=["Mongolia"], r=["isLocatedIn"], topk=3)
# [('Asia', 0.9894), ('Europe', 0.0158), ('Tadanari_Lee', 0.0125)]

keci.predict_topk(h=["Mongolia"], r=["isLocatedIn"], topk=3)
# [('Asia', 0.6522), ('Airport', 0.3656), ('Democratic_Party', 0.1960)]
```

**📖 [See download & evaluation examples →](tests/test_download_and_eval.py)**

</details>

## Knowledge graph foundation models

[ULTRA](docs/ultra.md) supports official pretrained checkpoints, zero-shot link prediction,
fine-tuning, and native DICE training objectives using pure PyTorch.

<details>
<summary>ULTRA checkpoint downloads and inference</summary>

Download any of the three official checkpoints:

```bash
wget -P checkpoints https://raw.githubusercontent.com/DeepGraphLearning/ULTRA/main/ckpts/ultra_3g.pth
wget -P checkpoints https://raw.githubusercontent.com/DeepGraphLearning/ULTRA/main/ckpts/ultra_4g.pth
wget -P checkpoints https://raw.githubusercontent.com/DeepGraphLearning/ULTRA/main/ckpts/ultra_50g.pth
```

Run zero-shot inference and filtered evaluation on the UMLS test set:

```bash
python -m dicee --model ULTRA --dataset_dir KGs/UMLS \
  --ultra_checkpoint checkpoints/ultra_3g.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test
```

Change `--ultra_checkpoint` to use another checkpoint and `--dataset_dir` to use
your own graph with `train.txt` and `test.txt`. See the [ULTRA guide](docs/ultra.md)
for training, fine-tuning, and prediction examples.

</details>

[TRIX](docs/trix.md) supports the official entity- and relation-prediction
checkpoints, with pure PyTorch training, fine-tuning, and zero-shot inference.
Its scores and gradients are verified against the official implementation.

<details>
<summary>TRIX checkpoint downloads and inference</summary>

```bash
mkdir -p checkpoints/trix
wget -P checkpoints/trix https://raw.githubusercontent.com/yuchengz99/TRIX/7596e14eefefe89e61396205a0550172cadeddb0/entity_prediction.pth
wget -P checkpoints/trix https://raw.githubusercontent.com/yuchengz99/TRIX/7596e14eefefe89e61396205a0550172cadeddb0/relation_prediction.pth

python -m dicee --model TRIX --dataset_dir KGs/UMLS \
  --trix_checkpoint checkpoints/trix/entity_prediction.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test

python -m dicee --model TRIXRelation --dataset_dir KGs/UMLS \
  --trix_checkpoint checkpoints/trix/relation_prediction.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --batch_size 8 --eval_model test
```

See the [TRIX guide](docs/trix.md) for Python inference, training, checkpoint
compatibility, and reproducible verification against the reference implementation.

</details>

[Flock](docs/flock.md) adds random-walk-based entity and relation prediction,
with both official checkpoints supported in pure PyTorch. Scores and gradients
are verified against the official models on identical recorded walks.

<details>
<summary>Flock checkpoint downloads and inference</summary>

```bash
mkdir -p checkpoints/flock
wget -P checkpoints/flock https://raw.githubusercontent.com/jw9730/flock/f35103d25a78bdf4075de5c673a51de4979aa4d7/checkpoints/flock_entity.pth
wget -P checkpoints/flock https://raw.githubusercontent.com/jw9730/flock/f35103d25a78bdf4075de5c673a51de4979aa4d7/checkpoints/flock_relation.pth

python -m dicee --model Flock --dataset_dir KGs/UMLS \
  --flock_checkpoint checkpoints/flock/flock_entity.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 2 --eval_model test

python -m dicee --model FlockRelation --dataset_dir KGs/UMLS \
  --flock_checkpoint checkpoints/flock/flock_relation.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --batch_size 2 --eval_model test
```

Flock is stochastic in evaluation mode. The [Flock guide](docs/flock.md) explains
sampling budgets, reproducibility, training, and verification against upstream.

</details>

## KGFM Link Prediction

Test-set entity prediction with released checkpoints and no fine-tuning, using
training triples plus inverse edges as the inference graph and filtered head/tail
ranking over all entities; see the [protocol and checkpoints](docs/kgfm_benchmarks.md).
Tie strategy: **pessimistic** (worst rank among exactly equal scores after filtering).

<details>
<summary>Show results</summary>

**Bold** marks the best result per dataset and metric, **Yes** marks target graphs used in pretraining.

| Dataset | Model | Target graph used in pretraining? | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|:---:|---:|---:|---:|---:|
| YAGO3-10 | ULTRA-3g | No | **0.4799** | **0.3832** | **0.5346** | **0.6583** |
| YAGO3-10 | TRIX | No | 0.4094 | 0.3024 | 0.4574 | 0.6265 |
| YAGO3-10 | Flock | No | 0.3998 | 0.3092 | 0.4526 | 0.5636 |
| FB15k-237 | ULTRA-3g | Yes | **0.3693** | **0.2718** | **0.4101** | **0.5620** |
| FB15k-237 | TRIX | Yes | 0.3618 | 0.2649 | 0.3989 | 0.5546 |
| FB15k-237 | Flock | Yes | 0.3116 | 0.2215 | 0.3442 | 0.4912 |
| WN18RR | ULTRA-3g | Yes | 0.3691 | 0.2924 | 0.3923 | 0.5329 |
| WN18RR | TRIX | Yes | 0.5065 | 0.4592 | 0.5217 | 0.6040 |
| WN18RR | Flock | Yes | **0.5303** | **0.4783** | **0.5482** | **0.6367** |
| UMLS | ULTRA-3g | No | 0.6960 | 0.5983 | 0.7474 | 0.8956 |
| UMLS | TRIX | No | 0.7256 | 0.6430 | 0.7632 | 0.8986 |
| UMLS | Flock | No | **0.7768** | **0.7005** | **0.8169** | **0.9244** |
| Countries-S1 | ULTRA-3g | No | **0.9375** | **0.8750** | **1.0000** | **1.0000** |
| Countries-S1 | TRIX | No | 0.9271 | 0.8542 | **1.0000** | **1.0000** |
| Countries-S1 | Flock | No | 0.9271 | 0.8542 | **1.0000** | **1.0000** |
| Countries-S2 | ULTRA-3g | No | 0.8715 | 0.7500 | **1.0000** | **1.0000** |
| Countries-S2 | TRIX | No | **0.8854** | **0.7708** | **1.0000** | **1.0000** |
| Countries-S2 | Flock | No | **0.8854** | **0.7708** | **1.0000** | **1.0000** |
| Countries-S3 | ULTRA-3g | No | 0.2354 | **0.0625** | 0.2917 | 0.6458 |
| Countries-S3 | TRIX | No | **0.3625** | **0.0625** | **0.5833** | **0.8958** |
| Countries-S3 | Flock | No | 0.2533 | 0.0208 | 0.4583 | 0.5000 |

</details>

<details>
<summary>Inference speed</summary>

Warm all-entity inference speedup over the authors’ official implementations:
RTX 4070 Ti SUPER, float32, matched query batches, and five-repeat medians on sampled test queries.

| Dataset | ULTRA-3g | TRIX | Flock* |
|---|---:|---:|---:|
| FB15k-237 | 3.66× | 31.75× | 2.64× |
| WN18RR | 1.15× | 18.71× | 1.83× |
| YAGO3-10 | 7.44× | 35.38× | 8.61× |

\* Flock includes independently sampled walks at the same budget; identical-walk neural speedups are **1.23–2.01×**.
YAGO3-10 Flock timings varied more. See the [full comparison and validation](docs/kgfm_inference.md).

</details>

## Link Prediction Benchmarks

In the below, we provide a brief overview of the link prediction results. Results are sorted in descending order of the size of the respective dataset.

<details>
<summary>Tie handling</summary>

| Policy | Target rank within a tied group | Example: three candidates tied for first |
|---|---|---|
| `sort` (default) | Position in the sorted score list; ties follow the sorting routine’s order | 1, 2, or 3 according to that order |
| `optimistic` | Best possible rank | 1 |
| `random` | Uniformly sampled integer rank | 1, 2, or 3 with equal probability |
| `pessimistic` | Worst possible rank | 3 |

Set `--eval_tie_policy` to rank exactly equal scores after filtering; `--eval_tie_seed` seeds `random` and defaults to `random_seed`.

</details>

#### YAGO3-10 ####

<details> <summary> To see the results </summary>

|                    |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll     | val   | 0.374 |  0.308 |  0.402 |   0.501 |
| ComplEx-KvsAll     | test  | 0.372 |  0.302 |  0.404 |   0.505 |
| ComplEx-KvsAll-SWA | train | 0.998 |  0.997 |  1.000 |   1.000 |
| ComplEx-KvsAll-SWA | val   | 0.345 |  0.279 |  0.372 |   0.474 |
| ComplEx-KvsAll-SWA | test  | 0.341 |  0.272 |  0.374 |   0.474 |
| ComplEx-KvsAll-ASWA | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll-ASWA | val   | 0.404 |  0.335 |  0.448 |   0.531 |
| ComplEx-KvsAll-ASWA | test  | 0.398 |  0.325 |  0.449 |   0.530 |
| Keci-KvsAll        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll        | val   | 0.337 |  0.268 |  0.370 |   0.468 |
| Keci-KvsAll        | test  | 0.343 |  0.274 |  0.376 |   0.343 |
| Keci-KvsAll-SWA    | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll-SWA    | val   | 0.325 |  0.253 |  0.358 |   0.459 |
| Keci-KvsAll-SWA    | test  | 0.334 |  0.263 |  0.367 |   0.470 |
| Keci-KvsAll-ASWA   | train | 0.978 |  0.969 |  0.985 |   0.991 |
| Keci-KvsAll-ASWA   | val   | 0.400 |  0.324 |  0.439 |   0.540 |
| Keci-KvsAll-ASWA   | test  | 0.394 |  0.317 |  0.439 |   0.539 |

```--embedding_dim 256 --num_epochs 300 --batch_size 1024 --optim Adam 0.1``` leading to 31.6M params.
Observations: A severe overfitting. ASWA improves the generalization better than SWA.

</details>

#### FB15k-237 ####

<details> <summary> To see the results </summary>

|                    |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll     | val   | 0.197 |  0.140 |  0.211 |   0.307 |
| ComplEx-KvsAll     | test  | 0.192 |  0.137 |  0.204 |   0.300 |
| ComplEx-KvsAll-SWA | train | 0.911 |  0.875 |  0.938 |   0.970 |
| ComplEx-KvsAll-SWA | val   | 0.169 |  0.121 |  0.178 |   0.264 |
| ComplEx-KvsAll-SWA | test  | 0.166 |  0.118 |  0.176 |   0.261 |
| ComplEx-KvsAll-ASWA | train | 0.780 | 0.719 | 0.822  |   0.886 |
| ComplEx-KvsAll-ASWA | val   | 0.220 | 0.158 | 0.240  |   0.342 |
| ComplEx-KvsAll-ASWA | test  | 0.217 | 0.155 |  0.234 |   0.337 |
| Keci-KvsAll        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll        | val   | 0.158 |  0.107 |  0.166 |   0.259 |
| Keci-KvsAll        | test  | 0.155 |  0.105 |  0.162 |   0.253 |
| Keci-KvsAll-SWA    | train | 0.941 |  0.909 |  0.967 |   0.990 |
| Keci-KvsAll-SWA    | val   | 0.188 |  0.133 |  0.200 |   0.298 |
| Keci-KvsAll-SWA    | test  | 0.183 |  0.128 |  0.195 |   0.292 |
| Keci-KvsAll-ASWA   | train | 0.745 |  0.666 |  0.799 |   0.886 |
| Keci-KvsAll-ASWA   | val   | 0.221 |  0.158 |  0.237 |   0.346 |
| Keci-KvsAll-ASWA   | test  | 0.216 |  0.153 |  0.233 |   0.342 |

</details>


#### WN18RR ####

<details> <summary> To see the results </summary>

|                    |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll     | val   | 0.346 |  0.337 |  0.353 |   0.359 |
| ComplEx-KvsAll     | test  | 0.343 |  0.333 |  0.349 |   0.357 |
| ComplEx-KvsAll-SWA | train | 0.335 |  0.242 |  0.365 |   0.522 |
| ComplEx-KvsAll-SWA | val   | 0.014 |  0.004 |  0.013 |   0.029 |
| ComplEx-KvsAll-SWA | test  | 0.018 |  0.007 |  0.017 |   0.036 |
| ComplEx-KvsAll-ASWA | train | 0.999 | 0.999 | 0.999  |   0.999 |
| ComplEx-KvsAll-ASWA | val   | 0.352 | 0.348 | 0.355  |   0.359 |
| ComplEx-KvsAll-ASWA | test  | 0.348 | 0.342 |  0.351 |   0.355 |
| Keci-KvsAll        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll        | val   | 0.316 |  0.296 |  0.331 |   0.352 |
| Keci-KvsAll        | test  | 0.308 |  0.285 |  0.326 |   0.346 |
| Keci-KvsAll-SWA    | train | 0.608 |  0.516 |  0.657 |   0.787 |
| Keci-KvsAll-SWA    | val   | 0.069 |  0.038 |  0.073 |   0.130 |
| Keci-KvsAll-SWA    | test  | 0.061 |  0.032 |  0.064 |   0.119 |
| Keci-KvsAll-ASWA   | train | 0.951 |  0.916 |  0.985 |   0.997 |
| Keci-KvsAll-ASWA   | val   | 0.356 |  0.353 |  0.356 |   0.360 |
| Keci-KvsAll-ASWA   | test  | 0.350 |  0.347 |  0.350 |   0.355 |

</details>

#### UMLS ####

<details> <summary> To see the results </summary>

|                       |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|-----------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll        | val   | 0.684 |  0.557 |  0.771 |   0.928 |
| ComplEx-KvsAll        | test  | 0.680 |  0.563 |  0.750 |   0.918 |
| ComplEx-AllvsAll      | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll      | val   | 0.771 |  0.670 |  0.847 |   0.949 |
| ComplEx-AllvsAll      | test  | 0.778 |  0.678 |  0.850 |   0.957 |
| ComplEx-KvsAll-SWA    | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll-SWA    | val   | 0.762 |  0.666 |  0.825 |   0.941 |
| ComplEx-KvsAll-SWA    | test  | 0.757 |  0.653 |  0.833 |   0.939 |
| ComplEx-AllvsAll-SWA  | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll-SWA  | val   | 0.817 |  0.736 |  0.879 |   0.953 |
| ComplEx-AllvsAll-SWA  | test  | 0.827 |  0.748 |  0.883 |   0.967 |
| ComplEx-KvsAll-ASWA   | train | 0.998 |  0.997 |  0.999 |   1.000 |
| ComplEx-KvsAll-ASWA   | val   | 0.799 |  0.712 |  0.863 |   0.946 |
| ComplEx-KvsAll-ASWA   | test  | 0.804 |  0.720 |  0.866 |   0.948 |
| ComplEx-AllvsAll-ASWA | train | 0.998 |  0.997 |  0.998 |   0.999 |
| ComplEx-AllvsAll-ASWA | val   | 0.879 |  0.824 |  0.926 |   0.964 |
| ComplEx-AllvsAll-ASWA | test  | 0.877 |  0.819 |  0.924 |   0.971 |
| Keci-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll           | val   | 0.538 |  0.401 |  0.595 |   0.829 |
| Keci-KvsAll           | test  | 0.543 |  0.411 |  0.610 |   0.815 |
| Keci-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll         | val   | 0.672 |  0.556 |  0.742 |   0.909 |
| Keci-AllvsAll         | test  | 0.684 |  0.567 |  0.759 |   0.914 |
| Keci-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll-SWA       | val   | 0.633 |  0.509 |  0.705 |   0.877 |
| Keci-KvsAll-SWA       | test  | 0.628 |  0.498 |  0.710 |   0.868 |
| Keci-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll-SWA     | val   | 0.697 |  0.584 |  0.770 |   0.911 |
| Keci-AllvsAll-SWA     | test  | 0.711 |  0.606 |  0.775 |   0.921 |
| Keci-KvsAll-ASWA      | train | 0.996 |  0.993 |  0.999 |   1.000 |
| Keci-KvsAll-ASWA      | val   | 0.767 |  0.668 |  0.836 |   0.944 |
| Keci-KvsAll-ASWA      | test  | 0.762 |  0.660 |  0.830 |   0.949 |
| Keci-AllvsAll-ASWA    | train | 0.998 |  0.997 |  0.999 |   1.000 |
| Keci-AllvsAll-ASWA    | val   | 0.852 |  0.793 |  0.896 |   0.955 |
| Keci-AllvsAll-ASWA    | test  | 0.848 |  0.787 |  0.886 |   0.951 |



```--embedding_dim 256 --num_epochs 300 --batch_size 1024 --optim Adam 0.1``` leading to 58.1K params.
Observations: 
+ A severe overfitting. 
+ AllvsAll improves the generalization more than KvsAll does
+ ASWA improves the generalization more than SWA does

```bash
dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "KvsSample" --embedding_dim 256 --num_epochs 100 --batch_size 32 --num_core 10
# Epoch 99: 100%|███████████| 13/13 [00:00<00:00, 29.56it/s, loss_step=6.46e-6, loss_epoch=8.35e-6]
# *** Save Trained Model ***
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.33358895705521474, 'H@3': 0.5253067484662577, 'H@10': 0.7576687116564417, 'MRR': 0.46992150194876076}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.3320726172465961, 'H@3': 0.5098335854765507, 'H@10': 0.7594553706505295, 'MRR': 0.4633434701052234}
```
Increasing cores increases the runtimes if there is a preprocessing step at the batch generation.
```bash
dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "KvsAll" --embedding_dim 256 --num_epochs 100 --batch_size 32
# Epoch 99: 100%|██████████| 13/13 [00:00<00:00, 101.94it/s, loss_step=8.11e-6, loss_epoch=8.92e-6]
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.348159509202454, 'H@3': 0.5659509202453987, 'H@10': 0.7883435582822086, 'MRR': 0.4912162082105331}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.34568835098335854, 'H@3': 0.5544629349470499, 'H@10': 0.7776096822995462, 'MRR': 0.48692617590763265}
```

```bash
dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "AllvsAll" --embedding_dim 256 --num_epochs 100 --batch_size 32
# Epoch 99: 100%|██████████████| 98/98 [00:01<00:00, 88.95it/s, loss_step=0.000, loss_epoch=0.0655]
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9976993865030674, 'H@3': 0.9997124233128835, 'H@10': 0.9999041411042945, 'MRR': 0.9987183437408705}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.3197852760736196, 'H@3': 0.5398773006134969, 'H@10': 0.7714723926380368, 'MRR': 0.46912531544840963}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.329803328290469, 'H@3': 0.5711043872919819, 'H@10': 0.7934947049924357, 'MRR': 0.4858500337837166}
```
In KvsAll and AllvsAll, a single data point **z=(x,y)** corresponds to a tuple of input indices **x** and multi-label output vector **y**.
**x** is a tuple of indices of a unique entity and relation pair.
**y** contains a binary vector of size of the number of unique entities.

To mitigate the rate of overfitting, many regularization techniques can be applied ,e.g.,
Stochastic Weight Averaging (SWA), Adaptive Stochastic Weight Averaging (ASWA), or Dropout.
Use ```--swa``` to apply Stochastic Weight Averaging
```bash
dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "KvsAll" --embedding_dim 256 --num_epochs 100 --batch_size 32 --swa
# Epoch 99: 100%|███████████| 13/13 [00:00<00:00, 85.61it/s, loss_step=8.11e-6, loss_epoch=8.92e-6]
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.45858895705521474, 'H@3': 0.6510736196319018, 'H@10': 0.8458588957055214, 'MRR': 0.5845156794070833}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.4636913767019667, 'H@3': 0.651285930408472, 'H@10': 0.8456883509833586, 'MRR': 0.5877221440365971}
# Total Runtime: 25.417 seconds
```
Use ```--adaptive_swa``` to apply Adaptive Stochastic Weight Averaging. Currently, ASWA should not be used with DDP on multi GPUs.
We are working on it.
```bash
CUDA_VISIBLE_DEVICES=0 dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "KvsAll" --embedding_dim 256 --num_epochs 100 --batch_size 32 --adaptive_swa
# Epoch 99: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 93.86it/s, loss_step=0.0978, loss_epoch=0.143]
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9974118098159509, 'H@3': 0.9992331288343558, 'H@10': 0.9996165644171779, 'MRR': 0.9983922084274367}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.7668711656441718, 'H@3': 0.8696319018404908, 'H@10': 0.9440184049079755, 'MRR': 0.828767705987023}
# Evaluate Keci on Test set: Evaluate Keci on Test set
#{'H@1': 0.7844175491679274, 'H@3': 0.8888048411497731, 'H@10': 0.9546142208774584, 'MRR': 0.8460991515345323}
```
```bash
CUDA_VISIBLE_DEVICES=0 dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "KvsAll" --embedding_dim 256 --input_dropout_rate 0.1 --num_epochs 100 --batch_size 32 --adaptive_swa
# Epoch 99: 100%|██████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 93.49it/s, loss_step=0.600, loss_epoch=0.553]
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9970283742331288, 'H@3': 0.9992331288343558, 'H@10': 0.999808282208589, 'MRR': 0.9981489117237927}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.8473926380368099, 'H@3': 0.9049079754601227, 'H@10': 0.9470858895705522, 'MRR': 0.8839172788777631}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.8381240544629349, 'H@3': 0.9167927382753404, 'H@10': 0.9568835098335855, 'MRR': 0.8829572716873321}

CUDA_VISIBLE_DEVICES=0 dicee --dataset_dir "KGs/UMLS" --model "Keci" --p 0 --q 1 --trainer "PL" --scoring_technique "KvsAll" --embedding_dim 256 --input_dropout_rate 0.2 --num_epochs 100 --batch_size 32 --adaptive_swa
# Epoch 99: 100%|██████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 94.43it/s, loss_step=0.108, loss_epoch=0.111]
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9818826687116564, 'H@3': 0.9942484662576687, 'H@10': 0.9972200920245399, 'MRR': 0.9885307022708297}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.8581288343558282, 'H@3': 0.9156441717791411, 'H@10': 0.9447852760736196, 'MRR': 0.8930935122236525}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.8494704992435703, 'H@3': 0.9334341906202723, 'H@10': 0.9667170953101362, 'MRR': 0.8959156201718665}
```

</details>

#### Countries-S1 ####

<details> <summary> To see the results </summary>

|                        |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|------------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll         | val   | 0.218 |  0.104 |  0.250 |   0.479 |
| ComplEx-KvsAll         | test  | 0.184 |  0.104 |  0.167 |   0.375 |
| ComplEx-AllvsAll       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll       | val   | 0.160 |  0.083 |  0.167 |   0.271 |
| ComplEx-AllvsAll       | test  | 0.131 |  0.042 |  0.146 |   0.312 |
| ComplEx-KvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll-SWA     | val   | 0.228 |  0.125 |  0.229 |   0.479 |
| ComplEx-KvsAll-SWA     | test  | 0.184 |  0.104 |  0.188 |   0.375 |
| ComplEx-AllvsAll-SWA   | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll-SWA   | val   | 0.143 |  0.062 |  0.125 |   0.292 |
| ComplEx-AllvsAll-SWA   | test  | 0.109 |  0.021 |  0.125 |   0.292 |
| ComplEx-KvsAll-ASWA    | train | 0.127 |  0.055 |  0.131 |   0.261 |
| ComplEx-KvsAll-ASWA    | val   | 0.217 |  0.083 |  0.271 |   0.458 |
| ComplEx-KvsAll-ASWA    | test  | 0.249 |  0.104 |  0.396 |   0.542 |
| ComplEx-AllvsAll-ASWA  | train | 0.068 |  0.029 |  0.060 |   0.115 |
| ComplEx-AllvsAll-ASWA  | val   | 0.232 |  0.167 |  0.250 |   0.312 |
| ComplEx-AllvsAll-ASWA  | test  | 0.210 |  0.125 |  0.250 |   0.354 |
| Keci-KvsAll            | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll            | val   | 0.095 |  0.021 |  0.104 |   0.188 |
| Keci-KvsAll            | test  | 0.162 |  0.062 |  0.229 |   0.292 |
| Keci-AllvsAll          | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll          | val   | 0.206 |  0.125 |  0.208 |   0.333 |
| Keci-AllvsAll          | test  | 0.118 |  0.021 |  0.083 |   0.354 |
| Keci-KvsAll-SWA        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll-SWA        | val   | 0.143 |  0.083 |  0.104 |   0.271 |
| Keci-KvsAll-SWA        | test  | 0.198 |  0.104 |  0.271 |   0.354 |
| Keci-AllvsAll-SWA      | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll-SWA      | val   | 0.220 |  0.146 |  0.208 |   0.396 |
| Keci-AllvsAll-SWA      | test  | 0.163 |  0.062 |  0.188 |   0.375 |
| Keci-KvsAll-ASWA       | train | 0.991 |  0.984 |  1.000 |   1.000 |
| Keci-KvsAll-ASWA       | val   | 0.286 |  0.167 |  0.333 |   0.562 |
| Keci-KvsAll-ASWA       | test  | 0.370 |  0.271 |  0.396 |   0.688 |
| Keci-AllvsAll-ASWA     | train | 1.000 |  0.999 |  1.000 |   1.000 |
| Keci-AllvsAll-ASWA     | val   | 0.258 |  0.188 |  0.250 |   0.438 |
| Keci-AllvsAll-ASWA     | test  | 0.264 |  0.167 |  0.292 |   0.396 |
| QMult-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-KvsAll           | val   | 0.144 |  0.104 |  0.104 |   0.167 |
| QMult-KvsAll           | test  | 0.161 |  0.062 |  0.146 |   0.375 |
| QMult-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-AllvsAll         | val   | 0.111 |  0.062 |  0.104 |   0.229 |
| QMult-AllvsAll         | test  | 0.146 |  0.062 |  0.146 |   0.250 |
| QMult-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-KvsAll-SWA       | val   | 0.106 |  0.042 |  0.104 |   0.146 |
| QMult-KvsAll-SWA       | test  | 0.148 |  0.062 |  0.146 |   0.292 |
| QMult-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-AllvsAll-SWA     | val   | 0.117 |  0.062 |  0.125 |   0.229 |
| QMult-AllvsAll-SWA     | test  | 0.105 |  0.021 |  0.104 |   0.271 |
| QMult-KvsAll-ASWA      | train | 0.190 |  0.105 |  0.199 |   0.356 |
| QMult-KvsAll-ASWA      | val   | 0.294 |  0.167 |  0.396 |   0.542 |
| QMult-KvsAll-ASWA      | test  | 0.198 |  0.062 |  0.250 |   0.500 |
| QMult-AllvsAll-ASWA    | train | 0.256 |  0.150 |  0.284 |   0.455 |
| QMult-AllvsAll-ASWA    | val   | 0.169 |  0.062 |  0.188 |   0.396 |
| QMult-AllvsAll-ASWA    | test  | 0.111 |  0.021 |  0.083 |   0.333 |
| OMult-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-KvsAll           | val   | 0.041 |  0.000 |  0.021 |   0.104 |
| OMult-KvsAll           | test  | 0.033 |  0.000 |  0.000 |   0.042 |
| OMult-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-AllvsAll         | val   | 0.029 |  0.000 |  0.000 |   0.104 |
| OMult-AllvsAll         | test  | 0.025 |  0.000 |  0.000 |   0.062 |
| OMult-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-KvsAll-SWA       | val   | 0.031 |  0.000 |  0.000 |   0.083 |
| OMult-KvsAll-SWA       | test  | 0.031 |  0.000 |  0.000 |   0.042 |
| OMult-AllvsAll-SWA     | train | 0.999 |  0.999 |  1.000 |   1.000 |
| OMult-AllvsAll-SWA     | val   | 0.027 |  0.000 |  0.000 |   0.042 |
| OMult-AllvsAll-SWA     | test  | 0.023 |  0.000 |  0.000 |   0.062 |
| OMult-KvsAll-ASWA      | train | 0.146 |  0.069 |  0.158 |   0.280 |
| OMult-KvsAll-ASWA      | val   | 0.232 |  0.146 |  0.250 |   0.438 |
| OMult-KvsAll-ASWA      | test  | 0.209 |  0.083 |  0.312 |   0.417 |
| OMult-AllvsAll-ASWA    | train | 0.390 |  0.265 |  0.433 |   0.636 |
| OMult-AllvsAll-ASWA    | val   | 0.109 |  0.062 |  0.083 |   0.208 |
| OMult-AllvsAll-ASWA    | test  | 0.075 |  0.000 |  0.062 |   0.208 |
| DistMult-KvsAll        | train | 0.998 |  0.996 |  1.000 |   1.000 |
| DistMult-KvsAll        | val   | 0.168 |  0.104 |  0.146 |   0.312 |
| DistMult-KvsAll        | test  | 0.107 |  0.062 |  0.104 |   0.188 |
| DistMult-AllvsAll      | train | 0.977 |  0.961 |  0.991 |   0.997 |
| DistMult-AllvsAll      | val   | 0.090 |  0.021 |  0.083 |   0.250 |
| DistMult-AllvsAll      | test  | 0.067 |  0.000 |  0.062 |   0.229 |
| DistMult-KvsAll-SWA    | train | 0.999 |  0.999 |  1.000 |   1.000 |
| DistMult-KvsAll-SWA    | val   | 0.092 |  0.021 |  0.083 |   0.250 |
| DistMult-KvsAll-SWA    | test  | 0.062 |  0.000 |  0.042 |   0.208 |
| DistMult-AllvsAll-SWA  | train | 0.958 |  0.923 |  0.991 |   0.996 |
| DistMult-AllvsAll-SWA  | val   | 0.128 |  0.042 |  0.146 |   0.354 |
| DistMult-AllvsAll-SWA  | test  | 0.129 |  0.062 |  0.083 |   0.354 |
| DistMult-KvsAll-ASWA   | train | 0.959 |  0.930 |  0.984 |   0.997 |
| DistMult-KvsAll-ASWA   | val   | 0.222 |  0.125 |  0.229 |   0.417 |
| DistMult-KvsAll-ASWA   | test  | 0.140 |  0.062 |  0.104 |   0.292 |
| DistMult-AllvsAll-ASWA | train | 0.933 |  0.887 |  0.983 |   0.999 |
| DistMult-AllvsAll-ASWA | val   | 0.299 |  0.208 |  0.354 |   0.438 |
| DistMult-AllvsAll-ASWA | test  | 0.195 |  0.083 |  0.250 |   0.375 |
| TransE-KvsAll          | train | 0.505 |  0.233 |  0.738 |   0.923 |
| TransE-KvsAll          | val   | 0.636 |  0.375 |  0.896 |   0.979 |
| TransE-KvsAll          | test  | 0.686 |  0.438 |  0.979 |   1.000 |
| TransE-AllvsAll        | train | 0.497 |  0.314 |  0.599 |   0.850 |
| TransE-AllvsAll        | val   | 0.798 |  0.646 |  0.958 |   1.000 |
| TransE-AllvsAll        | test  | 0.843 |  0.729 |  0.938 |   1.000 |
| TransE-KvsAll-SWA      | train | 0.653 |  0.381 |  0.918 |   0.992 |
| TransE-KvsAll-SWA      | val   | 0.844 |  0.688 |  1.000 |   1.000 |
| TransE-KvsAll-SWA      | test  | 0.872 |  0.750 |  1.000 |   1.000 |
| TransE-AllvsAll-SWA    | train | 0.622 |  0.372 |  0.859 |   0.976 |
| TransE-AllvsAll-SWA    | val   | 0.819 |  0.646 |  1.000 |   1.000 |
| TransE-AllvsAll-SWA    | test  | 0.868 |  0.750 |  1.000 |   1.000 |
| TransE-KvsAll-ASWA     | train | 0.651 |  0.367 |  0.934 |   0.982 |
| TransE-KvsAll-ASWA     | val   | 0.885 |  0.771 |  1.000 |   1.000 |
| TransE-KvsAll-ASWA     | test  | 0.858 |  0.729 |  0.979 |   1.000 |
| TransE-AllvsAll-ASWA   | train | 0.603 |  0.360 |  0.817 |   0.972 |
| TransE-AllvsAll-ASWA   | val   | 0.927 |  0.854 |  1.000 |   1.000 |
| TransE-AllvsAll-ASWA   | test  | 0.938 |  0.875 |  1.000 |   1.000 |
| DeCaL-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-KvsAll           | val   | 0.137 |  0.062 |  0.125 |   0.250 |
| DeCaL-KvsAll           | test  | 0.224 |  0.125 |  0.229 |   0.396 |
| DeCaL-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll         | val   | 0.129 |  0.042 |  0.146 |   0.250 |
| DeCaL-AllvsAll         | test  | 0.144 |  0.062 |  0.146 |   0.292 |
| DeCaL-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-KvsAll-SWA       | val   | 0.152 |  0.083 |  0.146 |   0.333 |
| DeCaL-KvsAll-SWA       | test  | 0.216 |  0.083 |  0.292 |   0.354 |
| DeCaL-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll-SWA     | val   | 0.112 |  0.021 |  0.125 |   0.229 |
| DeCaL-AllvsAll-SWA     | test  | 0.134 |  0.042 |  0.125 |   0.312 |
| DeCaL-KvsAll-ASWA      | train | 0.995 |  0.993 |  0.996 |   0.998 |
| DeCaL-KvsAll-ASWA      | val   | 0.263 |  0.146 |  0.333 |   0.500 |
| DeCaL-KvsAll-ASWA      | test  | 0.251 |  0.104 |  0.312 |   0.521 |
| DeCaL-AllvsAll-ASWA    | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll-ASWA    | val   | 0.320 |  0.229 |  0.333 |   0.562 |
| DeCaL-AllvsAll-ASWA    | test  | 0.286 |  0.208 |  0.292 |   0.458 |

`--embedding_dim 256 --num_epochs 100 --batch_size 32`

</details>

#### Countries-S2 ####

<details> <summary> To see the results </summary>

|                        |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|------------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll         | val   | 0.195 |  0.104 |  0.229 |   0.354 |
| ComplEx-KvsAll         | test  | 0.137 |  0.062 |  0.146 |   0.312 |
| ComplEx-AllvsAll       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll       | val   | 0.148 |  0.062 |  0.167 |   0.312 |
| ComplEx-AllvsAll       | test  | 0.153 |  0.083 |  0.167 |   0.271 |
| ComplEx-KvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll-SWA     | val   | 0.176 |  0.083 |  0.188 |   0.354 |
| ComplEx-KvsAll-SWA     | test  | 0.138 |  0.062 |  0.146 |   0.312 |
| ComplEx-AllvsAll-SWA   | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll-SWA   | val   | 0.146 |  0.083 |  0.125 |   0.312 |
| ComplEx-AllvsAll-SWA   | test  | 0.152 |  0.083 |  0.167 |   0.271 |
| ComplEx-KvsAll-ASWA    | train | 0.113 |  0.042 |  0.117 |   0.255 |
| ComplEx-KvsAll-ASWA    | val   | 0.237 |  0.125 |  0.271 |   0.479 |
| ComplEx-KvsAll-ASWA    | test  | 0.296 |  0.188 |  0.375 |   0.521 |
| ComplEx-AllvsAll-ASWA  | train | 0.997 |  0.996 |  0.997 |   0.999 |
| ComplEx-AllvsAll-ASWA  | val   | 0.186 |  0.083 |  0.250 |   0.354 |
| ComplEx-AllvsAll-ASWA  | test  | 0.178 |  0.083 |  0.188 |   0.375 |
| Keci-KvsAll            | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll            | val   | 0.209 |  0.146 |  0.188 |   0.375 |
| Keci-KvsAll            | test  | 0.204 |  0.104 |  0.188 |   0.458 |
| Keci-AllvsAll          | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll          | val   | 0.124 |  0.083 |  0.083 |   0.188 |
| Keci-AllvsAll          | test  | 0.076 |  0.000 |  0.042 |   0.229 |
| Keci-KvsAll-SWA        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll-SWA        | val   | 0.194 |  0.125 |  0.167 |   0.354 |
| Keci-KvsAll-SWA        | test  | 0.200 |  0.104 |  0.188 |   0.396 |
| Keci-AllvsAll-SWA      | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll-SWA      | val   | 0.121 |  0.062 |  0.104 |   0.208 |
| Keci-AllvsAll-SWA      | test  | 0.108 |  0.021 |  0.104 |   0.292 |
| Keci-KvsAll-ASWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll-ASWA       | val   | 0.294 |  0.229 |  0.271 |   0.417 |
| Keci-KvsAll-ASWA       | test  | 0.253 |  0.146 |  0.271 |   0.479 |
| Keci-AllvsAll-ASWA     | train | 0.927 |  0.895 |  0.952 |   0.977 |
| Keci-AllvsAll-ASWA     | val   | 0.246 |  0.125 |  0.292 |   0.500 |
| Keci-AllvsAll-ASWA     | test  | 0.197 |  0.104 |  0.208 |   0.438 |
| QMult-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-KvsAll           | val   | 0.054 |  0.000 |  0.042 |   0.146 |
| QMult-KvsAll           | test  | 0.109 |  0.042 |  0.104 |   0.229 |
| QMult-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-AllvsAll         | val   | 0.146 |  0.104 |  0.125 |   0.208 |
| QMult-AllvsAll         | test  | 0.076 |  0.021 |  0.083 |   0.125 |
| QMult-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-KvsAll-SWA       | val   | 0.051 |  0.000 |  0.021 |   0.125 |
| QMult-KvsAll-SWA       | test  | 0.120 |  0.062 |  0.125 |   0.229 |
| QMult-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-AllvsAll-SWA     | val   | 0.141 |  0.104 |  0.125 |   0.167 |
| QMult-AllvsAll-SWA     | test  | 0.066 |  0.021 |  0.042 |   0.125 |
| QMult-KvsAll-ASWA      | train | 0.157 |  0.083 |  0.161 |   0.290 |
| QMult-KvsAll-ASWA      | val   | 0.185 |  0.062 |  0.250 |   0.417 |
| QMult-KvsAll-ASWA      | test  | 0.282 |  0.208 |  0.292 |   0.479 |
| QMult-AllvsAll-ASWA    | train | 0.169 |  0.083 |  0.186 |   0.328 |
| QMult-AllvsAll-ASWA    | val   | 0.138 |  0.062 |  0.146 |   0.312 |
| QMult-AllvsAll-ASWA    | test  | 0.168 |  0.062 |  0.208 |   0.375 |
| OMult-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-KvsAll           | val   | 0.068 |  0.021 |  0.083 |   0.146 |
| OMult-KvsAll           | test  | 0.037 |  0.000 |  0.042 |   0.062 |
| OMult-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-AllvsAll         | val   | 0.024 |  0.000 |  0.000 |   0.042 |
| OMult-AllvsAll         | test  | 0.036 |  0.000 |  0.021 |   0.083 |
| OMult-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-KvsAll-SWA       | val   | 0.066 |  0.021 |  0.062 |   0.125 |
| OMult-KvsAll-SWA       | test  | 0.038 |  0.000 |  0.042 |   0.083 |
| OMult-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-AllvsAll-SWA     | val   | 0.024 |  0.000 |  0.000 |   0.062 |
| OMult-AllvsAll-SWA     | test  | 0.035 |  0.000 |  0.021 |   0.104 |
| OMult-KvsAll-ASWA      | train | 0.117 |  0.051 |  0.113 |   0.228 |
| OMult-KvsAll-ASWA      | val   | 0.150 |  0.062 |  0.167 |   0.292 |
| OMult-KvsAll-ASWA      | test  | 0.165 |  0.083 |  0.208 |   0.354 |
| OMult-AllvsAll-ASWA    | train | 0.264 |  0.154 |  0.286 |   0.490 |
| OMult-AllvsAll-ASWA    | val   | 0.097 |  0.042 |  0.062 |   0.250 |
| OMult-AllvsAll-ASWA    | test  | 0.099 |  0.042 |  0.083 |   0.208 |
| DistMult-KvsAll        | train | 0.999 |  0.999 |  1.000 |   1.000 |
| DistMult-KvsAll        | val   | 0.101 |  0.021 |  0.125 |   0.229 |
| DistMult-KvsAll        | test  | 0.148 |  0.083 |  0.188 |   0.271 |
| DistMult-AllvsAll      | train | 0.963 |  0.930 |  0.993 |   1.000 |
| DistMult-AllvsAll      | val   | 0.231 |  0.146 |  0.292 |   0.375 |
| DistMult-AllvsAll      | test  | 0.202 |  0.083 |  0.271 |   0.417 |
| DistMult-KvsAll-SWA    | train | 0.999 |  0.998 |  1.000 |   1.000 |
| DistMult-KvsAll-SWA    | val   | 0.121 |  0.042 |  0.167 |   0.271 |
| DistMult-KvsAll-SWA    | test  | 0.162 |  0.062 |  0.208 |   0.375 |
| DistMult-AllvsAll-SWA  | train | 0.969 |  0.944 |  0.992 |   0.998 |
| DistMult-AllvsAll-SWA  | val   | 0.116 |  0.021 |  0.167 |   0.271 |
| DistMult-AllvsAll-SWA  | test  | 0.122 |  0.062 |  0.125 |   0.229 |
| DistMult-KvsAll-ASWA   | train | 0.184 |  0.092 |  0.193 |   0.379 |
| DistMult-KvsAll-ASWA   | val   | 0.244 |  0.083 |  0.333 |   0.542 |
| DistMult-KvsAll-ASWA   | test  | 0.238 |  0.104 |  0.312 |   0.500 |
| DistMult-AllvsAll-ASWA | train | 0.961 |  0.927 |  0.993 |   0.998 |
| DistMult-AllvsAll-ASWA | val   | 0.264 |  0.208 |  0.292 |   0.354 |
| DistMult-AllvsAll-ASWA | test  | 0.214 |  0.083 |  0.292 |   0.396 |
| TransE-KvsAll          | train | 0.544 |  0.238 |  0.827 |   0.950 |
| TransE-KvsAll          | val   | 0.549 |  0.250 |  0.854 |   0.979 |
| TransE-KvsAll          | test  | 0.561 |  0.250 |  0.854 |   1.000 |
| TransE-AllvsAll        | train | 0.465 |  0.233 |  0.629 |   0.880 |
| TransE-AllvsAll        | val   | 0.515 |  0.250 |  0.792 |   0.979 |
| TransE-AllvsAll        | test  | 0.510 |  0.229 |  0.708 |   0.979 |
| TransE-KvsAll-SWA      | train | 0.638 |  0.347 |  0.925 |   0.991 |
| TransE-KvsAll-SWA      | val   | 0.684 |  0.458 |  0.979 |   1.000 |
| TransE-KvsAll-SWA      | test  | 0.653 |  0.396 |  0.979 |   1.000 |
| TransE-AllvsAll-SWA    | train | 0.599 |  0.330 |  0.857 |   0.984 |
| TransE-AllvsAll-SWA    | val   | 0.677 |  0.458 |  0.917 |   1.000 |
| TransE-AllvsAll-SWA    | test  | 0.660 |  0.417 |  0.938 |   1.000 |
| TransE-KvsAll-ASWA     | train | 0.643 |  0.344 |  0.940 |   0.994 |
| TransE-KvsAll-ASWA     | val   | 0.688 |  0.458 |  1.000 |   1.000 |
| TransE-KvsAll-ASWA     | test  | 0.658 |  0.375 |  0.979 |   1.000 |
| TransE-AllvsAll-ASWA   | train | 0.415 |  0.179 |  0.579 |   0.840 |
| TransE-AllvsAll-ASWA   | val   | 0.756 |  0.583 |  0.958 |   0.979 |
| TransE-AllvsAll-ASWA   | test  | 0.720 |  0.500 |  0.958 |   1.000 |
| DeCaL-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-KvsAll           | val   | 0.193 |  0.125 |  0.188 |   0.354 |
| DeCaL-KvsAll           | test  | 0.188 |  0.083 |  0.208 |   0.396 |
| DeCaL-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll         | val   | 0.148 |  0.104 |  0.104 |   0.292 |
| DeCaL-AllvsAll         | test  | 0.243 |  0.146 |  0.271 |   0.417 |
| DeCaL-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-KvsAll-SWA       | val   | 0.184 |  0.104 |  0.208 |   0.333 |
| DeCaL-KvsAll-SWA       | test  | 0.216 |  0.125 |  0.229 |   0.396 |
| DeCaL-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll-SWA     | val   | 0.125 |  0.042 |  0.146 |   0.312 |
| DeCaL-AllvsAll-SWA     | test  | 0.227 |  0.146 |  0.271 |   0.438 |
| DeCaL-KvsAll-ASWA      | train | 0.997 |  0.996 |  0.999 |   1.000 |
| DeCaL-KvsAll-ASWA      | val   | 0.297 |  0.229 |  0.312 |   0.396 |
| DeCaL-KvsAll-ASWA      | test  | 0.183 |  0.083 |  0.208 |   0.354 |
| DeCaL-AllvsAll-ASWA    | train | 0.978 |  0.967 |  0.986 |   0.997 |
| DeCaL-AllvsAll-ASWA    | val   | 0.246 |  0.167 |  0.271 |   0.417 |
| DeCaL-AllvsAll-ASWA    | test  | 0.215 |  0.125 |  0.229 |   0.396 |

`--embedding_dim 256 --num_epochs 100 --batch_size 32`

</details> 

#### Countries-S3 ####

<details> <summary> To see the results </summary>

|                        |       |   MRR | Hits@1 | Hits@3 | Hits@10 |
|------------------------|-------|------:|-------:|-------:|--------:|
| ComplEx-KvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll         | val   | 0.144 |  0.083 |  0.146 |   0.229 |
| ComplEx-KvsAll         | test  | 0.061 |  0.021 |  0.042 |   0.125 |
| ComplEx-AllvsAll       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll       | val   | 0.121 |  0.062 |  0.146 |   0.188 |
| ComplEx-AllvsAll       | test  | 0.058 |  0.021 |  0.062 |   0.125 |
| ComplEx-KvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll-SWA     | val   | 0.153 |  0.104 |  0.146 |   0.250 |
| ComplEx-KvsAll-SWA     | test  | 0.058 |  0.021 |  0.042 |   0.104 |
| ComplEx-AllvsAll-SWA   | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-AllvsAll-SWA   | val   | 0.116 |  0.062 |  0.125 |   0.208 |
| ComplEx-AllvsAll-SWA   | test  | 0.060 |  0.021 |  0.062 |   0.146 |
| ComplEx-KvsAll-ASWA    | train | 1.000 |  1.000 |  1.000 |   1.000 |
| ComplEx-KvsAll-ASWA    | val   | 0.151 |  0.083 |  0.167 |   0.271 |
| ComplEx-KvsAll-ASWA    | test  | 0.048 |  0.000 |  0.021 |   0.146 |
| ComplEx-AllvsAll-ASWA  | train | 0.999 |  0.999 |  0.999 |   0.999 |
| ComplEx-AllvsAll-ASWA  | val   | 0.120 |  0.083 |  0.083 |   0.229 |
| ComplEx-AllvsAll-ASWA  | test  | 0.076 |  0.042 |  0.042 |   0.167 |
| Keci-KvsAll            | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll            | val   | 0.079 |  0.042 |  0.062 |   0.104 |
| Keci-KvsAll            | test  | 0.068 |  0.021 |  0.042 |   0.188 |
| Keci-AllvsAll          | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll          | val   | 0.048 |  0.021 |  0.042 |   0.062 |
| Keci-AllvsAll          | test  | 0.094 |  0.042 |  0.062 |   0.188 |
| Keci-KvsAll-SWA        | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-KvsAll-SWA        | val   | 0.081 |  0.042 |  0.062 |   0.146 |
| Keci-KvsAll-SWA        | test  | 0.074 |  0.021 |  0.042 |   0.208 |
| Keci-AllvsAll-SWA      | train | 1.000 |  1.000 |  1.000 |   1.000 |
| Keci-AllvsAll-SWA      | val   | 0.050 |  0.021 |  0.042 |   0.062 |
| Keci-AllvsAll-SWA      | test  | 0.101 |  0.042 |  0.104 |   0.229 |
| Keci-KvsAll-ASWA       | train | 0.910 |  0.873 |  0.937 |   0.973 |
| Keci-KvsAll-ASWA       | val   | 0.200 |  0.104 |  0.229 |   0.396 |
| Keci-KvsAll-ASWA       | test  | 0.207 |  0.104 |  0.229 |   0.417 |
| Keci-AllvsAll-ASWA     | train | 0.487 |  0.383 |  0.530 |   0.697 |
| Keci-AllvsAll-ASWA     | val   | 0.117 |  0.062 |  0.083 |   0.229 |
| Keci-AllvsAll-ASWA     | test  | 0.180 |  0.125 |  0.167 |   0.292 |
| QMult-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-KvsAll           | val   | 0.088 |  0.042 |  0.083 |   0.125 |
| QMult-KvsAll           | test  | 0.092 |  0.021 |  0.042 |   0.312 |
| QMult-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-AllvsAll         | val   | 0.099 |  0.042 |  0.062 |   0.208 |
| QMult-AllvsAll         | test  | 0.094 |  0.062 |  0.062 |   0.146 |
| QMult-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-KvsAll-SWA       | val   | 0.085 |  0.042 |  0.062 |   0.125 |
| QMult-KvsAll-SWA       | test  | 0.091 |  0.021 |  0.062 |   0.292 |
| QMult-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| QMult-AllvsAll-SWA     | val   | 0.096 |  0.042 |  0.062 |   0.208 |
| QMult-AllvsAll-SWA     | test  | 0.084 |  0.042 |  0.062 |   0.125 |
| QMult-KvsAll-ASWA      | train | 0.138 |  0.071 |  0.136 |   0.263 |
| QMult-KvsAll-ASWA      | val   | 0.149 |  0.062 |  0.188 |   0.250 |
| QMult-KvsAll-ASWA      | test  | 0.152 |  0.083 |  0.125 |   0.312 |
| QMult-AllvsAll-ASWA    | train | 0.379 |  0.269 |  0.413 |   0.606 |
| QMult-AllvsAll-ASWA    | val   | 0.123 |  0.083 |  0.104 |   0.188 |
| QMult-AllvsAll-ASWA    | test  | 0.124 |  0.062 |  0.104 |   0.208 |
| OMult-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-KvsAll           | val   | 0.054 |  0.021 |  0.021 |   0.125 |
| OMult-KvsAll           | test  | 0.041 |  0.000 |  0.021 |   0.104 |
| OMult-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-AllvsAll         | val   | 0.043 |  0.000 |  0.021 |   0.104 |
| OMult-AllvsAll         | test  | 0.089 |  0.021 |  0.104 |   0.229 |
| OMult-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| OMult-KvsAll-SWA       | val   | 0.054 |  0.021 |  0.021 |   0.125 |
| OMult-KvsAll-SWA       | test  | 0.040 |  0.000 |  0.021 |   0.104 |
| OMult-AllvsAll-SWA     | train | 0.999 |  0.999 |  0.999 |   0.999 |
| OMult-AllvsAll-SWA     | val   | 0.044 |  0.000 |  0.042 |   0.104 |
| OMult-AllvsAll-SWA     | test  | 0.097 |  0.042 |  0.104 |   0.229 |
| OMult-KvsAll-ASWA      | train | 0.321 |  0.224 |  0.346 |   0.516 |
| OMult-KvsAll-ASWA      | val   | 0.139 |  0.083 |  0.146 |   0.229 |
| OMult-KvsAll-ASWA      | test  | 0.089 |  0.000 |  0.104 |   0.271 |
| OMult-AllvsAll-ASWA    | train | 0.234 |  0.139 |  0.256 |   0.421 |
| OMult-AllvsAll-ASWA    | val   | 0.105 |  0.000 |  0.146 |   0.312 |
| OMult-AllvsAll-ASWA    | test  | 0.058 |  0.000 |  0.042 |   0.188 |
| DistMult-KvsAll        | train | 0.999 |  0.999 |  1.000 |   1.000 |
| DistMult-KvsAll        | val   | 0.170 |  0.083 |  0.208 |   0.292 |
| DistMult-KvsAll        | test  | 0.207 |  0.146 |  0.229 |   0.292 |
| DistMult-AllvsAll      | train | 0.969 |  0.943 |  0.995 |   0.999 |
| DistMult-AllvsAll      | val   | 0.200 |  0.125 |  0.188 |   0.354 |
| DistMult-AllvsAll      | test  | 0.188 |  0.104 |  0.208 |   0.375 |
| DistMult-KvsAll-SWA    | train | 1.000 |  0.999 |  1.000 |   1.000 |
| DistMult-KvsAll-SWA    | val   | 0.142 |  0.062 |  0.167 |   0.271 |
| DistMult-KvsAll-SWA    | test  | 0.227 |  0.167 |  0.250 |   0.312 |
| DistMult-AllvsAll-SWA  | train | 0.963 |  0.930 |  0.994 |   0.998 |
| DistMult-AllvsAll-SWA  | val   | 0.120 |  0.042 |  0.167 |   0.250 |
| DistMult-AllvsAll-SWA  | test  | 0.101 |  0.042 |  0.062 |   0.250 |
| DistMult-KvsAll-ASWA   | train | 0.985 |  0.973 |  0.996 |   0.998 |
| DistMult-KvsAll-ASWA   | val   | 0.215 |  0.125 |  0.229 |   0.417 |
| DistMult-KvsAll-ASWA   | test  | 0.248 |  0.188 |  0.250 |   0.396 |
| DistMult-AllvsAll-ASWA | train | 0.958 |  0.920 |  0.994 |   0.998 |
| DistMult-AllvsAll-ASWA | val   | 0.238 |  0.188 |  0.229 |   0.312 |
| DistMult-AllvsAll-ASWA | test  | 0.165 |  0.104 |  0.208 |   0.271 |
| TransE-KvsAll          | train | 0.505 |  0.188 |  0.792 |   0.942 |
| TransE-KvsAll          | val   | 0.120 |  0.021 |  0.083 |   0.354 |
| TransE-KvsAll          | test  | 0.123 |  0.000 |  0.125 |   0.417 |
| TransE-AllvsAll        | train | 0.424 |  0.169 |  0.592 |   0.906 |
| TransE-AllvsAll        | val   | 0.115 |  0.000 |  0.104 |   0.354 |
| TransE-AllvsAll        | test  | 0.116 |  0.000 |  0.083 |   0.375 |
| TransE-KvsAll-SWA      | train | 0.606 |  0.286 |  0.920 |   0.991 |
| TransE-KvsAll-SWA      | val   | 0.130 |  0.000 |  0.125 |   0.375 |
| TransE-KvsAll-SWA      | test  | 0.151 |  0.000 |  0.208 |   0.479 |
| TransE-AllvsAll-SWA    | train | 0.579 |  0.280 |  0.870 |   0.983 |
| TransE-AllvsAll-SWA    | val   | 0.161 |  0.000 |  0.104 |   0.646 |
| TransE-AllvsAll-SWA    | test  | 0.178 |  0.000 |  0.229 |   0.688 |
| TransE-KvsAll-ASWA     | train | 0.508 |  0.203 |  0.779 |   0.951 |
| TransE-KvsAll-ASWA     | val   | 0.169 |  0.021 |  0.146 |   0.500 |
| TransE-KvsAll-ASWA     | test  | 0.155 |  0.000 |  0.146 |   0.542 |
| TransE-AllvsAll-ASWA   | train | 0.400 |  0.155 |  0.562 |   0.849 |
| TransE-AllvsAll-ASWA   | val   | 0.255 |  0.083 |  0.333 |   0.604 |
| TransE-AllvsAll-ASWA   | test  | 0.245 |  0.042 |  0.375 |   0.625 |
| DeCaL-KvsAll           | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-KvsAll           | val   | 0.062 |  0.021 |  0.042 |   0.125 |
| DeCaL-KvsAll           | test  | 0.038 |  0.000 |  0.021 |   0.125 |
| DeCaL-AllvsAll         | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll         | val   | 0.047 |  0.000 |  0.000 |   0.146 |
| DeCaL-AllvsAll         | test  | 0.055 |  0.000 |  0.062 |   0.125 |
| DeCaL-KvsAll-SWA       | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-KvsAll-SWA       | val   | 0.076 |  0.042 |  0.042 |   0.167 |
| DeCaL-KvsAll-SWA       | test  | 0.045 |  0.000 |  0.021 |   0.125 |
| DeCaL-AllvsAll-SWA     | train | 1.000 |  1.000 |  1.000 |   1.000 |
| DeCaL-AllvsAll-SWA     | val   | 0.051 |  0.000 |  0.021 |   0.125 |
| DeCaL-AllvsAll-SWA     | test  | 0.053 |  0.000 |  0.021 |   0.125 |
| DeCaL-KvsAll-ASWA      | train | 0.689 |  0.604 |  0.740 |   0.846 |
| DeCaL-KvsAll-ASWA      | val   | 0.143 |  0.062 |  0.146 |   0.354 |
| DeCaL-KvsAll-ASWA      | test  | 0.137 |  0.062 |  0.146 |   0.271 |
| DeCaL-AllvsAll-ASWA    | train | 0.998 |  0.997 |  0.999 |   0.999 |
| DeCaL-AllvsAll-ASWA    | val   | 0.136 |  0.083 |  0.125 |   0.229 |
| DeCaL-AllvsAll-ASWA    | test  | 0.090 |  0.021 |  0.083 |   0.208 |

`--embedding_dim 256 --num_epochs 100 --batch_size 32`

</details>

## Docker
<details> <summary> Details</summary>
To build the Docker image:
```
docker build -t dice-embeddings .
```

To test the Docker image:
```
docker run --rm -v ~/.local/share/dicee/KGs:/dicee/KGs dice-embeddings ./main.py --model AConEx --embedding_dim 16
```
</details>

## How to cite
Currently, we are working on our manuscript describing our framework. 
If you really like our work and want to cite it now, feel free to choose one :) 
```
#ASWA
@inproceedings{sapkota2025parameter,
  author    = {Sapkota, Rupesh and Demir, Caglar and Sharma, Arnab and Ngonga Ngomo, Axel-Cyrille},
  title     = {Parameter Averaging in Link Prediction},
  booktitle = {Proceedings of the Knowledge Capture Conference 2025 (K-CAP '25)},
  year      = {2025},
  address   = {Dayton, OH, USA},
  publisher = {ACM},
  organization = {K-CAP},
  pages     = {1--8},
  doi       = {10.1145/3731443.3771365},
  url       = {https://papers.dice-research.org/2025/KCAP_ASWA/public.pdf},
  keywords  = {dice sailproject kiowl enexa sapkota demir ngonga sharma}
}

# DeCaL
@incollection{kamdem2024embedding,
  title={Embedding Knowledge Graphs in Degenerate Clifford Algebras},
  author={Kamdem Teyou, Louis Mozart and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={ECAI 2024},
  pages={1293--1300},
  year={2024},
  publisher={IOS Press}
}
# LFMult
@inproceedings{kamdem2024embedding,
  title={Embedding Knowledge Graphs in Function Spaces},
  author={Kamdem Teyou, Louis Mozart and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={1070--1079},
  year={2024}
}
# Keci
@inproceedings{demir2023clifford,
  title={Clifford Embeddings--A Generalized Approach for Embedding in Normed Algebras},
  author={Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={567--582},
  year={2023},
  organization={Springer}
}
# LitCQD
@inproceedings{demir2023litcqd,
  title={LitCQD: Multi-Hop Reasoning in Incomplete Knowledge Graphs with Numeric Literals},
  author={Demir, Caglar and Wiebesiek, Michel and Lu, Renzhong and Ngonga Ngomo, Axel-Cyrille and Heindorf, Stefan},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={617--633},
  year={2023},
  organization={Springer}
}
# DICE Embedding Framework
@article{demir2022hardware,
  title={Hardware-agnostic computation for large-scale knowledge graph embeddings},
  author={Demir, Caglar and Ngomo, Axel-Cyrille Ngonga},
  journal={Software Impacts},
  year={2022},
  publisher={Elsevier}
}
# KronE
@inproceedings{demir2022kronecker,
  title={Kronecker decomposition for knowledge graph embeddings},
  author={Demir, Caglar and Lienen, Julian and Ngonga Ngomo, Axel-Cyrille},
  booktitle={Proceedings of the 33rd ACM Conference on Hypertext and Social Media},
  pages={1--10},
  year={2022}
}
# QMult, OMult, ConvQ, ConvO
@InProceedings{pmlr-v157-demir21a,
  title = 	 {Convolutional Hypercomplex Embeddings for Link Prediction},
  author =       {Demir, Caglar and Moussallem, Diego and Heindorf, Stefan and Ngonga Ngomo, Axel-Cyrille},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  pages = 	 {656--671},
  year = 	 {2021},
  editor = 	 {Balasubramanian, Vineeth N. and Tsang, Ivor},
  volume = 	 {157},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--19 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v157/demir21a/demir21a.pdf},
  url = 	 {https://proceedings.mlr.press/v157/demir21a.html},
}
# ConEx
@inproceedings{demir2021convolutional,
title={Convolutional Complex Knowledge Graph Embeddings},
author={Caglar Demir and Axel-Cyrille Ngonga Ngomo},
booktitle={Eighteenth Extended Semantic Web Conference - Research Track},
year={2021},
url={https://openreview.net/forum?id=6T45-4TFqaX}}
# Shallom
@inproceedings{demir2021shallow,
  title={A shallow neural model for relation prediction},
  author={Demir, Caglar and Moussallem, Diego and Ngomo, Axel-Cyrille Ngonga},
  booktitle={2021 IEEE 15th International Conference on Semantic Computing (ICSC)},
  pages={179--182},
  year={2021},
  organization={IEEE}
```
For any questions or wishes, please contact:  ```caglar.demir@upb.de```
