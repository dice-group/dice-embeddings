## Dicee Manual

**Version:** dicee 0.1.3.2

**GitHub repository:** [https://github.com/dice-group/dice-embeddings](https://github.com/dice-group/dice-embeddings)

**Publisher and maintainer:** [Caglar Demir](https://github.com/Demirrr)

**Contact**: [caglar.demir@upb.de](mailto:caglar.demir@upb.de)

**License:** OSI Approved :: MIT License

--------------------------------------------

Dicee is a hardware-agnostic framework for large-scale knowledge graph embeddings.


Knowledge graph embedding research has mainly focused on learning continuous 
representations of knowledge graphs towards the link prediction problem. 
Recently developed frameworks can be effectively applied in a wide range 
of research-related applications. Yet, using these frameworks in real-world
applications becomes more challenging as the size of the knowledge graph 
grows

We developed the DICE Embeddings framework (dicee) to compute embeddings for large-scale knowledge graphs in a hardware-agnostic manner.
To achieve this goal, we rely on
1. **[Pandas](https://pandas.pydata.org/) & Co.** to use parallelism at preprocessing a large knowledge graph,
2. **[PyTorch](https://pytorch.org/) & Co.** to learn knowledge graph embeddings via multi-CPUs, GPUs, TPUs or computing cluster, and
3. **[Huggingface](https://huggingface.co/)** to ease the deployment of pre-trained models.

**Why [Pandas](https://pandas.pydata.org/) & Co. ?**
A large knowledge graph can be read and preprocessed (e.g. removing literals) by pandas, modin, or polars in parallel.
Through polars, a knowledge graph having more than 1 billion triples can be read in parallel fashion. 
Importantly, using these frameworks allow us to perform all necessary computations on a single CPU as well as a cluster of computers.

**Why [PyTorch](https://pytorch.org/) & Co. ?**
PyTorch is one of the most popular machine learning frameworks available at the time of writing. 
PytorchLightning facilitates scaling the training procedure of PyTorch without boilerplate.
In our framework, we combine [PyTorch](https://pytorch.org/) & [PytorchLightning](https://www.pytorchlightning.ai/).
Users can choose the trainer class (e.g., DDP by Pytorch) to train large knowledge graph embedding models with billions of parameters.
PytorchLightning allows us to use state-of-the-art model parallelism techniques (e.g. Fully Sharded Training, FairScale, or DeepSpeed)
without extra effort.
With our framework, practitioners can directly use PytorchLightning for model parallelism to train gigantic embedding models.

**Why [Hugging-face Gradio](https://huggingface.co/gradio)?**
Deploy a pre-trained embedding model without writing a single line of code.


## Installation

### Installation from Source
``` bash
git clone https://github.com/dice-group/dice-embeddings.git
conda create -n dice python=3.10.13 --no-default-packages && conda activate dice && cd dice-embeddings &&
pip3 install -e .
```
or
```bash
pip install dicee
```
## Download Knowledge Graphs
```bash
wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate && unzip KGs.zip
```
To test the Installation
```bash
python -m pytest -p no:warnings -x # Runs >114 tests leading to > 15 mins
python -m pytest -p no:warnings --lf # run only the last failed test
python -m pytest -p no:warnings --ff # to run the failures first and then the rest of the tests.
```



## Knowledge Graph Embedding Models

1. TransE, DistMult, ComplEx, ConEx, QMult, OMult, ConvO, ConvQ, Keci
2. All 44 models available in https://github.com/pykeen/pykeen#models

> For more, please refer to `examples`.


## How to Train

To Train a KGE model (KECI) and evaluate it on the train, validation, and test sets of the UMLS benchmark dataset.
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
# See the Keci_UMLS folder embeddings and all other files
```
where the data is in the following form
```bash
$ head -3 KGs/UMLS/train.txt 
acquired_abnormality    location_of     experimental_model_of_disease
anatomical_abnormality  manifestation_of        physiologic_function
alga    isa     entity
```
A KGE model can also be trained from the command line
```bash
dicee --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
```
dicee automaticaly detects available GPUs and trains a model with distributed data parallels technique. Under the hood, dicee uses lighning as a default trainer.
```bash
# Train a model by only using the GPU-0
CUDA_VISIBLE_DEVICES=0 dicee --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
# Train a model by only using GPU-1
CUDA_VISIBLE_DEVICES=1 dicee --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python dicee/scripts/run.py --trainer PL --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
```
Under the hood, dicee executes run.py script and uses lighning as a default trainer
```bash
# Two equivalent executions
# (1)
dicee --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9518788343558282, 'H@3': 0.9988496932515337, 'H@10': 1.0, 'MRR': 0.9753123402351737}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.6932515337423313, 'H@3': 0.9041411042944786, 'H@10': 0.9754601226993865, 'MRR': 0.8072362996241839}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.6951588502269289, 'H@3': 0.9039334341906202, 'H@10': 0.9750378214826021, 'MRR': 0.8064032293278861}

# (2)
CUDA_VISIBLE_DEVICES=0,1 python dicee/scripts/run.py --trainer PL --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
# Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9518788343558282, 'H@3': 0.9988496932515337, 'H@10': 1.0, 'MRR': 0.9753123402351737}
# Evaluate Keci on Train set: Evaluate Keci on Train set
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.6932515337423313, 'H@3': 0.9041411042944786, 'H@10': 0.9754601226993865, 'MRR': 0.8072362996241839}
# Evaluate Keci on Test set: Evaluate Keci on Test set
# {'H@1': 0.6951588502269289, 'H@3': 0.9039334341906202, 'H@10': 0.9750378214826021, 'MRR': 0.8064032293278861}
```
Similarly, models can be easily trained with torchrun
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee/scripts/run.py --trainer torchDDP --dataset_dir "KGs/UMLS" --model Keci --eval_model "train_val_test"
# Evaluate Keci on Train set: Evaluate Keci on Train set: Evaluate Keci on Train set
# {'H@1': 0.9518788343558282, 'H@3': 0.9988496932515337, 'H@10': 1.0, 'MRR': 0.9753123402351737}
# Evaluate Keci on Validation set: Evaluate Keci on Validation set
# {'H@1': 0.6932515337423313, 'H@3': 0.9041411042944786, 'H@10': 0.9754601226993865, 'MRR': 0.8072499937521418}
# Evaluate Keci on Test set: Evaluate Keci on Test set
{'H@1': 0.6951588502269289, 'H@3': 0.9039334341906202, 'H@10': 0.9750378214826021, 'MRR': 0.8064032293278861}
```
You can also train a model in multi-node multi-gpu setting.
```bash
torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 0 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula  dicee/scripts/run.py --trainer torchDDP --dataset_dir KGs/UMLS
torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 1 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula dicee/scripts/run.py --trainer torchDDP --dataset_dir KGs/UMLS
```
Train a KGE model by providing the path of a single file and store all parameters under newly created directory
called `KeciFamilyRun`.
```bash
dicee --path_single_kg "KGs/Family/family-benchmark_rich_background.owl" --model Keci --path_to_store_single_run KeciFamilyRun --backend rdflib
```
where the data is in the following form
```bash
$ head -3 KGs/Family/train.txt 
_:1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#Ontology> .
<http://www.benchmark.org/family#hasChild> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#ObjectProperty> .
<http://www.benchmark.org/family#hasParent> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#ObjectProperty> .
```
**Apart from n-triples or standard link prediction dataset formats, we support ["owl", "nt", "turtle", "rdf/xml", "n3"]***.
Moreover, a KGE model can be also trained  by providing **an endpoint of a triple store**.
```bash
dicee --sparql_endpoint "http://localhost:3030/mutagenesis/" --model Keci
```
For more, please refer to `examples`.


## Creating an Embedding Vector Database

##### Learning Embeddings
```bash
# Train an embedding model
dicee --dataset_dir KGs/Countries-S1 --path_to_store_single_run CountryEmbeddings --model Keci --p 0 --q 1 --embedding_dim 32 --adaptive_swa
```
#### Loading Embeddings into Qdrant Vector Database
```bash
# Ensure that Qdrant available
# docker pull qdrant/qdrant && docker run -p 6333:6333 -p 6334:6334      -v $(pwd)/qdrant_storage:/qdrant/storage:z      qdrant/qdrant
diceeindex --path_model "CountryEmbeddings" --collection_name "dummy" --location "localhost"
```
#### Launching Webservice
```bash
diceeserve --path_model "CountryEmbeddings" --collection_name "dummy" --collection_location "localhost"
```
##### Retrieve and Search 

Get embedding of germany
```bash
curl -X 'GET' 'http://0.0.0.0:8000/api/get?q=germany' -H 'accept: application/json'
```

Get most similar things to europe
```bash
curl -X 'GET' 'http://0.0.0.0:8000/api/search?q=europe' -H 'accept: application/json'
{"result":[{"hit":"europe","score":1.0},
{"hit":"northern_europe","score":0.67126536},
{"hit":"western_europe","score":0.6010134},
{"hit":"puerto_rico","score":0.5051694},
{"hit":"southern_europe","score":0.4829831}]}
```




## Answering Complex Queries

```python
# pip install dicee
# wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate & unzip KGs.zip
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
# (1) Train a KGE model
args = Namespace()
args.model = 'Keci'
args.p=0
args.q=1
args.optim = 'Adam'
args.scoring_technique = "AllvsAll"
args.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
args.backend = "rdflib"
args.num_epochs = 200
args.batch_size = 1024
args.lr = 0.1
args.embedding_dim = 512
result = Execute(args).start()
# (2) Load the pre-trained model
pre_trained_kge = KGE(path=result['path_experiment_folder'])
# (3) Single-hop query answering
# Query: ?E : \exist E.hasSibling(E, F9M167)
# Question: Who are the siblings of F9M167?
# Answer: [F9M157, F9F141], as (F9M167, hasSibling, F9M157) and (F9M167, hasSibling, F9F141)
predictions = pre_trained_kge.answer_multi_hop_query(query_type="1p",
                                                     query=('http://www.benchmark.org/family#F9M167',
                                                            ('http://www.benchmark.org/family#hasSibling',)),
                                                     tnorm="min", k=3)
top_entities = [topk_entity for topk_entity, query_score in predictions]
assert "http://www.benchmark.org/family#F9F141" in top_entities
assert "http://www.benchmark.org/family#F9M157" in top_entities
# (2) Two-hop query answering
# Query: ?D : \exist E.Married(D, E) \land hasSibling(E, F9M167)
# Question: To whom a sibling of F9M167 is married to?
# Answer: [F9F158, F9M142] as (F9M157 #married F9F158) and (F9F141 #married F9M142)
predictions = pre_trained_kge.answer_multi_hop_query(query_type="2p",
                                                     query=("http://www.benchmark.org/family#F9M167",
                                                            ("http://www.benchmark.org/family#hasSibling",
                                                             "http://www.benchmark.org/family#married")),
                                                     tnorm="min", k=3)
top_entities = [topk_entity for topk_entity, query_score in predictions]
assert "http://www.benchmark.org/family#F9M142" in top_entities
assert "http://www.benchmark.org/family#F9F158" in top_entities
# (3) Three-hop query answering
# Query: ?T : \exist D.type(D,T) \land Married(D,E) \land hasSibling(E, F9M167)
# Question: What are the type of people who are married to a sibling of F9M167?
# (3) Answer: [Person, Male, Father] since  F9M157 is [Brother Father Grandfather Male] and F9M142 is [Male Grandfather Father]

predictions = pre_trained_kge.answer_multi_hop_query(query_type="3p", query=("http://www.benchmark.org/family#F9M167",
                                                                             ("http://www.benchmark.org/family#hasSibling",
                                                                             "http://www.benchmark.org/family#married",
                                                                             "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),
                                                     tnorm="min", k=5)
top_entities = [topk_entity for topk_entity, query_score in predictions]
print(top_entities)
assert "http://www.benchmark.org/family#Person" in top_entities
assert "http://www.benchmark.org/family#Father" in top_entities
assert "http://www.benchmark.org/family#Male" in top_entities
```
For more, please refer to `examples/multi_hop_query_answering`.


## Predicting Missing Links

```python
from dicee import KGE
# (1) Train a knowledge graph embedding model..
# (2) Load a pretrained model
pre_trained_kge = KGE(path='..')
# (3) Predict missing links through head entity rankings
pre_trained_kge.predict_topk(h=[".."],r=[".."],topk=10)
# (4) Predict missing links through relation rankings
pre_trained_kge.predict_topk(h=[".."],t=[".."],topk=10)
# (5) Predict missing links through tail entity rankings
pre_trained_kge.predict_topk(r=[".."],t=[".."],topk=10)
```



## Downloading Pretrained Models 

```python
from dicee import KGE
# (1) Load a pretrained ConEx on DBpedia 
model = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/KINSHIP-Keci-dim128-epoch256-KvsAll")
```

- For more please look at [dice-research.org/projects/DiceEmbeddings/](https://files.dice-research.org/projects/DiceEmbeddings/)



## How to Deploy

```python
from dicee import KGE
KGE(path='...').deploy(share=True,top_k=10)
```

## Docker
To build the Docker image:
```
docker build -t dice-embeddings .
```

To test the Docker image:
```
docker run --rm -v ~/.local/share/dicee/KGs:/dicee/KGs dice-embeddings ./main.py --model AConEx --embedding_dim 16
```

## Coverage Report

The coverage report is generated using [coverage.py](https://coverage.readthedocs.io/en/7.6.0/):

```
Name                                                   Stmts   Miss  Cover   Missing
------------------------------------------------------------------------------------
dicee/__init__.py                                          7      0   100%
dicee/abstracts.py                                       201     82    59%   104-105, 123, 146-147, 152, 165, 197, 240-254, 257-260, 263-266, 301, 314-317, 320-324, 364-375, 390-398, 413, 424-428, 555-575, 581-585, 589-591
dicee/callbacks.py                                       245    102    58%   50-55, 67-73, 76, 88-93, 98-103, 106-109, 116-133, 138-142, 146-147, 276-280, 286-287, 305-311, 314, 319-320, 332-338, 344-353, 358-360, 405, 416-429, 433-468, 480-486
dicee/config.py                                           93      2    98%   141-142
dicee/dataset_classes.py                                 299     74    75%   41, 54, 87, 93, 99-106, 109, 112, 115-139, 195-201, 204, 207-209, 314, 325-328, 344, 410-411, 429, 528-536, 539, 543-557, 700-707, 710-714
dicee/eval_static_funcs.py                               227     95    58%   101, 106, 111, 258-353, 360-411
dicee/evaluator.py                                       262     51    81%   46, 51, 56, 84, 89-90, 93, 109, 126, 137, 141, 146, 177-188, 195-206, 314, 344-367, 455, 465, 482-487
dicee/executer.py                                        113      4    96%   116, 258-259, 291
dicee/knowledge_graph.py                                  65      3    95%   79, 110, 114
dicee/knowledge_graph_embeddings.py                      636    443    30%   27, 30-31, 39-52, 57-90, 93-127, 131-139, 170-184, 215-228, 254-274, 324-327, 330-333, 346, 381-426, 484-486, 502-503, 509-517, 522-525, 528-533, 538, 547, 592-598, 630, 688-1053, 1084-1145, 1149-1177, 1200, 1227-1265
dicee/models/__init__.py                                   9      0   100%
dicee/models/base_model.py                               234     31    87%   54, 56, 82, 88-103, 157, 190, 230, 236, 245, 248, 252, 259, 263, 265, 280, 288-289, 296-297, 351, 354, 427, 439
dicee/models/clifford.py                                 556    357    36%   31-42, 68-117, 122-133, 156-168, 190-220, 235, 237, 241, 248-249, 276-280, 303-311, 325-327, 332-333, 364-384, 406, 413, 417-478, 495-499, 511, 514, 519, 524, 571-607, 625-631, 644, 647, 652, 657, 686-692, 705, 708, 713, 718, 728-737, 753-754, 774-845, 856-859, 884-909, 933-966, 1002-1006, 1019, 1029, 1032, 1037, 1042, 1047, 1051, 1055, 1064-1065, 1095, 1102, 1107, 1135-1139, 1167-1176, 1186-1194, 1212-1214, 1232-1234, 1250-1252
dicee/models/complex.py                                  151     15    90%   86-109
dicee/models/dualE.py                                     59     10    83%   93-102, 142-156
dicee/models/function_space.py                           262    221    16%   10-24, 28-37, 40-49, 53-70, 77-86, 89-98, 101-110, 114-126, 134-156, 159-165, 168-185, 188-194, 197-205, 208, 213-234, 243-246, 250-254, 258-267, 271-292, 301-307, 311-328, 332-335, 344-352, 355, 366-372, 392-406, 424-438, 443-453, 461-465, 474-478
dicee/models/octonion.py                                 227     83    63%   21-44, 320-329, 334-345, 348-370, 374-416, 426-474
dicee/models/pykeen_models.py                             50      5    90%   60-63, 118
dicee/models/quaternion.py                               192     69    64%   7-21, 30-55, 68-72, 107, 185, 328-342, 345-364, 368-389, 399-426
dicee/models/real.py                                      61     12    80%   36-39, 66-69, 87, 103-106
dicee/models/static_funcs.py                              10      0   100%
dicee/models/transformers.py                             236    189    20%   24-43, 46, 60-75, 84-102, 105-116, 123-125, 128, 134-151, 155-180, 186-190, 193-197, 203-207, 210-212, 229-256, 265-268, 271-276, 279-304, 310-315, 319-372, 376-398, 404-414
dicee/query_generator.py                                 374    346     7%   18-52, 56, 62-65, 69-70, 78-92, 100-147, 155-188, 192-206, 212-269, 274-303, 307-443, 453-472, 480-501, 508-512, 517, 522-528
dicee/read_preprocess_save_load_kg/__init__.py             3      0   100%
dicee/read_preprocess_save_load_kg/preprocess.py         256     41    84%   34, 40, 78, 102-127, 133, 138-151, 184, 214, 388-389, 444
dicee/read_preprocess_save_load_kg/read_from_disk.py      36     11    69%   33, 38-40, 47, 55, 58-72
dicee/read_preprocess_save_load_kg/save_load_disk.py      45     18    60%   39-60
dicee/read_preprocess_save_load_kg/util.py               219    126    42%   65-67, 72-73, 91-97, 100-102, 107-109, 121, 134, 140-143, 148-156, 161-167, 172-177, 182-187, 199-220, 226-282, 286-290, 294-295, 299, 303-304, 334, 351, 356, 363-364
dicee/sanity_checkers.py                                  54     23    57%   8-12, 21-31, 46, 51, 58, 64-79, 85, 89, 96
dicee/static_funcs.py                                    418    163    61%   40, 50, 56-61, 83, 105-106, 115, 138, 152, 157-159, 163-165, 167, 194-198, 246, 254, 263-268, 290-304, 316-336, 340-357, 362, 386-387, 392-393, 410-411, 413-414, 416-417, 419-420, 428, 446-450, 467-470, 474-479, 483-487, 491-492, 498-500, 526-527, 539-542, 547-550, 559-610, 615-627, 644-658, 661-669
dicee/static_funcs_training.py                           123     63    49%   118-215, 223-224
dicee/static_preprocess_funcs.py                         100     44    56%   17-25, 52, 56, 64, 67, 78, 91-115, 120-123, 128-131, 136-139
dicee/trainer/__init__.py                                  1      0   100%
dicee/trainer/dice_trainer.py                            126     13    90%   27-32, 91, 98, 103-108, 147
dicee/trainer/torch_trainer.py                            79      4    95%   31, 196, 207-208
dicee/trainer/torch_trainer_ddp.py                       152    128    16%   13-14, 43, 47-72, 83-112, 131-137, 140-149, 164-194, 204-217, 226-246, 251-260, 263-272, 275-299, 302-309
------------------------------------------------------------------------------------
TOTAL                                                   6181   2828    54%
```

## How to cite
Currently, we are working on our manuscript describing our framework. 
If you really like our work and want to cite it now, feel free to chose one :) 
```
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
