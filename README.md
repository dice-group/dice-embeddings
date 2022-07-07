# DICE Embeddings: Hardware-agnostic Framework for of Large-scale Knowledge Graph Embeddings

Knowledge graph embedding research has mainly focused on learning continuous representations of knowledge graphs towards the link prediction problem. 
Recently developed frameworks can be effectively applied in wide range of research-related applications.
Yet, using these frameworks in real-world applications becomes more challenging as the size of the knowledge graph grows.

We developed DICE Embeddings framework based on Pytorch Lightning and Hugging Face to compute embeddings for large-scale knowledge graphs in a hardware-agnostic manner.
By this, we rely on
1. [Pandas](https://github.com/pandas-dev/pandas) & [DASK](https://dask.org/) to use parallelism at preprocessing a large knowledge graph,
2. [PytorchLightning](https://www.pytorchlightning.ai/) to learn knowledge graph embeddings via multi-CPUs, GPUs, TPUs or computing cluster, and
3. [Gradio](https://gradio.app/) to ease the deployment of pre-trained models.

**Why Pandas & DASK?**
Pandas allows us to read, preprocess (removing literals) and indexed input knowledge graph efficiently.
Through parquet within pandas or dask, a billion of triples can be read in parallel fashion. 
Importantly, dask allow us to perform all necessary computations on a single CPU as well as a cluster of computers.

**Why Pytorch-lightning ?**
Scale the training without the boilerplate.
Importantly, Pytorch-lightning provides state-of-the-art training techniques (e.g. Fully Sharded Training, FairScale, and DeepSpeed) to train
gigantic models (>10B parameters). These techniques do not simply copy a model into all GPUs, hence, allow us to use our hardware efficiently. 

**Why Hugging-face Gradio?**
Deploy a pre-trained embedding model without writing a single line of code.


## Installation
Clone the repository:
```
git clone https://github.com/dice-group/dice-embeddings.git
```
To install dependencies:
```
# python=3.10 with torch cuda nncl https://discuss.pytorch.org/t/issues-on-using-nn-dataparallel-with-python-3-10-and-pytorch-1-11/146745/13
conda create -n dice python=3.9.12
conda activate dice
pip3 install pandas==1.4.2
pip3 install swifter==1.1.2 # we can remove it later
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install pytorch-lightning==1.6.4
pip3 install "dask[complete]"==2022.6.0
pip3 install scikit-learn==1.1.1
pip3 install pytest==6.2.5
pip3 install gradio==3.0.17
pip3 install pyarrow==8.0.0
```
To test the Installation
```
wget https://hobbitdata.informatik.uni-leipzig.de/KG/KGs.zip
unzip KGs.zip
pytest -p no:warnings -x # it takes circa 15 minutes
pytest -p no:warnings --lf # run only the last one
pytest -p no:warnings --ff # to run the failures first and then the rest of the tests.
```
## Pre-trained Models
Please contact:  ```caglar.demir@upb.de ``` or ```caglardemir8@gmail.com ``` , if you lack hardware resources to obtain embeddings of a specific knowledge Graph.
- [DBpedia version: 03-2021 Embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/DBpediaQMultEmbeddings_03_07):
  - 114,747,963 entities, 13,906 relations, and 375,900,264 triples.
- [YAGO3-10 ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/YAGO3-10.zip)
- [FB15K-237 ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/FB15K-237.zip)
- [FB15K ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/FB15K.zip)
- [WN18RR ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/WN18RR.zip)
- [WN18 ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/WN18.zip)
- [Hepatitis ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/ConEx_Hepatitis.zip)
- [Lymphography ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/ConEx_Lymphography.zip)
- [Mammographic ConEx embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/ConEx_Mammographic.zip)
- For more please look at [Hobbit Data](https://hobbitdata.informatik.uni-leipzig.de/KGE/)

## Training
please see examples/Training.md.

## Interactive Link Prediction on DBpedia
```python
from core import KGE
# (1) Download this folder into your local machine https://hobbitdata.informatik.uni-leipzig.de/KGE/DBpediaQMultEmbeddings_03_07/
# (2) Give the path of serialized (1).
pre_trained_kge = KGE(path_of_pretrained_model_dir='QMultDBpedia10Epoch')
# (3) Triple score.
pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],tail_entity=["http://dbpedia.org/resource/Ulm"])
# expected output => tensor([0.9948])
```

## How to Deploy
Any pretrained model can be deployed with an ease. Moreover, anyone on the internet can use the pretrained model with ```--share``` parameter.
```
python deploy.py --path_of_experiment_folder 'DAIKIRI_Storage/QMultFamily' --share
Loading Model...
Model is loaded!
Running on local URL:  http://127.0.0.1:7860/
Running on public URL: https://54886.gradio.app

This share link expires in 72 hours. For free permanent hosting, check out Spaces (https://huggingface.co/spaces)
```
![alt text](core/figures/deploy_qmult_family.png)
### Documentation
We aim to document each function by adding input and output types along with concise description of the performt computation.
Yet, if something is unclear, please let us know.
## How to cite
Currently, we are working on our manuscript describing our framework. 
If you really like our work and want to cite it now, feel free to chose one :) 
```
@article{demir2022kronecker,
  title={Kronecker Decomposition for Knowledge Graph Embeddings},
  author={Demir, Caglar and Lienen, Julian and Ngomo, Axel-Cyrille Ngonga},
  journal={arXiv preprint arXiv:2205.06560},
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
For any questions or wishes, please contact:  ```caglar.demir@upb.de``` or ```caglardemir8@gmail.com.de```

