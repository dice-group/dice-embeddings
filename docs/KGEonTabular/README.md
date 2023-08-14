# Knowledge Graph Embedding Boost on Tabular Data

## Installation

```bash
# (1) Clone the repositories. 
git clone https://github.com/dice-group/dice-embeddings && git clone https://github.com/dice-group/vectograph.git
# (2) Create a virtual environment and install the dependencies pertaining frameworks.
conda create -n dice python=3.9.12 && conda activate dice
pip3 install -r dice-embeddings/requirements.txt && pip3 install -e vectograph/.
```
## Usage
```bash
# (1) Fetch a tabular data (Regression Benchmark)
python vectograph/create_toy_data.py --toy_dataset_name "california"
# (2) Remove labels signals to ensure 
python -c "import pandas as pd;df=pd.read_csv('california.csv',index_col=0);df.drop(columns=['labels'],inplace=True);df.to_csv('california_wo_labels.csv')"
# (3) Create a knowledge graph from a tabular data
python vectograph/main.py --tabularpath "california_wo_labels.csv" --kg_name "california_wo_labels.nt" --num_quantile=10 --min_unique_val_per_column=12
# (4) Preparation for KGE
# (4.1) Create an experiment folder and Move the RDF knowledge graph into (6.1) and rename it
mkdir Example && mv california_wo_labels.nt Example/train.txt
# (5) Generate Embeddings
python dice-embeddings/main.py --path_dataset_folder "Example" --model "QMult" --embedding_dim 4 --num_epochs 1 --save_embeddings_as_csv True
# e.g. see Experiments folder
# (6)
python  regression.py --path_kg "Example/train.txt" --path_tabular_csv "california.csv" --path_entity_embeddings "Experiments/2022-12-06 16:27:18.133871/QMult_entity_embeddings.csv"
```
## Benchmarking

### 10FOLD CV results on California housing dataset (regression). ####
| Linear Regression | Mean Squared Error |
|-------------------|-------------------:|
| Tabular           |           1090+-47 |
| KGE               |           1653+-51 |
| Tabular & KGE     |            898+-22 | 

#### Tabular 10Fold Detailed
Tabular data:	X:(20640, 8) | y:(20640,)
Linear Regression: results of 10 splits
count      10.000000
mean     1090.287544
std        47.823181
min      1023.775281
25%      1063.200679
50%      1093.047391
75%      1107.809817
max      1190.704755
dtype: float64

#### KGE 10Fold Detailed

Tabular data:	X:(20640, 32) | y:(20640,)
Linear Regression: results of 10 splits
count      10.000000
mean     1653.883685
std        51.896592
min      1571.634447
25%      1627.806948
50%      1656.219531
75%      1666.211732
max      1746.601478
dtype: float64
#### TABULAR & KGE 10Fold Detailed
Tabular data:	X:(20640, 40) | y:(20640,)
Linear Regression: results of 10 splits
count     10.000000
mean     898.226317
std       22.614693
min      866.631540
25%      879.495965
50%      896.230564
75%      915.368515
max      929.895093
dtype: float64

## 10FOLD CV results on TODO (binary classification). ####
| Logistic Regression | Accuracy |
|---------------------|---------:|
| Tabular             |        ? |
| KGE                 |        ? |
| Tabular & KGE       |        ? | 
