# Training KGE
```python
from dicee.executer import Execute
from dicee.config import Namespace
args = Namespace()
args.path_dataset_folder = 'KGs/UMLS'
args.scoring_technique = 'AllvsAll'
args.model="Keci"
args.embedding_dim=32
args.num_epochs=100
args.eval_model = 'train_val_test'
result1 = Execute(args).start()
```
Expected output
```bash
Evaluate Keci on Train set: Evaluate Keci on Train set
{'H@1': 0.9450728527607362, 'H@3': 0.9968366564417178, 'H@10': 0.9994248466257669, 'MRR': 0.9708919751909051}
Evaluate Keci on Validation set: Evaluate Keci on Validation set
{'H@1': 0.7154907975460123, 'H@3': 0.9026073619631901, 'H@10': 0.9647239263803681, 'MRR': 0.8161878520625896}
Evaluate Keci on Test set: Evaluate Keci on Test set
{'H@1': 0.7140695915279879, 'H@3': 0.916036308623298, 'H@10': 0.970499243570348, 'MRR': 0.819776678394008}
Total Runtime: 25.139 seconds
```
To generate table of results.
**pip install rich**
You do not need install another library like rich to report results in csv of markdown format.
You can use **df.to_markdown(index=False)**

### Generating Queries and get test dataset ready
@TODO. This shouldn't be the case => (Mappings and Query Creation for UMLS in KGs folder)
The input KG folder should not be modified, hence please update this example to create a folder where
queries are stored.
```bash
python mappings.py --datapath "./KGs/UMLS" --map_to_ids --indexify_files
python create_queries.py --dataset "UMLS" --gen_test_num 10 --gen_test --save_name --gen_all
python mappings.py --datapath "./KGs/UMLS" --unmap_to_text --join_queries --file_type unmapped
```

## Approximate Query answering

```bash
path_pretrained_model=?
python complex_query_answering.py --datapath "./KGs/UMLS" --experiment "$path_pretrained_model" --tnorm 'prod' --neg_norm 'yager' --k_ 4 --lambda_ 0.4
python complex_query_answering.py --datapath "./KGs/UMLS" --experiment "$path_pretrained_model" --tnorm 'prod' --neg_norm 'yager' --k_ 4 --lambda_ 0.4
```

**SOME RESULTS**

The table below shows the result for KECI model (trained with KvsALL) (embedding_dim=32, epochs=100, p=0, q=1) on UMLS test queries 5000 for each type.

| Query Name | MRR | H1 | H3 | H10 |
| ---------- | --- | -- | -- | --- |
| 3p | 0.21361 | 0.11917 | 0.20161 | 0.42826 |
| 2i | 0.83583 | 0.73475 | 0.92881 | 0.99204 |
| 3i | 0.82573 | 0.72115 | 0.92022 | 0.99469 |
| ip | 0.47873 | 0.35216 | 0.53032 | 0.74173 |
| pi | 0.68376 | 0.56729 | 0.75484 | 0.90672 |
| 2in | 0.50812 | 0.28471 | 0.68047 | 0.92180 |
| 3in | 0.57779 | 0.36202 | 0.75598 | 0.96908 |
| pin | 0.48788 | 0.38308 | 0.52423 | 0.69204 |
| pni | 0.36234 | 0.21707 | 0.38021 | 0.74431 |
| inp | 0.40255 | 0.28733 | 0.41792 | 0.66911 |
| 2u | 0.70567 | 0.59493 | 0.76820 | 0.92183 |
| up | 0.16408 | 0.06499 | 0.15254 | 0.37866 |

Keci Trained with AllvsAll
┏━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Query Name ┃ MRR     ┃ H1      ┃ H3      ┃ H10     ┃
┡━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ 2p         │ 0.36980 │ 0.23333 │ 0.43333 │ 0.63333 │
│ 3p         │ 0.32340 │ 0.16667 │ 0.48333 │ 0.66667 │
│ 2i         │ 0.74678 │ 0.68929 │ 0.73929 │ 0.95000 │
│ 3i         │ 0.78993 │ 0.62083 │ 0.96250 │ 1.00000 │
│ ip         │ 0.57916 │ 0.46667 │ 0.70000 │ 0.81667 │
│ pi         │ 0.69915 │ 0.66667 │ 0.66667 │ 0.78333 │
│ 2in        │ 0.55685 │ 0.37500 │ 0.70000 │ 0.80000 │
│ 3in        │ 0.48735 │ 0.30000 │ 0.61167 │ 0.81500 │
│ pin        │ 0.24562 │ 0.10000 │ 0.30000 │ 0.70000 │
│ pni        │ 0.27930 │ 0.14000 │ 0.24000 │ 0.75000 │
│ inp        │ 0.47969 │ 0.40000 │ 0.45000 │ 0.70000 │
│ 2u         │ 0.73831 │ 0.65833 │ 0.76667 │ 0.87500 │
│ up         │ 0.36253 │ 0.30333 │ 0.30333 │ 0.60667 │
└────────────┴─────────┴─────────┴─────────┴─────────┘
