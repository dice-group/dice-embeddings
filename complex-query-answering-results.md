**Generate Queries and get test dataset ready**
(Mappings and Query Creation for UMLS in KGs folder)

python mappings.py --datapath "./KGs/UMLS" --map_to_ids --indexify_files

python create_queries.py --dataset "UMLS" --gen_test_num 10 --gen_test --save_name --gen_all

python mappings.py --datapath "./KGs/UMLS" --unmap_to_text --join_queries --file_type unmapped


**Complex query answering** 

python complex_query_answering.py --datapath "./KGs/UMLS" --experiment pretrained KGE model path --tnorm 'prod' --neg_norm 'yager' --k_ 4 --lambda_ 0.4

**SOME RESULTS**

The table below shows the result for KECI model (embedding_dim=32, epochs=100, p=0, q=1) on UMLS test queries 5000 for each type  

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

