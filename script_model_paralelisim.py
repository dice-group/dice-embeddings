import torch
import dicee
from dicee import DistMult
from dicee import NegSampleDataset
import polars as pl
import time
import pandas as pd

print("Reading KG...", end=" ")
start_time = time.time()

# data = pl.read_parquet("dbpedia-2022-12-nt.parquet.snappy")  # ,n_rows=100_000_000)
data = pl.read_csv("KGs/UMLS/train.txt",
                   has_header=False,
                   low_memory=False,
                   columns=[0, 1, 2],
                   dtypes=[pl.Utf8],  # str
                   new_columns=['subject', 'relation', 'object'],
                   separator="\t")

print(f"took {time.time() - start_time}")
print("Unique entities...", end=" ")
start_time = time.time()
unique_entities = pl.concat((data.get_column('subject'), data.get_column('object'))).unique().rename('entity').to_list()
print(f"took {time.time() - start_time}")

print("Unique relations...", end=" ")
start_time = time.time()
unique_relations = data.unique(subset=["relation"]).select("relation").to_series().to_list()
print(f"took {time.time() - start_time}")

print("Entity index mapping...", end=" ")
start_time = time.time()
entity_to_idx = {ent: idx for idx, ent in enumerate(unique_entities)}
print(f"took {time.time() - start_time}")

print("Relation index mapping...", end=" ")
start_time = time.time()
rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
print(f"took {time.time() - start_time}")

print("Constructing training data...", end=" ")
start_time = time.time()
data = data.with_columns(pl.col("subject").map_dict(entity_to_idx).alias("subject"),
                         pl.col("relation").map_dict(rel_to_idx).alias("relation"),
                         pl.col("object").map_dict(entity_to_idx).alias("object")).to_numpy()
print(f"took {time.time() - start_time}")

data = NegSampleDataset(train_set=data,
                        num_entities=len(unique_entities), num_relations=len(unique_relations),
                        neg_sample_ratio=1.0)
data = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
print("KGE model...", end=" ")
start_time = time.time()
model1 = DistMult(args={"num_entities": len(entity_to_idx), "num_relations": len(rel_to_idx),
                        "embedding_dim": 20, 'learning_rate': 0.01})
model2 = DistMult(args={"num_entities": len(entity_to_idx), "num_relations": len(rel_to_idx),
                        "embedding_dim": 20, 'learning_rate': 0.01})
model1.to(torch.device("cuda:0"))
model2.to(torch.device("cuda:1"))

print(f"took {time.time() - start_time}", end=" ")
print("Optimizer...")
start_time = time.time()
optim1 = model1.configure_optimizers()
optim2 = model2.configure_optimizers()

loss_function = model1.loss_function
print("Training...")
for e in range(1):
    epoch_loss = 0
    for (x, y) in data:
        x = x.flatten(start_dim=0, end_dim=1)
        y = y.flatten(start_dim=0, end_dim=1)

        optim1.zero_grad(set_to_none=True)
        optim2.zero_grad(set_to_none=True)
        # cpu =
        yhat = (model1.forward(x.pin_memory().to("cuda:0", non_blocking=True)).to("cpu") + model2.forward(x.pin_memory().to("cuda:1", non_blocking=True)).to("cpu")) / 2

        batch_positive_loss = loss_function(yhat, y)
        epoch_loss += batch_positive_loss.item()
        batch_positive_loss.backward()
        optim1.step()
        optim2.step()

    print(l / len(data))

print(model1.entity_embeddings.weight[:3])
print(model2.entity_embeddings.weight[:3])
print('DONE')
