import torch
import dicee
from dicee import DistMult
from dicee import NegSampleDataset
import polars as pl
import time
import pandas as pd
import numpy as np

if True:
    print("Reading KG...")
    start_time = time.time()
    data = pl.read_parquet("dbpedia-2022-12-nt.parquet.snappy",n_rows=10000)
#data = pl.read_csv("KGs/UMLS/train.txt",
#                   has_header=False,
#                   low_memory=False,
#                   columns=[0, 1, 2],
#                   dtypes=[pl.Utf8],  # str
#                   new_columns=['subject', 'relation', 'object'],
#                   separator="\t")
    print(f"took {time.time() - start_time}")
    print("Unique entities...")
    start_time = time.time()
    unique_entities = pl.concat((data.get_column('subject'), data.get_column('object'))).unique().rename('entity').to_list()
    print(f"took {time.time() - start_time}")

    print("Unique relations...")
    start_time = time.time()
    unique_relations = data.unique(subset=["relation"]).select("relation").to_series().to_list()
    print(f"took {time.time() - start_time}")

    print("Entity index mapping...")
    start_time = time.time()
    entity_to_idx = {ent: idx for idx, ent in enumerate(unique_entities)}
    print(f"took {time.time() - start_time}")

    print("Relation index mapping...")
    start_time = time.time()
    rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
    print(f"took {time.time() - start_time}")

    print("Constructing training data...")
    start_time = time.time()
    data = data.with_columns(pl.col("subject").map_dict(entity_to_idx).alias("subject"),
                         pl.col("relation").map_dict(rel_to_idx).alias("relation"),
                         pl.col("object").map_dict(entity_to_idx).alias("object")).to_numpy()
    print(f"took {time.time() - start_time}")

    num_entities=len(unique_entities)
    num_relations=len(unique_relations)



    with open("data.npy", 'wb') as f:
        np.save(f, data)
else:
    with open("data.npy", 'rb') as f:
        data = np.load(f)

    num_entities=1+max(max(data[:,0]),max(data[:,2]))
    num_relations=1+max(data[:,1])

data = NegSampleDataset(train_set=data,
                        num_entities=num_entities, num_relations=num_relations,
                        neg_sample_ratio=1.0)
data = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=True)
print("KGE model...")
start_time = time.time()
model1 = DistMult(args={"optim":"SGD","num_entities": num_entities, "num_relations": num_relations,"embedding_dim": 10, 'learning_rate': 0.001})
print(model1)
model2 = DistMult(args={"optim":"SGD","num_entities": num_entities, "num_relations": num_relations,"embedding_dim": 10, 'learning_rate': 0.001})
print(model2)

#model1 = torch.compile(model1)
#model2 = torch.compile(model2)
# TODO: We need to cpu-offlaoding
# Do not sent the eintiere model to GPU but only the batch
#model1.to(device=torch.device("cuda:0"),dtype=torch.float16)
#model2.to(device=torch.device("cuda:1"),dtype=torch.float16)


print(f"took {time.time() - start_time}")

print("Optimizer...")
start_time = time.time()
optim1 = model1.configure_optimizers()
optim2 = model2.configure_optimizers()

from tqdm import tqdm


loss_function = model1.loss_function
print("Training...")
for e in range(1):
    epoch_loss = 0

    for (x, y) in tqdm(data):
        # if we have space in GPU, get the next batch
        x = x.flatten(start_dim=0, end_dim=1)
        y = y.flatten(start_dim=0, end_dim=1)
        optim1.zero_grad(set_to_none=True)
        optim2.zero_grad(set_to_none=True)

        start_time=time.time()
        # CPU
        
        yhat=(model1.score(*model1.get_triple_representation(x)) + model2.score(*model2.get_triple_representation(x)))/2

        batch_positive_loss = loss_function(yhat, y)
        epoch_loss += batch_positive_loss.item()
        batch_positive_loss.backward()
        optim1.step()
        optim2.step()

    print(epoch_loss / len(data))

print('DONE')
