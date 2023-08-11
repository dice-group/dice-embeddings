import torch
import dicee
from dicee import Keci
from dicee import NegSampleDataset
import polars as pl
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

if True:
    print("Reading KG...")
    start_time = time.time()
    data = pl.read_parquet("dbpedia-2022-12-nt.parquet.snappy")
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
    print("Loading the index numpy KG")
    with open("data.npy", 'rb') as f:
        data = np.load(f)

    num_entities=1+max(max(data[:,0]),max(data[:,2]))
    num_relations=1+max(data[:,1])

data = NegSampleDataset(train_set=data,
                        num_entities=num_entities, num_relations=num_relations,
                        neg_sample_ratio=1.0)
data = torch.utils.data.DataLoader(data, batch_size=10_000_000, shuffle=True,num_workers=30)
print("KGE model...")
start_time = time.time()
model1 = Keci(args={"optim":"Adam","p":0,"q":1,"num_entities": num_entities, "num_relations": num_relations,"embedding_dim": 50, 'learning_rate': 0.01})
print(model1)
print(model1.summarize())
model1 = torch.compile(model1)
print(f"took {time.time() - start_time}")
print(model1)
print("Optimizer...")
start_time = time.time()
optimizer = model1.configure_optimizers()


loss_function = model1.loss_function
print("Training...")

num_epochs=10
device="cuda:0"
for e in range(num_epochs):
    epoch_loss = 0

    for (x, y) in tqdm(data):
        # (1) Shape the batch
        x = x.flatten(start_dim=0, end_dim=1)
        y = y.flatten(start_dim=0, end_dim=1)
        # (2) Empty the gradients
        optimizer.zero_grad(set_to_none=True)
        # (3) Forward Backward and Parameter Update
        start_time=time.time()
        # (3.1) Select embeddings of triples
        h1,r1,t1=model1.get_triple_representation(x)
        # (3.2) Move (3.1) into a single GPU
        h1,r1,t1,y=h1.pin_memory().to(device, non_blocking=True),r1.pin_memory().to(device, non_blocking=True),t1.pin_memory().to(device, non_blocking=True),y.pin_memory().to(device, non_blocking=True)
        # (3.3) Compute triple score (Forward Pass)
        yhat = model1.score(h1,r1,t1)
        # (3.4) Compute Loss
        batch_loss = loss_function(yhat, y)
        # (3.5) Compute gradients (Backward Pass)
        batch_loss.backward()
        # (3.6) Update parameters
        optimizer.step()
        # (3.7) Update epoch loss
        batch_loss=batch_loss.item()
        epoch_loss += batch_loss
        print(f"Batch Loss:{batch_loss}\tForward-Backward-Update: {time.time()-start_time}")
    
    print(epoch_loss / len(data))
    print('Saving')
    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model1._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, f"{model1._orig_mod.name}_{e}.torch")
print('DONE')
