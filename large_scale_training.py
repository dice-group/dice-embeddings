import torch
import dicee
from dicee import Keci
from dicee import NegSampleDataset
import polars as pl
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
"""
r:10	 p:0	 q:1
Keci(
  (loss): BCEWithLogitsLoss()
  (normalize_head_entity_embeddings): IdentityClass()
  (normalize_relation_embeddings): IdentityClass()
  (normalize_tail_entity_embeddings): IdentityClass()
  (hidden_normalizer): IdentityClass()
  (input_dp_ent_real): Dropout(p=0.0, inplace=False)
  (input_dp_rel_real): Dropout(p=0.0, inplace=False)
  (hidden_dropout): Dropout(p=0.0, inplace=False)
  (entity_embeddings): Embedding(257938325, 20)
  (relation_embeddings): Embedding(54811, 20)
  (q_coefficients): Embedding(1, 1)
)
   | Name                             | Type              | Params
------------------------------------------------------------------------
0  | loss                             | BCEWithLogitsLoss | 0     
1  | normalize_head_entity_embeddings | IdentityClass     | 0     
2  | normalize_relation_embeddings    | IdentityClass     | 0     
3  | normalize_tail_entity_embeddings | IdentityClass     | 0     
4  | hidden_normalizer                | IdentityClass     | 0     
5  | input_dp_ent_real                | Dropout           | 0     
6  | input_dp_rel_real                | Dropout           | 0     
7  | hidden_dropout                   | Dropout           | 0     
8  | entity_embeddings                | Embedding         | 5.2 B 
9  | relation_embeddings              | Embedding         | 1.1 M 
10 | q_coefficients                   | Embedding         | 1     
------------------------------------------------------------------------
5.2 B     Trainable params
0         Non-trainable params
5.2 B     Total params
20,639.451Total estimated model params size (MB)
r:10	 p:0	 q:1
took 42.364829778671265
OptimizedModule(
  (_orig_mod): Keci(
    (loss): BCEWithLogitsLoss()
    (normalize_head_entity_embeddings): IdentityClass()
    (normalize_relation_embeddings): IdentityClass()
    (normalize_tail_entity_embeddings): IdentityClass()
    (hidden_normalizer): IdentityClass()
    (input_dp_ent_real): Dropout(p=0.0, inplace=False)
    (input_dp_rel_real): Dropout(p=0.0, inplace=False)
    (hidden_dropout): Dropout(p=0.0, inplace=False)
    (entity_embeddings): Embedding(257938325, 20)
    (relation_embeddings): Embedding(54811, 20)
    (q_coefficients): Embedding(1, 1)
  )
)
Optimizer...
Training...
  0%|          | 0/111 [00:00<?, ?it/s]Batch Loss:1.1785708665847778	Forward-Backward-Update: 60.76826333999634
  1%|          | 1/111 [09:31<17:27:08, 571.17s/it]Batch Loss:1.111355185508728	Forward-Backward-Update: 37.08943176269531
  2%|▏         | 2/111 [10:09<7:48:22, 257.82s/it] Batch Loss:1.0394731760025024	Forward-Backward-Update: 34.5961971282959
  3%|▎         | 3/111 [10:45<4:41:41, 156.50s/it]Batch Loss:0.9841560125350952	Forward-Backward-Update: 33.15920901298523
  4%|▎         | 4/111 [11:21<3:14:09, 108.88s/it]Batch Loss:0.9383040070533752	Forward-Backward-Update: 33.949097871780396
  5%|▍         | 5/111 [11:56<2:25:26, 82.33s/it] Batch Loss:0.8980816006660461	Forward-Backward-Update: 33.626163721084595
  5%|▌         | 6/111 [12:31<1:55:54, 66.24s/it]Batch Loss:0.8625831007957458	Forward-Backward-Update: 34.746774196624756
  6%|▋         | 7/111 [13:07<1:37:41, 56.36s/it]Batch Loss:0.8332152962684631	Forward-Backward-Update: 33.99446940422058
  7%|▋         | 8/111 [13:43<1:25:14, 49.66s/it]Batch Loss:0.8098770976066589	Forward-Backward-Update: 33.45107460021973
  8%|▊         | 9/111 [14:17<1:16:30, 45.01s/it]Batch Loss:0.7909269332885742	Forward-Backward-Update: 33.88880896568298
  9%|▉         | 10/111 [14:53<1:10:40, 41.98s/it]Batch Loss:0.7751095294952393	Forward-Backward-Update: 34.984283685684204
 10%|▉         | 11/111 [15:30<1:07:47, 40.67s/it]Batch Loss:0.7614863514900208	Forward-Backward-Update: 33.006606578826904
 11%|█         | 12/111 [16:05<1:03:55, 38.74s/it]Batch Loss:0.7502735257148743	Forward-Backward-Update: 34.53621578216553
 12%|█▏        | 13/111 [16:44<1:03:35, 38.93s/it]Batch Loss:0.7408137321472168	Forward-Backward-Update: 34.38044762611389
 13%|█▎        | 14/111 [17:20<1:01:23, 37.97s/it]Batch Loss:0.7331563234329224	Forward-Backward-Update: 34.358346939086914
 14%|█▎        | 15/111 [17:55<59:39, 37.29s/it]  Batch Loss:0.7268505692481995	Forward-Backward-Update: 32.88389492034912

"""

if False:
    print("Reading KG...\n")
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
    print("Loading the index numpy KG..\n")
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

# Model Parallel 1
model1 = Keci(args={"optim":"Adam","p":0,"q":1,"num_entities": num_entities, "num_relations": num_relations,"embedding_dim": 20, 'learning_rate': 0.1})
print(model1)
print(model1.summarize())
# Model Parallel 2
model2 = Keci(args={"optim":"Adam","p":0,"q":1,"num_entities": num_entities, "num_relations": num_relations,"embedding_dim": 20, 'learning_rate': 0.1})
# Compute both models
model1 = torch.compile(model1)
model2 = torch.compile(model2)
print(f"took {time.time() - start_time}")
print(model1)
print("Optimizer...")
start_time = time.time()
# Initialize optimizers
optimizer = model1.configure_optimizers()
optimizer2 = model2.configure_optimizers()
# Define the loss function
loss_function = model1.loss_function
print("Training...")

num_epochs=10
device1="cuda:0"
device2="cuda:1"
for e in range(num_epochs):
    epoch_loss = 0

    for (x, y) in tqdm(data):
        # (1) Shape the batch
        x = x.flatten(start_dim=0, end_dim=1)
        y = y.flatten(start_dim=0, end_dim=1)
        # (2) Empty the gradients
        optimizer.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)
        # (3) Forward Backward and Parameter Update
        start_time=time.time()
        # (3.1) Select embeddings of triples
        h1,r1,t1=model1.get_triple_representation(x)
        # (3.2) Move (3.1) into a single GPU
        h1,r1,t1,y=h1.pin_memory().to(device1, non_blocking=True),r1.pin_memory().to(device1, non_blocking=True),t1.pin_memory().to(device1, non_blocking=True),y.pin_memory().to(device1, non_blocking=True)
        # (3.3) Compute triple score (Forward Pass)
        yhat1 = model1.score(h1,r1,t1)
        
        # (3.4) Select second part of the embeddings of triples
        h2,r2,t2=model2.get_triple_representation(x)
        # (3.5) Move (3.4) into a single GPU
        h2,r2,t2=h2.pin_memory().to(device2, non_blocking=True),r2.pin_memory().to(device2, non_blocking=True),t2.pin_memory().to(device2, non_blocking=True)
        # 3.6 Forward Pass
        yhat2 = model2.score(h2,r2,t2).to(device1)
        
        # (4) Composite Prediction
        yhat=(yhat1 + yhat2)/2   


        # (5.) Compute Loss
        batch_loss = loss_function(yhat, y)
        # (6.) Compute gradients (Backward Pass)
        batch_loss.backward()
        # (3.6) Update parameters
        optimizer.step()
        optimizer2.step()
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
            }, f"{model1._orig_mod.name}_1_{e}.torch")

    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model2._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer2.state_dict(),
            'loss': epoch_loss,
            }, f"{model1._orig_mod.name}_2_{e}.torch")

print('DONE')
