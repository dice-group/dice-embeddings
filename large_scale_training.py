"""
Example script to train KGE models via model parallel with GPU-off-loading
"""
from typing import Tuple
import os
import torch
import dicee
from dicee import Keci
from dicee import NegSampleDataset
import polars as pl
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from argparse import ArgumentParser


def input_arguments():
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_kg", type=str, default=None,  # "kinship.parquet.snappy",
                        help="path parquet formatted polars dataframe")
    parser.add_argument("--path_idx_kg", type=str, default="data.npy",
                        help="path to numpy ndarray")
    parser.add_argument("--path_checkpoint", type=str, default="Keci_1_9.torch"
                        )
    parser.add_argument("--path_checkpoint2", type=str, default="Keci_2_9.torch")

    parser.add_argument("--batch_size", type=int, default=10_000_000)
    parser.add_argument("--neg_sample_ratio", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=40)

    return parser.parse_args()


def get_data(args) -> Tuple[np.ndarray, int, int]:
    if args.path_kg:
        """ do this """
        print("Reading KG...\n")
        start_time = time.time()
        data = pl.read_parquet(args.path_kg)
        print(f"took {time.time() - start_time}")
        print("Unique entities...")
        start_time = time.time()
        unique_entities = pl.concat((data.get_column('subject'), data.get_column('object'))).unique().rename(
            'entity')
        # @TODO Store unique_relations
        unique_entities = unique_entities.to_list()
        print(f"took {time.time() - start_time}")

        print("Unique relations...")
        start_time = time.time()
        unique_relations = data.unique(subset=["relation"]).select("relation").to_series()
        # @TODO Store unique_relations
        unique_entities = unique_entities

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

        num_entities = len(unique_entities)
        num_relations = len(unique_relations)

        with open("data.npy", 'wb') as f:
            np.save(f, data)

        return data, num_entities, num_relations

    elif args.path_idx_kg:
        print("Loading the index numpy KG..\n")
        #data=np.load('data.npy', mmap_mode='r')
        with open("data.npy", 'rb') as f:
            data = np.load(f)
        num_entities = 1 + max(max(data[:, 0]), max(data[:, 2]))
        num_relations = 1 + max(data[:, 1])
        return data, num_entities, num_relations
    else:
        raise RuntimeError


def init_model(args, num_entities, num_relations):
    start_time = time.time()
    model1 = Keci(
        args={"optim": "Adam", "p": 0, "q": 1, "num_entities": num_entities, "num_relations": num_relations,
              "embedding_dim": args.embedding_dim, 'learning_rate': 0.1})
    model2 = Keci(
        args={"optim": "Adam", "p": 0, "q": 1, "num_entities": num_entities, "num_relations": num_relations,
              "embedding_dim": args.embedding_dim, 'learning_rate': 0.1})
    print(f"took {time.time() - start_time}")
    return (model1, model2), (model1.configure_optimizers(), model2.configure_optimizers())


def get_model(args, num_entities: int, num_relations: int):
    # Initialize |GPUs| models on a single node
    models, optimizers = init_model(args, num_entities, num_relations)
    if args.path_checkpoint:
        """ Load the checkpoint"""
        # update models
        model1, model2 = models
        opt1, opt2 = optimizers

        checkpoint1 = torch.load(args.path_checkpoint,map_location='cpu')
        model1.load_state_dict(checkpoint1['model_state_dict'])
        #opt1.load_state_dict(checkpoint1['optimizer_state_dict'])

        checkpoint2 = torch.load(args.path_checkpoint2,map_location='cpu')
        model2.load_state_dict(checkpoint2['model_state_dict'])
        #opt2.load_state_dict(checkpoint2['optimizer_state_dict'])
        models = (model1, model2)
        optimizers = (opt1, opt2)
    return models, optimizers


def get_train_loader(args):
    data: np.ndarray
    data, num_ent, num_rel = get_data(args)
    data: torch.utils.data.DataLoader
    print('Creating dataset...')
    data: NegSampleDataset
    data = NegSampleDataset(train_set=data,
                            num_entities=num_ent, num_relations=num_rel,
                            neg_sample_ratio=1.0)
    data: torch.utils.data.DataLoader
    data = torch.utils.data.DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True,
                                       num_workers=10)
    return data, num_ent, num_rel


def run(args):
    # (1) Get training data
    dataloader: torch.utils.data.DataLoader
    dataloader, num_ent, num_rel = get_train_loader(args)
    num_triples = len(dataloader.dataset)
    print('Number of triples', num_triples)
    # (2) Get model
    models, optimizers = get_model(args, num_ent, num_rel)

    # (3) Compile models
    model1, model2 = models
    print('####### Model 1 #######')
    print(model1)
    print(model1.summarize())
    model1 = torch.compile(model1)
    print(model1)
    print('######## Model2 #######')
    print(model2)
    print(model2.summarize())
    model2 = torch.compile(model2)
    print(model2)
    # (4) Get optim
    opt1, opt2 = optimizers
    print(opt1)
    print(opt2)
    # (5) Get loss func
    loss_function = model1.loss_function
    print("Training...")

    device1 = "cuda:0"
    device2 = "cuda:1"

    for e in range(args.num_epochs):
        epoch_loss = 0

        for ith, (x, y) in enumerate(tqdm(dataloader)):
            # (1) Shape the batch
            x = x.flatten(start_dim=0, end_dim=1)
            y = y.flatten(start_dim=0, end_dim=1)
            
            # (2) Empty the gradients
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            
            # (3) Forward Backward and Parameter Update
            start_time = time.time()
            # (3.1) Select embeddings of triples
            h1, r1, t1 = model1.get_triple_representation(x)
            # (3.2) Move (3.1) into a single GPU
            if "cuda" in device1:
                h1, r1, t1, y = h1.pin_memory().to(device1, non_blocking=True), r1.pin_memory().to(device1,
                                                                                                   non_blocking=True), t1.pin_memory().to(
                    device1, non_blocking=True), y.pin_memory().to(device1, non_blocking=True)
            # (3.3) Compute triple score (Forward Pass)
            yhat1 = model1.score(h1, r1, t1)

            # (3.4) Select second part of the embeddings of triples
            h2, r2, t2 = model2.get_triple_representation(x)
            if "cuda" in device2:
                # (3.5) Move (3.4) into a single GPU
                h2, r2, t2 = h2.pin_memory().to(device2, non_blocking=True), r2.pin_memory().to(device2,
                                                                                                non_blocking=True), t2.pin_memory().to(
                    device2, non_blocking=True)
            # 3.6 Forward Pass
            yhat2 = model2.score(h2, r2, t2).to(device1)

            # (3.7) Composite Prediction
            yhat = yhat1 + yhat2
            # (3.8) Compute Loss
            batch_loss = loss_function(yhat, y)
            # (3.9) Compute gradients (Backward Pass)
            batch_loss.backward()
            # (3.10) Update parameters
            opt1.step()
            opt2.step()
            # (4) Update epoch loss
            batch_loss = batch_loss.item()
            epoch_loss += batch_loss
            if ith % 1 == 0: # init an argument
                print(f"Batch Loss:{batch_loss}\tForward-Backward-Update: {time.time() - start_time}")
            
        print(f"Epoch loss:{epoch_loss}")
        print('Saving..')
        torch.save({
            'model_state_dict': model1._orig_mod.state_dict(),
            'optimizer_state_dict': opt1.state_dict(),
        }, f"{model1._orig_mod.name}_1_{e}.torch")

        torch.save({
            'model_state_dict': model2._orig_mod.state_dict(),
            'optimizer_state_dict': opt2.state_dict(),
        }, f"{model1._orig_mod.name}_2_{e}.torch")

    print('DONE')


if __name__ == '__main__':
    run(input_arguments())

# Note mode1 and model2 keci with p=0, q=1
# model1 real_m1:[] complex_m1[]
# model2 real_m2:[] complex_m2[]
# y1 y2
# Final model = real_m1[],real_m2[], complex_m1[] complex_m2[]
"""

if False:
    print("Reading KG...\n")
    start_time = time.time()
    data = pl.read_parquet("dbpedia-2022-12-nt.parquet.snappy")
    print(f"took {time.time() - start_time}")
    print("Unique entities...")
    start_time = time.time()
    unique_entities = pl.concat((data.get_column('subject'), data.get_column('object'))).unique().rename(
        'entity').to_list()
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

    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    with open("data.npy", 'wb') as f:
        np.save(f, data)
else:
    print("Loading the index numpy KG..\n")
    with open("data.npy", 'rb') as f:
        data = np.load(f)

    num_entities = 1 + max(max(data[:, 0]), max(data[:, 2]))
    num_relations = 1 + max(data[:, 1])

data = NegSampleDataset(train_set=data,
                        num_entities=num_entities, num_relations=num_relations,
                        neg_sample_ratio=1.0)
data = torch.utils.data.DataLoader(data, batch_size=10_000_000, shuffle=True, num_workers=os.cpu_count() - 1)
print("KGE model...")
start_time = time.time()

# Model Parallel 1
model1 = Keci(args={"optim": "Adam", "p": 0, "q": 1, "num_entities": num_entities, "num_relations": num_relations,
                    "embedding_dim": 20, 'learning_rate': 0.1})
print(model1)
print(model1.summarize())
# Model Parallel 2
model2 = Keci(args={"optim": "Adam", "p": 0, "q": 1, "num_entities": num_entities, "num_relations": num_relations,
                    "embedding_dim": 20, 'learning_rate': 0.1})
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

num_epochs = 10
device1 = "cuda:0"
device2 = "cuda:1"
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
        start_time = time.time()
        # (3.1) Select embeddings of triples
        h1, r1, t1 = model1.get_triple_representation(x)
        # (3.2) Move (3.1) into a single GPU
        h1, r1, t1, y = h1.pin_memory().to(device1, non_blocking=True), r1.pin_memory().to(device1,
                                                                                           non_blocking=True), t1.pin_memory().to(
            device1, non_blocking=True), y.pin_memory().to(device1, non_blocking=True)
        # (3.3) Compute triple score (Forward Pass)
        yhat1 = model1.score(h1, r1, t1)

        # (3.4) Select second part of the embeddings of triples
        h2, r2, t2 = model2.get_triple_representation(x)
        # (3.5) Move (3.4) into a single GPU
        h2, r2, t2 = h2.pin_memory().to(device2, non_blocking=True), r2.pin_memory().to(device2,
                                                                                        non_blocking=True), t2.pin_memory().to(
            device2, non_blocking=True)
        # 3.6 Forward Pass
        yhat2 = model2.score(h2, r2, t2).to(device1)

        # (4) Composite Prediction
        yhat = (yhat1 + yhat2) / 2

        # (5.) Compute Loss
        batch_loss = loss_function(yhat, y)
        # (6.) Compute gradients (Backward Pass)
        batch_loss.backward()
        # (3.6) Update parameters
        optimizer.step()
        optimizer2.step()
        # (3.7) Update epoch loss
        batch_loss = batch_loss.item()
        epoch_loss += batch_loss
        print(f"Batch Loss:{batch_loss}\tForward-Backward-Update: {time.time() - start_time}")

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
"""
