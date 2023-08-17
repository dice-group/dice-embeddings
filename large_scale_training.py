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
import pickle

def input_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path_kg", type=str, default="dbpedia-2022-12-nt-wo-lit-polars.parquet.snappy",
                        help="path parquet formatted polars dataframe")
    parser.add_argument("--path_idx_kg", type=str, default="data.npy",
                        help="path to numpy ndarray")
    parser.add_argument("--path_checkpoint", type=str, default="Keci_1_14.torch"
                        )
    parser.add_argument("--path_checkpoint2", type=str, default="Keci_1_14.torch")

    parser.add_argument("--batch_size", type=int, default=10_000_000)
    parser.add_argument("--neg_sample_ratio", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--read_only", default=None)
    parser.add_argument("--lr", type=float, default=0.1)

    return parser.parse_args()

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ To avoid the excessive time spent to fetch the first batch at each new epoch
    See https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def get_data(args) -> Tuple[np.ndarray, int, int]:
    if args.path_kg:
        """ do this """
        print("Reading KG...\n")
        start_time = time.time()
        data = pl.read_parquet(source=args.path_kg,n_rows=args.read_only)
        print(f"took {time.time() - start_time}")
        print("Unique entities...")
        start_time = time.time()
        unique_entities = pl.concat((data.get_column('subject'), data.get_column('object'))).unique().rename(
            'entity')
        print(f"Number of unique entities:{len(unique_entities)}")
        unique_entities = unique_entities.to_list()
        print(f"took {time.time() - start_time}")

        print("Unique relations...")
        start_time = time.time()
        unique_relations = data.unique(subset=["relation"]).select("relation").to_series()
        print(f"Number of unique relations:{len(unique_relations)}")
        unique_relations = unique_relations.to_list()
        print(f"took {time.time() - start_time}")

        print("Entity index mapping...")
        start_time = time.time()
        entity_to_idx = {ent: idx for idx, ent in enumerate(unique_entities)}
        pickle.dump(entity_to_idx, open("entity_to_idx.p", "wb"))

        print(f"took {time.time() - start_time}")

        print("Relation index mapping...")
        start_time = time.time()
        rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
        pickle.dump(rel_to_idx, open("relation_to_idx.p","wb")) 
        print(f"took {time.time() - start_time}")
        print("Constructing training data...")
        start_time = time.time()
        data = data.with_columns(pl.col("subject").map_dict(entity_to_idx).alias("subject"),
                                 pl.col("relation").map_dict(rel_to_idx).alias("relation"),
                                 pl.col("object").map_dict(entity_to_idx).alias("object")).to_numpy()
        print(f"took {time.time() - start_time}")

        num_entities = len(unique_entities)
        num_relations = len(unique_relations)
        # TODO: maybe save the data into some designated folter
        with open("data.npy", 'wb') as f:
            np.save(f, data)

        return data, num_entities, num_relations

    elif args.path_idx_kg:
        print("Loading the index numpy KG..\n")
        #data=np.load(args.path_idx_kg, mmap_mode='r')
        with open(args.path_idx_kg, 'rb') as f:
            data = np.load(f)
        num_entities = 1 + max(max(data[:, 0]), max(data[:, 2]))
        num_relations = 1 + max(data[:, 1])
        return data, num_entities, num_relations
    else:
        raise RuntimeError


def init_model(args, num_entities, num_relations):
    start_time = time.time()
    print('Initializing models...')
    model1 = Keci(
        args={"optim": "Adam", "p": 0, "q": 1, "num_entities": num_entities, "num_relations": num_relations,
              "embedding_dim": args.embedding_dim, 'learning_rate': args.lr})
    model2 = Keci(
        args={"optim": "Adam", "p": 0, "q": 1, "num_entities": num_entities, "num_relations": num_relations,
            "embedding_dim": args.embedding_dim, 'learning_rate': args.lr})
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

        model1.load_state_dict(torch.load(args.path_checkpoint,map_location='cpu'))
        model2.load_state_dict(torch.load(args.path_checkpoint,map_location='cpu'))
        models = (model1, model2)
        optimizers = (opt1, opt2)
    return models, optimizers


def get_train_loader(args):
    data: np.ndarray
    data, num_ent, num_rel = get_data(args)
    data: torch.utils.data.DataLoader
    print('Creating dataset...')
    data: NegSampleDataset
    # TODO: neg_sample_ratio is not used at the moment
    data = NegSampleDataset(train_set=data,
                            num_entities=num_ent, num_relations=num_rel,
                            neg_sample_ratio=1.0)
    data: torch.utils.data.DataLoader
    data = MultiEpochsDataLoader(dataset=data,
            batch_size=args.batch_size, shuffle=True,
                                       num_workers=32)
    print('Number of triples', len(data.dataset))
    return data, num_ent, num_rel

def run_epoch(loss_function,dataloader,model1,model2,opt1,opt2):
    device1 = "cuda:0"
    device2 = "cuda:1"
    epoch_loss = 0.0
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
        h1, r1, t1, y = h1.pin_memory().to(device1, non_blocking=True), r1.pin_memory().to(device1,non_blocking=True), t1.pin_memory().to(device1, non_blocking=True), y.pin_memory().to(device1, non_blocking=True)
        # (3.3) Compute triple score (Forward Pass)
        yhat1 = model1.score(h1, r1, t1)

        # (3.4) Select second part of the embeddings of triples
        h2, r2, t2 = model2.get_triple_representation(x)
        # (3.5) Move (3.4) into a single GPU
        h2, r2, t2 = h2.pin_memory().to(device2, non_blocking=True), r2.pin_memory().to(device2, non_blocking=True), t2.pin_memory().to(device2, non_blocking=True)
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
        numpy_batch_loss = batch_loss.item()
        epoch_loss += numpy_batch_loss
        print(f"\tBatch Loss:{numpy_batch_loss}\tForward-Backward-Update: {time.time() - start_time}")
    print(f"Epoch Loss:{epoch_loss}")

def run(args):
    # (1) Get training data
    dataloader: torch.utils.data.DataLoader
    dataloader, num_ent, num_rel = get_train_loader(args)
    # (2) Get model
    models, optimizers = get_model(args, num_ent, num_rel)
    print("Compiling...")
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
    # @TODO: Ensure the multi-node training
    for e in range(args.num_epochs):
        epoch_loss = 0
        if e==-1:
            args.batch_size+=args.batch_size
            print(f"Increase Batch size to {args.batch_size}")
            args.batch_size+=args.batch_size
            dataloader = MultiEpochsDataLoader(dataset=dataloader.dataset,batch_size=args.batch_size, shuffle=True,num_workers=32)


        run_epoch(loss_function,dataloader,model1,model2,opt1,opt2)
        """
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
            h1, r1, t1, y = h1.pin_memory().to(device1, non_blocking=True), r1.pin_memory().to(device1,non_blocking=True), t1.pin_memory().to(device1, non_blocking=True), y.pin_memory().to(device1, non_blocking=True)
            # (3.3) Compute triple score (Forward Pass)
            yhat1 = model1.score(h1, r1, t1)

            # (3.4) Select second part of the embeddings of triples
            h2, r2, t2 = model2.get_triple_representation(x)
            # (3.5) Move (3.4) into a single GPU
            h2, r2, t2 = h2.pin_memory().to(device2, non_blocking=True), r2.pin_memory().to(device2, non_blocking=True), t2.pin_memory().to(device2, non_blocking=True)
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
            numpy_batch_loss = batch_loss.item()
            epoch_loss += numpy_batch_loss
            if ith % 1 == 0: # init an argument
                print(f"\tBatch Loss:{numpy_batch_loss}\tForward-Backward-Update: {time.time() - start_time}")
        print(f"Epoch:{e}\tEpoch Loss:{epoch_loss}")
        """
    print("Saving....")
    start_time=time.time()
    model1.to("cpu")
    model2.to("cpu")
    print(model1._orig_mod.state_dict())
    torch.save(model1._orig_mod.state_dict(),f"{model1._orig_mod.name}_1_{e}.torch")
    print(model2._orig_mod.state_dict())
    torch.save(model2._orig_mod.state_dict(),f"{model1._orig_mod.name}_2_{e}.torch")
    print('DONE')
    print(f"took {time.time() - start_time}")


if __name__ == '__main__':
    run(input_arguments())

# @TODO Post Processing
# Note mode1 and model2 keci with p=0, q=1
# model1 real_m1:[] complex_m1[]
# model2 real_m2:[] complex_m2[]
# y1 y2 => Final model = real_m1[],real_m2[], complex_m1[] complex_m2[]
