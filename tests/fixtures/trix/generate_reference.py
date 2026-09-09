"""Generate fixtures with the unmodified official TRIX models and rspmm kernel.

Use an isolated environment with torch 2.5.1, torch-geometric 2.4.0,
torch-scatter 2.1.2, easydict and ninja. Set PYTHONPATH=<TRIX checkout>/src
and CUDA_VISIBLE_DEVICES='' to build/run its CPU kernel. No DICE code is used.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

import torch
from easydict import EasyDict
from torch_geometric.data import Data
from trix import models_entity, models_relation, tasks

COMMIT = "7596e14eefefe89e61396205a0550172cadeddb0"


def generate(root, destination):
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == COMMIT
    assert not subprocess.check_output(["git", "-C", str(root), "diff", "--", "src"], text=True)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0],
                            [0, 1, 1], [4, 2, 3], [3, 0, 5], [5, 1, 4], [2, 2, 2], [5, 2, 0]])
    edges = torch.cat((triples[:, [0, 2, 1]], triples[:, [2, 0, 1]] + torch.tensor([0, 0, 3])))
    graph = Data(edge_index=edges[:, :2].T, edge_type=edges[:, 2], num_nodes=7, num_relations=torch.tensor(6))
    tasks.build_relation_graph(graph)
    queries = torch.tensor([[0, 1, 4], [4, 2, 1], [1, 0, 5], [2, 2, 2], [6, 1, 0]])
    tails, heads = tasks.all_negative(graph, queries[:, [0, 2, 1]])
    relation_batch = tasks.all_negative_relation(graph, queries[:, [0, 2, 1]])
    # Mixed tail/head corruption, fixed candidates in upstream (h,t,r) order.
    entity_grouped = torch.tensor([[[0, 1, 0], [0, 0, 0], [0, 6, 0]],
                                   [[1, 2, 1], [0, 2, 1], [6, 2, 1]]])
    relation_grouped = torch.tensor([[[0, 1, 0], [0, 1, 1], [0, 1, 2]],
                                     [[2, 2, 2], [2, 2, 0], [2, 2, 1]]])
    for task in ("entity", "relation"):
        for tiny in (True, False):
            dim = 8 if tiny else 32
            def config(layers):
                return dict(input_dim=dim, hidden_dims=[dim] * layers, message_func="distmult",
                            aggregate_func="sum", short_cut=True, layer_norm=True)
            torch.manual_seed(42)
            if task == "entity":
                model = models_entity.TRIX(config(3), config(2), config(4))
                grouped = entity_grouped
            else:
                model = models_relation.TRIX(config(2), config(2), EasyDict(num_layer=3, feature_dim=dim, num_mlp_layer=2))
                grouped = relation_grouped
            fixture = dict(triples=triples, queries=queries, grouped=grouped[:, :, [0, 2, 1]],
                           num_entities=7, num_relations=3, dim=dim, upstream_commit=COMMIT,
                           relation_graph={role: (g.edge_index, g.edge_type) for role, g in graph.relation_adj.items()})
            if tiny:
                fixture["state_dict"] = model.state_dict()
            else:
                path = root / (task + "_prediction.pth")
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
                fixture["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            model.eval()
            with torch.no_grad():
                if task == "entity":
                    fixture["relations"] = model.relation(graph, tails)
                    fixture["tails"] = model(graph, tails)
                    fixture["heads"] = model(graph, heads)
                else:
                    fixture["scores"] = model(graph, relation_batch)
            model.train()
            scores = model(graph, grouped)
            fixture["training_scores"] = scores.detach()
            scores.sum().backward()
            fixture["gradients"] = {name: p.grad for name, p in model.named_parameters()}
            name = task + ("_tiny" if tiny else "_prediction")
            torch.save(fixture, destination / (name + ".pt"))
            print(name, "generated", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.upstream, args.destination)
