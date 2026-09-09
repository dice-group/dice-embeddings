"""Generate Flock fixtures using unmodified upstream models and compiled walker.

Run once per task with PYTHONPATH=<upstream>/src_entity or src_relation in an
isolated environment containing PyG 2.4.0, torch-scatter, easydict and the
upstream graph-walker package. No DICE code is imported. Only the walks method
is instrumented to record its return values and the graph it was given.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

import graph_walker
import numpy as np
import torch
from flock.models import Flock, to_csr_tensor
from torch_geometric.data import Data

COMMIT = "f35103d25a78bdf4075de5c673a51de4979aa4d7"


def generate(root, destination, task):
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == COMMIT
    assert not subprocess.check_output(["git", "-C", str(root), "diff", "--", "src_entity", "src_relation", "graph-walker/src"], text=True)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0],
                            [0, 1, 1], [4, 2, 3], [3, 0, 5], [5, 1, 4], [2, 2, 2], [5, 2, 0]])
    edges = torch.cat((triples[:, [0, 2, 1]], triples[:, [2, 0, 1]] + torch.tensor([0, 0, 3])))
    graph = Data(edge_index=edges[:, :2].T, edge_type=edges[:, 2], num_nodes=7,
                 num_relations=torch.tensor(6), device=torch.device("cpu"))
    queries = torch.tensor([[0, 1, 4], [2, 2, 2], [6, 1, 0]])
    entity_grouped = torch.tensor([[[0, 0, 1], [0, 0, 0], [0, 0, 6]],
                                   [[1, 1, 2], [0, 1, 2], [6, 1, 2]]])
    relation_grouped = torch.tensor([[[0, 0, 1], [0, 1, 1], [0, 2, 1]],
                                     [[2, 2, 2], [2, 0, 2], [2, 1, 2]]])
    for tiny in (True, False):
        dim, length, refinements = (8, 8, 2) if tiny else (64, 128, 6)
        config = dict(verbose=False, walk_num=2, walk_len=length, record_neighbors=False, refinements=refinements,
                      attention_scatter=True, attention_scatter_n_heads=4, additive_refinement=True,
                      embed_only_first_refinement=False, embedding_tying_across_refinements=False,
                      parameter_tying_across_refinements=False, net="gru", hidden_dim=dim, n_layers=1, dtype="float32")
        torch.manual_seed(42)
        np.random.seed(42)
        model = Flock(config)
        fixture = dict(triples=triples, queries=queries, num_entities=7, num_relations=3,
                       dim=dim, walk_num=2, walk_len=length, refinements=refinements,
                       upstream_commit=COMMIT, cases={})
        if tiny:
            fixture["state_dict"] = model.state_dict()
        else:
            path = root / "checkpoints" / ("flock_" + task + ".pth")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
            fixture["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        captured = []
        original_walks = model.walks

        def capture(data, *args, **kwargs):
            records = original_walks(data, *args, **kwargs)
            captured.append(dict(records=records, edge_index=data.edge_index.clone(), edge_type=data.edge_type.clone()))
            return records

        model.walks = capture
        groups = queries[:, None].repeat(1, 7 if task == "entity" else 3, 1)
        groups[..., 2 if task == "entity" else 1] = torch.arange(groups.shape[1])
        cases = [("tails" if task == "entity" else "relations", groups, False)]
        if task == "entity":
            groups = queries[:, None].repeat(1, 7, 1)
            groups[..., 0] = torch.arange(7)
            cases.append(("heads", groups, False))
        cases.append(("training", entity_grouped if task == "entity" else relation_grouped, True))
        for name, groups, training in cases:
            model.train(training)
            torch.manual_seed(100 + len(captured))
            np.random.seed(200 + len(captured))
            with torch.set_grad_enabled(training):
                scores = model(graph, groups[:, :, [0, 2, 1]])
            case = captured[-1]
            case.update(groups=groups, scores=scores.detach())
            fixture["cases"][name] = case
            if training:
                scores.sum().backward()
                fixture["gradients"] = {name: p.grad for name, p in model.named_parameters()}
        name = task + ("_tiny" if tiny else "_pretrained")
        torch.save(fixture, destination / (name + ".pt"))
        print(name, "generated", flush=True)

    if task == "entity":
        # Compare edge-type/direction distributions independently of the model.
        walks = np.tile(np.array([[0, 1, 2, 2, 6, 6]], dtype=np.uint32), (10000, 1))
        types, directions = graph_walker._parse_edge_types_and_directions(
            walks, np.zeros_like(walks, dtype=bool), *to_csr_tensor(graph.edge_index, graph.edge_type, 7),
            *to_csr_tensor(graph.edge_index[[1, 0]], graph.edge_type, 7), 42)
        types = torch.tensor(types.astype(np.int32), dtype=torch.long)
        directions = torch.tensor(directions.astype(np.int32), dtype=torch.long)
        types[types == -1], directions[directions == -1] = 6, 3
        counts = torch.stack([torch.bincount(types[:, i] * 4 + directions[:, i], minlength=28) for i in range(walks.shape[1])])
        torch.save(dict(walk=torch.tensor(walks[0].astype(np.int64)), counts=counts, triples=triples), destination / "parser.pt")
        untyped = graph.edge_index.unique(dim=1)
        untyped = untyped[:, untyped[0] != untyped[1]]
        prefixes = torch.arange(7).repeat(2000)
        walks, _ = graph_walker.random_walks_fast(
            Data(edge_index=untyped, num_nodes=7, is_directed_hash=False), n_walks=1, walk_len=6,
            no_backtrack=True, prefix=prefixes, seed=991, verbose=False)
        walks = torch.tensor(walks.astype(np.int64))
        first = torch.bincount(walks[:, 0] * 7 + walks[:, 1], minlength=49).view(7, 7)
        second = torch.bincount((walks[:, 0] * 7 + walks[:, 1]) * 7 + walks[:, 2], minlength=343).view(49, 7)
        torch.save(dict(first=first, second=second, triples=triples), destination / "transitions.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("task", choices=("entity", "relation"))
    args = parser.parse_args()
    generate(args.upstream, args.destination, args.task)
