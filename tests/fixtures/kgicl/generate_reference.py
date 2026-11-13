"""Generate KG-ICL fixtures with the pinned official code plus upstream-fixes.patch.

Use an isolated environment with torch, torch-scatter 2.1.2, numpy, scipy and
networkx (the reference environment used torch 2.7.1, numpy 1.24.0). No DICE
code is imported. The generator checks the checkout revision and that its
tracked files are unmodified, applies the committed patch to a temporary copy,
processes a fixture graph with the official prompt extraction, loads it with the
official data loader and records the official model's outputs on CPU:

    CUDA_VISIBLE_DEVICES='' python tests/fixtures/kgicl/generate_reference.py \
        /path/to/KG-ICL tests/fixtures/kgicl
"""
import argparse
import hashlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

COMMIT = "6a3166e347ae468acdfb30a70a2cf3608b66b8f1"
HERE = Path(__file__).resolve().parent


def patched_copy(root, workdir):
    """Pinned sources with upstream-fixes.patch applied, importable from ``workdir``."""
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert head == COMMIT, head
    assert not subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src", "run_your_own_dataset"], text=True)
    for name in ("src", "run_your_own_dataset"):
        shutil.copytree(root / name, workdir / name)
    subprocess.run(["patch", "-p1", "-s", "-d", str(workdir), "-i", str(HERE / "upstream-fixes.patch")], check=True)
    sys.path[:0] = [str(workdir / "src"), str(workdir / "run_your_own_dataset")]


def fixture_graph():
    """A small graph with hubs, cycles, parallel facts, self-loops and rare relations."""
    rng = random.Random(7)
    facts = set()
    while len(facts) < 90:
        h, t = rng.randrange(30), rng.randrange(30)
        if h != t:
            facts.add((h, rng.choice((0, 0, 0, 3)), t))
    facts |= {(30, 1, 31), (31, 1, 32), (2, 1, 30),        # three facts: fewer cases than shots
              (33, 2, 34),                                 # one fact
              (5, 4, 6), (6, 4, 33),                       # two facts
              (7, 3, 7), (8, 3, 8), (5, 0, 6), (5, 3, 6),   # self-loops and parallel facts
              (34, 0, 35), (35, 0, 36), (36, 3, 37), (37, 0, 38)}  # a chain away from the core
    train = sorted(facts)
    valid = [(0, 0, 4), (9, 3, 12), (30, 1, 2), (33, 0, 35), (14, 0, 21), (38, 0, 34), (6, 4, 5), (19, 0, 3)]
    test = [(1, 0, 2), (31, 1, 33), (33, 2, 6), (7, 3, 9), (6, 5, 10), (36, 0, 39), (12, 4, 30), (38, 3, 36)]
    return train, valid, test, 41, 6  # entity 40 never occurs; relation 5 has no training facts


def write_dataset(utils_all, train, valid, test, num_entities, num_relations, destination):
    """The official transductive layout: cases from training facts, test background = train + valid."""
    np.random.seed(0)
    random.seed(0)
    kg = utils_all.KG(train, num_entities, num_relations)
    cases = kg.build_cases_for_large_graph(case_num=25, enclosing=False, hop=3)
    for split in ("train", "valid", "test"):
        directory = destination / split
        directory.mkdir(parents=True)
        background = train + valid if split == "test" else train
        utils_all.write_triple(directory / "background.txt", background)
        utils_all.write_triple(directory / "facts.txt", {"train": train, "valid": valid, "test": test}[split])
        utils_all.write_triple(directory / "filter.txt", train + valid + (test if split == "test" else []))
        utils_all.write_dict(directory / "entity2id.txt", {f"e{i}": i for i in range(num_entities)})
        utils_all.write_dict(directory / "relation2id.txt", {f"r{i}": i for i in range(num_relations)})
        for relation in range(num_relations):
            (directory / "cases" / f"r{relation}").mkdir(parents=True)
            for i, case in enumerate(cases[relation]):
                utils_all.write_cases(str(directory / "cases" / f"r{relation}" / str(i)), case)
    extracted = {}
    for relation, items in cases.items():
        extracted[relation] = []
        for local_triples, labels, entities in items:
            order = sorted(entities)
            example = local_triples[0]
            extracted[relation].append(dict(
                example=(entities[example[0]], entities[example[2]]), entities=[entities[k] for k in order],
                labels=[list(map(int, labels[k])) for k in order],
                triples=sorted({(entities[h], int(r), entities[t]) for h, r, t in local_triples})))
    return extracted


def settings(dim, attn_dim, layers, prompt_layers):
    return SimpleNamespace(
        finetune=False, train_batch_size=8, test_batch_size=64, shot=5, device=torch.device("cpu"),
        use_attn=True, attn_type="Sigmoid", AGG="max", AGG_rel="max", MSG="concat", use_augment=False,
        use_token_set=True, use_prompt_graph=True, prompt_graph_type="all", path_hop=3, hidden_dim=dim,
        attn_dim=attn_dim, n_relation_encoder_layer=prompt_layers, n_layer=layers, act="idd", dropout=0.0,
        relation_mask_rate=0.0, use_rspmm=False)


def slot_prompts(loader, relation, shot):
    """Per-shot prompt graphs exactly as the patched get_case_graph assembles them."""
    kg = loader.kg
    edge_index, edge_type, heads, tails, _, _, labels, _ = kg.get_case_graph([relation] * shot, False)
    graphs, node_start, edge_start = [], 0, 0
    for slot in range(shot):
        case_edges, _, _, _, nodes = kg.case_select(relation, id=slot)
        edges = case_edges.shape[1]
        graphs.append(dict(edge_index=edge_index[:, edge_start:edge_start + edges] - node_start,
                           edge_type=edge_type[edge_start:edge_start + edges] - slot * kg.relation_num,
                           labels=labels[node_start:node_start + nodes],
                           head=int(heads[slot]) - node_start, tail=int(tails[slot]) - node_start))
        node_start, edge_start = node_start + nodes, edge_start + edges
    return graphs


def generate(root, destination, checkpoint_dir=None):
    torch.set_num_threads(2)
    workdir = Path(tempfile.mkdtemp(prefix="kgicl-"))
    # The official loader derives split paths by replacing these words.
    assert "train" not in str(workdir) and "test" not in str(workdir) and "valid" not in str(workdir)
    try:
        patched_copy(root, workdir)
        import utils_all
        from data_loader import DataLoader
        from encoder.EntityEncoder import EntityEncoder

        train, valid, test, num_entities, num_relations = fixture_graph()
        data = workdir / "fixture"
        extracted = write_dataset(utils_all, train, valid, test, num_entities, num_relations, data)
        queries = torch.tensor([[0, 0], [5, 3], [30, 1], [33, 2], [5, 4], [7, 3], [6, 5], [40, 0],
                                [2, 6], [31, 7], [34, 8], [6, 10], [10, 11], [33, 9], [36, 6], [35, 0]])
        variants = [("tiny", 8, 3, 3, 2), ("official", 32, 5, 6, 3)]
        for name, dim, attn_dim, layers, prompt_layers in variants:
            args = settings(dim, attn_dim, layers, prompt_layers)
            loader = DataLoader(args, str(data / "test") + "/", "fixture")
            torch.manual_seed(42)
            model = EntityEncoder(args)
            fixture = dict(upstream_commit=COMMIT, patch_sha256=hashlib.sha256((HERE / "upstream-fixes.patch").read_bytes()).hexdigest(),
                           num_entities=num_entities, num_relations=num_relations, dim=dim, attn_dim=attn_dim,
                           num_layers=layers, prompt_layers=prompt_layers, shots=args.shot,
                           graph=torch.tensor(sorted(map(tuple, loader.kg.background[loader.kg.background[:, 1] < num_relations].tolist()))),
                           train=torch.tensor(train), valid=torch.tensor(valid), queries=queries,
                           answer_distance=[float(loader.kg.answer_distance[i]) for i in range(10)], cases=extracted)
            if name == "tiny":
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.3 * torch.randn_like(parameter))
                fixture["state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                if checkpoint_dir is None:
                    continue
                path = Path(checkpoint_dir) / "model_best.tar"
                state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
                model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()}, strict=True)
                fixture["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            model.eval()
            relations = sorted(set(queries[:, 1].tolist()))
            fixture["prompts"] = {q: slot_prompts(loader, q, args.shot) for q in relations}
            with torch.no_grad():
                fixture["prompt_outputs"] = {}
                for q in relations:
                    batch = loader.get_case_graph(np.array([q] * args.shot), False)
                    output = model.relation_encoder(*batch, loader, shot=args.shot)[0]
                    fixture["prompt_outputs"][q] = output[0]
                subs, rels = queries[:, 0].numpy(), queries[:, 1].numpy()
                fixture["scores"] = model(subs, rels, loader=loader, training=False)[0]
                single = torch.cat([model(subs[i:i + 1], rels[i:i + 1], loader=loader, training=False)[0]
                                    for i in range(len(subs))])
                torch.testing.assert_close(single, fixture["scores"], atol=1e-5, rtol=1e-5)
                loader.kg.answer_distance = {i: 1.0 for i in range(10)}
                fixture["unmasked_scores"] = model(subs, rels, loader=loader, training=False)[0]
            # Evaluation-mode gradients are deterministic after fix B1.
            weights = torch.randn(fixture["unmasked_scores"].shape, generator=torch.Generator().manual_seed(3))
            (model(subs, rels, loader=loader, training=False)[0] * weights).sum().backward()
            fixture["gradient_weights"] = weights
            fixture["gradients"] = {k: (None if p.grad is None else p.grad.clone()) for k, p in model.named_parameters()}
            torch.save(fixture, destination / f"{name}.pt")
            print(name, "generated", flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path, help="Pinned nju-websoft/KG-ICL checkout")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory with the official KG-ICL-6L model_best.tar")
    options = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    generate(options.upstream.resolve(), options.destination.resolve(), options.checkpoint_dir)
