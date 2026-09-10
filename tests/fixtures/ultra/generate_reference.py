"""Generate fixtures using an unmodified, pinned upstream ULTRA checkout.

Run in an isolated environment with torch 2.5.1+cpu, torch-geometric 2.6.1,
torch-scatter 2.1.2 and ninja. Set PYTHONPATH to the upstream checkout and
MAX_JOBS=1 when compiling its rspmm extension. No DICE model is imported.
"""
import argparse
import hashlib
from pathlib import Path

import torch
from torch_geometric.data import Data
from ultra import tasks
from ultra.models import Ultra


def generate(root, destination):
    torch.set_num_threads(2)
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0]])
    edges = torch.cat((triples[:, [0, 2, 1]], triples[:, [2, 0, 1]] + torch.tensor([0, 0, 3])))
    graph = Data(edge_index=edges[:, :2].T, edge_type=edges[:, 2], num_nodes=5, num_relations=6)
    tasks.build_relation_graph(graph)
    queries = triples[[0, 2, 4]][:, [0, 2, 1]]
    tails, heads = tasks.all_negative(graph, queries)
    # Fixed strict negative groups in upstream h,t,r order.
    grouped = torch.tensor([[[0, 1, 0], [0, 0, 0], [0, 4, 0]],
                            [[1, 2, 1], [0, 2, 1], [4, 2, 1]]])
    for name in ['tiny', 'ultra_3g', 'ultra_4g', 'ultra_50g']:
        dim, layers = (8, 2) if name == 'tiny' else (64, 6)
        config = dict(input_dim=dim, hidden_dims=[dim] * layers, message_func='distmult',
                      aggregate_func='sum', short_cut=True, layer_norm=True)
        torch.manual_seed(42)
        model = Ultra(dict(config, **{'class': 'RelNBFNet'}), dict(config, **{'class': 'EntityNBFNet'}))
        fixture = dict(triples=triples, queries=queries[:, [0, 2, 1]], grouped=grouped[:, :, [0, 2, 1]],
                       num_entities=5, num_relations=3, dim=dim, num_layers=layers,
                       rel_edge_index=graph.relation_graph.edge_index, rel_edge_type=graph.relation_graph.edge_type,
                       upstream_commit='427966ad8ed60420eef034063d44f3153addff90')
        if name == 'tiny':
            fixture['state_dict'] = model.state_dict()
        else:
            path = root / 'ckpts' / (name + '.pth')
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'], strict=True)
            fixture['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        model.eval()
        with torch.no_grad():
            fixture['relations'] = model.relation_model(graph.relation_graph, queries[:, 2])
            fixture['tails'] = model(graph, tails)
            fixture['heads'] = model(graph, heads)
        model.train()
        training_scores = model(graph, grouped)
        fixture['training_scores'] = training_scores.detach()
        if name == 'tiny':
            training_scores.sum().backward()
            fixture['gradients'] = {key: parameter.grad for key, parameter in model.named_parameters()}
        torch.save(fixture, destination / (name + '.pt'))
        print(name, fixture['tails'].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('upstream', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    generate(args.upstream, args.destination)
