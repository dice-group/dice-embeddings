"""Generate CPU oracles from pinned upstream code; never import DICE here.

Run each method in its own process with its repository on PYTHONPATH.
Optional reference dependencies belong in an isolated environment.
"""

import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import torch

A, B, C, D = (0, (0,)), (1, (2,)), (2, (1,)), (3, (3,))
QUERIES = {
    '1p': A, '2p': (0, (0, 2)), '3p': (0, (0, 2, 1)), '4p': (0, (0, 2, 1, 3)),
    '2i': (A, B), '3i': (A, B, C), '4i': (A, B, C, D),
    'ip': ((A, B), (2,)), 'pi': ((0, (0, 2)), B),
    '2u': (A, B, (-1,)), 'up': ((A, B, (-1,)), (2,)),
    '2in': (A, (1, (2, -2))), '3in': (A, C, (1, (2, -2))),
    'inp': ((A, (1, (2, -2))), (2,)), 'pin': ((0, (0, 2)), (1, (2, -2))),
    'pni': ((0, (0, 2, -2)), B),
}
DIRECT = [(0, 0, 1), (0, 0, 2), (1, 2, 3), (2, 2, 4), (3, 0, 5), (5, 2, 6), (4, 0, 0)]
TRIPLES = sorted({*DIRECT, *((t, r ^ 1, h) for h, r, t in DIRECT)})
N, R = 7, 4


def flatten(query):
    return [value for child in query for value in flatten(child)] if isinstance(query, tuple) else [query]


def shape(query):
    if isinstance(query[0], int) and len(query) == 2:
        return 'e', tuple('n' if r == -2 else 'r' for r in query[1])
    if len(query) == 2 and all(isinstance(x, int) for x in query[1]):
        return shape(query[0]), tuple('n' if r == -2 else 'r' for r in query[1])
    return tuple(('u',) if child == (-1,) else shape(child) for child in query)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def complex_checkpoint(path):
    if path is None:
        return None
    state = torch.load(path, map_location='cpu', weights_only=True)
    while any(key in state for key in ('model', 'model_state_dict', 'state_dict')):
        state = next(state[key] for key in ('model', 'model_state_dict', 'state_dict') if key in state)
    if set(state) != {'embeddings.0.weight', 'embeddings.1.weight'}:
        raise ValueError('Expected the released ComplEx embedding tensors')
    return {key: value[:N if key == 'embeddings.0.weight' else R].clone() for key, value in state.items()}


def cone(root, checkpoint=None):
    source = module(root / 'models.py', 'cone_reference')
    names = {shape(q): name + '-DNF' if name in ('2u', 'up') else name for name, q in QUERIES.items()}
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        for key, count in (('entity_embedding', N), ('axis_embedding', R), ('arg_embedding', R)):
            state[key] = state[key][:count].clone()
    dim, gamma = (state['entity_embedding'].shape[1], state['gamma'].item()) if state else (8, 12.)
    model = source.KGReasoning(N, R, dim, gamma, query_name_dict=names, center_reg=.02)
    model.cone_proj = source.ConeProjection(dim, state['cone_proj.layer1.weight'].shape[0] if state else 16, 2)
    if state:
        model.load_state_dict(state, strict=True)
    model.eval()
    scores = {name: model(None, torch.arange(N)[None], None,
                          {shape(q): torch.tensor([flatten(q)])}, {shape(q): [0]})[1][0]
              for name, q in QUERIES.items()}
    return [dict(name='cone', state=model.state_dict(), config={'center_reg': .02}, scores=scores)]


def cqd(root, checkpoint=None):
    from cqd.base import CQD
    state = complex_checkpoint(checkpoint)
    rank = state['embeddings.0.weight'].shape[1] // 2 if state else 4
    names = {shape(q): name + '-DNF' if name in ('2u', 'up') else name for name, q in QUERIES.items() if 'n' not in name}
    filters = {}
    for h, r, t in TRIPLES:
        filters.setdefault((h, r), []).append(t)
    result = []
    for hybrid in (False, True):
        for norm in ('prod', 'min'):
            model = CQD(N, R, rank, k=2, max_k=4, max_norm=.9 if hybrid else 1., filters=filters,
                        do_normalize=True, t_norm_name=norm, query_name_dict=names).eval()
            if state:
                model.load_state_dict(state, strict=True)
            else:
                for parameter in model.parameters():
                    parameter.normal_()
            scores = {name: model(None, None, None, {shape(q): torch.tensor([flatten(q)])}, {shape(q): [0]})[1][0]
                      for name, q in QUERIES.items() if 'n' not in name}
            result.append(dict(name='cqd-hybrid' if hybrid else 'cqd', state=model.state_dict(),
                               config=dict(beam_size=2, max_k=4, tnorm=norm), scores=scores))
    model = CQD(N, R, rank, k=6, max_k=1, max_norm=1., filters=filters,
                do_normalize=True, t_norm_name='prod', query_name_dict=names).eval()
    model.load_state_dict(result[0]['state'], strict=True)
    scores = {name: model(None, None, None, {shape(QUERIES[name]): torch.tensor([flatten(QUERIES[name])])},
                          {shape(QUERIES[name]): [0]})[1][0] for name in ('ip', 'up')}
    result.append(dict(name='cqd', state=model.state_dict(),
                       config=dict(beam_size=6, max_k=1, tnorm='prod'), scores=scores))
    return result


def qto(root, checkpoint=None):
    source = module(root / 'model.py', 'qto_reference')
    from src.models import ComplEx
    state = complex_checkpoint(checkpoint)
    rank = state['embeddings.0.weight'].shape[1] // 2 if state else 4
    backbone = ComplEx([N, R, N], rank=rank, init_size=1.)
    if state:
        backbone.load_state_dict(state, strict=True)
    rows = []
    for relation in range(R):
        edges = [(h, t) for h, r, t in TRIPLES if r == relation]
        value = source.neural_adj_matrix(backbone, relation, N, 'cpu', .08, edges)
        value = (value >= 1).float() * .9999 + (value < 1).float() * value
        for h, t in edges:
            value[h, t] = 1
        rows.append(value)
    model = source.KGReasoning.__new__(source.KGReasoning)
    torch.nn.Module.__init__(model)
    model.nentity, model.device, model.fraction, model.neg_scale = N, 'cpu', 2, 3.
    model.relation_embeddings = [[row[:N // 2].to_sparse(), row[N // 2:].to_sparse()] for row in rows]
    scores = {name: model.embed_query(torch.tensor([flatten(q)]), shape(q), 0)[0][0] for name, q in QUERIES.items()}
    return [dict(name='qto', state=backbone.state_dict(), config=dict(threshold=.08, negation_scale=3.), scores=scores,
                 adjacency=torch.stack(rows))]


def ultra(root, checkpoint=None):
    from torch_geometric.data import Data
    from ultra import datasets_query, tasks  # noqa: F401 - initialize the upstream circular import
    from ultra.models import Ultra
    from ultra.query_utils import Query
    from ultra.ultraquery import UltraQuery
    triples = torch.tensor(TRIPLES)
    graph = Data(edge_index=triples[:, [0, 2]].T, edge_type=triples[:, 1], num_nodes=N, num_relations=R)
    tasks.build_relation_graph(graph)
    result = []
    for checkpoint, threshold in ((None, 0.), ('ultraquery.pth', 0.), ('ultra_3g.pth', .8)):
        dim, layers = (8, 2) if checkpoint is None else (64, 6)
        config = dict(input_dim=dim, hidden_dims=[dim] * layers, message_func='distmult', aggregate_func='sum',
                      short_cut=True, layer_norm=True)
        backbone = Ultra(dict(config, **{'class': 'RelNBFNet'}), dict(config, **{'class': 'QueryNBFNet'}))
        model = UltraQuery(backbone, threshold=threshold).eval()
        if checkpoint:
            state = torch.load(root / 'ckpts' / checkpoint, weights_only=True, map_location='cpu')['model']
            if checkpoint == 'ultraquery.pth':
                model.load_state_dict(state, strict=True)
            else:
                backbone.load_state_dict(state, strict=True)
        scores = {name: model(graph, Query.from_nested(q)[None], symbolic_traversal=False)[0] for name, q in QUERIES.items()}
        item = dict(name='ultraquery', state=model.state_dict(), config=dict(threshold=threshold), scores=scores)
        if checkpoint:
            item['checkpoint'] = checkpoint
            item['checkpoint_sha256'] = hashlib.sha256((root / 'ckpts' / checkpoint).read_bytes()).hexdigest()
        result.append(item)
    return result


def clmpt(root, checkpoint=None):
    from src.language.foq import EFO1Query
    from src.language.grammar import parse_lstr_to_lformula
    from src.pipeline.clmpt import CLMPTLayer, CLMPTReasoner
    from src.structure.nbp_complex import ComplEx
    # Independent literal formulas in the author's query language.
    formulas = {
        '1p': 'r1(s1,f)', '2p': '(r1(s1,e1))&(r2(e1,f))',
        '3p': '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))',
        '4p': '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,e3)))&(r4(e3,f))',
        '2i': '(r1(s1,f))&(r2(s2,f))', '3i': '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))',
        '4i': '(((r1(s1,f))&(r2(s2,f)))&(r3(s3,f)))&(r4(s4,f))',
        'ip': '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))',
        'pi': '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))',
        '2in': '(r1(s1,f))&(!(r2(s2,f)))', '3in': '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))',
        'inp': '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))',
        'pin': '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))',
        'pni': '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))',
    }
    values = {'1p': [0, 0], '2p': [0, 0, 2], '3p': [0, 0, 2, 1], '4p': [0, 0, 2, 1, 3],
              '2i': [0, 0, 1, 2], '3i': [0, 0, 1, 2, 2, 1], '4i': [0, 0, 1, 2, 2, 1, 3, 3],
              'ip': [0, 0, 1, 2, 2], 'pi': [0, 0, 2, 1, 2], '2in': [0, 0, 1, 2],
              '3in': [0, 0, 2, 1, 1, 2], 'inp': [0, 0, 1, 2, 2], 'pin': [0, 0, 2, 1, 2],
              'pni': [0, 0, 2, 1, 2]}
    # Grounding order is the atom traversal's first occurrence of each symbol/relation.
    import re
    state = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint else None
    if state:
        for key, count in (('nbp._entity_embedding.weight', N), ('nbp._relation_embedding.weight', R)):
            state[key] = state[key][:count].clone()
    result = []
    modes = [any(key.startswith('transformer.encoder.') for key in state)] if state else (True, False)
    for pre_norm in modes:
        rank = state['nbp._entity_embedding.weight'].shape[1] // 2 if state else 4
        backbone = ComplEx(N, R, rank, init_size=1.)
        hidden_key = 'transformer.encoder.layers.0.ffn.layer1.weight' if pre_norm else 'transformer.layers.0.linear1.weight'
        hidden = state[hidden_key].shape[0] if state else 16
        layer = CLMPTLayer(hidden, backbone, layers=2, pre_norm=pre_norm).eval()
        if state:
            layer.load_state_dict(state, strict=True)
        reasoner = CLMPTReasoner(backbone, layer)

        def score(name, override=None, batch_overrides=None):
            formula = formulas[name]
            symbols = []
            for relation, source, target in re.findall(r'(r\d+)\((\w+),(\w+)\)', formula):
                for key in (source, relation):
                    if key.startswith(('s', 'r')) and key not in symbols:
                        symbols.append(key)
            query = EFO1Query(parse_lstr_to_lformula(formula))
            rows = batch_overrides if batch_overrides is not None else [values[name] if override is None else override]
            for row in rows:
                query.append_qa_instances(dict(zip(symbols, row)), {'f': []}, {'f': []})
            reasoner.initialize_with_query(query)
            reasoner.estimate_variable_embeddings()
            scores = torch.cosine_similarity(reasoner.get_ent_emb('f')[:, None], backbone.entity_embedding, dim=-1)
            return scores if batch_overrides is not None else scores[0]

        scores = {name: score(name) for name in formulas}
        scores['2u'] = torch.maximum(score('1p'), score('1p', [1, 2]))
        scores['up'] = torch.maximum(score('2p'), score('2p', [1, 2, 2]))
        batches = {'2p': dict(queries=[QUERIES['2p'], (1, (2, 3))], scores=score('2p', batch_overrides=[[0, 0, 2], [1, 2, 3]])),
                   '2i': dict(queries=[QUERIES['2i'], ((2, (1,)), (3, (3,)))],
                              scores=score('2i', batch_overrides=[[0, 0, 1, 2], [2, 1, 3, 3]]))}
        result.append(dict(name='clmpt', state=layer.state_dict(), config={}, scores=scores, batches=batches,
                           oracle_note='DNF branches evaluated separately with the released relation groundings'))
    return result


def gnnqe(root, checkpoint=None):
    # Import compatibility only; the neural and sparse inference code is unchanged.
    import collections
    import collections.abc
    import sys
    import types
    collections.Sequence = collections.abc.Sequence
    drawing = types.ModuleType('rdkit.Chem.Draw.mplCanvas')
    drawing.Canvas = object
    sys.modules[drawing.__name__] = drawing
    from gnnqe.data import Query
    from gnnqe.gnn import NeuralBellmanFordNetwork
    from gnnqe.model import QueryExecutor
    from torchdrug import data
    triples = torch.tensor(TRIPLES)[:, [0, 2, 1]]
    graph = data.Graph(triples, num_node=N, num_relation=R)
    state = None
    if checkpoint:
        with torch.serialization.safe_globals([data.Graph]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
        state = {key.removeprefix('model.'): value for key, value in state.items() if key not in ('fact_graph', 'graph')}
        dim = state['model.query.weight'].shape[1]
        count = len({key.split('.')[3] for key in state if key.startswith('model.model.layers.')})
        state['model.query.weight'] = state['model.query.weight'][:R].clone()
        for key in state:
            if '.relation_linear.' in key:
                state[key] = state[key][:R * dim].clone()
    else:
        dim, count = 8, 2
    result = []
    for aggregate in (['pna'] if checkpoint else ['sum', 'pna']):
        concat_hidden = state is not None and state['model.mlp.layers.0.weight'].shape[1] != 2 * dim
        backbone = NeuralBellmanFordNetwork(dim, [dim] * count, R, aggregate_func=aggregate,
                                            short_cut=True, layer_norm=True, concat_hidden=concat_hidden)
        model = QueryExecutor(backbone).eval()
        if state:
            model.load_state_dict(state, strict=True)
        scores = {name: model(graph, Query.from_nested(q)[None])[0] for name, q in QUERIES.items()}
        result.append(dict(name='gnnqe', state=model.state_dict(), config=dict(aggregate=aggregate), scores=scores))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['cone', 'cqd', 'qto', 'ultra', 'clmpt', 'gnnqe'])
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260926)
    with torch.no_grad():
        if args.checkpoint and args.method == 'ultra':
            parser.error('ULTRA generation uses the named public checkpoints under upstream/ckpts')
        cases = globals()[args.method](args.upstream, args.checkpoint)
    commit = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = dict(upstream_commit=commit, queries=QUERIES, triples=TRIPLES, num_entities=N, num_relations=R, cases=cases)
    if args.checkpoint:
        with args.checkpoint.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        fixture.update(checkpoint_sha256=checksum, checkpoint=str(args.checkpoint),
                       vocabulary='first 7 entities and 4 relations; learned non-vocabulary parameters unchanged')
    torch.save(fixture, args.output)
    print(f'{args.method}: {len(cases)} cases, {sum(len(case["scores"]) for case in cases)} score vectors', flush=True)
