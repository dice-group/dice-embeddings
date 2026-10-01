"""Export full-candidate validation oracles from upstream code, without DICE imports."""

import argparse
import collections
import collections.abc
import hashlib
import importlib.util
import json
import pickle
import platform
import random
import re
import subprocess
import sys
import types
from pathlib import Path
from types import FunctionType

import numpy as np
import torch
from torch.overrides import TorchFunctionMode


class ReferenceAttention(TorchFunctionMode):
    """Partition independent heads; retain each complete reference query batch."""
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.multi_head_attention_forward:
            namespace = func.__globals__ | {'scaled_dot_product_attention': lambda *a, **kw: self.__torch_function__(
                torch.nn.functional.scaled_dot_product_attention, (), a, kw)}
            copied = FunctionType(func.__code__, namespace, func.__name__, func.__defaults__, func.__closure__)
            return copied(*args, **kwargs)
        if func is torch.nn.functional.scaled_dot_product_attention:
            q, k, v = args[:3]
            if q.is_cuda and q.shape[-3] > 1 and q.shape[-2] * k.shape[-2] > 1024**2:
                mask = args[3] if len(args) > 3 else kwargs.get('attn_mask')
                dropout = args[4] if len(args) > 4 else kwargs.get('dropout_p', 0)
                if mask is not None or dropout or kwargs.get('enable_gqa', False):
                    raise ValueError('Reference attention expects unmasked inference')
                heads = [func(q.narrow(-3, i, 1), k.narrow(-3, i, 1), v.narrow(-3, i, 1), *args[3:], **kwargs)
                         for i in range(q.shape[-3])]
                return torch.cat(heads, dim=-3)
        return func(*args, **kwargs)


FORMULAS = {
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def flatten(value):
    return [i for part in value for i in flatten(part)] if isinstance(value, tuple) else [value]


def structure(query):
    if isinstance(query[0], int):
        return 'e', tuple('n' if r == -2 else 'r' for r in query[1])
    if len(query) == 2 and all(isinstance(r, int) for r in query[1]):
        return structure(query[0]), tuple('n' if r == -2 else 'r' for r in query[1])
    return tuple(('u',) if part == (-1,) else structure(part) for part in query)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# The pinned upstream revisions, read from the import-free catalog (no DICE imports).
COMMITS = {method: commit for method, (_, commit) in
           module(Path(__file__).resolve().parents[3] / 'dicee/query_answering/catalog.py', 'cqa_catalog').REFERENCES.items()}


def load_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    while any(key in state for key in ('model', 'model_state_dict', 'state_dict')):
        state = next(state[key] for key in ('model', 'model_state_dict', 'state_dict') if key in state)
    return state


def reference_dataset(bundle, entry, root):
    """Read published layouts independently; never import DICE's dataset loader."""
    name = entry['dataset']
    plus_h = {'FB15k237+H': 'FB15k-237+H', 'NELL995+H': 'NELL995+H', 'ICEWS18+H': 'ICEWS18+H'}
    transductive = {'FB15k237LogicalQuery': 'FB15k-237-betae', 'FB15kLogicalQuery': 'FB15k-betae',
                    'NELL995LogicalQuery': 'NELL-betae'}
    if name in plus_h:
        folder = Path('iscqa-compl-benchmarks/new_benchmarks') / plus_h[name]
    elif name in transductive:
        folder = Path(transductive[name])
    else:
        family, version = name.split(':')
        if family not in ('InductiveFB15k237Query', 'WikiTopicsQuery'):
            raise ValueError('Unsupported reference dataset')
        folder = Path(version) if family == 'InductiveFB15k237Query' else Path('WikiTopics_QE') / version
    prefix = str(Path(bundle['manifest']['data_root']) / folder) + '/'
    raw = Path(root) / prefix

    def unpickle(filename):
        with (raw / filename).open('rb') as stream:
            return pickle.load(stream)

    def triples(stem):
        path = raw / (stem + '.txt')
        if path.is_file():
            return [tuple(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]
        return list(map(tuple, torch.load(raw / (stem + '.pt'), map_location='cpu', weights_only=True).tolist()))

    def context(edges, n, nr, pairs):
        inverse = {a: b for h, t in pairs for a, b in ((h, t), (t, h))}
        edges = sorted(set(edges) | {(t, inverse[r], h) for h, r, t in edges if r in inverse})
        return dict(triples=edges, num_entities=n, num_relations=nr, inverse_relations=sorted(pairs))

    if name in plus_h or name in transductive:
        n, nr = len(unpickle('id2ent.pkl')), len(unpickle('id2rel.pkl'))
        stem = 'KG_splits/train' if name == 'ICEWS18+H' else 'train'
        train = triples(stem)
        test = train
        if name in plus_h and (entry.get('inference_graph') or 'train+valid') == 'train+valid':
            test = train + triples('KG_splits/valid' if name == 'ICEWS18+H' else 'valid')
        pairs = [(r, r + 1) for r in range(0, nr, 2)]
        graphs = {'valid': context(train, n, nr, pairs), 'test': context(test, n, nr, pairs)}
        structured, easy, hard = [unpickle('valid-' + kind + '.pkl') for kind in ('queries', 'easy-answers', 'hard-answers')]
    else:
        train, test = triples('train_graph'), triples('test_inference')
        if family == 'InductiveFB15k237Query':
            valid = triples('val_inference')
            vocabulary = train + valid + test
            n = max(entity for h, _, t in vocabulary for entity in (h, t)) + 1
            nr = max(r for _, r, _ in vocabulary) + 1
            pairs = [(r, r + nr // 2) for r in range(nr // 2)]
            graphs = {split: context(train + edges, n, nr, pairs) for split, edges in [('valid', valid), ('test', test)]}
        else:
            mapping = unpickle('og_mappings.pkl')['r2id'] if (raw / 'og_mappings.pkl').is_file() else None
            graphs = {}
            for split, edges in [('valid', train), ('test', test)]:
                n = max(entity for h, _, t in edges for entity in (h, t)) + 1
                nr = max(r for _, r, _ in edges) + 1
                pairs = ([(r, mapping[label + '_inv']) for label, r in mapping.items()
                          if label + '_inv' in mapping and r < nr and mapping[label + '_inv'] < nr]
                         if mapping else [(r, r + nr // 2) for r in range(nr // 2)])
                graphs[split] = context(edges, n, nr, pairs)
        structured, grouped_easy, grouped_hard = [unpickle('valid_' + kind + '.pkl')
                                                 for kind in ('queries', 'answers_easy', 'answers_hard')]
        easy = {query: answers for group in grouped_easy.values() for query, answers in group.items()}
        hard = {query: answers for group in grouped_hard.values() for query, answers in group.items()}
    graph = graphs['valid']
    return prefix, structured, easy, hard, graph, {split: digest(value) for split, value in graphs.items()}


def predictor(entry, upstream, checkpoint, n, nr, triples, queries, device, probe_queries=None):
    method, options = entry['method'], entry['options']
    names = {structure(query): shape + '-DNF' if shape in ('2u', 'up') else shape for shape, query in queries.values()}
    if method == 'cone':
        source, state = module(upstream / 'models.py', 'reference_cone'), load_state(checkpoint)
        dim = state['entity_embedding'].shape[1]
        model = source.KGReasoning(n, nr, dim, state['gamma'].item(), use_cuda=device.startswith('cuda'),
                                   query_name_dict=names, center_reg=options['center_reg'])
        model.cone_proj = source.ConeProjection(dim, state['cone_proj.layer1.weight'].shape[0], 2)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        def predict(shape, batch):
            encoded = {structure(batch[0]): torch.tensor([flatten(q) for q in batch], device=device)}
            indices = {structure(batch[0]): list(range(len(batch)))}
            return torch.cat([model(None, torch.arange(start, min(start + 1024, n), device=device)[None]
                                    .expand(len(batch), -1), None, encoded, indices)[1]
                              for start in range(0, n, 1024)], dim=-1)
        return predict
    if method in ('cqd', 'cqd-hybrid'):
        from cqd.base import CQD
        state = load_state(checkpoint)
        filters = collections.defaultdict(list)
        for h, r, t in triples:
            filters[h, r].append(t)
        model = CQD(n, nr, state['embeddings.0.weight'].shape[1] // 2, k=options['beam_size'], max_k=options['max_k'],
                    max_norm=.9 if method == 'cqd-hybrid' else 1., filters=filters,
                    do_normalize=options['normalize'], t_norm_name=options['tnorm'], query_name_dict=names)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        def predict(shape, batch):
            settings = options | options['per_shape'].get(shape, {})
            model.k, model.t_norm_name, model.max_k = settings['beam_size'], settings['tnorm'], settings['max_k']
            with torch.device(device):
                if settings.get('reference_batching') is False:
                    from cqd import discrete
                    from reference_cqd import query_bounded
                    return query_bounded(model, discrete, shape, torch.tensor([flatten(q) for q in batch], device=device),
                                         settings['row_batch_size'])
                if shape == '4p' and options.get('final_batch_size') and device.startswith('cuda'):
                    from cqd import discrete
                    from reference_cqd import query_4p
                    return query_4p(model, discrete, torch.tensor([flatten(q) for q in batch], device=device),
                                    options['final_batch_size'])
                return model(None, None, None, {structure(batch[0]): torch.tensor([flatten(q) for q in batch], device=device)},
                             {structure(batch[0]): list(range(len(batch)))})[1]
        return predict
    if method == 'clmpt':
        from src.language.foq import EFO1Query
        from src.language.grammar import parse_lstr_to_lformula
        from src.pipeline.clmpt import CLMPTLayer, CLMPTReasoner
        from src.structure.nbp_complex import ComplEx
        state = load_state(checkpoint)
        pre_norm = any(key.startswith('transformer.encoder.') for key in state)
        backbone = ComplEx(n, nr, state['nbp._entity_embedding.weight'].shape[1] // 2, device=device)
        hidden = state['transformer.encoder.layers.0.ffn.layer1.weight' if pre_norm else 'transformer.layers.0.linear1.weight'].shape[0]
        model = CLMPTLayer(hidden, backbone, layers=2, pre_norm=pre_norm, agg_func=options['aggregation'])
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        reasoner = CLMPTReasoner(backbone, model)

        def atomic(shape, batch):
            formula = FORMULAS[shape]
            symbols = []
            for relation, source, _ in re.findall(r'(r\d+)\((\w+),(\w+)\)', formula):
                for key in (source, relation):
                    if key.startswith(('s', 'r')) and key not in symbols:
                        symbols.append(key)
            query = EFO1Query(parse_lstr_to_lformula(formula))
            for row in batch:
                query.append_qa_instances(dict(zip(symbols, [value for value in flatten(row) if value >= 0])), {'f': []}, {'f': []})
            reasoner.initialize_with_query(query)
            with ReferenceAttention():
                reasoner.estimate_variable_embeddings()
            embeddings = reasoner.get_ent_emb('f')[:probe_queries]
            return torch.stack([torch.cat([torch.cosine_similarity(row[None], entities, dim=-1)
                                           for entities in backbone.entity_embedding.split(options['candidate_batch_size'])]) for row in embeddings])

        def predict(shape, batch):
            if shape == '2u':
                return torch.maximum(atomic('1p', [q[0] for q in batch]), atomic('1p', [q[1] for q in batch]))
            if shape == 'up':
                return torch.maximum(*[atomic('2p', [(q[0][branch][0], (*q[0][branch][1], *q[1])) for q in batch]) for branch in (0, 1)])
            return atomic(shape, batch)
        return predict
    if method in ('ultraquery', 'ultraquery-lp'):
        from torch_geometric.data import Data
        from ultra import datasets_query, tasks  # noqa: F401
        from ultra.models import Ultra
        from ultra.query_utils import Query
        from ultra.ultraquery import UltraQuery
        edges = torch.tensor(triples)
        graph = Data(edge_index=edges[:, [0, 2]].T, edge_type=edges[:, 1], num_nodes=n, num_relations=nr)
        tasks.build_relation_graph(graph)
        graph = graph.to(device)
        state = load_state(checkpoint)
        sentinel = 'relation_model.layers.0.linear.weight'
        prefix = next(p for p in ('model.model.', 'model.', '') if p + sentinel in state)
        state = {k[len(prefix):]: v for k, v in state.items()}
        key = sentinel
        dim = state[key].shape[0]
        layers = len({k.split('.')[2] for k in state if k.startswith('relation_model.layers.')})
        config = dict(input_dim=dim, hidden_dims=[dim] * layers, message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True)
        backbone = Ultra(dict(config, **{'class': 'RelNBFNet'}), dict(config, **{'class': 'QueryNBFNet'}))
        model = UltraQuery(backbone, threshold=options['threshold'], logic=options['logic'])
        backbone.load_state_dict(state, strict=True)
        model.to(device).eval()
        return lambda shape, batch: model(graph, torch.stack([Query.from_nested(q) for q in batch]).to(device), symbolic_traversal=False)
    if method in ('gnnqe', 'inductive-gnnqe', 'incoming-relation'):
        collections.Sequence = collections.abc.Sequence
        drawing = types.ModuleType('rdkit.Chem.Draw.mplCanvas')
        drawing.Canvas = object
        sys.modules[drawing.__name__] = drawing
        from gnnqe.data import Query
        from gnnqe.gnn import NeuralBellmanFordNetwork
        from gnnqe.model import QueryExecutor
        from torchdrug import data
        from torchdrug.layers import functional
        if not hasattr(functional, '_size_to_index'):
            functional._size_to_index = lambda size: torch.arange(len(size), device=size.device).repeat_interleave(size)
        graph = data.Graph(torch.tensor(triples)[:, [0, 2, 1]], num_node=n, num_relation=nr).to(device)
        if method == 'incoming-relation':
            from gnnqe.heuristic_baseline import HeuristicBaseline
            model = HeuristicBaseline().to(device).eval()
            seed = options.get('seed', torch.initial_seed())
            # Building TorchDrug's random perfect-hash index must not consume
            # the query's random ranking stream.
            graph.match(torch.tensor([[-1, -1, r] for r in range(nr)], device=device))

            def heuristic_predict(shape, batch):
                scores = []
                devices = [torch.device(device).index or 0] if device.startswith('cuda') else []
                for q in batch:
                    payload = json.dumps([seed, flatten(q)], separators=(',', ':')).encode()
                    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63)
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(local_seed)
                        scores.append(model(graph, Query.from_nested(q)[None].to(device))[0])
                return torch.stack(scores)
            return heuristic_predict
        with torch.serialization.safe_globals([data.Graph]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
        buffers = ('train_graph', 'valid_graph', 'test_graph', 'valid_nodes', 'test_nodes') if method == 'inductive-gnnqe' else ('fact_graph', 'graph')
        state = {key.removeprefix('model.'): value for key, value in state.items() if key not in buffers}
        dim = state['model.query.weight'].shape[1]
        count = len({key.split('.')[3] for key in state if key.startswith('model.model.layers.')})
        backbone = NeuralBellmanFordNetwork(dim, [dim] * count, nr, aggregate_func='pna', short_cut=True, layer_norm=True,
                                            concat_hidden=state['model.mlp.layers.0.weight'].shape[1] != 2 * dim)
        model = QueryExecutor(backbone, logic=options['logic'])
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return lambda shape, batch: model(graph, torch.stack([Query.from_nested(q) for q in batch]).to(device))
    if method == 'qto':
        source = module(upstream / 'model.py', 'reference_qto')
        from reference_qto import ReferenceRelations
        from src.models import ComplEx
        state = load_state(checkpoint)
        backbone = ComplEx([n, nr, n], rank=state['embeddings.0.weight'].shape[1] // 2, init_size=1.)
        backbone.load_state_dict(state, strict=True)
        backbone.to(device).eval()

        model = source.KGReasoning.__new__(source.KGReasoning)
        torch.nn.Module.__init__(model)
        model.nentity, model.device, model.fraction, model.neg_scale = n, device, min(n, 100), options['negation_scale']
        model.relation_embeddings = ReferenceRelations(source, backbone, triples, n, device, options['threshold'], model.fraction)
        return lambda shape, batch: model.embed_query(torch.tensor([flatten(q) for q in batch], device=device), structure(batch[0]), 0)[0]
    raise ValueError('Only upstream baselines have an independent reference exporter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--entry', required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pilot-queries', type=int, default=2)
    parser.add_argument('--probe-only', action='store_true', help='Score only the pilot queries after encoding complete reference batches')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.pilot_queries < 1:
        parser.error('Positive pilot budget required')
    bundle = json.loads((args.bundle / 'bundle.json').read_text())
    if sha(args.bundle / 'bundle.json') != json.loads((args.bundle / 'bundle.sha256.json').read_text())['sha256']:
        raise ValueError('Bundle checksum mismatch')
    entries = {entry['id']: entry for entry in bundle['manifest']['entries']}
    if args.entry not in entries:
        parser.error(f'No frozen entry {args.entry!r} in this bundle')
    entry = entries[args.entry]
    commit = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMITS.get(entry['method']):
        raise ValueError('Unexpected upstream revision')
    if subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Upstream tracked files differ from the pinned revision')
    for relative, expected in bundle['files'].items():
        if sha(args.input_root / relative) != expected:
            raise ValueError(f'Input checksum mismatch: {relative}')
    torch.set_num_threads(bundle['manifest']['threads'])
    seed = bundle['manifest']['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    sys.path.insert(0, str(args.upstream.resolve()))
    if entry['method'] == 'qto':
        sys.path.insert(0, str(args.upstream.resolve() / 'kbc'))
    prefix, structured, easy, hard, graph, graphs = reference_dataset(bundle, entry, args.input_root)
    n, nr, triples = graph['num_entities'], graph['num_relations'], graph['triples']
    name_shapes = {}
    for shape in entry['query_types']:
        # Infer structure labels independently from the source query syntax.
        example = {'1p': (0, (0,)), '2p': (0, (0, 0)), '3p': (0, (0, 0, 0)), '4p': (0, (0, 0, 0, 0)),
                   '2i': ((0, (0,)), (1, (0,))), '3i': ((0, (0,)), (1, (0,)), (2, (0,))),
                   '4i': ((0, (0,)), (1, (0,)), (2, (0,)), (3, (0,))),
                   'ip': (((0, (0,)), (1, (0,))), (0,)), 'pi': ((0, (0, 0)), (1, (0,))),
                   '2u': ((0, (0,)), (1, (0,)), (-1,)), 'up': (((0, (0,)), (1, (0,)), (-1,)), (0,)),
                   '2in': ((0, (0,)), (1, (0, -2))), '3in': ((0, (0,)), (1, (0,)), (2, (0, -2))),
                   'inp': (((0, (0,)), (1, (0, -2))), (0,)), 'pin': ((0, (0, 0)), (1, (0, -2))),
                   'pni': ((0, (0, 0, -2)), (1, (0,)))}[shape]
        name_shapes[structure(example)] = shape
    queries = {digest((name_shapes[struct], query, sorted(easy[query]), sorted(hard[query]))): (name_shapes[struct], query)
               for struct, group in structured.items() if struct in name_shapes for query in group}
    frozen_plan = bundle['plans'][f'{args.entry}/valid']
    if sha(args.bundle / frozen_plan['path']) != frozen_plan['sha256']:
        raise ValueError('Plan checksum mismatch')
    plan = json.loads((args.bundle / frozen_plan['path']).read_text())
    selected_inputs = {entry['checkpoint'], *entry.get('adapters', {}).values()}
    if entry.get('training'):
        selected_inputs.add(entry['training'])
    inputs = {key: value for key, value in bundle['files'].items() if key in selected_inputs or key.startswith(prefix)}
    oracle = dict(version=1, entry_sha256=digest({key: value for key, value in entry.items() if key != 'verification'}),
                  inputs_sha256=digest(inputs), validation_plan_sha256=frozen_plan['sha256'],
                  reference_commit=commit, exporter_sha256=sha(Path(__file__)), pilot_queries=args.pilot_queries,
                  probe_only=args.probe_only, scores={}, orders={})
    oracle['graphs'] = graphs
    oracle['runtime_adjustments'] = (['Set the default tensor device during CQD forward; upstream hybrid padding omits device.']
                                     if entry['method'] in ('cqd', 'cqd-hybrid') else [])
    if entry['method'] == 'qto':
        oracle['runtime_adjustments'] = ['Lazy canonical 100-head calibration; up to 100 max-reduction partitions instead of 10.']
        oracle['helpers_sha256'] = {'reference_qto.py': sha(Path(__file__).with_name('reference_qto.py'))}
    if entry['method'] == 'clmpt':
        oracle['runtime_adjustments'] = ['Partition independent attention heads; preserve complete query sequences and batches.']
    if entry['method'] == 'incoming-relation':
        oracle['runtime_adjustments'] = ['Seed each query independently before upstream random tie shuffling; preserve Boolean classes.']
    if entry['method'] in ('gnnqe', 'inductive-gnnqe', 'incoming-relation'):
        oracle['runtime_adjustments'].append('Configure TorchDrug size-to-index helper and Python/RDKit imports.')
    if entry['method'] in ('cqd', 'cqd-hybrid') and entry['options'].get('reference_batching') is False:
        oracle['runtime_adjustments'].append(
            'Bound every CQD atom stage with declared row batches and stage-global normalization before observed overrides; '
            'use upstream row-wise candidate selection and global padding, streaming final max, and released 3p/4p quirks.')
        oracle['arithmetic_profile'] = dict(name='cqd-bounded-rows-v1', row_batch_size=entry['options']['row_batch_size'],
                                            topk='row-wise', normalization='stage-global-before-observed-facts')
        oracle['helpers_sha256'] = {'reference_cqd.py': sha(Path(__file__).with_name('reference_cqd.py'))}
    elif (entry['method'] in ('cqd', 'cqd-hybrid') and entry['options'].get('final_batch_size')
            and args.device.startswith('cuda')):
        oracle['runtime_adjustments'].append('Stream final 4p projection with stage-global normalization; preserve released 4p quirks.')
        oracle['helpers_sha256'] = {'reference_cqd.py': sha(Path(__file__).with_name('reference_cqd.py'))}
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    oracle['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                                 driver=driver, precision='float32 IEEE; no autocast', threads=torch.get_num_threads())
    with torch.no_grad():
        predict = predictor(entry, args.upstream, args.input_root / entry['checkpoint'], n, nr, triples, queries,
                            args.device, args.pilot_queries if args.probe_only else None)
        counts, start = collections.Counter(), 0
        for end in plan['batch_ends']:
            keys = plan['queries'][start:end]
            shape = queries[keys[0]][0]
            if counts[shape] < args.pilot_queries:
                values = predict(shape, [queries[key][1] for key in keys])
                if args.probe_only:
                    keys = keys[:args.pilot_queries - counts[shape]]
                    values = values[:len(keys)]
                orders = values.argsort(dim=-1, descending=True)
                for key, value, order in zip(keys, values, orders):
                    oracle['scores'][key], oracle['orders'][key] = value.cpu(), order.cpu()
                counts[shape] += len(keys)
                print(json.dumps(dict(shape=shape, queries=counts[shape])), flush=True)
            start = end
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(oracle, args.output)


if __name__ == '__main__':
    main()
