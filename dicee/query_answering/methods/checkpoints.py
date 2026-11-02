"""Strict conversion of published inference checkpoints into native modules."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ...models.complex import ComplEx
from ..catalog import REFERENCES
from ..context import QueryContext, state_fingerprint
from ._common import QueryMethod
from .clmpt import CLMPT
from .cone import ConE
from .cqd import CQD
from .gnnqe import GNNQE
from .heuristic import IncomingRelationHeuristic
from .qto import QTO
from .ultraquery import UltraQuery


class _GraphPayload:
    """Data-only receiver for the graph buffers in released TorchDrug weights."""


def read_state(path, *, graph_context=None, graph_metadata=None, inductive=False):
    allowed = [(_GraphPayload, 'torchdrug.data.graph.Graph')] if graph_context is not None else []
    with torch.serialization.safe_globals(allowed):
        value = torch.load(path, map_location='cpu', weights_only=True)
    while isinstance(value, dict) and not all(isinstance(tensor, (torch.Tensor, _GraphPayload)) for tensor in value.values()):
        keys = [key for key in ('model', 'model_state_dict', 'state_dict') if key in value]
        if len(keys) != 1:
            raise ValueError('Expected an unambiguous tensor state dictionary')
        value = value[keys[0]]
    if isinstance(value, dict):
        value = dict(value)
        if inductive and any(not isinstance(value.get(key), _GraphPayload)
                             for key in ('train_graph', 'valid_graph', 'test_graph')):
            raise ValueError('Missing inductive checkpoint graph buffers')
        for key, graph in list(value.items()):
            if not isinstance(graph, _GraphPayload):
                continue
            if key not in (('train_graph', 'valid_graph', 'test_graph') if inductive else ('fact_graph', 'graph')):
                raise ValueError(f'Unexpected TorchDrug graph buffer {key}')
            _validate_graph_buffer(key, graph, graph_context, graph_metadata, inductive=inductive)
            del value[key]
        if inductive:
            for key in ('valid_nodes', 'test_nodes'):
                nodes = value.pop(key, None)
                if (not isinstance(nodes, torch.Tensor) or nodes.dtype != torch.long or nodes.ndim != 1
                        or (nodes < 0).any() or (nodes >= graph_context.num_entities).any()
                        or len(nodes.unique()) != len(nodes)):
                    raise ValueError(f'Invalid inductive checkpoint node buffer {key}')
                if graph_metadata is not None:
                    graph_metadata[key] = dict(sha256=state_fingerprint({'nodes': nodes}), used_for_inference=False)
    if not isinstance(value, dict) or not value or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()):
        raise ValueError('Expected a nonempty tensor state dictionary')
    if all(key.startswith('module.') for key in value):
        value = {key.removeprefix('module.'): tensor for key, tensor in value.items()}
    if any(not torch.isfinite(tensor).all() for tensor in value.values()):
        raise ValueError('Checkpoint contains nonfinite values')
    return value


def _validate_graph_buffer(name, graph, context, metadata, *, inductive=False):
    edges, weights = graph._edge_list, graph._edge_weight
    if (not isinstance(edges, torch.Tensor) or not isinstance(weights, torch.Tensor)
            or not 0 < int(graph.num_node) <= context.num_entities
            or (not inductive and int(graph.num_node) != context.num_entities)
            or int(graph.num_relation) != context.num_relations
            or edges.ndim != 2 or edges.shape[1] != 3 or edges.dtype != torch.long
            or weights.shape != (len(edges),) or not (weights == 1).all()):
        raise ValueError('TorchDrug graph buffer does not match the public vocabulary / unit-edge protocol')
    if (edges < 0).any() or (edges[:, :2] >= int(graph.num_node)).any() or (edges[:, 2] >= context.num_relations).any():
        raise ValueError('TorchDrug graph buffer contains out-of-vocabulary IDs')
    if name in ('fact_graph', 'train_graph'):
        observed = torch.tensor(context.triples, dtype=torch.long).reshape(-1, 3)
        keys = ((observed[:, 0] * context.num_relations + observed[:, 1]) * context.num_entities + observed[:, 2]).sort().values
        requested = (edges[:, 0] * context.num_relations + edges[:, 2]) * context.num_entities + edges[:, 1]
        positions = torch.searchsorted(keys, requested)
        if not len(keys) or (positions >= len(keys)).any() or not torch.equal(keys[positions], requested):
            raise ValueError('Checkpoint training facts are absent from the supplied graph; verify entity/relation ID maps')
    if metadata is not None:
        metadata[name] = dict(edges=len(edges), sha256=state_fingerprint({'edges': edges}),
                              used_for_inference=False, training_facts_verified=name in ('fact_graph', 'train_graph'))


def _strip(state, prefixes, sentinel):
    for prefix in prefixes:
        if prefix + sentinel in state and all(key.startswith(prefix) for key in state):
            return {key[len(prefix):]: tensor for key, tensor in state.items()}
    raise ValueError(f'Unrecognized checkpoint layout: expected {sentinel}')


def _complex(state, context):
    layouts = [('entity_embeddings.weight', 'relation_embeddings.weight'),
               ('embeddings.0.weight', 'embeddings.1.weight'),
               ('_entity_embedding.weight', '_relation_embedding.weight')]
    for entity_key, relation_key in layouts:
        if set(state) == {entity_key, relation_key}:
            entities, relations = state[entity_key], state[relation_key]
            if (entities.ndim != 2 or relations.ndim != 2 or entities.shape[1] != relations.shape[1]
                    or entities.shape[1] % 2 or entities.shape[0] != context.num_entities
                    or relations.shape[0] != context.num_relations):
                raise ValueError('ComplEx checkpoint dimensions do not match the public vocabulary')
            model = ComplEx(dict(model='ComplEx', num_entities=context.num_entities, num_relations=context.num_relations,
                                 embedding_dim=entities.shape[1], normalization=None)).to(dtype=entities.dtype)
            model.load_state_dict({'entity_embeddings.weight': entities, 'relation_embeddings.weight': relations}, strict=True)
            return model
    raise ValueError('Unsupported ComplEx checkpoint keys')


def load_method(method: str, path: str | Path, context: QueryContext, *, device: str | torch.device = 'cpu',
                **config: Any) -> QueryMethod:
    """Load a published method's released checkpoint into its native implementation.

    Every learned tensor loads strictly with ``weights_only=True``; entity and
    relation IDs are never inferred or permuted.

    Args:
        method: A name from ``REFERENCES``.
        path: Released checkpoint, or the heuristic's specification file.
        context: Inference graph whose IDs match the checkpoint.
        device: Inference device.
        **config: Method options, such as CQD's ``beam_size`` or QTO's ``threshold``.

    Returns:
        The method in evaluation mode, with checkpoint provenance in ``provenance``.

    Raises:
        ValueError: For unknown methods, or checkpoints that do not match.
    """
    if method not in REFERENCES:
        raise ValueError(f'Unknown method {method}')
    path = Path(path)
    graph_metadata = {}
    if method == 'incoming-relation':
        expected = dict(version=1, method=method, reference_commit=REFERENCES[method][1],
                        ranking='query-seeded random shuffle within Boolean classes')
        if json.loads(path.read_text()) != expected:
            raise ValueError('Invalid untrained heuristic specification')
        model = IncomingRelationHeuristic(context, **config)
        state = None
    else:
        state = read_state(path, graph_context=context if method in ('gnnqe', 'inductive-gnnqe') else None,
                           graph_metadata=graph_metadata, inductive=method == 'inductive-gnnqe')
    if method == 'incoming-relation':
        pass
    elif method in ('cqd', 'cqd-hybrid', 'qto'):
        backbone = _complex(state, context)
        model = (QTO(backbone, context, **config) if method == 'qto' else
                 CQD(backbone, context, hybrid=method == 'cqd-hybrid', **config))
    elif method in ('ultraquery', 'ultraquery-lp'):
        state = _strip(state, ('', 'model.model.', 'model.'), 'relation_model.layers.0.linear.weight')
        dim = state['relation_model.layers.0.linear.weight'].shape[0]
        layers = len({key.split('.')[2] for key in state if key.startswith('relation_model.layers.')})
        model = UltraQuery(context, dim=dim, num_layers=layers, **config)
        model.load_state_dict(state, strict=True)
    elif method == 'cone':
        dim = state['entity_embedding'].shape[1]
        projection_dim = state['cone_proj.layer1.weight'].shape[0]
        projection_layers = len([key for key in state if key.startswith('cone_proj.layer') and key.endswith('.weight')]) - 1
        model = ConE(context, dim=dim, gamma=state['gamma'].item(), projection_dim=projection_dim,
                     projection_layers=projection_layers, **config)
        model.load_state_dict(state, strict=True)
    elif method == 'clmpt':
        nbp = {key.removeprefix('nbp.'): tensor for key, tensor in state.items() if key.startswith('nbp.')}
        backbone = _complex(nbp, context)
        pre_norm = any(key.startswith('transformer.encoder.') for key in state)
        prefix = 'transformer.encoder.layers.' if pre_norm else 'transformer.layers.'
        hidden_key = prefix + ('0.ffn.layer1.weight' if pre_norm else '0.linear1.weight')
        layers = len({key[len(prefix):].split('.')[0] for key in state if key.startswith(prefix)})
        model = CLMPT(backbone, context, pre_norm=pre_norm, layers=layers, hidden_dim=state[hidden_key].shape[0], **config)
        translated = {key: tensor for key, tensor in state.items() if not key.startswith('nbp.')}
        translated.update({'model.' + key: tensor for key, tensor in backbone.state_dict().items()})
        model.load_state_dict(translated, strict=True)
    else:
        state = _strip(state, ('', 'model.'), 'model.query.weight')
        translated = {}
        for key, tensor in state.items():
            if key.startswith('model.model.layers.'):
                target = key.removeprefix('model.model.')
            elif key.startswith(('model.query.', 'model.mlp.')):
                target = key.removeprefix('model.')
            else:
                raise ValueError(f'Unexpected GNN-QE parameter {key}')
            translated[target] = tensor
        dim = translated['query.weight'].shape[1]
        layers = len({key.split('.')[1] for key in translated if key.startswith('layers.')})
        mlp_layers = len({key.split('.')[2] for key in translated if key.startswith('mlp.layers.')})
        config.setdefault('concat_hidden', translated['mlp.layers.0.weight'].shape[1] != 2 * dim)
        config.setdefault('layer_norm', 'layers.0.layer_norm.weight' in translated)
        config.setdefault('dependent', 'layers.0.relation_linear.weight' in translated)
        if translated['layers.0.linear.weight'].shape[1] == 13 * dim:
            config.setdefault('aggregate', 'pna')
        model = GNNQE(context, dim=dim, num_layers=layers, mlp_layers=mlp_layers, **config)
        model.load_state_dict(translated, strict=True)
    model.to(device).eval().requires_grad_(False)
    repo, commit = REFERENCES[method]
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    model.provenance = dict(method=method, reference=f'https://github.com/{repo}/tree/{commit}',
                            reference_commit=commit, checkpoint=str(path.resolve()),
                            checkpoint_sha256=checksum,
                            model_state_sha256=state_fingerprint(model), configuration=config,
                            public_ids='unchanged; checkpoint must use the dataset ID maps',
                            context_sha256=context.identity, checkpoint_graph_buffers=graph_metadata)
    if isinstance(model, CLMPT):
        model.provenance.update(post_norm_batch_coupling=not model.pre_norm,
                                dnf_up='separate grounded branches matching the released evaluation helper')
    if isinstance(model, CQD) and model.atomic_negation:
        model.provenance.update(
            negation_reference='https://github.com/EdinburghNLP/adaptive-cqd/tree/642ce042708be6247087c09c780b9deb47e941d3',
            negation_scope='CQD-A signed-atom negation with +H scoring')
    return model
