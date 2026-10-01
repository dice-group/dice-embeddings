"""Published complex-query benchmarks, the upstream code they reproduce, and recipe rules.

The benchmark launcher loads this file without PyTorch, so it imports nothing.
"""

# Query structures in the released grammar: 'e' anchor, 'r' relation, 'n' negation, 'u' union.
_ONE = ('e', ('r',))
QUERY_SHAPES = {
    '1p': _ONE, '2p': ('e', ('r', 'r')), '3p': ('e', ('r', 'r', 'r')),
    '2i': (_ONE, _ONE), '3i': (_ONE, _ONE, _ONE),
    'ip': ((_ONE, _ONE), ('r',)), 'pi': (('e', ('r', 'r')), _ONE),
    '2u': (_ONE, _ONE, ('u',)), 'up': ((_ONE, _ONE, ('u',)), ('r',)),
    '2in': (_ONE, ('e', ('r', 'n'))), '3in': (_ONE, _ONE, ('e', ('r', 'n'))),
    'inp': ((_ONE, ('e', ('r', 'n'))), ('r',)),
    'pin': (('e', ('r', 'r')), ('e', ('r', 'n'))),
    'pni': (('e', ('r', 'r', 'n')), _ONE),
    '4p': ('e', ('r', 'r', 'r', 'r')), '4i': (_ONE, _ONE, _ONE, _ONE),
}
ULTRAQUERY_SHAPES = ('1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up', '2in', '3in', 'inp', 'pin', 'pni')
PLUS_H_SHAPES = (*ULTRAQUERY_SHAPES, '4p', '4i')

# Released datasets of the UltraQuery suite, by public name.
TRANSDUCTIVE = {'FB15k237LogicalQuery': 'FB15k-237-betae', 'FB15kLogicalQuery': 'FB15k-betae', 'NELL995LogicalQuery': 'NELL-betae'}
INDUCTIVE_VERSIONS = ('106', '113', '122', '134', '150', '175', '217', '300', '550')
WIKITOPICS = ('art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax')
BENCHMARK_DATASETS = (*TRANSDUCTIVE, *(f'InductiveFB15k237Query:{v}' for v in INDUCTIVE_VERSIONS),
                      *(f'WikiTopicsQuery:{v}' for v in WIKITOPICS))

# The harder +H suite and its single release archive.
PLUS_H_FOLDERS = {'FB15k237+H': 'FB15k-237+H', 'NELL995+H': 'NELL995+H', 'ICEWS18+H': 'ICEWS18+H'}
PLUS_H_DATASETS = tuple(PLUS_H_FOLDERS)
PLUS_H_ARCHIVE = 'https://github.com/april-tools/is-cqa-complex/releases/download/benchs-1.0/iscqa-compl-benchmarks.zip'
PLUS_H_PREFIX = 'iscqa-compl-benchmarks/new_benchmarks'

# Pinned upstream implementations: method -> (repository, commit).
ULTRA = ('DeepGraphLearning/ULTRA', '427966ad8ed60420eef034063d44f3153addff90')
TRIX = ('yuchengz99/TRIX', '7596e14eefefe89e61396205a0550172cadeddb0')
INDUCTIVE_QE = ('DeepGraphLearning/InductiveQE', 'ab49a0e64e3449e687e484a42269d9898e1d1325')
IS_CQA = ('april-tools/is-cqa-complex', 'd1ce74164936a7c09d9147e83190da047cb39429')
REFERENCES = {
    'ultraquery': ULTRA,
    'ultraquery-lp': ULTRA,
    'inductive-gnnqe': INDUCTIVE_QE,
    'incoming-relation': INDUCTIVE_QE,
    'gnnqe': ('DeepGraphLearning/GNN-QE', '6620fe3d8d77ceed019f36541e84d67599d4f259'),
    'cqd': IS_CQA,
    'cqd-hybrid': IS_CQA,
    'qto': ('bys0318/QTO', '31f46b970d0f075e6b2be9c733cc11b41365d5fe'),
    'cone': ('MIRALab-USTC/QE-ConE', 'cc45b90e3cdce0f609670a257e6f756d28495a93'),
    'clmpt': ('qianlima-lab/CLMPT', '6c4f3b8a5e052e8bc1c0c93eaed18ef869a1bcd7'),
}
KGFM_ADAPTERS = ('ultra-adapter', 'trix-adapter')
METHODS = (*REFERENCES, *KGFM_ADAPTERS)
# Methods whose checkpoints score queries without the inference graph.
GRAPH_INDEPENDENT_METHODS = ('cone', 'clmpt', 'cqd')


def dataset_spec(name: str) -> tuple[str, str, str]:
    """Locate a released benchmark.

    Args:
        name: A name from ``BENCHMARK_DATASETS`` or ``PLUS_H_DATASETS``.

    Returns:
        The setting (``'transductive'``, ``'inductive-e'`` or ``'inductive-er'``),
        the dataset folder below the data root, and the release archive URL.

    Raises:
        ValueError: If ``name`` is not a released benchmark.
    """
    if name in PLUS_H_FOLDERS:
        return 'transductive', f'{PLUS_H_PREFIX}/{PLUS_H_FOLDERS[name]}', PLUS_H_ARCHIVE
    if name in TRANSDUCTIVE:
        return 'transductive', TRANSDUCTIVE[name], 'https://snap.stanford.edu/betae/KG_data.zip'
    family, _, version = name.partition(':')
    if family in ('InductiveFB15k237Query', 'InductiveFB15k237QueryExtendedEval') and version in INDUCTIVE_VERSIONS:
        return 'inductive-e', version, f'https://zenodo.org/records/7306046/files/{version}.zip'
    if family == 'WikiTopicsQuery' and version in WIKITOPICS:
        return 'inductive-er', f'WikiTopics_QE/{version}', 'https://reltrans.s3.us-east-2.amazonaws.com/WikiTopics_QE.zip'
    raise ValueError(f'Unknown benchmark {name!r}; choose from BENCHMARK_DATASETS or PLUS_H_DATASETS')


def query_types_for_dataset(name: str) -> tuple[str, ...]:
    """The query types released for benchmark ``name``: 16 for +H, 14 otherwise."""
    dataset_spec(name)
    return PLUS_H_SHAPES if name in PLUS_H_DATASETS else ULTRAQUERY_SHAPES


def inference_graph_for_split(name: str, split: str, inference_graph: str | None = None) -> str | None:
    """The facts that answer +H queries of ``split``; None for datasets with released inference graphs.

    Args:
        name: Benchmark name.
        split: ``'valid'`` or ``'test'``.
        inference_graph: ``'train'`` or ``'train+valid'`` for +H test queries
            (default: train+valid). Validation always uses training facts.

    Raises:
        ValueError: For an unknown graph, or an override outside +H.
    """
    if inference_graph not in (None, 'train', 'train+valid'):
        raise ValueError('Inference graph must be train or train+valid')
    if name not in PLUS_H_DATASETS:
        if inference_graph is not None:
            raise ValueError('Inference graph overrides are only supported for +H')
        return None
    return (inference_graph or 'train+valid') if split == 'test' else 'train'


def check_recipe(entry: dict, shapes: list[str] | tuple[str, ...]) -> None:
    """Check the method rules of an evaluation recipe.

    Shared by the evaluator and the benchmark manifests, so both reject the
    same recipes before any data or checkpoint is opened.

    Args:
        entry: Recipe with ``method`` and optional ``options``, ``query_batch_size``
            and, for KGFM adapters, ``operators``, ``adapters``, ``selection_protocol``
            and ``adapter_ablation``.
        shapes: Query types the recipe evaluates.

    Raises:
        ValueError: If the recipe breaks a rule of its method.
    """
    method, options = entry['method'], entry.get('options', {})
    if method not in METHODS:
        raise ValueError(f'Choose a supported query method: {", ".join(METHODS)}')
    if method in ('cqd', 'cqd-hybrid') and any('n' in shape for shape in shapes) and not options.get('atomic_negation'):
        raise ValueError('CQD negated query types require atomic_negation=true; otherwise select positive query types')
    size = entry.get('query_batch_size', 1)
    if type(size) is not int or size < 1:
        raise ValueError('Query batch size must be positive')
    if method not in KGFM_ADAPTERS:
        return
    if options.get('observed_facts', 'none') not in ('none', 'atomic', 'both'):
        raise ValueError('Observed facts must be none, atomic, or both')
    if type(entry.get('adapter_ablation', False)) is not bool:
        raise ValueError('Adapter ablation must be a boolean')
    if entry.get('selection_protocol') not in ('source-validation', 'target-validation'):
        raise ValueError('Declare the operator selection protocol')
    operators = entry.get('operators', {})
    if set(operators) != set(shapes) or set(operators.values()) - {'product', 'min'} or set(operators.values()) - entry.get('adapters', {}).keys():
        raise ValueError('Each query type needs one operator, product or min, with its matching adapter')
