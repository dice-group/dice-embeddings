"""One query grammar and composition algebra for inference and supervision."""

import torch

ONE = ('e', ('r',))
QUERY_SHAPES = {
    '1p': ONE, '2p': ('e', ('r', 'r')), '3p': ('e', ('r', 'r', 'r')),
    '2i': (ONE, ONE), '3i': (ONE, ONE, ONE),
    'ip': ((ONE, ONE), ('r',)), 'pi': (('e', ('r', 'r')), ONE),
    '2u': (ONE, ONE, ('u',)), 'up': ((ONE, ONE, ('u',)), ('r',)),
    '2in': (ONE, ('e', ('r', 'n'))), '3in': (ONE, ONE, ('e', ('r', 'n'))),
    'inp': ((ONE, ('e', ('r', 'n'))), ('r',)),
    'pin': (('e', ('r', 'r')), ('e', ('r', 'n'))),
    'pni': (('e', ('r', 'r', 'n')), ONE),
}


def nested(value):
    return tuple(nested(v) for v in value) if isinstance(value, (tuple, list)) else value


def index_query(query_type, query, entities, relations):
    """Translate named queries by structure, so operator-like names remain valid IDs."""
    if query_type not in QUERY_SHAPES:
        raise ValueError(f'Unknown query type {query_type!r}; choose from {sorted(QUERY_SHAPES)}')

    def visit(pattern, value):
        if pattern in ('e', 'r'):
            mapping = entities if pattern == 'e' else relations
            if value not in mapping:
                raise ValueError(f'Unknown {"entity" if pattern == "e" else "relation"}: {value!r}')
            return mapping[value]
        if pattern == 'n':
            if value not in ('n', 'not', -2):
                raise ValueError('Expected a negation marker (not, n, or -2)')
            return -2
        if pattern == 'u':
            if value not in ('u', 'union', 'or', -1):
                raise ValueError('Expected a union marker (union, u, or -1)')
            return -1
        # The named API historically inferred union from query_type alone.
        if isinstance(value, (tuple, list)) and pattern[-1] == ('u',) and len(value) == len(pattern) - 1:
            value = (*value, ('union',))
        if not isinstance(value, (tuple, list)) or len(pattern) != len(value):
            raise ValueError(f'Query does not match {query_type}: expected {pattern!r}')
        return tuple(visit(p, v) for p, v in zip(pattern, value))

    return visit(QUERY_SHAPES[query_type], query)


def compile_query(query):
    query = nested(query)
    if type(query) is int and query >= 0:
        return ('anchor', query)
    if not isinstance(query, tuple) or not query:
        raise ValueError('Expected an indexed structured query')
    if len(query) == 2 and isinstance(query[1], tuple) and query[1] and all(type(r) is int for r in query[1]) and query[1] != (-1,):
        node = compile_query(query[0])
        for relation in query[1]:
            if relation == -2:
                node = ('not', node)
            elif relation >= 0:
                node = ('project', relation, node)
            else:
                raise ValueError('Unknown relation/negation marker')
        return node
    union = query[-1] == (-1,)
    branches = query[:-1] if union else query
    if len(branches) < 2:
        raise ValueError('Composition requires at least two branches')
    return ('or' if union else 'and', *(compile_query(q) for q in branches))


def validate_tree(node, n, nr):
    if node[0] == 'anchor':
        if not 0 <= node[1] < n:
            raise ValueError('Anchor outside the entity vocabulary')
    elif node[0] == 'project':
        if not 0 <= node[1] < nr:
            raise ValueError('Relation outside the relation vocabulary')
        validate_tree(node[2], n, nr)
    else:
        for child in node[1:]:
            validate_tree(child, n, nr)


def positive(node):
    if node[0] == 'not':
        return False
    return all(positive(child) for child in node[1:] if isinstance(child, tuple))


def exact_answers(node, outgoing, num_entities):
    """Finite-domain set semantics shared by observed proofs and query generation."""
    op = node[0]
    if op == 'anchor':
        return {node[1]}
    if op == 'project':
        return set().union(*(outgoing.get(h, {}).get(node[1], set()) for h in exact_answers(node[2], outgoing, num_entities)))
    if op == 'not':
        return set(range(num_entities)) - exact_answers(node[1], outgoing, num_entities)
    branches = [exact_answers(c, outgoing, num_entities) for c in node[1:]]
    return set.intersection(*branches) if op == 'and' else set.union(*branches)


def log_complement(value):
    """Stable log(1-exp(value)), including exact zero/one memberships.

    Masked evaluation avoids undefined derivatives in inactive torch.where branches.
    """
    result = torch.empty_like(value)
    low = value < -0.6931471805599453
    result[low] = torch.log1p(-value[low].exp())
    result[~low] = torch.log(-torch.expm1(value[~low]))
    return result


def combine(values, operator, tnorm='prod', *, logits=False):
    values = torch.stack(values)
    if logits:
        if operator == 'and':
            return values.prod(0) if tnorm == 'prod' else values.min(0).values
        return 1 - (1 - values).prod(0) if tnorm == 'prod' else values.max(0).values
    if operator == 'and':
        return values.sum(0) if tnorm == 'prod' else values.min(0).values
    return log_complement(log_complement(values).sum(0)) if tnorm == 'prod' else values.max(0).values


def negate(value, norm='standard', parameter=0., *, logits=False):
    if not logits and norm == 'standard':
        return log_complement(value)
    p = value if logits else value.exp()
    if norm == 'standard':
        result = 1 - p
    elif norm == 'sugeno':
        result = (1 - p) / (1 + parameter * p)
    else:
        result = (1 - p.pow(parameter)).clamp_min(0).pow(1 / parameter)
    return result if logits else result.log()
