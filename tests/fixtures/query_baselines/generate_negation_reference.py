"""CQD-A negation oracles with +H atomic scoring; no DICE imports."""

import argparse
import hashlib
import subprocess
from pathlib import Path

import torch
from generate_reference import flatten, module

COMMIT = '642ce042708be6247087c09c780b9deb47e941d3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError('Unexpected CQD-A source revision')
    source = module(args.upstream / 'cqda/cqd/discrete.py', 'cqda_reference')
    base = Path(__file__).with_name('cqd.pt')
    fixture = torch.load(base, weights_only=True)
    queries = {name: value for name, value in fixture['queries'].items() if 'n' in name}
    cases = []
    torch.set_num_threads(2)
    with torch.no_grad():
        for parent in fixture['cases']:
            if '1p' not in parent['scores']:
                continue
            state, hybrid = parent['state'], parent['name'] == 'cqd-hybrid'
            entity = torch.nn.Embedding.from_pretrained(state['embeddings.0.weight'])
            relation = torch.nn.Embedding.from_pretrained(state['embeddings.1.weight'])
            norm = parent['config']['tnorm']
            k = fixture['num_entities'] if hybrid else 2

            def scoring(lhs, rel, rhs):
                hr, hi = lhs.chunk(2, -1)
                rr, ri = rel.chunk(2, -1)
                tr, ti = rhs.chunk(2, -1)
                raw = (hr * rr - hi * ri) @ tr.T + (hr * ri + hi * rr) @ ti.T
                result = (.9 if hybrid else 1.) * (raw - raw.min()) / (raw.max() - raw.min())
                if hybrid:
                    heads = (lhs[:, None] == entity.weight[None]).all(-1).int().argmax(-1)
                    rels = (rel[:, None] == relation.weight[None]).all(-1).int().argmax(-1)
                    for row, (h, r) in enumerate(zip(heads.tolist(), rels.tolist())):
                        for a, b, t in fixture['triples']:
                            if (a, b) == (h, r):
                                result[row, t] = 1
                return result

            def negation(value, **kwargs):
                return 1 - value

            scores = {}
            for name, query in queries.items():
                options = {'aim_run': None} if name in ('2in', '3in') else {'k': k}
                scores[name] = getattr(source, f'query_{name}')(
                    entity, relation, torch.tensor([flatten(query)]), scoring,
                    t_norm=torch.mul if norm == 'prod' else torch.minimum, negation=negation, **options)[0]
            cases.append(dict(name=parent['name'], state=state, scores=scores,
                              config=dict(atomic_negation=True, beam_size=k, max_k=k, tnorm=norm)))
    output = {key: fixture[key] for key in ('triples', 'num_entities', 'num_relations')}
    output.update(queries=queries, cases=cases, upstream_commit=commit,
                  source='https://github.com/EdinburghNLP/adaptive-cqd',
                  parent_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  scope='CQD-A signed-atom executor with +H calibration; hybrid full-width observed override.')
    torch.save(output, args.output)
    print(f'{len(cases)} cases, {sum(len(c["scores"]) for c in cases)} negation vectors')


if __name__ == '__main__':
    main()
