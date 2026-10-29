"""Access to the ProofPFN golden fixtures of the tests.

Small fixtures are copies in ``tests/fixtures/proofpfn`` (see its README). Large golden files stay outside the
repository: a test that needs one calls ``golden_large`` and skips unless the environment variable ``PROOFPFN_GOLDEN``
names the proofpfn-golden folder (or its ``golden`` subfolder).
"""
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest

from dicee.synthetic_kg.priors import FamilySpec, PriorConfig

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'proofpfn'
# Research hook modules of the record and real families (module:function 'sample'); s5_null:<generator> is the null.
RESEARCH_GENERATORS = {'pf_prior2f': 'categorical2', 'pf_rec1': 'rec1', 'pf_attr2': 'attr2', 'pf_list1': 'list1', 'pt_real': 'real'}


def research_prior(entry: dict) -> PriorConfig:
    """The prior of a research arm, from the --mix string, order name and hook records of a G1 arm entry."""
    specs = []
    for name, hook in sorted(entry['hooks'].items()):
        params = dict(hook['params'])
        budget = params.pop('swap_budget', None)
        if hook['module'] == 's5_null':
            generator = hook['function']
            assert budget is not None
        else:
            generator = RESEARCH_GENERATORS[hook['module']]
            assert hook['function'] == 'sample' and budget is None
        pairs = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in params.items())
        specs.append(FamilySpec(name, generator, hook['first'], pairs, budget))
    return PriorConfig(entry['mix_string'], entry['order_name'], tuple(specs))


def fixture(name: str) -> Any:
    """A JSON fixture of ``tests/fixtures/proofpfn``, gzip-compressed when the name ends with ``.gz``."""
    data = (FIXTURES / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)


def golden_large(name: str) -> Path:
    """A file below ``golden/large`` of the proofpfn-golden folder; skips the test when it is not available."""
    root = os.environ.get('PROOFPFN_GOLDEN')
    if root:
        for folder in (Path(root) / 'golden' / 'large', Path(root) / 'large'):
            if (folder / name).is_file():
                return folder / name
    pytest.skip(f'golden file large/{name} not available (set PROOFPFN_GOLDEN to the proofpfn-golden folder)')


def typed(value: Any) -> Any:
    """JSON with the Python type of every leaf, in the layout of the 3a research dumps (tuples, ints, floats kept apart)."""
    if isinstance(value, dict):
        return {str(k): typed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [typed(v) for v in value]}
    return {'type': type(value).__name__, 'value': value}
