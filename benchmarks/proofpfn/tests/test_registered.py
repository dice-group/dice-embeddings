"""The six registered analysis files: their freeze anchors, their frozen fast checks, the registered chain (golden G8)
and, on request, their simulations. The files run in subprocesses only (``conftest.py`` keeps pytest from importing them).

The frozen ``fv1_analysis.py`` refuses any environment other than the registered one (``REGISTERED_ENV``: Python 3.12.2
with numpy 1.26.4), so its checks, the chain and the simulations skip elsewhere. The table checks, the anchors and the
pins run everywhere.
"""
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.proofpfn.evaluate import write_json
from benchmarks.proofpfn.manifests import REPO, file_sha256, read_models

from .golden import GOLDEN, fixture, fixture_bytes

REGISTERED = REPO / 'benchmarks' / 'proofpfn' / 'registered'
ANCHORS = {'fv1_analysis.py': 'eb6af63af900a9967b84f2d3c60ef07e48fd2929a75de90a57ac7c9dfd08388d',
           'test_fv1_analysis.py': 'e2bcedaccd6f76059c251b2956a0fbda482ef6817300124e5703b971269557a5',
           'fv1_tables.py': '4d2eb850e9a5ba5901be4a2d3185c0db427183f4b9edb53035726a1dddb1dcd3',
           'test_fv1_tables.py': '43dde76e06b9b19a938fa91c8742e2d5749edafefa4e2c92e70ce00670bf045e',
           'fv1_drops.py': '3ca0e50d75af33604e83779b8dc9caa9e91338ab0f28e525555aa257b6eea014',
           'fv1_devrows.py': '6af5c62d07dfdd50d948d81f852d23a697caf1ab35e0a20a99adde6440a2538a'}
REGISTERED_ENVIRONMENT = pytest.mark.skipif((platform.python_version(), np.__version__) != ('3.12.2', '1.26.4'),
                                            reason='the registered analysis refuses other environments (registration, section 16)')


def python(*args: str, cwd: Path = REGISTERED, timeout: float = 1200) -> subprocess.CompletedProcess:
    """Run Python without bytecode files in a fresh interpreter, with the repository on its path."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (str(REPO), os.environ.get('PYTHONPATH')))))
    return subprocess.run([sys.executable, '-B', *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def frozen(script: str, *args: str, timeout: float = 1200) -> subprocess.CompletedProcess:
    """Run a registered script as registered: ``python -B <script>`` in its folder."""
    return python(str(REGISTERED / script), *args, timeout=timeout)


def test_files_equal_their_freeze_anchors():
    assert {name: file_sha256(REGISTERED / name) for name in ANCHORS} == ANCHORS
    assert sorted(p.name for p in REGISTERED.glob('*.py') if p.name != '__init__.py') == sorted(ANCHORS)
    assert ANCHORS == fixture('g8/g8_analysis.json.gz')['anchors']['expected']


def test_lint_and_test_collection_leave_the_files_alone():
    ruff = (REGISTERED / 'ruff.toml').read_text()
    assert 'extend-exclude = ["*.py"]' in ruff and 'force-exclude = true' in ruff
    assert "collect_ignore_glob = ['registered/*']" in (REGISTERED.parent / 'conftest.py').read_text()


def test_registered_systems_have_the_keys_of_the_model_pins():
    analysis = (REGISTERED / 'fv1_analysis.py').read_text()
    for arm, family in read_models().families.items():
        assert f"'{arm}': ('{family.key}', OUR_SEEDS)" in analysis
    for key in ('trix', 'ultra3g', 'flockoff'):
        assert f"('{key}', (0,))" in analysis


def test_frozen_table_checks(tmp_path):
    tables = frozen('test_fv1_tables.py', '--tmp', str(tmp_path))
    assert tables.returncode == 0, tables.stdout[-3000:] + tables.stderr[-3000:]
    assert tables.stdout.strip().splitlines()[-1].startswith('PASS all checks')


@REGISTERED_ENVIRONMENT
def test_frozen_analysis_checks():
    analysis = frozen('test_fv1_analysis.py')
    assert analysis.returncode == 0, analysis.stdout[-3000:] + analysis.stderr[-3000:]
    assert analysis.stdout.strip().splitlines()[-1].startswith('0 failed checks, 0 documented limitations exceeded')


def tree_digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


@pytest.mark.slow
@REGISTERED_ENVIRONMENT
def test_registered_chain_reproduces_golden_g8(tmp_path):
    """Simulated rows -> the registered fv1_tables.py -> the registered fv1_analysis.py, byte for byte as golden G8."""
    case = tmp_path / 'expected'
    case.mkdir()
    chain = python('-m', 'benchmarks.proofpfn.tests.g8_chain', '999', '50000', cwd=case)
    assert chain.returncode == 0, chain.stdout[-3000:] + chain.stderr[-3000:]
    record = fixture('g8/g8_analysis.json.gz')['expected_design']
    assert tree_digests(case / 'rows') == record['row_files']
    assert {name: file_sha256(case / name) for name in record['files']} == record['files']
    for name in ('manifest.json', 'block3.json', 'models.json', 'gates.json', 'dropped.json', 'tables/mrr.csv.gz', 'tables/cost.csv.gz',
                 'tables/tables.meta.json.gz', 'result-b999.json.gz', 'result-b50000.json.gz'):
        assert (case / name.removesuffix('.gz')).read_bytes() == fixture_bytes(f'g8/expected/{name}'), name
    assert hashlib.sha256(gzip.decompress((GOLDEN / 'g8' / 'expected' / 'result-b999.json.gz').read_bytes())).hexdigest() == \
        record['files']['result-b999.json']
    # the clean writer serializes a row file as the research harness did, so the chain reads its files byte for byte alike
    copy = tmp_path / 'copy.json'
    for path in (case / 'rows').rglob('*.json'):
        if not any(part.startswith('.') for part in path.relative_to(case / 'rows').parts):
            write_json(copy, json.loads(path.read_bytes()))
            assert copy.read_bytes() == path.read_bytes(), path
    shutil.rmtree(case)   # about 0.5 GB of simulated rows and tables in the temporary folder


@pytest.mark.simulate
@REGISTERED_ENVIRONMENT
@pytest.mark.skipif(os.environ.get('PROOFPFN_SIMULATE') != '1', reason='the frozen simulations take about 9 minutes; set PROOFPFN_SIMULATE=1')
def test_frozen_simulations():
    result = frozen('test_fv1_analysis.py', '--simulate', timeout=3600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert result.stdout.strip().splitlines()[-1].startswith('0 failed checks')
