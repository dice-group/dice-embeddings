"""The checkpoint format, its loader and the converter of research exports.

G5 (g5_cpu.json.gz, g5_gpu.json.gz): the clean raw and EMA states of 41 research exports of all seven arms. G3
(g3_init.json.gz): the initial adapter tensors, which the exports' frozen adapters must still equal. Scores
(umls_research_scores.json, recorded in phase 3a from the research models) and per-seed raw ranks (G6,
g6_ranks.json.gz): converted exports of S, CU and UL on UMLS. The export files are large goldens (PROOFPFN_GOLDEN).
"""
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.arms import ARMS, initial_model
from dicee.synthetic_kg.checkpoint import FORMAT, KEYS, CheckpointError, check_checkpoint, from_research_export, load_model, make_checkpoint, read_checkpoint, save_checkpoint
from dicee.synthetic_kg.determinism import CUBLAS_WORKSPACES, deterministic_execution
from dicee.synthetic_kg.digests import file_sha256, tensor_digest
from tests.proofpfn_fixtures import fixture, golden_large

ARM_OF = {arm.research_arm: arm.name for arm in ARMS.values()}
G3 = fixture('g3_init.json.gz')
INIT_OF = {'S': 'S', 'R': 'S', 'M': 'S', 'N': 'S', 'HT': 'S', 'CU': 'CU'}
UMLS = Path(__file__).resolve().parents[1] / 'KGs' / 'UMLS'


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def checkpoint_of(arm: str, seed: int = 42, **changes) -> dict:
    model = initial_model(arm, seed)
    values = dict(arm=arm, seed=seed, updates=0, weights='raw', order_sha256=ARMS[arm].pinned_order(seed), provenance={'source': 'test'})
    return make_checkpoint(model.state_dict(), **{**values, **changes})


def test_checkpoint_round_trip(tmp_path):
    checkpoint = checkpoint_of('S')
    path = save_checkpoint(checkpoint, tmp_path / 'S-42.pt')
    stored = torch.load(path, map_location='cpu', weights_only=True)
    assert set(stored) == set(KEYS) and stored['format'] == FORMAT and stored['model_config'] == ARMS['S'].model_config()
    assert tensor_digest(read_checkpoint(path)['model']) == tensor_digest(checkpoint['model'])
    before = torch.get_rng_state()
    model = load_model(path)
    assert torch.equal(torch.get_rng_state(), before)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert (model.relation_weighting, model.pair_channels) == ('normalized', 'on')
    assert tensor_digest(model.state_dict()) == tensor_digest(checkpoint['model'])
    direct = ULTRA(dict(checkpoint['model_config'])).load_pretrained(path)   # the plain dicee loader reads the format
    assert tensor_digest(direct.state_dict()) == tensor_digest(checkpoint['model'])
    with pytest.raises(ValueError, match='saved with'):
        ULTRA(ARMS['UL'].model_config()).load_pretrained(path)


def test_checkpoint_checks(tmp_path):
    with pytest.raises(CheckpointError, match='keys and shapes'):
        make_checkpoint(initial_model('S', 42).state_dict(), arm='CU', seed=42, updates=0, weights='raw',
                        order_sha256=ARMS['CU'].pinned_order(42), provenance={})
    with pytest.raises(CheckpointError, match='order_sha256'):
        checkpoint_of('S', order_sha256=ARMS['M'].pinned_order(42))
    with pytest.raises(CheckpointError, match='weights'):
        checkpoint_of('UL', weights='swa')
    checkpoint_of('UL', seed=7, order_sha256='0' * 64)   # an unpinned seed takes any order digest
    good = checkpoint_of('CU')
    for change in (dict(format='other'), dict(arm='S0'), dict(model_config=ARMS['S'].model_config()), dict(seed=-1)):
        with pytest.raises(CheckpointError):
            check_checkpoint({**good, **change})
    with pytest.raises(CheckpointError, match='keys'):
        check_checkpoint({k: v for k, v in good.items() if k != 'provenance'})
    torch.save({**good, 'format': 'other'}, tmp_path / 'bad.pt')
    with pytest.raises(CheckpointError, match='format'):
        load_model(tmp_path / 'bad.pt')


def research_exports() -> list[tuple[str, dict, str]]:
    """(golden folder below large/, G5 record, arm) of every recorded research export."""
    out = []
    for name, folder in (('g5_cpu.json.gz', 'g4'), ('g5_gpu.json.gz', 'g10')):
        for path, record in fixture(name)['exports'].items():
            out.append((f'{folder}/{path}', record, ARM_OF[path.split('/')[1]]))
    return out


EXPORTS = research_exports()


def test_exports_cover_every_arm():
    assert len(EXPORTS) == 41 and {arm for _, _, arm in EXPORTS} == set(ARMS)


@pytest.mark.parametrize('relative, record, arm', EXPORTS, ids=[e[0].split('/', 2)[2] for e in EXPORTS])
def test_research_exports_convert_to_the_golden_states(relative, record, arm, tmp_path):
    path = golden_large(relative)
    assert file_sha256(path) == record['file_sha256']
    for weights in ('raw', 'ema'):
        checkpoint = from_research_export(path, arm, weights)
        assert tensor_digest(checkpoint['model']) == record['weights'][weights]['clean']['digest'], weights
        assert (checkpoint['arm'], checkpoint['seed'], checkpoint['updates'], checkpoint['weights']) == (arm, record['fields']['seed'], record['fields']['updates'], weights)
        assert checkpoint['order_sha256'] == record['fields']['run']['order_sha256'] == ARMS[arm].pinned_order(checkpoint['seed'])
        provenance = checkpoint['provenance']
        assert provenance['research_export_sha256'] == record['file_sha256'] and provenance['research_arm'] == record['fields']['run']['arm']
        expected_dropped = set(record['weights'][weights]['inert'].get('dropped', []))
        assert {k for k in provenance['dropped'] if not k.endswith(':]')} == expected_dropped
    model = load_model(save_checkpoint(checkpoint, tmp_path / 'converted.pt'))
    assert tensor_digest(model.state_dict()) == record['weights']['ema']['clean']['digest']
    if arm in INIT_OF:   # the frozen adapters and the markers still have their initial values (golden G3)
        initial = G3['arms'][INIT_OF[arm]][str(checkpoint['seed'])]['dropped']['per_key']
        saved = torch.load(path, map_location='cpu', weights_only=True)['arms']['decay']
        for weights in ('raw', 'ema'):
            for key, value in saved[weights].items():
                if key.startswith(('context_adapter.', 'role_adapter.', '_')):
                    assert tensor_digest({key: value}) == initial[key], (weights, key)


def export_path(arm: str, seed: int = 42) -> Path:
    return golden_large(f'g4/golden-runs/{ARMS[arm].research_arm}/training/{seed}/export-20.pt')


def changed(tmp_path, arm: str, change) -> Path:
    saved = torch.load(export_path(arm), map_location='cpu', weights_only=True)
    change(saved)
    torch.save(saved, tmp_path / 'changed.pt')
    return tmp_path / 'changed.pt'


def test_converter_refuses_exports_that_are_not_inert_or_not_the_arms(tmp_path):
    def output(saved):
        saved['arms']['decay']['ema']['context_adapter.2.weight'][0, 0] = 1e-3

    def trained(saved):
        saved['arms']['decay']['raw']['role_adapter.0.weight'][0, 0] += 1.0

    def marker(saved):
        saved['arms']['decay']['raw']['_pair_model_mode'][1] = 0

    def pair_row(saved):
        saved['arms']['decay']['ema']['relation_model.layers.3.relation.weight'][5, 0] = 1e-6

    def restart(saved):
        saved['run']['warm_restart'] = 1000

    for arm, change, message in (('S', output, 'adapter output'), ('S', trained, 'trained'), ('S', marker, 'marker'),
                                 ('CU', pair_row, 'pair rows'), ('S', restart, 'warm_restart')):
        with pytest.raises(CheckpointError, match=message):
            from_research_export(changed(tmp_path, arm, change), arm)
    for other, message in (('CU', 'pair_channels'), ('HT', 'loss'), ('N', 'hooks'), ('UL', 'model_kind'), ('M', 'mix')):
        with pytest.raises(CheckpointError, match=message):
            from_research_export(export_path('S'), other)
    with pytest.raises(CheckpointError, match='order digest'):
        from_research_export(changed(tmp_path, 'S', lambda saved: saved['run'].update(order_sha256='0' * 64)), 'S')
    with pytest.raises(CheckpointError, match='weights'):
        from_research_export(export_path('S'), 'S', 'swa')


# ------------------------------------------------------------------------------------------- scores on UMLS

def research_ids(folder: Path):
    """The research loader of the evaluation: ids are the sorted names over all splits, triples in file order."""
    splits = {s: [tuple(line.split()) for line in (folder / f'{s}.txt').read_text().splitlines() if line.strip()]
              for s in ('train', 'valid', 'test')}
    entities = sorted({e for rows in splits.values() for h, _, t in rows for e in (h, t)})
    relations = sorted({r for rows in splits.values() for _, r, _ in rows})
    ei, ri = {e: i for i, e in enumerate(entities)}, {r: i for i, r in enumerate(relations)}
    ids = {s: [(ei[h], ri[r], ei[t]) for h, r, t in rows] for s, rows in splits.items()}
    return ids, len(entities), len(relations)


def query_groups(test: list) -> list:
    groups: dict = {}
    for index, (h, r, t) in enumerate(test):
        for head in (False, True):
            groups.setdefault((head, r, t if head else h), []).append(index)
    return sorted(groups.items(), key=lambda item: item[0][1])


def scores_and_raw_ranks(model: ULTRA, device: str) -> tuple[str, str, str]:
    """The research protocol for one model: scores per query group, log-softmax, filtered pessimistic rank."""
    ids, n, nr = research_ids(UMLS)
    test = ids['test']
    known: dict = {False: defaultdict(set), True: defaultdict(set)}
    for h, r, t in ids['train'] + test + ids['valid']:
        known[False][r, h].add(t)
        known[True][r, t].add(h)
    model.query_batch_size = 1
    model.set_graph(torch.tensor(ids['train'], dtype=torch.long), n, nr)
    model.to(device).eval().requires_grad_(False)
    scores, ranks, identity = hashlib.sha256(), [], []
    for (head, r, anchor), members in query_groups(test):
        h0, _, t0 = test[members[0]]
        query = torch.tensor([[r, t0]] if head else [[h0, r]], device=model.device)
        with torch.no_grad():
            s = (model.forward_k_vs_all_heads(query) if head else model.forward_k_vs_all(query))[0].cpu()
        assert torch.isfinite(s).all()
        scores.update(s.contiguous().numpy().tobytes())
        v = torch.log_softmax(s.double(), 0)
        filtered = known[head][(r, anchor)]
        for index in members:
            h, _, t = test[index]
            target = h if head else t
            mask = torch.ones(n, dtype=torch.bool)
            mask[torch.tensor(sorted(filtered - {target}), dtype=torch.long)] = False
            ranks.append(int(((v >= v[target]) & mask).sum()))
            identity.append([index, int(head)])
    digest = hashlib.sha256(np.asarray(ranks, dtype='<i8').tobytes()).hexdigest()
    return scores.hexdigest(), digest, hashlib.sha256(np.asarray(identity, dtype='<i8').tobytes()).hexdigest()


@pytest.mark.slow
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_converted_exports_score_umls_like_the_research_models(device, tmp_path):
    reference = fixture('umls_research_scores.json')['devices'][device]
    if not UMLS.is_dir():
        pytest.skip('KGs/UMLS is not available')
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('needs CUDA')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_WORKSPACES:
            pytest.skip('deterministic cuBLAS needs CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts')
        if torch.cuda.get_device_name(0) != reference['environment']['gpu']:
            pytest.skip(f"the CUDA reference was recorded on {reference['environment']['gpu']}")
    g6 = fixture('g6_ranks.json.gz')['graphs']['UMLS']
    for key, arm in (('gs', 'S'), ('gcu', 'CU'), ('gul', 'UL')):
        for seed in (42, 43, 44):
            path = export_path(arm, seed)
            expected = reference['models'][f'{key}.seed{seed}']
            assert file_sha256(path) == expected['export_sha256']
            model = load_model(save_checkpoint(from_research_export(path, arm), tmp_path / f'{key}-{seed}.pt'))
            with deterministic_execution():
                scores, ranks, identity = scores_and_raw_ranks(model, device)
            assert identity == reference['rows_identity'] == g6['rows_identity']
            assert scores == expected['scores_sha256'], (key, seed)
            assert ranks == expected['raw_rank_sha256'], (key, seed)
            if device == 'cuda':   # the research harness's own rows (golden G6) were computed on this GPU
                assert ranks == g6['per_key'][f'{key}.seed{seed}:raw'], (key, seed)
