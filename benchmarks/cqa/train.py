"""Train missing UltraQuery comparison weights using clean, pinned author code."""

import collections
import collections.abc
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import types
import zlib
from pathlib import Path

from .manifests import catalog, read_manifest, suite_directory

PUBLIC = suite_directory('ultraquery')
# Upstream checkout below Experiments/query-baselines/upstream and its pinned commit.
PINS = {'qto': ('qto', catalog().REFERENCES['qto'][1]),
        'inductive-gnnqe': ('inductiveqe', catalog().INDUCTIVE_QE[1])}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_source(source, pin):
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != pin:
        raise ValueError('Unexpected training source revision')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Training source has modified tracked files')


def verify_v2(folder, version):
    release = json.loads((PUBLIC / 'inductive-v2-files.json').read_text())
    for filename, expected in release['files'][version].items():
        path = folder / filename
        if not path.is_file():
            raise ValueError(f'Missing v2 training input {path}; download the complete official {version}.zip')
        crc = 0
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                crc = zlib.crc32(block, crc)
        if path.stat().st_size != expected['bytes'] or crc != expected['crc32']:
            raise ValueError(f'Input differs from corrected v2 release: {path}')


def compatibility():
    """Python 3.11 import compatibility; neither GNN arithmetic nor losses change."""
    collections.Sequence = collections.abc.Sequence
    drawing = types.ModuleType('rdkit.Chem.Draw.mplCanvas')
    drawing.Canvas = object
    sys.modules[drawing.__name__] = drawing


def train_qto(job, folder, source, output, args):
    import numpy as np
    import torch
    sys.path.insert(0, str(source / 'kbc/src'))
    import datasets
    import engines

    # Keep the public integer vocabulary: the authors' text preprocessor
    # reindexes first-seen strings, which cannot be used directly by our loader.
    name = 'FB15K'
    prepared = output / 'kbc-data'
    prepared.mkdir()
    arrays = {}
    for split in ('train', 'valid', 'test'):
        arrays[split] = np.array([tuple(map(int, line.split())) for line in (folder / f'{split}.txt').read_text().splitlines()
                                  if line.strip()], dtype=np.int64)
        with (prepared / f'{split}.pickle').open('wb') as stream:
            pickle.dump(arrays[split], stream)
    with (folder / 'id2ent.pkl').open('rb') as stream:
        n = len(pickle.load(stream))
    with (folder / 'id2rel.pkl').open('rb') as stream:
        nr = len(pickle.load(stream))
    filters = {'lhs': collections.defaultdict(set), 'rhs': collections.defaultdict(set)}
    for values in arrays.values():
        for h, r, t in values:
            filters['rhs'][int(h), int(r)].add(int(t))
            filters['lhs'][int(t), int(r) + nr].add(int(h))
    with (prepared / 'to_skip.pickle').open('wb') as stream:
        pickle.dump({side: {key: sorted(value) for key, value in groups.items()} for side, groups in filters.items()}, stream)
    config = dict(dataset=name, device=args.device, reciprocal=False, cache_eval=None,
                  model_cache_path=str(output) + '/', alias='', seed=str(args.seed), model='ComplEx',
                  rank=8 if args.smoke else 1000, init=1e-3, regularizer='N3', lmbda=.01,
                  optimizer='Adagrad', learning_rate=.1, decay1=.9, decay2=.999, dropout=0,
                  world='LCWA', num_neg=0, score_rel=True, score_rhs=True, score_lhs=False,
                  w_rel=.1, w_lhs=1., max_epochs=args.epochs or (1 if args.smoke else 100),
                  batch_size=args.batch_size or (2 if args.smoke else 100), valid=1 if args.smoke else 5)
    def setup_ds(options):
        dataset = datasets.Dataset(options, data_path=prepared)
        dataset.n_entities, dataset.n_predicates = n, nr
        return dataset
    engines.setup_ds = setup_ds
    declared_config = dict(config)
    engine = engines.KBCEngine(config)
    sampler = engine.dataset.get_sampler('train')
    best, best_epoch, updates = -1., None, 0
    for epoch in range(config['max_epochs']):
        engine.model.train()
        # The upstream while-is_epoch loop consumes the first batch of the
        # next epoch before its counter advances. Consume exactly one pass.
        for _ in range(0, sampler.size, config['batch_size']):
            batch = sampler.batchify(config['batch_size'], args.device)
            scores, factors = engine.model(batch, score_rel=True, score_rhs=True, score_lhs=False)
            loss = engine.loss(scores[0], batch[:, 2]) + .1 * engine.loss(scores[1], batch[:, 1])
            loss = loss + engine.regularizer.penalty(batch, factors)[0]
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite QTO training loss')
            engine.optimizer.zero_grad()
            loss.backward()
            engine.optimizer.step()
            updates += 1
            if args.smoke:
                break
        if epoch % config['valid'] == 0:
            engine.model.eval()
            mrrs, _, _ = engine.dataset.eval(engine.model, 'valid', -1)
            score = sum(float(v) for v in mrrs.values()) / len(mrrs)
            if score > best:
                best, best_epoch = score, epoch + 1
                torch.save(engine.model.state_dict(), output / 'checkpoint.pt')
    engine.writer.close()
    return dict(configuration=declared_config, model_shape=list(engine.dataset.get_shape()),
                best_validation_mrr=best, best_epoch=best_epoch, updates=updates,
                adjustments=['Preserve public integer IDs.',
                             'Correct the upstream epoch rollover: exactly one pass per epoch.',
                             'Select checkpoints on validation only; omit repeated test diagnostics.'])


def train_inductive(job, folder, source, output, args):
    import torch
    compatibility()
    sys.path.insert(0, str(source))
    from gnnqe import dataset, gnn, model, task, util  # noqa: F401  (imports register TorchDrug classes)
    from torchdrug import utils
    from torchdrug.layers import functional
    if not hasattr(functional, '_size_to_index'):
        functional._size_to_index = lambda size: torch.arange(len(size), device=size.device).repeat_interleave(size)
    from unittest.mock import patch

    config = util.load_config(source / 'config/complex_query/gnnqe_main.yaml',
                              context={'ratio': int(job['dataset'].split(':')[1]), 'gpus': 'null' if args.device == 'cpu' else [torch.device(args.device).index or 0]})
    config.engine.batch_size = args.batch_size or (2 if args.smoke else 64)
    config.train.num_epoch = args.epochs or (1 if args.smoke else 10)
    if args.smoke:
        config.train.batch_per_epoch = 1
        config.task.model.model.input_dim = 8
        config.task.model.model.hidden_dims = [8, 8]
        config.fast_test = 1
    # Use the authors' local data reader to avoid their unconditional download.
    with patch.object(utils, 'download', return_value=str(folder.parent / (folder.name + '.zip'))):
        data = dataset.InductiveFB15k237Comp(str(folder.parent), ratio=int(folder.name), verbose=0)
    if len(data.split()[1]) < config.fast_test:
        config.fast_test = len(data.split()[1])
    solver = util.build_solver(config, data)
    config.optimizer.pop('params', None)
    # Save the best complete state directly; evaluation validates the graph
    # buffers and loads learned tensors into the public inference context.
    best, best_epoch, updates = -1., None, 0
    previous = Path.cwd()
    os.chdir(output)
    try:
        for epoch in range(config.train.num_epoch):
            solver.model.split = 'train'
            solver.train(num_epoch=1, batch_per_epoch=config.train.batch_per_epoch)
            updates += config.train.batch_per_epoch
            solver.model.split = 'valid'
            score = float(solver.evaluate('valid')[config.metric])
            if score > best:
                best, best_epoch = score, epoch + 1
                solver.save('checkpoint.pt')
    finally:
        os.chdir(previous)
    return dict(configuration=json.loads(json.dumps(config, default=str)), best_validation_mrr=best,
                best_epoch=best_epoch, updates=updates,
                adjustments=['Use local corrected v2 inputs; omit final test inference during training.',
                             'Configure TorchDrug size-to-index helper and Python/RDKit imports.'])


def add_arguments(parser):
    parser.add_argument('--methods', nargs='+', choices=tuple(PINS), required=True)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--data-root', default='KGs/UltraQuery')
    parser.add_argument('--output', type=Path, required=True, help='Training directory, below the input root for Docker')
    parser.add_argument('--manifest-output-root', type=Path, help='Path of OUTPUT as later evaluations see it, below the input root')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--reference-deps', type=Path, nargs='+', default=[], help='Extra import roots for author dependencies')
    parser.add_argument('--smoke', action='store_true', help='Tiny model and one-step fixture test; never paper weights')


def run(args):
    """Train the selected recipes of ``trained_baselines.json``, each author codebase in its own process."""
    if args.threads < 1 or any(v is not None and v < 1 for v in (args.batch_size, args.epochs)):
        raise ValueError('Positive threads, batch size and epoch count required')
    recipes = read_manifest(PUBLIC / 'trained_baselines.json')
    jobs = [e for e in recipes['entries'] if e['method'] in args.methods
            and (args.datasets is None or e['dataset'] in args.datasets)]
    if not jobs or (args.datasets and set(args.datasets) - {e['dataset'] for e in jobs}):
        raise ValueError('Select compatible methods/datasets from trained_baselines.json')
    if args.dry_run:
        print(json.dumps(dict(device=args.device, jobs=[dict(id=j['id'], dataset=j['dataset'], method=j['method']) for j in jobs],
                              seed=args.seed, smoke=args.smoke), indent=2))
        return
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for path in reversed(args.reference_deps):
        sys.path.insert(0, str(path.resolve()))
        if (path / 'bin').is_dir():
            os.environ['PATH'] = str(path.resolve() / 'bin') + os.pathsep + os.environ['PATH']
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    args.input_root, args.output = args.input_root.resolve(), args.output.resolve()
    manifest_root = (args.manifest_output_root or args.output).resolve()
    if args.manifest_output_root and not manifest_root.is_relative_to(args.input_root):
        raise ValueError('Manifest output root must be below input-root')
    for job in jobs:
        # Separate interpreters prevent registry/name collisions between authors.
        if len(jobs) > 1:
            command = [sys.executable, '-m', 'benchmarks.cqa', 'ultraquery', 'train', '--methods', job['method'],
                       '--datasets', job['dataset'], '--input-root', str(args.input_root), '--data-root', args.data_root,
                       '--output', str(args.output), '--device', args.device, '--threads', str(args.threads), '--seed', str(args.seed)]
            for option, value in [('--batch-size', args.batch_size), ('--epochs', args.epochs)]:
                if value is not None:
                    command += [option, str(value)]
            if args.manifest_output_root:
                command += ['--manifest-output-root', str(manifest_root)]
            if args.reference_deps:
                command += ['--reference-deps', *map(str, args.reference_deps)]
            if args.smoke:
                command.append('--smoke')
            subprocess.run(command, check=True)
            continue
        checkout, pin = PINS[job['method']]
        source = args.input_root / 'Experiments/query-baselines/upstream' / checkout
        verify_source(source, pin)
        folder = args.input_root / args.data_root / ('FB15k-betae' if job['method'] == 'qto' else job['dataset'].split(':')[1])
        if job['method'] == 'inductive-gnnqe' and not args.smoke:
            verify_v2(folder, job['dataset'].split(':')[1])
        output = args.output / job['id']
        if output.exists():
            raise FileExistsError(f'Use a fresh training output: {output}')
        output.mkdir(parents=True)
        provenance = dict(version=1, method=job['method'], dataset=job['dataset'], source_commit=pin,
                          launcher_sha256=sha(__file__), seed=args.seed, device=args.device, threads=args.threads,
                          profile='smoke' if args.smoke else 'custom' if args.epochs or args.batch_size else 'author-recipe',
                          dataset_release='fixture' if args.smoke else 'InductiveQE v2.0' if job['method']=='inductive-gnnqe' else 'BetaE',
                          inputs={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
                          environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                           image_id=os.environ.get('DICEE_IMAGE_ID'),
                                           packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                                                     if d.metadata['Name']}),
                          checkpoint_selection='validation MRR only', status='running')
        write(output / 'training.json', provenance)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        started = time.monotonic()
        try:
            result = (train_qto if job['method']=='qto' else train_inductive)(job, folder, source, output, args)
            provenance.update(result, status='complete', seconds=time.monotonic()-started,
                              checkpoint_sha256=sha(output / 'checkpoint.pt'))
            write(output / 'training.json', provenance)
            logical = manifest_root / job['id']
            def manifest_path(filename):
                path = logical / filename
                return str(path.relative_to(args.input_root) if path.is_relative_to(args.input_root) else path)
            local = dict(recipes, seed=args.seed, threads=args.threads, data_root=args.data_root,
                         entries=[dict(job, checkpoint=manifest_path('checkpoint.pt'), training=manifest_path('training.json'))])
            local['entries'][0]['reference']['training_profile'] = provenance['profile']
            local['entries'][0]['blockers'] = ['Training smoke checkpoint; not a paper baseline'] if args.smoke else []
            write(output / 'manifest.json', local)
        except Exception as error:
            provenance.update(status='failed', error=str(error))
            write(output / 'training.json', provenance)
            raise
