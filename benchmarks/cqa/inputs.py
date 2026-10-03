"""Fetch official benchmark inputs with the standard library; never load models.

Downloads and extraction use 1 MiB buffers. Existing inputs are checked and
never replaced silently.
"""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .manifests import REPO, catalog

CHUNK = 1024 * 1024
RESERVE = 1024 * 1024 * 1024
MODEL_URL = ('https://rssiste-my.sharepoint.com/:u:/g/personal/'
             'cosimo_gregucci_ms_informatik_uni-stuttgart_de/'
             'EccKx4K6sZ1Nhy_fJTL8L44BFtpNRZwmI43-ffPRSmb28g?e=kO5dYk&download=1')
MODEL_PREFIX = 'checkpoints/query-baselines/'
# SHA-256 of the released files, extracted from the archive pinned in the +H
# manifests. The standalone weights are from the pinned ULTRA/TRIX trees.
WEIGHT_HASHES = {
    'Experiments/query-baselines/upstream/ultra/ckpts/ultraquery.pth': '9b6dc20801c35e7c9ebe65764acb4ed9cb5bffc6b0a0a727bd48935589e9388d',
    'checkpoints/query-baselines/iscqa-compl-models/models/CLMPT/CLMPT-fb15k237.ckpt': '4ce1e2abeda74199970c9350e3da10d928d34f974f3c1f42ead553a9c13799b9',
    'checkpoints/query-baselines/iscqa-compl-models/models/CLMPT/CLMPT-icews18.ckpt': '1a450d5cd0c726eec10ff302249143894f2fb3329370204b453ec09d447c227d',
    'checkpoints/query-baselines/iscqa-compl-models/models/CLMPT/CLMPT-nell995.ckpt': 'c61fa9089e6233faf38cabe8e2e0ef066eed5d4123871108de3226e21cafcd18',
    'checkpoints/query-baselines/iscqa-compl-models/models/CQD:CQD-HYBRID:QTO/ICEWS18/checkpoint': 'ba873863eeec2abeeb70e9ae484171fbc56a1f5cfd194cb8a7fac24cae03f9f7',
    'checkpoints/query-baselines/iscqa-compl-models/models/CQD:CQD-HYBRID:QTO/QTO/fb15k237checkpoint': '631bd1fd3c5d8095eb2c99b78052c3ea4c9c2dd88f7ab044cbb2c610ed547c18',
    'checkpoints/query-baselines/iscqa-compl-models/models/CQD:CQD-HYBRID:QTO/QTO/nellcheckpoint': '42b46696588918dba83262dd88b4884d9b6a27c8975f2c6d5a6bc7c65764168f',
    'checkpoints/query-baselines/iscqa-compl-models/models/ConE/FB15k237/checkpoint': 'c576cd8cf55bebc642b6007e079650ccdfd4f627f1c6a4780f5680ba001ef829',
    'checkpoints/query-baselines/iscqa-compl-models/models/ConE/ICEWS18/checkpoint': '44f0f0178e099206ad8ce0dca39815d57df6eed44992698eb6fd9e73ff92f144',
    'checkpoints/query-baselines/iscqa-compl-models/models/ConE/NELL995/checkpoint': 'f0ac18ae0bf3eacff15207d9320e19d010cd4a494e3bae96e8475b891e886787',
    'checkpoints/query-baselines/iscqa-compl-models/models/GNN-QE/model/model_FB15k237.pth': 'cb45202f9aa68da61a59b88721c76b5883274db309d4cf14037b8fc2c72a93b2',
    'checkpoints/query-baselines/iscqa-compl-models/models/GNN-QE/model/model_ICEWS18.pth': '6b8c1fd64cc7ba96fe2e4834634a2721c4d3914658c1d81f61fdd5526d1b5a13',
    'checkpoints/query-baselines/iscqa-compl-models/models/GNN-QE/model/model_NELL995.pth': '086619b977d7b45658c11b4804b6e0e1c92a356cd4f427b4bf5b8ea1ee7016b3',
    'checkpoints/trix/entity_prediction.pth': '8f6e7266093c2d15ad88d41e9825e4b327171143543cf060cfee907d9a890342',
    'checkpoints/ultra_3g.pth': 'fdedc01b0045fc089d2ad5da08569466b7b221691fe978a18e91082bbb133c18',
    'checkpoints/ultra_4g.pth': '48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0',
    'checkpoints/ultra_50g.pth': 'f1c5377b2cf547aaa67520ffb6fce27b75b6eb3417b86dc9cf9dba89964ed10f',
}


def weight_urls():
    ultra, trix = (f'https://raw.githubusercontent.com/{repository}/{commit}'
                   for repository, commit in (catalog().ULTRA, catalog().TRIX))
    return {'Experiments/query-baselines/upstream/ultra/ckpts/ultraquery.pth': f'{ultra}/ckpts/ultraquery.pth',
            'checkpoints/ultra_3g.pth': f'{ultra}/ckpts/ultra_3g.pth',
            'checkpoints/ultra_4g.pth': f'{ultra}/ckpts/ultra_4g.pth',
            'checkpoints/ultra_50g.pth': f'{ultra}/ckpts/ultra_50g.pth',
            'checkpoints/trix/entity_prediction.pth': f'{trix}/entity_prediction.pth'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            result.update(block)
    return result.hexdigest()


def target(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative:
        raise ValueError(f'Unsafe input path: {relative}')
    result = root / relative
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError(f'Unsafe input path: {relative}')
    return result


def space(root, required):
    available = shutil.disk_usage(root).free
    if available < required + RESERVE:
        raise OSError(f'Need {required / 2**30:.2f} GiB plus 1 GiB reserve; '
                      f'only {available / 2**30:.2f} GiB free at {root}')


def install(stream, path, expected=None):
    """Write atomically, rejecting a different existing file or bad checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.input-', dir=path.parent)
    temporary = Path(temporary)
    result = hashlib.sha256()
    try:
        with os.fdopen(handle, 'wb') as output:
            while block := stream.read(CHUNK):
                result.update(block)
                output.write(block)
        checksum = result.hexdigest()
        if expected is not None and checksum != expected:
            raise ValueError(f'SHA-256 mismatch for {path}')
        if path.exists():
            if digest(path) != checksum:
                raise ValueError(f'Existing input differs; refusing to replace {path}')
        else:
            os.replace(temporary, path)
        return checksum
    finally:
        temporary.unlink(missing_ok=True)


def download(url, path, expected=None, size=None):
    print(f'Downloading {url}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'dicee-benchmark-inputs/1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = size or int(response.headers.get('Content-Length') or 0)
        if length:
            space(path.parent, length)
        return install(response, path, expected)


def transductive_files():
    return {'id2ent.pkl', 'id2rel.pkl', 'train.txt', 'valid.txt', 'test.txt',
            *(f'{split}-{kind}.pkl' for split in ('valid', 'test')
              for kind in ('queries', 'easy-answers', 'hard-answers'))}


def plus_h_files(icews=False):
    files = transductive_files()
    if icews:
        files -= {'train.txt', 'valid.txt', 'test.txt'}
        files |= {f'KG_splits/{split}.txt' for split in ('train', 'valid', 'test')}
    else:
        files |= {f'test-query-reduction/{shape}/all/test-{kind}.pkl'
                  for shape in ('2in', 'pni') for kind in ('queries', 'easy-answers', 'hard-answers')}
        reductions = {'ip': ('1p', '2i', '2p', 'ip'), 'pi': ('1p', '2i', '2p', 'pi'),
                      'up': ('1p', '2u', 'up')}
        reductions.update({f'{n}{op}': ('1p', *(f'{i}{op}' for i in range(2, n + 1)))
                           for op in ('p', 'i') for n in (2, 3, 4)})
        reductions.update({s: ('pos-exist', 'pos-only-miss') for s in ('3in', 'pin', 'inp')})
        files |= {f'test-query-reduction/{shape}/{label}/test-hard-answers.pkl'
                  for shape, labels in reductions.items() for label in labels}
    return files


def plan(manifest, suite):
    """Dataset archives and checkpoints that a resolved manifest needs."""
    c = catalog()
    names = {entry['dataset'] for entry in manifest['entries']}
    weights = sorted({entry['checkpoint'] for entry in manifest['entries']})
    root = manifest['data_root']
    if suite == 'plus_h':
        archive = json.loads((REPO / 'benchmarks/plus_h/baselines.json').read_text())['archives']['datasets']
        folders = {f'{c.PLUS_H_PREFIX}/{folder}': plus_h_files(name == 'ICEWS18+H')
                   for name, folder in c.PLUS_H_FOLDERS.items() if name in names}
        return [dict(id='plus-h-datasets', url=c.PLUS_H_ARCHIVE, root=root, folders=folders,
                     sha256=archive['sha256'], size=archive['bytes'], selective=True)], weights
    jobs = []
    folders = {folder: transductive_files() for name, folder in c.TRANSDUCTIVE.items() if name in names}
    if folders:
        jobs.append(dict(id='ultra-transductive', url='https://snap.stanford.edu/betae/KG_data.zip', root=root, folders=folders))
    required = {f'{split}_{kind}.pkl' for split in ('valid', 'test') for kind in ('queries', 'answers_easy', 'answers_hard')}
    for version in c.INDUCTIVE_VERSIONS:
        if f'InductiveFB15k237Query:{version}' in names:
            jobs.append(dict(id=f'ultra-inductive-{version}', url=f'https://zenodo.org/records/7306046/files/{version}.zip', root=root,
                             folders={version: required | {'train_graph.txt', 'val_inference.txt', 'test_inference.txt'}}))
    folders = {f'WikiTopics_QE/{topic}': required | {'train_graph.txt', 'test_inference.txt', 'og_mappings.pkl'}
               for topic in c.WIKITOPICS if f'WikiTopicsQuery:{topic}' in names}
    if folders:
        jobs.append(dict(id='ultra-wikitopics', url='https://reltrans.s3.us-east-2.amazonaws.com/WikiTopics_QE.zip',
                         root=root, folders=folders))
    return jobs, weights


def missing_files(root, job):
    return [f"{job['root']}/{folder}/{name}" for folder, names in job['folders'].items()
            for name in sorted(names) if not target(root, f"{job['root']}/{folder}/{name}").is_file()]


def extract(archive, root, select):
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        # Validate the entire archive before installing any input.
        for member in members:
            target(root, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Symlink in archive: {member.filename}')
        chosen = [(m, select(m.filename)) for m in members if not m.is_dir()]
        chosen = [(m, target(root, name)) for m, name in chosen if name is not None]
        space(root, sum(m.file_size for m, _ in chosen))
        checksums = {}
        for member, path in chosen:
            with zipped.open(member) as stream:
                checksums[path.relative_to(root).as_posix()] = install(stream, path)
        return checksums


def fetch_dataset(root, job):
    state = root / '.benchmark-downloads'
    state.mkdir(parents=True, exist_ok=True)
    receipt = state / f"{job['id']}.json"
    if not missing_files(root, job):
        if not receipt.is_file():
            print(f"Dataset inputs already present: {job['id']} (freeze pins their hashes)", flush=True)
            return
        saved = json.loads(receipt.read_text())
        if saved['url'] == job['url'] and all(target(root, p).is_file() and digest(target(root, p)) == h
                                              for p, h in saved['files'].items()):
            print(f"Verified dataset {job['id']}", flush=True)
            return
        raise ValueError(f"Dataset receipt mismatch: {receipt}")
    with tempfile.TemporaryDirectory(prefix='.fetch-', dir=root) as temporary:
        archive = Path(temporary) / 'data.zip'
        checksum = download(job['url'], archive, job.get('sha256'), job.get('size'))

        def select(name):
            for folder, required in job['folders'].items():
                if name.startswith(folder + '/'):
                    relative = name[len(folder) + 1:]
                    if not job.get('selective') or relative in required or relative == 'stats.txt':
                        return f"{job['root']}/{name}"
            return None

        files = extract(archive, root, select)
        missing = missing_files(root, job)
        if missing:
            raise ValueError(f"Archive lacks required inputs: {missing}")
        payload = json.dumps(dict(url=job['url'], archive_sha256=checksum, files=files), indent=2).encode()
        # A receipt is metadata, not an immutable benchmark input.
        temp = receipt.with_suffix('.tmp')
        temp.write_bytes(payload)
        os.replace(temp, receipt)
        print(f"Installed {len(files)} files for {job['id']}", flush=True)


def fetch_weights(root, weights, models_archive=None):
    urls = weight_urls()
    pending = []
    for name in weights:
        path = target(root, name)
        if path.is_file():
            if name in WEIGHT_HASHES and digest(path) != WEIGHT_HASHES[name]:
                raise ValueError(f'Existing checkpoint checksum mismatch: {path}')
        else:
            if name not in WEIGHT_HASHES:
                raise ValueError(f'No public download is configured for {path}; provide this custom checkpoint first')
            pending.append(name)
    author = [name for name in pending if name not in urls]
    if author:
        meta = json.loads((REPO / 'benchmarks/plus_h/baselines.json').read_text())['archives']['models']
        with tempfile.TemporaryDirectory(prefix='.models-', dir=root) as temporary:
            archive = models_archive or Path(temporary) / 'models.zip'
            if models_archive:
                if digest(archive) != meta['sha256']:
                    raise ValueError(f'Official model archive checksum mismatch: {archive}')
            else:
                try:
                    download(MODEL_URL, archive, meta['sha256'], meta['bytes'])
                except urllib.error.HTTPError as error:
                    raise RuntimeError('The authors\' model host rejected the download. '
                                       'Use --models-archive /path/to/iscqa-compl-models.zip, '
                                       'or copy the released checkpoints into the manifest paths and retry.') from error
            wanted = {name[len(MODEL_PREFIX):]: name for name in author}
            extract(archive, root, wanted.get)
    for name in pending:
        path = target(root, name)
        if name in urls:
            download(urls[name], path, WEIGHT_HASHES[name])
        if not path.is_file() or digest(path) != WEIGHT_HASHES[name]:
            raise ValueError(f'Released checkpoint missing or checksum mismatch: {path}')
    print(f'Checkpoint inputs present: {len(weights)}; released files checked by SHA-256', flush=True)


def ensure_inputs(manifest, input_root, *, suite, models_archive=None, download_missing=True):
    """Set up only requested public inputs; custom data/checkpoints stay explicit."""
    root = Path(input_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs, weights = plan(manifest, suite)
    missing = [p for p in weights if not target(root, p).is_file()]
    public = json.loads((REPO / 'benchmarks' / suite / 'baselines.json').read_text())
    official = manifest.get('archives', {}).get('datasets') == public['archives']['datasets']
    if official:
        missing += [p for job in jobs for p in missing_files(root, job)]
    extras = {p for e in manifest['entries'] for p in e.get('adapters', {}).values()}
    extras |= {e['checkpoint'] for e in manifest['entries'] if e['checkpoint'].endswith('.json')}
    for name in sorted(extras):
        path = target(root, name)
        if not path.is_file():
            source = target(REPO, name)
            if not source.is_file() or not download_missing:
                missing.append(name)
                continue
            space(root, source.stat().st_size)
            with source.open('rb') as stream:
                install(stream, path)
    weights = [name for name in weights if name not in extras]
    missing = [p for p in missing if not target(root, p).is_file()]
    if not download_missing:
        if missing:
            raise FileNotFoundError(f'Inputs missing with --no-setup: {missing}')
        return
    if weights:
        fetch_weights(root, weights, models_archive)
    if official:
        for job in jobs:
            if missing_files(root, job):
                fetch_dataset(root, job)


def describe(manifest, root, *, suite):
    """Print where each input of ``manifest`` comes from; no downloads or writes."""
    jobs, weights = plan(manifest, suite)
    urls = weight_urls()
    for job in jobs:
        print(f"{job['id']}: {job['url']} -> {root / job['root']}")
    for name in weights:
        source = urls.get(name, MODEL_URL if name in WEIGHT_HASHES else 'User-provided input')
        print(f'{source} -> {root / name}')


def check(manifest, root, *, suite):
    """Print missing inputs and checksum mismatches; return whether there were any."""
    jobs, weights = plan(manifest, suite)
    adapters = {path for entry in manifest['entries'] for path in entry.get('adapters', {}).values()}
    missing = [path for job in jobs for path in missing_files(root, job)]
    missing += [path for path in [*weights, *sorted(adapters)] if not target(root, path).is_file()]
    bad = [path for path in weights if path in WEIGHT_HASHES and target(root, path).is_file()
           and digest(target(root, path)) != WEIGHT_HASHES[path]]
    for name in missing:
        print(f'Missing: {name}')
    for name in bad:
        print(f'Checksum mismatch: {name}')
    print(f'{len(jobs)} dataset archives; {len(weights)} checkpoints; {len(missing)} missing files; {len(bad)} bad checkpoints.')
    return bool(missing or bad)
