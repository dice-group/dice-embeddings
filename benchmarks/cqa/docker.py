"""Build the benchmark image and run commands in it with immutable inputs.

Containers have no network, mount the input root read-only at ``/inputs`` and
one results directory at ``/results``, and record the image ID they used.
"""

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from .manifests import REPO

HERE = Path(__file__).parent
# Tracked sources copied into the image: the library, this harness, and suite data.
SOURCES = ('dicee', 'pyproject.toml', 'benchmarks/__init__.py', 'benchmarks/cqa', 'benchmarks/adapters',
           'benchmarks/plus_h', 'benchmarks/ultraquery')


def inspect(image):
    return json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]


def build(image, *, cache=None, network='default', training_base=None):
    """Build the pinned runtime, or the CPU author-training runtime on top of ``training_base``."""
    if training_base is not None:
        subprocess.run(['docker', 'build', '--network', network, '-t', image,
                        '--build-arg', f'BENCHMARK_IMAGE={inspect(training_base)["Id"]}', str(HERE / 'training')], check=True)
        return

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(REPO), *arguments], text=True)

    commit = git('rev-parse', 'HEAD').strip()
    dirty = bool(git('status', '--porcelain', '--untracked-files=no', '--', *SOURCES).strip())
    cache = Path(cache or Path.home() / '.cache' / 'dicee' / 'cqa').resolve()
    wheels, pip_cache = cache / 'wheels', Path.home() / '.cache' / 'pip'
    wheels.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)
    base = (HERE / 'Dockerfile').read_text().splitlines()[0].split()[1]
    # Resolve hash-pinned wheels once, in the image's own Python, then build offline.
    subprocess.run(['docker', 'run', '--rm', '--network', network, '--user', f'{os.getuid()}:{os.getgid()}',
                    '--env', 'PIP_CACHE_DIR=/pip-cache',
                    '--mount', f'type=bind,source={wheels},target=/wheels',
                    '--mount', f'type=bind,source={pip_cache},target=/pip-cache',
                    '--mount', f'type=bind,source={HERE / "requirements.lock"},target=/requirements.lock,readonly',
                    base, 'python', '-m', 'pip', 'download', '--require-hashes', '--only-binary=:all:',
                    '--timeout', '180', '--retries', '10', '--dest', '/wheels', '-r', '/requirements.lock'], check=True)
    with tempfile.TemporaryDirectory(prefix='build-', dir=cache) as directory:
        shutil.copytree(wheels, Path(directory) / 'wheels', copy_function=os.link)
        for relative in git('ls-files', '-z', '--', *SOURCES).split('\0'):
            source, target = REPO / relative, Path(directory) / relative
            if relative and source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        subprocess.run(['docker', 'build', '--network', network, '-t', image, '-f', 'benchmarks/cqa/Dockerfile',
                        '--build-arg', f'SOURCE_COMMIT={commit}', '--build-arg', f'SOURCE_DIRTY={str(dirty).lower()}',
                        directory], check=True)


def run(image, arguments, *, inputs, results, gpu='runtime', dry_run=False):
    """Run ``python -m benchmarks.cqa ARGUMENTS`` in ``image``.

    Every container of a results directory must use the same image ID; the
    first launch records it in ``container-image.json``.
    """
    inputs, results = Path(inputs).resolve(), Path(results).resolve()
    mounts = ['--mount', f'type=bind,source={inputs},target=/inputs,readonly',
              '--mount', f'type=bind,source={results},target=/results',
              '--mount', f'type=bind,source={results / "cache"},target=/cache']
    devices = {'runtime': ['--gpus', 'all'], 'cdi': ['--device', 'nvidia.com/gpu=all'], 'none': []}[gpu]
    command = ['docker', 'run', '--rm', '--network', 'none', '--read-only', '--shm-size', '2g',
               '--tmpfs', '/tmp:rw,exec,nosuid,size=4g', '--user', f'{os.getuid()}:{os.getgid()}',
               '--env', 'HOME=/tmp', '--env', 'XDG_CACHE_HOME=/cache', '--env', 'TORCHINDUCTOR_CACHE_DIR=/cache/inductor',
               '--env', 'TRITON_CACHE_DIR=/cache/triton', *mounts, *devices]
    if dry_run:
        print(shlex.join([*command, image, *arguments]))
        return
    identity = inspect(image)
    pin = results / 'container-image.json'
    if pin.exists() and json.loads(pin.read_text())['image_id'] != identity['Id']:
        raise ValueError('Image differs from this study; use the original image ID or a new output directory')
    (results / 'cache').mkdir(parents=True, exist_ok=True)
    if not pin.exists():
        pin.write_text(json.dumps(dict(image_id=identity['Id'], repo_digests=identity.get('RepoDigests', [])), indent=2) + '\n')
    command += ['--env', f'DICEE_IMAGE_ID={identity["Id"]}', identity['Id'], *arguments]
    (results / 'container-launch.json').write_text(json.dumps(dict(image_id=identity['Id'], command=command), indent=2) + '\n')
    subprocess.run(command, check=True)
