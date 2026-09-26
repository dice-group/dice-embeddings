"""Atomic, single-writer benchmark checkpoints."""

import json
import os
from pathlib import Path

from .context import fingerprint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class BenchmarkCheckpoint:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = fingerprint(identity)
        self.lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / '.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.lock.write(b'0')
                self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            raise RuntimeError(f'Benchmark already running: {self.directory}') from error
        try:
            manifest = self.directory / 'manifest.json'
            expected = dict(version=1, identity=self.identity)
            if manifest.exists():
                if json.loads(manifest.read_text()) != expected:
                    raise ValueError('Benchmark inputs/settings changed; use a new checkpoint directory')
            else:
                write_json(manifest, expected)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def load(self):
        path = self.directory / 'state.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record.get('identity') != self.identity or record.get('sha256') != fingerprint(record['state']):
            raise ValueError('Corrupt benchmark checkpoint')
        return record['state']

    def save(self, state):
        write_json(self.directory / 'state.json', dict(identity=self.identity, sha256=fingerprint(state), state=state))

    def __exit__(self, *_):
        if self.lock is not None:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            self.lock.close()
