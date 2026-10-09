"""Apply three mechanical TorchDrug 0.2.1 fixes for the pinned PyTorch runtime.

No kernel arithmetic changes: update two moved includes and pass loader options
by keyword because the modern PyTorch positional signature has changed.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def apply(root):
    replacements = {
        'layers/functional/extension/spmm.h': ('#include <ATen/SparseTensorUtils.h>',
                                             '#include <ATen/native/SparseTensorUtils.h>'),
        'layers/functional/extension/rspmm.h': ('#include <ATen/SparseTensorUtils.h>',
                                              '#include <ATen/native/SparseTensorUtils.h>'),
        'utils/torch.py': ('self.extra_ldflags, self.extra_include_paths, self.build_directory,\n'
                          '                                  self.verbose, **self.kwargs)',
                          'extra_ldflags=self.extra_ldflags, extra_include_paths=self.extra_include_paths,\n'
                          '                                  build_directory=self.build_directory, verbose=self.verbose, **self.kwargs)'),
    }
    pending = []
    for relative, (old, new) in replacements.items():
        path = root / relative
        original = path.read_text()
        if old not in original and new not in original:
            raise ValueError(f'Unexpected TorchDrug source; refusing to patch {path}')
        pending.append((path, original, original.replace(old, new)))
    report = {}
    for path, original, updated in pending:
        if original != updated:
            path.write_text(updated)
        report[str(path.relative_to(root))] = dict(
            before_sha256=hashlib.sha256(original.encode()).hexdigest(),
            after_sha256=hashlib.sha256(updated.encode()).hexdigest())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path)
    args = parser.parse_args()
    root = args.root
    if root is None:
        root = Path(importlib.util.find_spec('torchdrug').origin).parent
    print(json.dumps(apply(root), indent=2))
