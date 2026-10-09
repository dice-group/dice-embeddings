"""Small optional native CPU inference loop (no PyTorch extension build).

Compile lazily with the system C compiler, without fast math or contraction. The
loop is single threaded and releases the GIL. Missing compilers use the literal
PyTorch implementation; training and unsupported shapes/dtypes do so as well.
The private temporary library lives for this process only.
"""

import ctypes
import os
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache

import torch

_SOURCE = r'''
#include <stdint.h>
#include <float.h>
#include <math.h>
#define DEFINE(NAME, TYPE, LIMIT) \
void NAME(const TYPE *s, const TYPE *r, TYPE *o, \
          const int64_t *p, const int64_t *src, const int64_t *rel, \
          int64_t n, int64_t b, int64_t d, \
          int64_t sn, int64_t sb, int64_t sd, \
          int64_t rn, int64_t rb, int64_t rd, int kind) { \
    for (int64_t row = 0; row < n; ++row) { \
        for (int64_t batch = 0; batch < b; ++batch) { \
            TYPE *out = o + (row * b + batch) * d; \
            for (int64_t f = 0; f < d; ++f) \
                out[f] = kind == 0 ? 0 : kind == 1 ? -LIMIT : LIMIT; \
            for (int64_t e = p[row]; e < p[row + 1]; ++e) { \
                const TYPE *state = s + src[e] * sn + batch * sb; \
                const TYPE *relation = r + rel[e] * rn + batch * rb; \
                for (int64_t f = 0; f < d; ++f) { \
                    TYPE value = state[f * sd] * relation[f * rd]; \
                    TYPE old = out[f]; \
                    out[f] = kind == 0 ? old + value : \
                        kind == 1 ? (isnan(old) || old >= value ? old : value) : \
                                    (isnan(old) || old <= value ? old : value); \
                } \
            } \
        } \
    } \
}
DEFINE(reduce_float, float, FLT_MAX)
DEFINE(reduce_double, double, DBL_MAX)
'''
_BUILD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _library():
    compiler = shutil.which('cc')
    if compiler is None or os.name != 'posix':
        return None
    directory = tempfile.TemporaryDirectory(prefix='dicee-ordered-')
    try:
        source = os.path.join(directory.name, 'ordered.c')
        binary = os.path.join(directory.name, 'ordered.so')
        with open(source, 'w') as stream:
            stream.write(_SOURCE)
        subprocess.run([compiler, '-std=c11', '-O3', '-shared', '-fPIC',
                        '-fno-fast-math', '-ffp-contract=off', source, '-o', binary],
                       check=True, capture_output=True, timeout=30)
        library = ctypes.CDLL(binary)
        library._directory = directory
        for name in ('reduce_float', 'reduce_double'):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 9 + [ctypes.c_int]
            function.restype = None
        return library
    except (OSError, subprocess.SubprocessError):
        directory.cleanup()
        return None


def ordered_reduce(states, relations, edges, reduction):
    """Return a native result, or None when the literal implementation is needed."""
    if (states.device.type != 'cpu' or relations.device.type != 'cpu'
            or states.dtype not in (torch.float32, torch.float64)
            or relations.dtype != states.dtype or states.ndim != 3 or relations.ndim != 3
            or states.shape[1:] != relations.shape[1:]
            or states.layout != torch.strided or relations.layout != torch.strided
            or states.is_neg() or relations.is_neg()
            or (torch.is_grad_enabled() and (states.requires_grad or relations.requires_grad))
            or reduction not in ('sum', 'max', 'min')):
        return None
    pointers, sources, types = edges
    if any(t.device.type != 'cpu' or t.dtype != torch.int64 or t.ndim != 1
           or not t.is_contiguous() for t in edges):
        return None
    nodes, batch, dim = states.shape
    # Native memory accesses require validated CSR bounds. Keep malformed inputs
    # on the Python path so they retain PyTorch's indexing/error semantics.
    if (len(pointers) != nodes + 1 or len(sources) != len(types)
            or int(pointers[0]) != 0 or int(pointers[-1]) != len(sources)
            or bool((pointers[1:] < pointers[:-1]).any())
            or (sources.numel() and (int(sources.min()) < 0 or int(sources.max()) >= nodes
                                    or int(types.min()) < 0 or int(types.max()) >= len(relations)))):
        return None
    if not states.numel():
        return torch.empty(states.shape, dtype=states.dtype, device=states.device)
    with _BUILD_LOCK:
        library = _library()
    if library is None:
        return None
    output = torch.empty(states.shape, dtype=states.dtype, device=states.device)
    function = library.reduce_float if states.dtype == torch.float32 else library.reduce_double
    function(states.data_ptr(), relations.data_ptr(), output.data_ptr(),
             pointers.data_ptr(), sources.data_ptr(), types.data_ptr(),
             nodes, batch, dim, *states.stride(), *relations.stride(),
             ('sum', 'max', 'min').index(reduction))
    return output
