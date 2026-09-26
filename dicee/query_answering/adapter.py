"""Small monotone row transforms shared by adapter fitting and query inference.

Memberships are fuzzy scores, not a claim of probability calibration.
"""

import hashlib
import json
import math
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .context import state_fingerprint, state_token

FEATURE_COUNTS = {'global': 1, 'context': 4, 'context_scores_v1': 8}


def score_features(raw, observed, base, mode):
    if mode not in FEATURE_COUNTS:
        raise ValueError(f'Unknown feature mode: {mode}')
    raw = raw.to(torch.float64)
    if raw.ndim != 2 or not raw.shape[1] or observed.shape != raw.shape or base.shape != (len(raw), 4):
        raise ValueError('Features require complete [rows, entities] scores and four context features')
    if not torch.isfinite(raw).all() or not torch.isfinite(base).all():
        raise ValueError('Atomic rows and features must be finite')
    if mode == 'global':
        return base[:, :1]
    if mode == 'context':
        return base
    n = raw.shape[1]
    mean = raw.mean(1)
    centered = raw - raw.max(1, keepdim=True).values
    exp = centered.exp()
    normalizer = exp.sum(1)
    entropy = normalizer.log() - (exp * centered).sum(1) / normalizer
    if n > 1:
        entropy = entropy / math.log(n)
        top = raw.topk(2, dim=1).values
        gap = ((top[:, 0] - top[:, 1]) / 4).tanh()
    else:
        entropy, gap = torch.zeros_like(mean), torch.zeros_like(mean)
    count = observed.sum(1)
    obs_mean = (raw * observed).sum(1) / count.clamp_min(1)
    contrast = torch.where(count > 0, ((obs_mean - mean) / 4).tanh(), 0.)
    return torch.cat((base, torch.stack(((mean / 4).tanh(), entropy, gap, contrast), dim=1)), dim=1)


class QueryScoreAdapter(nn.Module):
    """Positive row scale and bias; zero weights give the sigmoid baseline.

    ``global`` learns only two parameters. ``context`` and ``context_scores_v1``
    learn eight and sixteen. Context observations can independently override or
    interpolate memberships. Newly fitted adapters bind to a backbone state hash.
    """

    def __init__(self, feature_mode='context', observed_mix=0., *, weights=None, metadata=None,
                 bias_bound=4., scale_bound=2., normalization='none', hidden_dim=0, hidden_weights=None, seed=0):
        super().__init__()
        if feature_mode not in FEATURE_COUNTS:
            raise ValueError(f'Unknown feature mode: {feature_mode}')
        if not math.isfinite(observed_mix) or not 0 <= observed_mix <= 1:
            raise ValueError('observed_mix must be in [0, 1]')
        if not math.isfinite(bias_bound) or bias_bound <= 0:
            raise ValueError('bias_bound must be finite and positive')
        if scale_bound is not None and (not math.isfinite(scale_bound) or scale_bound <= 1):
            raise ValueError('scale_bound must exceed one, or be None for unrestricted scaling')
        if normalization not in ('none', 'standard'):
            raise ValueError('Choose none or standard row normalization')
        if type(hidden_dim) is not int or hidden_dim < 0:
            raise ValueError('hidden_dim must be a nonnegative integer')
        self.feature_mode, self.observed_mix = feature_mode, float(observed_mix)
        self.bias_bound, self.normalization, self.hidden_dim = float(bias_bound), normalization, hidden_dim
        self.scale_bound = None if scale_bound is None else float(scale_bound)
        features = FEATURE_COUNTS[feature_mode]
        size = (2, hidden_dim or features)
        value = torch.zeros(size, dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        if value.shape != size or not torch.isfinite(value).all():
            raise ValueError(f'Expected finite adapter weights of shape {size}')
        self.weights = nn.Parameter(value.clone())
        if hidden_dim:
            value = (torch.randn(hidden_dim, features, dtype=torch.float64,
                                 generator=torch.Generator().manual_seed(seed)) / math.sqrt(features)
                     if hidden_weights is None else torch.as_tensor(hidden_weights, dtype=torch.float64))
            if value.shape != (hidden_dim, features) or not torch.isfinite(value).all():
                raise ValueError('Invalid hidden adapter weights')
            self.hidden_weights = nn.Parameter(value.clone())
        else:
            if hidden_weights is not None:
                raise ValueError('Hidden weights require hidden_dim')
            self.register_parameter('hidden_weights', None)
        self.metadata = dict(metadata or {})

    @property
    def configuration(self):
        return dict(feature_mode=self.feature_mode, observed_mix=self.observed_mix, bias_bound=self.bias_bound,
                    scale_bound=self.scale_bound, normalization=self.normalization, hidden_dim=self.hidden_dim)

    def prepare(self, raw, observed=None, base=None):
        """Prepare parameter-independent inputs once for frozen training banks."""
        raw = raw.to(dtype=torch.float64)
        if raw.ndim != 2:
            raise ValueError('Adapter requires [rows, entities] logits')
        if observed is None or base is None:
            if self.feature_mode != 'global' or self.observed_mix:
                raise ValueError('Context features/observations require an explicit context graph')
            observed = torch.zeros_like(raw, dtype=torch.bool)
            base = raw.new_zeros((len(raw), 4))
            base[:, 0] = 1
        features = score_features(raw, observed, base, self.feature_mode)
        if self.normalization == 'standard':
            raw = (raw - raw.mean(1, keepdim=True)) / raw.std(1, correction=0, keepdim=True).clamp_min(1e-6)
        return raw, observed, features

    def transform(self, raw, observed, features):
        weights = self.weights.to(device=raw.device)
        if self.hidden_weights is not None:
            features = (features @ self.hidden_weights.to(device=raw.device).T).tanh()
        u, v = (features @ weights.T).unbind(1)
        # Match the derivative at initialization across scale bounds.
        log_scale = math.log(2) * u
        if self.scale_bound is not None:
            limit = math.log(self.scale_bound)
            log_scale = limit * (u * (math.log(2) / limit)).tanh()
        scale = log_scale.exp()
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise FloatingPointError('Nonfinite or zero adapter scale')
        logits = scale[:, None] * raw + self.bias_bound * v.tanh()[:, None]
        logs = F.logsigmoid(logits)
        if self.observed_mix == 1:
            logs = logs.masked_fill(observed, 0.)
        elif self.observed_mix:
            mixed = torch.logaddexp(logs.new_tensor(math.log(self.observed_mix)), math.log1p(-self.observed_mix) + logs)
            logs = torch.where(observed, mixed, logs)
        return logs

    def forward(self, raw, observed=None, base=None):
        return self.transform(*self.prepare(raw, observed, base))

    def verify_model(self, model):
        expected = self.metadata.get('backbone_state_sha256')
        if not expected:
            return
        token = state_token(model)
        previous = getattr(self, '_verified_model', None)
        if (token is not None and previous is not None and previous() is model
                and getattr(self, '_verified_state', None) == (expected, token)):
            return
        if state_fingerprint(model) != expected:
            raise ValueError('Adapter belongs to different or modified backbone weights')
        self._verified_model = weakref.ref(model)
        self._verified_state = (expected, token) if token is not None else None

    def to_dict(self):
        return dict(self.metadata, version=3, **self.configuration, weights=self.weights.detach().cpu().tolist(),
                    hidden_weights=None if self.hidden_weights is None else self.hidden_weights.detach().cpu().tolist())

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, path, *, model, key=None, checkpoint=None):
        """Load an adapter artifact or an explicitly selected catalog entry.

        Legacy catalogs contain a checkpoint file hash. Supply that original file:
        both its bytes and its actual tensor state are checked, not just its name.
        """
        payload = json.loads(Path(path).read_text())
        if key is not None:
            if key not in payload:
                raise ValueError(f'Adapter catalog has no entry {key!r}')
            payload = payload[key]
        if 'weights' not in payload:
            raise ValueError('Select a catalog entry with key=...')
        if payload.get('version', 1) not in (1, 2, 3):
            raise ValueError('Unsupported adapter artifact version')
        fields = ('weights', 'feature_mode', 'observed_mix', 'version', 'bias_bound', 'scale_bound', 'normalization', 'hidden_dim', 'hidden_weights')
        metadata = {k: v for k, v in payload.items() if k not in fields}
        if 'backbone_state_sha256' not in metadata:
            if checkpoint is None or 'backbone_checkpoint_sha256' not in metadata:
                raise ValueError('Adapter needs verified backbone state or its original checkpoint')
            checkpoint = Path(checkpoint)
            with checkpoint.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if checksum != metadata['backbone_checkpoint_sha256']:
                raise ValueError('Adapter checkpoint checksum mismatch')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = state.get('model', state)
            metadata['backbone_state_sha256'] = state_fingerprint(state)
        adapter = cls(payload.get('feature_mode', 'context'), payload.get('observed_mix', 0.),
                      weights=payload['weights'], metadata=metadata, bias_bound=payload.get('bias_bound', 4.),
                      scale_bound=payload.get('scale_bound', 2.),
                      normalization=payload.get('normalization', 'none'), hidden_dim=payload.get('hidden_dim', 0),
                      hidden_weights=payload.get('hidden_weights'))
        adapter.verify_model(model)
        return adapter
