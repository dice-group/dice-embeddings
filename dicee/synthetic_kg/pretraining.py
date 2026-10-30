"""ProofPFN pretraining: ULTRA trained on one generated world per update, from an arm's initial weights.

A run is an arm (``arms.ARMS``) and a seed. Update ``i`` trains on the world of slot ``i`` of the arm's world order
(``priors.build_order``, 24,000 slots), prepared by worker processes (``prepare.prepare_slot``). One update:

- per task of the world (in the world's order): attach the task's prepared graph to the model; the tail queries, then
  the head queries, in chunks of ``MICROBATCH``; the scores of all entities through the model's scoring path in
  training mode (``GraphKGE.forward_query_facts``, which keeps every graph edge); the context answers masked to
  ``-inf``; ``log_softmax``; the target is the RB-W row ``soft`` or, for the hard target, a one-hot answer drawn from
  it (``hard_target``, with the update's own generator, ``streams.hard_target_seed``); the loss ``-sum(target * logp)``
  averaged over the chunk; backward of ``loss * rows / queries``;
- the gradient norm over the tensors with a gradient, one AdamW step (weight decay 0.01, default betas and eps, no
  clipping); the learning rate is ``LEARNING_RATE`` for 20,000 updates, then linear to 0 over 4,000 (``learning_rate``);
- the EMA (decay ``EMA_DECAY``) of every floating tensor of the state, from update 0.

The logged loss of an update is the research float loop ``value += number * size / queries`` over the chunks, in
chunk order. The update draws no global random number (the hard target has its own generator), so the run depends only
on its configuration, its device and torch's thread count, not on the number of workers.

Files of a run folder: ``run.json`` (the run record and its provenance), ``export-<N>-raw.pt`` and ``export-<N>-ema.pt``
(``proofpfn-checkpoint-v1`` checkpoints, ``checkpoint.load_model``), ``state-20000.pt`` and ``last.pt`` (the training
state: model, AdamW state, EMA, history; ``last.pt`` every ``save_every`` updates and at the end), ``progress.json``.
A run whose folder holds a ``last.pt`` resumes from it; the result equals the uninterrupted run.

This is the research trainer (``prior-fix-v1-runtime/pt_train.py`` with ``pt_ladder``, ``pt_ultra``, ``pt_hard`` and
``statworlds-scale-v1-runtime/fast_pipeline``) for the ported arms, without its adapters, screens, restarts and skips.
dicee's trainers are not used: an update trains on its own generated graph, which their datasets do not express.
"""
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Mapping, NamedTuple, Optional, Sequence, Union

import numpy
import torch

from .. import __version__ as DICEE_VERSION
from ..models import _relation_weights, graph_model, ultra
from ..models._inference import to_device
from ..models.ultra import ULTRA
from .arms import ArmPreset, arm_preset, initial_model
from .checkpoint import atomic_save, make_checkpoint, save_checkpoint
from .determinism import deterministic_execution, require_cublas_workspace
from .digests import file_sha256
from .prepare import PreparedWorld, prepare_slot
from .priors import PriorConfig, Slot, build_order, order_sha256, real
from .recipe import CONSTANT_UPDATES, DECAY_UPDATES, EMA_DECAY, LEARNING_RATE, MICROBATCH, REAL_SOURCE_SHA256, TOTAL_UPDATES, WEIGHT_DECAY
from .streams import hard_target_seed

logger = logging.getLogger(__name__)

RUN_FORMAT = 'proofpfn-pretraining-v1'
STATE_FORMAT = 'proofpfn-training-state-v1'
EXPORTS = (4000, 8000, 16000, 20000, 24000)
"""The default export updates (raw and EMA weights); the last update of a run is always exported."""
STATE_AT = (CONSTANT_UPDATES,)
"""The updates with a kept training state ``state-<N>.pt``: the end of the constant learning rate."""
SAVE_EVERY = 250


@dataclass(frozen=True)
class PretrainConfig:
    """A pretraining run.

    Attributes:
        arm: The arm name (``'S'``, ``'R'``, ``'M'``, ``'N'``, ``'HT'``, ``'CU'``, ``'UL'``).
        seed: The run seed.
        total: The run's last update; the learning rate keeps its schedule, so a shorter run is the start of the full
            one (tests).
        stop_after: Stop after this update with ``last.pt`` (rerun to resume); ``None`` runs to ``total``.
        exports: The export updates; ``None``: ``EXPORTS`` up to ``total``. ``total`` is always exported.
        save_every: ``last.pt`` every this many updates.
        workers: Worker processes that generate and prepare the worlds ahead; 0 prepares them in this process.
        prefetch: Worlds in flight; ``None``: twice the workers.
        worker_threads: Torch threads of a worker.
        device: The training device.
        threads: Torch threads of the training process (CPU results depend on it).
        real_sources: The folder of the real source graphs (arms R and M); ``None`` reads ``PROOFPFN_REAL_SOURCES``.
        allow_code_change: Resume although the code that the run record binds has changed.
    """

    arm: str
    seed: int
    total: int = TOTAL_UPDATES
    stop_after: Optional[int] = None
    exports: Optional[tuple[int, ...]] = None
    save_every: int = SAVE_EVERY
    workers: int = 4
    prefetch: Optional[int] = None
    worker_threads: int = 1
    device: str = 'cuda'
    threads: int = 2
    real_sources: Union[str, Path, None] = None
    allow_code_change: bool = False

    def __post_init__(self) -> None:
        arm_preset(self.arm)
        if not 1 <= self.total <= TOTAL_UPDATES:
            raise ValueError(f'total must lie in 1..{TOTAL_UPDATES}')
        if self.stop_after is not None and not 1 <= self.stop_after <= self.total:
            raise ValueError('stop_after must lie inside the run')
        if self.exports is not None and not all(1 <= e <= self.total for e in self.exports):
            raise ValueError('every export must lie inside the run')
        if min(self.save_every, self.worker_threads, self.threads) < 1 or self.workers < 0 or (self.prefetch is not None and self.prefetch < 1):
            raise ValueError('save_every, threads and prefetch must be positive, workers nonnegative')

    @property
    def export_updates(self) -> list[int]:
        """The export updates, sorted."""
        chosen = self.exports if self.exports is not None else [e for e in EXPORTS if e <= self.total]
        return sorted({*chosen, self.total})

    @property
    def state_updates(self) -> list[int]:
        """The updates with a kept training state."""
        return [s for s in STATE_AT if s <= self.total]


class PretrainResult(NamedTuple):
    """The end of a call of ``pretrain``.

    Attributes:
        status: ``'complete'`` (the last update is done), ``'stopped-after'`` (at ``stop_after``) or ``'stopped'`` (by
            the ``stop`` callback).
        done: The updates done.
        folder: The run folder.
    """

    status: str
    done: int
    folder: Path


# ------------------------------------------------------------------------------------------------------ the update

def learning_rate(update: int, constant: int = CONSTANT_UPDATES, decay: int = DECAY_UPDATES) -> float:
    """The learning rate of a 0-based update: ``LEARNING_RATE`` for ``constant`` updates, then linear to 0 over ``decay``."""
    return LEARNING_RATE if update < constant else LEARNING_RATE * (1 - (update - constant) / decay)


def hard_target(soft: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """One-hot rows of the shape of ``soft``: in each row one answer, drawn on the CPU with probability ``soft[row]``.

    Args:
        soft: RB-W target rows.
        generator: The update's CPU generator.

    Returns:
        The one-hot targets on the device and in the dtype of ``soft``.

    Raises:
        ValueError: When a row does not sum to 1.
    """
    probs = soft.detach().float().cpu()
    sums = probs.sum(-1)
    if not bool(((sums > 0.999) & (sums < 1.001)).all()):
        raise ValueError('an RB-W target row does not sum to 1')
    index = torch.multinomial(probs, 1, generator=generator).flatten()
    hard = torch.zeros_like(soft)
    hard[torch.arange(len(index), device=soft.device), index.to(soft.device)] = 1.0
    return hard


def _backward(model: ULTRA, prepared: PreparedWorld, generator: Optional[torch.Generator], microbatch: int) -> float:
    """Accumulate the gradients of the update's loss; returns the logged loss."""
    total = prepared.num_queries
    device, dtype = model.device, next(model.parameters()).dtype
    losses = []
    for task in prepared.tasks.values():
        task.graph.attach(model)
        queries = to_device(task.queries, device)
        soft = to_device(task.soft, device).to(dtype=dtype)
        context = to_device(task.context_mask, device)
        for head in (False, True):
            rows_of_direction = (task.head_prediction == head).nonzero().flatten().tolist()
            for start in range(0, len(rows_of_direction), microbatch):
                rows = rows_of_direction[start:start + microbatch]
                place = to_device(torch.tensor(rows, dtype=torch.long), device)
                scores = model.forward_query_facts(queries[place], head)
                observed = context[place]
                target = soft[place] if generator is None else hard_target(soft[place], generator)
                if generator is not None and bool((target.bool() & observed).any()):
                    raise ValueError('a drawn answer is a context answer')
                logp = scores.masked_fill(observed, float('-inf')).log_softmax(-1)
                loss = -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1).mean()
                (loss * len(rows) / total).backward()
                losses.append((len(rows), loss.detach()))
                del scores, logp, loss
    numbers = torch.stack([loss for _, loss in losses]).tolist()
    value = 0.0
    for (size, _), number in zip(losses, numbers):
        if not math.isfinite(number):
            raise FloatingPointError('the loss is not finite')
        value += number * size / total
    return value


def update(model: ULTRA, optimizer: torch.optim.Optimizer, prepared: PreparedWorld, generator: Optional[torch.Generator] = None,
           microbatch: int = MICROBATCH) -> dict[str, float]:
    """One pretraining update on a prepared world; the caller sets the learning rate.

    Args:
        model: ULTRA in training mode, with the options of the preparation.
        optimizer: Its AdamW.
        prepared: The prepared world.
        generator: The update's CPU generator for the hard target (``streams.hard_target_seed``); ``None`` trains on the
            RB-W rows.
        microbatch: Query rows per backward chunk.

    Returns:
        ``loss`` (the logged loss), ``grad_norm`` and ``seconds``.

    Raises:
        FloatingPointError: For a loss, gradient norm or parameter that is not finite.
    """
    if not model.training:
        raise ValueError('pretraining scores in training mode')
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss = _backward(model, prepared, generator, microbatch)
    norm = torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm()
    if not torch.isfinite(norm):
        raise FloatingPointError('the gradient norm is not finite')
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError('a parameter is not finite after the step')
    return dict(loss=loss, grad_norm=float(norm), seconds=time.monotonic() - started)


@torch.no_grad()
def ema_update(ema: Mapping[str, torch.Tensor], model: torch.nn.Module, decay: float = EMA_DECAY) -> None:
    """Move the EMA of every floating tensor of the model's state towards its value: ``ema = decay * ema + (1 - decay) * value``."""
    for key, value in model.state_dict().items():
        if value.dtype.is_floating_point:
            ema[key].mul_(decay).add_(value.detach(), alpha=1 - decay)


# ------------------------------------------------------------------------------------------------------ the run

def source_hashes() -> dict[str, str]:
    """SHA-256 of the sources that define the updates: ``dicee/synthetic_kg`` and the ULTRA model files, by path below ``dicee``."""
    package = Path(__file__).resolve().parent
    root = package.parent
    files = sorted(package.rglob('*.py')) + [Path(m.__file__).resolve() for m in (ultra, graph_model, _relation_weights) if m.__file__]
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in files}


def _real_sources(prior: PriorConfig, folder: Union[str, Path, None]) -> dict[str, str]:
    """The SHA-256 of the real source graphs that a prior reads, checked against their pins (empty without real worlds)."""
    if not any(spec.generator == 'real' for spec in prior.families):
        return {}
    root = real.source_folder(folder)
    found = {name: file_sha256(root / name / 'data.pt') for name in REAL_SOURCE_SHA256}
    differing = [name for name, digest in found.items() if digest != REAL_SOURCE_SHA256[name]]
    if differing:
        raise ValueError(f'the real source graphs {differing} in {root} differ from their pins')
    return found


def run_record(config: PretrainConfig, preset: ArmPreset, digest: str) -> dict[str, Any]:
    """What determines the updates of a run (a resumed run must have the same record).

    Args:
        config: The run configuration.
        preset: Its arm.
        digest: The digest of the arm's world order.

    Returns:
        A JSON-safe dictionary.
    """
    prior = preset.prior
    families = [dict(name=s.name, generator=s.generator, first=s.first, params={k: list(v) if isinstance(v, tuple) else v for k, v in s.params},
                     null_swap_budget=s.null_swap_budget) for s in prior.families]
    return dict(format=RUN_FORMAT, arm=preset.name, research_arm=preset.research_arm, seed=config.seed, total=config.total,
                order_sha256=digest, order_slots=TOTAL_UPDATES, prior=dict(mix=prior.mix, order_name=prior.order_name, families=families),
                model_config=preset.model_config(), target=preset.target, constant_updates=CONSTANT_UPDATES, decay_updates=DECAY_UPDATES,
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY, microbatch=MICROBATCH,
                device=torch.device(config.device).type, threads=config.threads, deterministic=True)


def provenance(config: PretrainConfig, preset: ArmPreset) -> dict[str, Any]:
    """Where the run's results come from: versions, device, the source and data hashes.

    Args:
        config: The run configuration.
        preset: Its arm.

    Returns:
        A JSON-safe dictionary.
    """
    device = torch.device(config.device)
    cuda = device.type == 'cuda'
    return dict(dicee=DICEE_VERSION, python=sys.version.split()[0], torch=str(torch.__version__), numpy=str(numpy.__version__),
                device=str(device), gpu=torch.cuda.get_device_name(device) if cuda else None, cuda=torch.version.cuda if cuda else None,
                cudnn=torch.backends.cudnn.version() if cuda else None, cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                sources=source_hashes(), real_sources=_real_sources(preset.prior, config.real_sources))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(temporary, path)


def _worker_init(threads: int) -> None:
    """Worker processes leave stopping to the training process and use few torch threads."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    torch.set_num_threads(threads)


def prepared_worlds(seed: int, slots: Sequence[Slot], prior: PriorConfig, options: Mapping[str, str], workers: int = 4,
                    prefetch: Optional[int] = None, worker_threads: int = 1) -> Generator[PreparedWorld, None, None]:
    """The prepared worlds of the slots, in slot order, generated ahead by a pool of spawned worker processes.

    At most ``prefetch`` worlds are in flight. A generator error in a worker is raised here (the run stops on it).

    Args:
        seed: The run seed.
        slots: The order slots.
        prior: The prior of the order.
        options: ``ultra_relation_weighting`` and ``ultra_pair_channels`` of the model.
        workers: Worker processes; 0 prepares the worlds in this process.
        prefetch: Worlds in flight; ``None``: twice the workers.
        worker_threads: Torch threads of a worker.

    Yields:
        One prepared world per slot.
    """
    if workers == 0:
        for slot in slots:
            yield prepare_slot(seed, slot, prior, options)
        return
    ahead = prefetch if prefetch is not None else 2 * workers
    pool = ProcessPoolExecutor(workers, mp_context=get_context('spawn'), initializer=_worker_init, initargs=(worker_threads,))
    try:
        pending: deque[Future[PreparedWorld]] = deque()
        remaining = iter(slots)
        for slot in remaining:
            pending.append(pool.submit(prepare_slot, seed, slot, prior, dict(options)))
            if len(pending) >= ahead:
                break
        while pending:
            prepared = pending.popleft().result()
            following = next(remaining, None)
            if following is not None:
                pending.append(pool.submit(prepare_slot, seed, following, prior, dict(options)))
            yield prepared
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@contextmanager
def _run_environment(config: PretrainConfig) -> Iterator[None]:
    """Torch threads, deterministic execution and the real-source folder of a run; all restored afterwards."""
    threads = torch.get_num_threads()
    previous = os.environ.get(real.SOURCES_ENV)
    torch.set_num_threads(config.threads)
    if config.real_sources is not None:
        os.environ[real.SOURCES_ENV] = str(config.real_sources)
    try:
        with deterministic_execution():
            yield
    finally:
        torch.set_num_threads(threads)
        if config.real_sources is not None:
            if previous is None:
                del os.environ[real.SOURCES_ENV]
            else:
                os.environ[real.SOURCES_ENV] = previous


def _cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _check_start(folder: Path, record: Mapping[str, Any], sources: Mapping[str, str], allow_code_change: bool) -> None:
    """Refuse to continue a run folder whose run record differs, or whose bound sources changed."""
    path = folder / 'run.json'
    if not path.exists():
        return
    stored = json.loads(path.read_text())
    if stored['run'] != record:
        changed = sorted(k for k in {*stored['run'], *record} if stored['run'].get(k) != record.get(k))
        raise ValueError(f'{folder} holds another run (it differs in {changed}); choose another folder')
    changed = sorted(k for k in {*stored['provenance']['sources'], *sources} if stored['provenance']['sources'].get(k) != sources.get(k))
    if changed and not allow_code_change:
        raise ValueError(f'the sources {changed} changed since the run started; resume with allow_code_change or start a new folder')


def pretrain(config: PretrainConfig, out_dir: Union[str, Path], stop: Optional[Callable[[], bool]] = None) -> PretrainResult:
    """Train an arm on its world order, or resume its run from ``last.pt``.

    Args:
        config: The run.
        out_dir: The run folder.
        stop: Called after every update; when it returns ``True`` the run saves ``last.pt`` and stops (rerun to resume).

    Returns:
        Where the run ended.

    Raises:
        ValueError: For an order that differs from the arm's pin, or a folder of another run.
        RuntimeError: For CUDA without a deterministic cuBLAS workspace setting.
    """
    preset = arm_preset(config.arm)
    folder = Path(out_dir)
    device = torch.device(config.device)
    if device.type == 'cuda':
        require_cublas_workspace()
    order = build_order(config.seed, preset.prior)
    digest = order_sha256(order)
    pinned = preset.pinned_order(config.seed)
    if pinned is not None and digest != pinned:
        raise ValueError(f'the world order of arm {preset.name}, seed {config.seed} has digest {digest}, its pin is {pinned}')
    record = run_record(config, preset, digest)
    with _run_environment(config):
        origin = provenance(config, preset)
        _check_start(folder, record, origin['sources'], config.allow_code_change)
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'run.json').exists():
            _write_json(folder / 'run.json', dict(run=record, provenance=origin))
        model = initial_model(preset, config.seed).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history: list[dict[str, Any]] = []
        start = 0
        last = folder / 'last.pt'
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=True)
            if state.get('format') != STATE_FORMAT or state.get('run') != record:
                raise ValueError(f'{last} is not a training state of this run')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            ema = {k: v.to(device) for k, v in state['ema'].items()}
            history, start = list(state['history']), int(state['next'])
            logger.info('resuming %s from update %d', folder, start)
        end = config.total if config.stop_after is None else config.stop_after
        exports, states = set(config.export_updates), set(config.state_updates)
        checkpoint_provenance = dict(source='pretraining', run=record, **origin)

        def save_state(path: Path, done: int) -> None:
            atomic_save(dict(format=STATE_FORMAT, run=record, next=done, model=_cpu(model.state_dict()), optimizer=optimizer.state_dict(),
                             ema=_cpu(ema), history=history), path)

        done, stopped = start, False
        began = time.monotonic()
        worlds = prepared_worlds(config.seed, order[start:end], preset.prior, preset.model_config(), config.workers, config.prefetch,
                                 config.worker_threads)
        with closing(worlds):
            for i, prepared in enumerate(worlds, start):
                if tuple(prepared.slot) != tuple(order[i]):
                    raise RuntimeError(f'update {i} got the world of slot {tuple(prepared.slot)}, not {tuple(order[i])}')
                lr = learning_rate(i)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                generator = torch.Generator().manual_seed(hard_target_seed(config.seed, i)) if preset.target == 'hard' else None
                metrics = update(model, optimizer, prepared, generator)
                ema_update(ema, model)
                done = i + 1
                slot, meta = prepared.slot, prepared.meta
                history.append(dict(update=done, family=slot.family, source=slot.source, index=slot.index, lr=lr, loss=metrics['loss'],
                                    grad_norm=metrics['grad_norm'], seconds=metrics['seconds'], n_entities=meta['n_entities'],
                                    n_context=meta['n_context'], program_slot=bool(meta.get('program_slot'))))
                del prepared
                if done in exports:
                    for weights, values in (('raw', model.state_dict()), ('ema', ema)):
                        save_checkpoint(make_checkpoint(values, arm=preset, seed=config.seed, updates=done, weights=weights, order_sha256=digest,
                                                        provenance=checkpoint_provenance), folder / f'export-{done}-{weights}.pt')
                if done in states:
                    save_state(folder / f'state-{done}.pt', done)
                stopping = stop is not None and done < end and stop()
                if done % config.save_every == 0 or done == end or stopping:
                    save_state(last, done)
                if done % 50 == 0 or done == end or stopping:
                    logger.info('update %d of %d (%s %s %d): loss %.6f, gradient norm %.4f, learning rate %.3g, %.0f s',
                                done, config.total, slot.family, slot.source, slot.index, metrics['loss'], metrics['grad_norm'], lr,
                                time.monotonic() - began)
                    _write_json(folder / 'progress.json', dict(done=done, total=config.total, last=history[-1], seconds=time.monotonic() - began))
                if stopping:
                    stopped = True
                    break
    status = 'stopped' if stopped else 'complete' if done >= config.total else 'stopped-after'
    return PretrainResult(status, done, folder)
