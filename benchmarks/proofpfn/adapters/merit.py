"""MERIT (Gregucci et al., arXiv 2610.15638): its five release checkpoints as reference systems of the benchmark (README,
section MERIT).

Port of the research harness ``allgraph-v1/ag_merit.py`` with ``merit-ref-v1-runtime/merit_ref.py``. MERIT is ULTRA's
entity NBFNet conditioned on relation representations from a Transformer over relation tokens, which reads a motif
tensor of the inference graph:

- Model (``MERIT``): dicee's ``ULTRA`` with MERIT's own relation Transformer (``kgfm.models.merit.RelTransformerMERIT``
  with ``CONFIG``, the configuration of MERIT's inference configs) and motif tensor (``kgfm.tasks.build_motif_tensor``).
  A release checkpoint holds 74 tensors of the relation Transformer and 52 of the entity NBFNet, whose names and shapes
  equal dicee's ULTRA (checked on loading).
- A head query is a tail query of the inverse relation (dicee's convention), and MERIT conditions its relation
  Transformer on that converted relation, where ULTRA uses the direct relation.
- The motif tensor comes from the inference graph with both edge directions, once per graph.
- The relation Transformer draws Gaussian noise on its tokens: the noise of a query is seeded with
  ``merit_seed(0, head, relation)`` inside ``torch.random.fork_rng``, so ranks do not depend on the query order.
- Attention: ``flex`` on CUDA (under MERIT's float16 autocast); the CPU runs MERIT's ``mha`` path (smoke tests only).
- Compiler (``fresh_compiler``): a fresh compiler cache per graph, and a recompilation is an error: torch 2.5.1 falls back
  from compiled flex attention to eager attention after eight shapes, which changes the scores.

The five systems ``merit<S>`` run together through ``evaluate.evaluate_graph``, so their row file has the reference keys
and the model hash of each checkpoint, plus ``merit_meta``. MERIT's package imports only with its own requirements, so
``run_in_child`` runs the scoring in a child process in MERIT's environment (``--python`` or ``$MERIT_PYTHON``).
"""
import logging
import os
import shlex
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Mapping, Optional, Sequence, cast

import torch
from torch import nn

from dicee.models._inference import candidate_slice
from dicee.models.graph_model import GraphKGE
from dicee.models.ultra import ULTRA

from .. import evaluate
from ..manifests import REPO, GraphEntry, Manifest, PathLike, file_sha256, read_models
from ..systems import Reference, frozen
from . import AdapterSetupError, ExternalStepError, call, check_commit, check_file, external_folder, external_pins

logger = logging.getLogger(__name__)

SEEDS = (1024, 2025, 2026, 4000, 8000)
KEYS = tuple(f'merit{s}' for s in SEEDS)
CONFIG = dict(input_dim=64, hidden_dims=[64] * 6, num_heads=8, ff_dim=256, dropout=0.0, rni=True, num_channels=4,
              attn_impl='flex', role_features='max,logdeg', role_dim=8)   # config/merit/*/MERIT_inference.yaml of MERIT
NOISE_SEED = 0
UPSTREAM = ('kgfm/models/merit.py', 'kgfm/tasks.py')
CHECKPOINTS = Path('merit') / 'ckpts' / 'merit'
ATTENTION_POLICY = 'fresh-compiler-per-graph-no-recompile-v1'
GRAPH_HOURS = 6.0
"""The time per graph after which the child process stops (``run_in_child``)."""
VARIABLE, PYTHON_VARIABLE = 'MERIT_HOME', 'MERIT_PYTHON'


def merit_seed(noise_seed: int, head: int, relation: int) -> int:
    """The seed of the relation tokens' noise for a query with this head entity and (converted) relation."""
    return (noise_seed * 1_000_003 + head * 4_099 + relation) % (2**63 - 1)


@dataclass(frozen=True)
class Upstream:
    """MERIT's own code that the model uses.

    Attributes:
        relation_model: ``RelTransformerMERIT``, called with ``CONFIG``.
        motif_tensor: ``build_motif_tensor``, called with a graph namespace (``edge_index``, ``edge_type``, ``num_nodes``,
            ``num_relations``).
        files: The SHA-256 of the two source files.
    """

    relation_model: Callable[..., nn.Module]
    motif_tensor: Callable[[Any], torch.Tensor]
    files: Mapping[str, str] = field(default_factory=dict)


def import_upstream(home: Path, pins: Mapping[str, str]) -> Upstream:
    """``RelTransformerMERIT`` and ``build_motif_tensor`` from MERIT's package, which must load from ``home``.

    Needs MERIT's environment with ``home`` on the module path (the child process of ``run_in_child``).

    Raises:
        AdapterSetupError: When the package does not import, loads from elsewhere, or its files differ from the pins.
    """
    try:
        from kgfm.models.merit import RelTransformerMERIT  # type: ignore[import-not-found]
        from kgfm.tasks import build_motif_tensor  # type: ignore[import-not-found]
    except ImportError as error:
        raise AdapterSetupError(f"MERIT's package does not import ({error}); run it in MERIT's environment with its folder on "
                                'PYTHONPATH (adapter merit --python)') from None
    files = {}
    for name, function in zip(UPSTREAM, (RelTransformerMERIT, build_motif_tensor)):
        source = Path(sys.modules[function.__module__].__file__ or '').resolve()
        if source != (home / name).resolve():
            raise AdapterSetupError(f'MERIT: {function.__name__} loads from {source}, not from {home / name}')
        files[name] = check_file(source, pins[name], 'MERIT')
    return Upstream(relation_model=RelTransformerMERIT, motif_tensor=build_motif_tensor, files=files)


class MERIT(ULTRA):
    """dicee's ULTRA with MERIT's relation Transformer and motif tensor (module docstring).

    Args:
        args: The model options of ULTRA, and ``merit_noise_seed`` (default 0).
        upstream: MERIT's code.
    """

    name = 'MERIT'
    checkpoint_hint = 'MERIT release checkpoint: 74 relation-Transformer and 52 ULTRA entity tensors'
    deterministic_inference = False

    def __init__(self, args: Mapping[str, Any], upstream: Upstream) -> None:
        super().__init__(dict(args))
        self.upstream = upstream
        self.relation_model = cast(Any, upstream.relation_model(**CONFIG))   # in place of ULTRA's RelNBFNet
        self.register_buffer('motif', None, persistent=False)
        self.noise_seed = int(args.get('merit_noise_seed', NOISE_SEED))

    def _build_relation_graph(self) -> None:
        """Build the motif tensor of the inference graph (both edge directions) instead of ULTRA's relation graph."""
        self.clear_inference_cache()
        graph = types.SimpleNamespace(edge_index=self.edge_index, edge_type=self.edge_type, num_nodes=self.num_entities,
                                      num_relations=2 * self.num_direct_relations)
        self.motif = self.upstream.motif_tensor(graph)
        relation_model: Any = self.relation_model
        relation_model._prof_cache = None   # MERIT's memo of the motif row summary

    def _score(self, heads: torch.Tensor, relations: torch.Tensor, candidates: torch.Tensor, query_relations: torch.Tensor,
               edges: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Scores of the candidates; the relation Transformer reads the converted relation and seeds its noise per query."""
        output = []
        struct = types.SimpleNamespace(num_nodes=2 * self.num_direct_relations)
        devices = [self.device] if self.device.type == 'cuda' else []
        relation_model: Any = self.relation_model
        relation_model.attn_impl = CONFIG['attn_impl'] if self.device.type == 'cuda' else 'mha'
        for start in range(0, len(heads), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(merit_seed(self.noise_seed, int(heads[start]), int(relations[start])))
                rels = relation_model(struct, relations[sl], self.motif)
            output.append(self.entity_model(*edges, self.num_entities, rels, heads[sl], relations[sl], candidate_slice(candidates, sl), None))
        return torch.cat(output) if output else next(self.parameters()).new_empty((0, candidates.shape[1]))


@dataclass(frozen=True)
class MeritReference(Reference):
    """One MERIT release checkpoint as a reference system of the benchmark (key ``merit<seed>``).

    Attributes:
        upstream: MERIT's code; required to build the model.
    """

    upstream: Optional[Upstream] = field(default=None, compare=False)
    KEYS: ClassVar[tuple[str, ...]] = KEYS

    def build_model(self, num_entities: int, num_relations: int, setting: Optional[Mapping[str, Any]] = None) -> GraphKGE:
        """The frozen MERIT of this checkpoint on the CPU (query batch 1), built with the global random state restored.

        Raises:
            AdapterSetupError: Without MERIT's code.
        """
        if self.upstream is None:
            raise AdapterSetupError(f"{self.key}: MERIT's code is not loaded")
        with torch.random.fork_rng(devices=[]):
            model = MERIT(dict(num_entities=num_entities, num_relations=num_relations, ultra_query_batch_size=1,
                               merit_noise_seed=NOISE_SEED), self.upstream)
        model.load_pretrained(self.path)
        return frozen(model)

    def spec(self) -> dict[str, Any]:
        """The system's record."""
        return dict(kind='reference', model='MERIT', checkpoint=str(self.path), query_batch_size=1, noise_seed=NOISE_SEED,
                    config=CONFIG, attention_policy=ATTENTION_POLICY)


@dataclass
class MeritSetup:
    """A checked MERIT setup: the code folder and the five pinned checkpoints.

    Attributes:
        home: The MERIT code folder.
        pins: The pins of ``models.json`` (``external.merit``).
        checkpoints: System key to checkpoint file.
        sha256: System key to the pinned SHA-256 of its checkpoint.
    """

    home: Path
    pins: Mapping[str, str]
    checkpoints: Mapping[str, Path]
    sha256: Mapping[str, str]
    _upstream: Optional[Upstream] = field(default=None, init=False, repr=False)

    def upstream(self) -> Upstream:
        """MERIT's code, imported on first use (needs MERIT's environment)."""
        if self._upstream is None:
            self._upstream = import_upstream(self.home, self.pins)
        return self._upstream

    def systems(self) -> list[MeritReference]:
        """The five MERIT systems, in seed order."""
        upstream = self.upstream()
        return [MeritReference(key=key, path=self.checkpoints[key], sha256=self.sha256[key], upstream=upstream) for key in KEYS]

    def record(self) -> dict[str, Any]:
        """The setup for the logs, with the files of MERIT's package that imported (needs MERIT's environment)."""
        return dict(home=str(self.home), code_commit=self.pins['code_commit'], upstream_sha256={k: self.pins[k] for k in UPSTREAM},
                    checkpoints={k: str(p) for k, p in self.checkpoints.items()}, imported=dict(self.upstream().files))


def setup(code: Optional[PathLike] = None, models: Optional[PathLike] = None, checkpoint_root: PathLike = REPO / 'checkpoints') -> MeritSetup:
    """The checked MERIT setup: the folder (``code``, else ``$MERIT_HOME``) at the pinned commit with the pinned source files,
    and the five checkpoints below ``checkpoint_root`` with the SHA-256 of ``models.json``.

    Raises:
        AdapterSetupError: For a missing folder or file, another commit or a file that differs from its pin.
    """
    pins = external_pins('merit', models)
    home = external_folder(code, VARIABLE, 'MERIT')
    check_commit(home, pins['code_commit'], 'MERIT')
    for name in UPSTREAM:
        check_file(home / name, pins[name], 'MERIT')
    references = read_models(models).references
    files = {key: Path(checkpoint_root) / CHECKPOINTS / f'merit_s{seed}.pth' for key, seed in zip(KEYS, SEEDS)}
    sha = {key: references[key].sha256 for key in KEYS}
    for key in KEYS:
        check_file(files[key], sha[key], key)
    return MeritSetup(home=home, pins=pins, checkpoints=files, sha256=sha)


@contextmanager
def fresh_compiler() -> Iterator[None]:
    """A fresh compiler cache, and recompilation as an error, for the scoring of one graph.

    Raises:
        AdapterSetupError: When a compiler cache limit is below 1 (MERIT would run eager attention, which changes its scores).
    """
    from torch._dynamo import config, reset
    if min(config.cache_size_limit, config.accumulated_cache_size_limit) < 1:
        raise AdapterSetupError('MERIT requires a positive compiler cache limit; eager attention changes its ranks')
    reset()
    with config.patch(error_on_recompile=True):
        yield


def evaluate_graph(manifest: Manifest, entry: GraphEntry, out: PathLike, merit: MeritSetup, *, root: PathLike = REPO,
                   device: str = 'cuda') -> str:
    """Rank one graph with the five MERIT systems and write their row file (``evaluate.evaluate_graph``), with ``merit_meta``.

    Args:
        manifest: The graph set.
        entry: The graph.
        out: The output root.
        merit: The checked setup (MERIT's code imports here).
        root: The data root of the manifest paths.
        device: The scoring device.

    Returns:
        ``'exists'`` or ``'done'``.

    Raises:
        ExternalStepError: When CUDA runs out of memory.
    """
    meta = dict(adapter_sha256=file_sha256(Path(__file__)), upstream_sha256=dict(merit.upstream().files), merit_home=str(merit.home),
                code_commit=merit.pins['code_commit'], noise_seed=NOISE_SEED, config=CONFIG, attention_policy=ATTENTION_POLICY)
    try:
        with fresh_compiler():
            return evaluate.evaluate_graph(manifest, entry, merit.systems(), out, root=root, device=device, extra=dict(merit_meta=meta))
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise ExternalStepError(dict(stage='score', reason='CUDA out of memory', systems=list(KEYS), error=str(error).splitlines()[0][:400])) from None


def child_command(argv: Sequence[str], merit: MeritSetup, python: Optional[str] = None) -> tuple[list[str], dict[str, str]]:
    """The command and environment of the child process that runs the scoring in MERIT's environment.

    Args:
        argv: The arguments of this command line (``adapter merit ...``).
        merit: The checked setup.
        python: The command of MERIT's Python (default ``$MERIT_PYTHON``, else this interpreter).

    Returns:
        ``(command, environment)``: ``python -B -m benchmarks.proofpfn <argv> --worker`` with ``PYTHONPATH`` = the repository
        and the MERIT folder.
    """
    command = shlex.split(python or os.environ.get(PYTHON_VARIABLE) or shlex.quote(sys.executable))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(REPO), str(merit.home))), PYTHONDONTWRITEBYTECODE='1', MERIT_HOME=str(merit.home))
    return [*command, '-B', '-m', 'benchmarks.proofpfn', *argv, '--worker'], env


def run_in_child(argv: Sequence[str], graphs: int, *, code: Optional[PathLike] = None, models: Optional[PathLike] = None,
                 checkpoint_root: PathLike = REPO / 'checkpoints', python: Optional[str] = None) -> Optional[int]:
    """Check the setup here, then run ``adapter merit`` in a child process in MERIT's environment.

    Args:
        argv: The arguments of this command line.
        graphs: The number of graphs of the run; the child stops after ``GRAPH_HOURS`` per graph.
        code: The MERIT folder (default ``$MERIT_HOME``).
        models: The pins (default ``manifests/models.json``).
        checkpoint_root: The root of ``merit/ckpts/merit/``.
        python: The command of MERIT's Python (default ``$MERIT_PYTHON``, else this interpreter).

    Returns:
        The child's exit code, or ``None`` when it ran past its time.

    Raises:
        AdapterSetupError: When the setup check fails or the Python of the child does not start.
    """
    merit = setup(code, models, checkpoint_root)
    command, env = child_command(argv, merit, python)
    logger.info('MERIT: %s', ' '.join(command))
    try:
        exit_code = call(command, timeout=GRAPH_HOURS * 3600 * max(1, graphs), env=env).code
    except OSError as error:   # no such Python
        raise AdapterSetupError(f'MERIT: cannot run {command[0]} ({error})') from None
    if exit_code is None:
        logger.error('MERIT: the child process ran past %s h per graph and was stopped', GRAPH_HOURS)
    return exit_code
