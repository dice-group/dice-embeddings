"""The systems of the ProofPFN benchmark: pretrained checkpoints of this package and the released reference models.

- ``Checkpoint``: one ``proofpfn-checkpoint-v1`` file (``dicee.synthetic_kg.checkpoint``), which carries its own model
  options. ``--model NAME=TEMPLATE`` names one or more of them: a template with ``{seed}`` gives one system per seed
  with the label ``NAME.seed<S>``, and a template without it one system with the label ``NAME``.
- ``Reference``: TRIX (``trix``) or ULTRA-3g (``ultra3g``), the released checkpoints, pinned by their SHA-256.
- ``OfficialFlock`` (``flock``): the released Flock checkpoint at the graph's official walk count with the official
  passes and sampling seed 42 (``flock_setting``). With ``one_microbatch`` (the setting of the thesis runs) the passes
  of one query run as one microbatch of the walk sampler; without it they run one after another (the same law, other
  walks).

A system has a ``label``, ``check``, ``build_model``, ``query_batch_size``, ``spec`` and ``sha256``. Every model is built
on the CPU with the global random state restored, in evaluation mode without gradients. The evaluation attaches the
inference graph with ``attach``.
"""
import math
import string
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Optional, Sequence, TypeVar, Union

import torch

from dicee.models.flock import Flock
from dicee.models.graph_model import GraphKGE
from dicee.models.trix import TRIX
from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.checkpoint import CheckpointError, model_from_checkpoint, read_checkpoint
from dicee.synthetic_kg.variants import SEEDS

from .data import GraphData, inference_stats
from .manifests import REPO, GraphEntry, InputError, file_sha256

CHECKPOINTS = REPO / 'checkpoints'
FLOCK_SEED = 42
REFERENCES: Mapping[str, tuple[str, str]] = MappingProxyType({
    'trix': ('trix/entity_prediction.pth', '8f6e7266093c2d15ad88d41e9825e4b327171143543cf060cfee907d9a890342'),
    'ultra3g': ('ultra_3g.pth', 'fdedc01b0045fc089d2ad5da08569466b7b221691fe978a18e91082bbb133c18'),
    'flock': ('flock/flock_entity.pth', 'd32ae78c0b0b5731ce183fd7e05f6443c03bbb4dd795481d0077818b8ba67850'),
})
"""The released reference checkpoints: the file below the checkpoint root and its SHA-256, by label."""
Model = TypeVar('Model', bound=GraphKGE)


class SystemCheckError(InputError):
    """A system that cannot run: a missing or invalid file, a hash that differs from its pin, or a label of another system."""


# ------------------------------------------------------------------------------------------- Flock's official setting

BASE_WALKS, LOW_WALKS, HIGH_WALKS = 128, 16, 512
# The inference-graph statistics (distinct entities, distinct triples) of Flock's pretraining graphs.
FLOCK_PRETRAIN = {'FB15k-237': (14505, 272115), 'WN18RR': (40559, 86835), 'codex-m': (17050, 185584)}
V_TRAIN = sum(v for v, _ in FLOCK_PRETRAIN.values()) / len(FLOCK_PRETRAIN)
E_TRAIN = sum(e for _, e in FLOCK_PRETRAIN.values()) / len(FLOCK_PRETRAIN)
FLOCK_RULE = ('walks: 2^round(log2(128 x HM(|V|/24038, |E|/181511.3))) clamped to 16..512, |V| and |E| of the inference graph '
              '(train.txt: distinct entities, distinct triples); passes: the official script on its 54 suite graphs, else '
              '16 (walks <= 128), 4 (256), 2 (512); the script value is used on its 54 suite graphs')


def _script_table() -> Mapping[str, tuple[int, int]]:
    """(walks, passes) of the official script ``scripts/entity_zeroshot.sh`` per suite id."""
    table = dict.fromkeys(('FB15k237Inductive:v1', 'FB15k237Inductive:v2', 'FB15k237Inductive:v3', 'FB15k237Inductive:v4',
                           'WN18RRInductive:v1', 'WN18RRInductive:v2', 'WN18RRInductive:v3', 'WN18RRInductive:v4',
                           'NELLInductive:v1', 'NELLInductive:v2', 'NELLInductive:v3', 'NELLInductive:v4', 'ILPC2022:small', 'HM:1k',
                           'FBIngram:25', 'FBIngram:50', 'FBIngram:75', 'FBIngram:100', 'WKIngram:25', 'WKIngram:50', 'WKIngram:75',
                           'WKIngram:100', 'NLIngram:0', 'NLIngram:25', 'NLIngram:50', 'NLIngram:75', 'NLIngram:100',
                           'WikiTopicsMT1:tax', 'WikiTopicsMT1:health', 'WikiTopicsMT2:sci', 'WikiTopicsMT4:sci', 'WikiTopicsMT4:health',
                           'Metafam', 'FBNELL', 'WDsinger', 'CoDExSmall'), (16, 16))
    table.update(dict.fromkeys(('HM:3k', 'WikiTopicsMT2:org', 'WikiTopicsMT3:art', 'WikiTopicsMT3:infra', 'NELL23k', 'FB15k237_10'), (32, 16)))
    table.update(dict.fromkeys(('ILPC2022:large', 'HM:5k', 'FB15k237_20', 'FB15k237_50'), (64, 16)))
    table.update(dict.fromkeys(('HM:indigo', 'NELL995', 'ConceptNet100k'), (128, 16)))
    table.update({'AristoV4': (256, 4), 'YAGO310': (512, 8)})
    table.update(dict.fromkeys(('CoDExLarge', 'DBpedia100k', 'Hetionet'), (512, 2)))
    assert len(table) == 54
    return MappingProxyType(table)


FLOCK_SCRIPT = _script_table()
"""The official script's (walks, passes) of the 54 standard-suite graphs, by suite id."""


def rule_walks(entities: int, edges: int) -> tuple[float, int]:
    """The raw value and the walk count of Flock's test-time rule for an inference graph.

    Args:
        entities: Distinct entities of the inference graph.
        edges: Distinct triples of the inference graph.

    Returns:
        ``(raw, walks)``: 128 times the harmonic mean of ``entities / V_TRAIN`` and ``edges / E_TRAIN``, and the walk
        count ``2^round(log2(raw))`` clamped to 16..512. ``V_TRAIN`` and ``E_TRAIN`` are the mean sizes of Flock's
        pretraining graphs.

    Raises:
        ValueError: For an empty inference graph.
    """
    if entities <= 0 or edges <= 0:
        raise ValueError(f'the rule needs a non-empty inference graph ({entities} entities, {edges} edges)')
    a, b = entities / V_TRAIN, edges / E_TRAIN
    harmonic = 2.0 / (1.0 / a + 1.0 / b)
    raw = BASE_WALKS * harmonic
    return raw, int(min(HIGH_WALKS, max(LOW_WALKS, 2 ** round(math.log2(raw)))))


def rule_passes(walks: int) -> int:
    """The passes of the rule for a graph outside the official script.

    Args:
        walks: The walk count of the rule (``rule_walks``).

    Returns:
        16 for at most 128 walks, 4 for 256 walks, else 2.
    """
    return 16 if walks <= 128 else (4 if walks == 256 else 2)


def flock_setting(entities: int, edges: int, suite_id: Optional[str] = None) -> dict[str, Any]:
    """Flock's official setting of a graph: the script's (walks, passes) on its suite graphs, else the rule's.

    Args:
        entities: Distinct entities of the inference graph.
        edges: Distinct triples of the inference graph.
        suite_id: The graph's standard-suite id, if it has one.

    Returns:
        The setting with the rule's values for the record.
    """
    raw, n = rule_walks(entities, edges)
    script = FLOCK_SCRIPT.get(suite_id) if suite_id else None
    walks, passes = script if script else (n, rule_passes(n))
    return dict(suite_id=suite_id, entities=int(entities), edges=int(edges), rule_raw=raw, rule_walks=n, rule_passes=rule_passes(n),
                script=list(script) if script else None, source='script' if script else 'rule', walks=walks, passes=passes)


def official_setting(entry: GraphEntry, graph: GraphData) -> dict[str, Any]:
    """The official Flock setting of a graph, after checking train.txt against the manifest's size.

    Args:
        entry: The graph's manifest entry, with ``size`` and ``suite_id``.
        graph: The loaded graph.

    Returns:
        The fields of ``flock_setting`` and ``rule`` (``FLOCK_RULE``).

    Raises:
        SystemCheckError: When the inference graph's statistics differ from the manifest (the walk count would not follow
            the stated rule).
    """
    setting = flock_setting(entry.size['entities'], entry.size['inference_edges'], entry.suite_id)
    found = inference_stats(graph.train)
    if found != (setting['entities'], setting['edges']):
        raise SystemCheckError(f'{entry.name}: train.txt has {found[0]} entities and {found[1]} distinct triples, the manifest '
                               f"{setting['entities']} and {setting['edges']}: no official Flock setting")
    return dict(rule=FLOCK_RULE, **setting)


# ------------------------------------------------------------------------------------------------------ systems

def frozen(model: Model) -> Model:
    """The model in evaluation mode without gradients.

    Args:
        model: A graph model.

    Returns:
        The same model, changed in place.
    """
    model.eval().requires_grad_(False)
    return model


def attach(model: Model, train: torch.Tensor, num_entities: int, num_relations: int, device: Union[str, torch.device],
           query_batch_size: int = 1) -> Model:
    """Attach the inference graph on the CPU, then move the model to the device.

    Args:
        model: A frozen graph model on the CPU.
        train: ``[N, 3]`` ids of the inference graph.
        num_entities: The number of entities of all splits.
        num_relations: The number of relations of all splits.
        device: The scoring device.
        query_batch_size: The model's microbatch of queries (Flock: of query passes).

    Returns:
        The model, in evaluation mode without gradients.
    """
    model.query_batch_size = query_batch_size
    model.set_graph(train, num_entities, num_relations)
    model.to(device)
    return frozen(model)


@dataclass
class Checkpoint:
    """A pretrained checkpoint of this package (``proofpfn-checkpoint-v1``).

    ``check`` reads the file once; the checked checkpoint stays with the system for ``build_model``.

    Attributes:
        label: The label of the result files.
        path: The checkpoint file.
    """

    label: str
    path: Path
    _checkpoint: Optional[Mapping[str, Any]] = field(default=None, init=False, repr=False)
    _sha256: Optional[str] = field(default=None, init=False, repr=False)

    def check(self) -> None:
        """Read the file and check its format.

        Raises:
            SystemCheckError: For a missing file or a file that is not a valid checkpoint.
        """
        if self._checkpoint is not None:
            return
        if not Path(self.path).is_file():
            raise SystemCheckError(f'{self.label}: missing {self.path}')
        try:
            self._checkpoint = read_checkpoint(self.path)
        except CheckpointError as error:
            raise SystemCheckError(f'{self.label}: {error}') from None
        self._sha256 = file_sha256(self.path)

    def build_model(self, num_entities: int, num_relations: int, setting: Optional[Mapping[str, Any]] = None) -> GraphKGE:
        """The frozen model on the CPU, without a graph (``check`` runs first if it has not).

        Args:
            num_entities: The number of entities of all splits. The checkpoint's model does not need it.
            num_relations: The number of relations of all splits. The checkpoint's model does not need it.
            setting: The official Flock setting of the graph. The checkpoint's model does not use it.

        Returns:
            ULTRA with the options and weights of the checkpoint (``model_from_checkpoint``).
        """
        self.check()
        assert self._checkpoint is not None
        return model_from_checkpoint(self._checkpoint)

    def query_batch_size(self, setting: Optional[Mapping[str, Any]]) -> int:
        """One query per microbatch.

        Args:
            setting: The official Flock setting of the graph, or ``None``. The checkpoint's model does not use it.

        Returns:
            The microbatch of queries: 1.
        """
        return 1

    def spec(self) -> dict[str, Any]:
        """The system's record: the file and what the checkpoint says about itself.

        Returns:
            ``kind`` (``'checkpoint'``), ``checkpoint`` (the file), ``model_config``, ``variant``, ``seed``, ``updates`` and
            ``weights`` of the checkpoint.
        """
        self.check()
        c = self._checkpoint
        assert c is not None
        return dict(kind='checkpoint', checkpoint=str(self.path), model_config=dict(c['model_config']), variant=c['variant'], seed=c['seed'],
                    updates=c['updates'], weights=c['weights'])

    def sha256(self) -> str:
        """The SHA-256 of the checkpoint file.

        Returns:
            The hex digest.
        """
        self.check()
        assert self._sha256 is not None
        return self._sha256


@dataclass(frozen=True)
class Reference:
    """A released reference checkpoint: TRIX (``trix``) or ULTRA-3g (``ultra3g``), and the base of official Flock.

    Attributes:
        label: The label, one of ``LABELS``.
        path: The checkpoint file.
        pin: The pinned SHA-256 of the file.
    """

    label: str
    path: Path
    pin: str
    LABELS: ClassVar[tuple[str, ...]] = ('trix', 'ultra3g')

    def __post_init__(self) -> None:
        if self.label not in self.LABELS:
            raise SystemCheckError(f'unknown reference {self.label!r}; choose from {self.LABELS}')

    def check(self) -> None:
        """Refuse a missing file or a file hash that differs from the pin.

        Raises:
            SystemCheckError: For a missing file or a SHA-256 that differs from the pin.
        """
        if not Path(self.path).is_file():
            raise SystemCheckError(f'{self.label}: missing {self.path}')
        if file_sha256(self.path) != self.pin:
            raise SystemCheckError(f'{self.label}: the SHA-256 of {self.path} differs from the released checkpoint')

    def build_model(self, num_entities: int, num_relations: int, setting: Optional[Mapping[str, Any]] = None) -> GraphKGE:
        """The frozen model on the CPU (query batch 1), without a graph.

        Args:
            num_entities: The number of entities of all splits.
            num_relations: The number of relations of all splits.
            setting: The official Flock setting of the graph, or ``None``. TRIX and ULTRA do not use it.

        Returns:
            TRIX for ``trix``, else ULTRA, with the weights of the checkpoint file.
        """
        with torch.random.fork_rng(devices=[]):
            if self.label == 'trix':
                model: GraphKGE = TRIX(dict(num_entities=num_entities, num_relations=num_relations, trix_query_batch_size=1))
            else:
                model = ULTRA(dict(num_entities=num_entities, num_relations=num_relations, ultra_query_batch_size=1))
        model.load_pretrained(self.path)
        return frozen(model)

    def query_batch_size(self, setting: Optional[Mapping[str, Any]]) -> int:
        """One query per microbatch.

        Args:
            setting: The official Flock setting of the graph, or ``None``. The reference does not use it.

        Returns:
            The microbatch of queries: 1.
        """
        return 1

    def spec(self) -> dict[str, Any]:
        """The reference's record.

        Returns:
            ``kind`` (``'reference'``), ``model`` (``'TRIX'`` or ``'ULTRA'``), ``checkpoint`` (the file) and
            ``query_batch_size`` (1).
        """
        return dict(kind='reference', model='TRIX' if self.label == 'trix' else 'ULTRA', checkpoint=str(self.path), query_batch_size=1)

    def sha256(self) -> str:
        """The pinned SHA-256 of the file.

        Returns:
            The hex digest.
        """
        return self.pin


@dataclass(frozen=True)
class OfficialFlock(Reference):
    """Flock at its official test-time setting (label ``flock``, checkpoint ``flock/flock_entity.pth``).

    Attributes:
        one_microbatch: Run the passes of one query as one microbatch of the walk sampler (the setting of the thesis runs).
    """

    one_microbatch: bool = True
    LABELS: ClassVar[tuple[str, ...]] = ('flock',)

    def build_model(self, num_entities: int, num_relations: int, setting: Optional[Mapping[str, Any]] = None) -> GraphKGE:
        """The frozen Flock of a graph's setting on the CPU: the official walk count and passes, seed 42.

        Args:
            num_entities: The number of entities of all splits.
            num_relations: The number of relations of all splits.
            setting: The official Flock setting of the graph (``official_setting``), with ``walks`` and ``passes``.

        Returns:
            Flock with the weights of the checkpoint file and the query batch of ``query_batch_size``, without a graph.

        Raises:
            SystemCheckError: Without the graph's setting.
        """
        if setting is None:
            raise SystemCheckError('official Flock needs the graph setting')
        with torch.random.fork_rng(devices=[]):
            model = Flock(dict(num_entities=num_entities, num_relations=num_relations, flock_query_batch_size=self.query_batch_size(setting),
                               flock_walk_num=setting['walks'], flock_test_samples=setting['passes'], flock_seed=FLOCK_SEED))
        model.load_pretrained(self.path)
        return frozen(model)

    def query_batch_size(self, setting: Optional[Mapping[str, Any]]) -> int:
        """The passes of one query with ``one_microbatch``, else 1.

        Args:
            setting: The official Flock setting of the graph (``official_setting``), or ``None``.

        Returns:
            The microbatch of query passes: ``setting['passes']`` with ``one_microbatch`` and a setting, else 1.
        """
        return int(setting['passes']) if self.one_microbatch and setting is not None else 1

    def spec(self) -> dict[str, Any]:
        """The reference's record.

        Returns:
            ``kind`` (``'reference'``), ``model`` (``'Flock'``), ``checkpoint`` (the file), ``setting`` (a fixed text),
            ``seed`` (42) and ``one_microbatch``.
        """
        return dict(kind='reference', model='Flock', checkpoint=str(self.path), setting='official walk count and passes',
                    seed=FLOCK_SEED, one_microbatch=self.one_microbatch)


System = Union[Checkpoint, Reference]


def checkpoints(spec: str, seeds: Sequence[int] = SEEDS) -> list[Checkpoint]:
    """The checkpoint systems of a ``NAME=TEMPLATE`` option.

    A template with the field ``{seed}`` gives one system per seed, labelled ``NAME.seed<S>``. A template without a
    field gives one system, labelled ``NAME``.

    Args:
        spec: The value of a ``--model`` option.
        seeds: The seeds of a template with ``{seed}``.

    Returns:
        The systems. ``check`` reads their files.

    Raises:
        SystemCheckError: For a malformed option, a template with another field than ``{seed}``, a name with a character
            other than a letter, a digit, ``-`` or ``_``, or the name of a reference.
    """
    name, separator, template = spec.partition('=')
    if not separator or not name or not template:
        raise SystemCheckError(f'--model {spec!r}: expected NAME=TEMPLATE')
    if not (name[0].isalpha() and name.isascii() and all(c.isalnum() or c in '-_' for c in name)):
        raise SystemCheckError(f'model name {name!r}: letters, digits, - and _, starting with a letter')
    if name in REFERENCES:
        raise SystemCheckError(f'{name} is the label of a reference')
    try:
        fields = {field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name is not None}
    except ValueError as error:   # a single brace
        raise SystemCheckError(f'{name}: the checkpoint template {template!r}: {error}') from None
    if not fields:
        return [Checkpoint(name, Path(template))]
    if fields != {'seed'}:
        raise SystemCheckError(f'{name}: the checkpoint template may hold {{seed}} and no other field')
    return [Checkpoint(f'{name}.seed{seed}', Path(template.format(seed=seed))) for seed in seeds]


def reference(label: str, root: Union[str, Path] = CHECKPOINTS, one_microbatch: bool = True) -> Reference:
    """A reference system by label (``trix``, ``ultra3g``, ``flock``) with its released checkpoint below ``root``.

    Args:
        label: The label.
        root: The checkpoint root (default ``<repository>/checkpoints``).
        one_microbatch: For ``flock``: run the passes of one query as one microbatch of the walk sampler.

    Returns:
        An ``OfficialFlock`` for ``flock``, else a ``Reference``, with the file below ``root`` and its pinned SHA-256.

    Raises:
        SystemCheckError: For an unknown label.
    """
    if label not in REFERENCES:
        raise SystemCheckError(f'unknown reference {label!r}; choose from {tuple(REFERENCES)}')
    name, pin = REFERENCES[label]
    path = Path(root) / name
    if label == 'flock':
        return OfficialFlock(label=label, path=path, pin=pin, one_microbatch=one_microbatch)
    return Reference(label=label, path=path, pin=pin)
