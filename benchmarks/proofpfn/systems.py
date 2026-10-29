"""The systems of the ProofPFN benchmark: ProofPFN families and the native reference models.

- ``ProofPFNFamily``: the three seeds (42, 43, 44) of one arm, each a clean checkpoint (``proofpfn-checkpoint-v1``) or a
  research export, which ``dicee.synthetic_kg.checkpoint.from_research_export`` converts (EMA weights) after its
  inertness checks. Under a key registered in ``models.json``, every file must match the arm's research export pins:
  a research export by its SHA-256, a clean checkpoint by the ``research_export_sha256`` of its provenance. The row
  file records the research export's SHA-256 as the model hash when there is one (so rows of both formats compare
  with the registered pins), else the clean file's.
- ``Reference``: TRIX (``trix``) or ULTRA-3g (``ultra3g``), the released checkpoints.
- ``OfficialFlock`` (``flockoff``): the released Flock checkpoint at the graph's official walk count with the official
  passes and sampling seed 42 (``flock_setting``). With ``one_microbatch`` (the research setting) the passes of one
  query run as one microbatch of the walk sampler; without it they run one after another (the same law, other walks).

Every model is built on the CPU with the global random state restored, in evaluation mode without gradients. The
evaluation attaches the inference graph with ``attach``.
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, TypeVar, Union

import torch

from dicee.models.flock import Flock
from dicee.models.graph_model import GraphKGE
from dicee.models.trix import TRIX
from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.arms import ARMS, SEEDS
from dicee.synthetic_kg.checkpoint import FORMAT, CheckpointError, check_checkpoint, from_research_export

from .data import GraphData, inference_stats
from .manifests import REPO, GraphEntry, ModelPins, file_sha256

logger = logging.getLogger(__name__)

CHECKPOINTS = REPO / 'checkpoints'
FLOCK_SEED = 42
REFERENCE_KEYS = ('trix', 'ultra3g', 'flockoff')
Label = tuple[str, Optional[int]]
Model = TypeVar('Model', bound=GraphKGE)


class SystemCheckError(ValueError):
    """A system that cannot run: a missing file, a hash that differs from its pins, or a key of another system."""


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
    """The passes of the rule for a graph outside the official script."""
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


def official_setting(entry: GraphEntry, graph: GraphData, keys: Sequence[str]) -> dict[str, Any]:
    """The official Flock setting of a graph for the given keys, after checking train.txt against the manifest's size.

    Raises:
        SystemCheckError: When the inference graph's statistics differ from the manifest (the walk count would not follow
            the stated rule).
    """
    setting = flock_setting(entry.size['entities'], entry.size['inference_edges'], entry.suite_id)
    found = inference_stats(graph.train)
    if found != (setting['entities'], setting['edges']):
        raise SystemCheckError(f'{entry.key}: train.txt has {found[0]} entities and {found[1]} distinct triples, the manifest '
                           f"{setting['entities']} and {setting['edges']}: no official Flock setting")
    return dict(graph=entry.key, rule=FLOCK_RULE, **setting, entities_all=graph.num_entities, relations_all=graph.num_relations,
                keys={k: dict(walks=setting['walks'], passes=setting['passes']) for k in keys})


# ------------------------------------------------------------------------------------------------------ systems

def frozen(model: Model) -> Model:
    """The model in evaluation mode without gradients."""
    model.eval().requires_grad_(False)
    return model


def attach(model: Model, train: torch.Tensor, num_entities: int, num_relations: int, device: Union[str, torch.device],
           query_batch_size: int = 1) -> Model:
    """Attach the inference graph on the CPU, then move the model to the device (the research order).

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


def ultra_from_checkpoint(checkpoint: Mapping[str, Any]) -> ULTRA:
    """ULTRA with a checked checkpoint's options and weights (as ``ULTRA.load_pretrained``), frozen, on the CPU."""
    check_checkpoint(checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = ULTRA(dict(checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    return frozen(model)


@dataclass(frozen=True)
class Member:
    """One loaded seed of a family: the checkpoint and the hashes of its file."""

    checkpoint: Mapping[str, Any]
    file_sha256: str
    research_export_sha256: Optional[str]

    @property
    def model_sha256(self) -> str:
        """The model hash of the row file: the research export's where there is one, else the clean file's."""
        return self.research_export_sha256 or self.file_sha256


def read_member(path: Union[str, Path], arm: str) -> Member:
    """Read a clean checkpoint or convert a research export of an arm (EMA weights).

    Raises:
        CheckpointError: When the file is neither a valid checkpoint of the arm nor an inert research export of it.
    """
    stored = torch.load(path, map_location='cpu', weights_only=True)
    digest = file_sha256(path)
    if isinstance(stored, dict) and stored.get('format') == FORMAT:
        check_checkpoint(stored)
        if stored['arm'] != arm:
            raise CheckpointError(f'{Path(path).name} is a checkpoint of arm {stored["arm"]}, not {arm}')
        provenance = stored['provenance']
        return Member(stored, digest, provenance.get('research_export_sha256') if provenance.get('source') == 'research-export' else None)
    checkpoint = from_research_export(path, arm, 'ema')
    return Member(checkpoint, digest, digest)


@dataclass
class ProofPFNFamily:
    """The three seeds of one arm.

    ``check`` reads the three files once; the checked checkpoints stay with the family for ``build``.

    Attributes:
        key: The system key of the row file (``<key>``, ``<key>.seed<S>``).
        arm: The arm (``S``, ``R``, ``M``, ``N``, ``HT``, ``CU``, ``UL``).
        checkpoints: Seed to the checkpoint or research export file.
        pins: Seed to the research export SHA-256 that each file must match, or ``None`` for an unregistered key.
    """

    key: str
    arm: str
    checkpoints: Mapping[int, Path]
    pins: Optional[Mapping[int, str]] = None
    members: dict[int, Member] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise SystemCheckError(f'{self.key}: unknown arm {self.arm!r}')
        if tuple(sorted(self.checkpoints)) != SEEDS:
            raise SystemCheckError(f'{self.key}: a family needs the seeds {SEEDS}')

    @property
    def labels(self) -> list[Label]:
        """The models of the family, as (key, seed)."""
        return [(self.key, seed) for seed in SEEDS]

    def check(self) -> None:
        """Read every seed (format, arm, inertness of a research export) and compare it with the pins.

        Raises:
            SystemCheckError: For a missing file or a hash that differs from the pins.
        """
        for seed in SEEDS:
            path = Path(self.checkpoints[seed])
            if not path.is_file():
                raise SystemCheckError(f'{self.key}: missing {path}')
            member = read_member(path, self.arm)
            if self.pins is not None and member.research_export_sha256 != self.pins[seed]:
                raise SystemCheckError(f'{self.key} seed {seed}: {path.name} is not the pinned research export of arm {self.arm}')
            self.members[seed] = member

    def build(self) -> dict[Label, GraphKGE]:
        """The three frozen models on the CPU (``check`` runs first if it has not)."""
        if set(self.members) != set(SEEDS):
            self.check()
        return {(self.key, seed): ultra_from_checkpoint(self.members[seed].checkpoint) for seed in SEEDS}

    def query_batch_size(self, setting: Optional[Mapping[str, Any]]) -> int:
        """One query per microbatch."""
        return 1

    def spec(self) -> dict[str, Any]:
        """The family's record: arm, model options, checkpoint files and pin state."""
        return dict(kind='proofpfn', arm=self.arm, model_config=ARMS[self.arm].model_config(), weights='ema',
                    checkpoints={str(s): str(self.checkpoints[s]) for s in SEEDS}, pinned=self.pins is not None)

    def model_sha256(self) -> dict[str, str]:
        """``{seed: model hash}`` of the row file's ``allgraph.model_sha256``."""
        return {str(s): self.members[s].model_sha256 for s in SEEDS}

    def provenance(self) -> dict[str, Any]:
        """Per seed: the file's SHA-256 and format, and the checkpoint's own provenance."""
        return {str(s): dict(file_sha256=m.file_sha256, research_export_sha256=m.research_export_sha256, updates=m.checkpoint['updates'],
                             order_sha256=m.checkpoint['order_sha256'], provenance=m.checkpoint['provenance'])
                for s, m in sorted(self.members.items())}


@dataclass(frozen=True)
class Reference:
    """A released reference checkpoint: TRIX (``trix``) or ULTRA-3g (``ultra3g``).

    Attributes:
        key: The system key.
        path: The checkpoint file.
        sha256: The pinned SHA-256 of the file.
    """

    key: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.key not in ('trix', 'ultra3g'):
            raise SystemCheckError(f'unknown reference {self.key!r}')

    @property
    def labels(self) -> list[Label]:
        """The one model of the reference."""
        return [(self.key, None)]

    def check(self) -> None:
        """Refuse a missing file or a file hash that differs from the pin."""
        if not Path(self.path).is_file():
            raise SystemCheckError(f'{self.key}: missing {self.path}')
        if file_sha256(self.path) != self.sha256:
            raise SystemCheckError(f'{self.key}: the SHA-256 of {self.path} differs from models.json')

    def build_model(self, num_entities: int, num_relations: int) -> GraphKGE:
        """The frozen model on the CPU (query batch 1)."""
        with torch.random.fork_rng(devices=[]):
            if self.key == 'trix':
                model: GraphKGE = TRIX(dict(num_entities=num_entities, num_relations=num_relations, trix_query_batch_size=1))
            else:
                model = ULTRA(dict(num_entities=num_entities, num_relations=num_relations, ultra_query_batch_size=1))
        model.load_pretrained(self.path)
        return frozen(model)

    def query_batch_size(self, setting: Optional[Mapping[str, Any]]) -> int:
        """One query per microbatch."""
        return 1

    def spec(self) -> dict[str, Any]:
        """The reference's record."""
        return dict(kind='reference', model='TRIX' if self.key == 'trix' else 'ULTRA', checkpoint=str(self.path), query_batch_size=1)

    def model_sha256(self) -> dict[str, str]:
        """``{"model": hash}``."""
        return {'model': self.sha256}


@dataclass(frozen=True)
class OfficialFlock:
    """Flock at its official test-time setting (key ``flockoff``).

    Attributes:
        path: The checkpoint file (``flock/flock_entity.pth``).
        sha256: The pinned SHA-256 of the file.
        one_microbatch: Run the passes of one query as one microbatch of the walk sampler (the research setting).
        key: The system key.
    """

    path: Path
    sha256: str
    one_microbatch: bool = True
    key: str = 'flockoff'

    @property
    def labels(self) -> list[Label]:
        """The one model of the reference."""
        return [(self.key, None)]

    def check(self) -> None:
        """Refuse a missing file or a file hash that differs from the pin."""
        if not Path(self.path).is_file():
            raise SystemCheckError(f'{self.key}: missing {self.path}')
        if file_sha256(self.path) != self.sha256:
            raise SystemCheckError(f'{self.key}: the SHA-256 of {self.path} differs from models.json')

    def build_model(self, num_entities: int, num_relations: int, setting: Mapping[str, Any]) -> GraphKGE:
        """The frozen Flock of a graph's setting on the CPU: the official walk count and passes, seed 42."""
        walks, passes = setting['walks'], setting['passes']
        with torch.random.fork_rng(devices=[]):
            model = Flock(dict(num_entities=num_entities, num_relations=num_relations, flock_query_batch_size=self.query_batch_size(setting),
                               flock_walk_num=walks, flock_test_samples=passes, flock_seed=FLOCK_SEED))
        model.load_pretrained(self.path)
        return frozen(model)

    def query_batch_size(self, setting: Optional[Mapping[str, Any]]) -> int:
        """The passes of one query with ``one_microbatch``, else 1."""
        return int(setting['passes']) if self.one_microbatch and setting is not None else 1

    def spec(self) -> dict[str, Any]:
        """The reference's record."""
        return dict(kind='reference', model='Flock', checkpoint=str(self.path), setting='official walk count and passes',
                    seed=FLOCK_SEED, one_microbatch=self.one_microbatch)

    def model_sha256(self) -> dict[str, str]:
        """``{"model": hash}``."""
        return {'model': self.sha256}


System = Union[ProofPFNFamily, Reference, OfficialFlock]


def family(key: str, arm: str, template: str, pins: ModelPins) -> ProofPFNFamily:
    """A family from a path template with ``{seed}``, with the pins of its key.

    A key registered in ``models.json`` must be registered for this arm and have pins; any other key must not be a
    registered key (a test family, run without pins).

    Raises:
        SystemCheckError: For a key of another arm or system, or a registered key whose pins are pending.
    """
    if '{seed}' not in template:
        raise SystemCheckError(f'{key}: the checkpoint template needs {{seed}}')
    registered = pins.family_of_key(key)
    if key in pins.references or key in REFERENCE_KEYS:
        raise SystemCheckError(f'{key} is a reference key')
    if registered is not None:
        if registered.arm != arm or not registered.loadable:
            raise SystemCheckError(f'{key} is the registered key of arm {registered.arm}, not of {arm}')
        if registered.export_sha256 is None:
            raise SystemCheckError(f'{key}: models.json has no pins for arm {arm} yet')
    if not key.isalnum() or not key.islower():
        raise SystemCheckError(f'family key {key!r}: lower-case letters and digits only')
    return ProofPFNFamily(key=key, arm=arm, checkpoints=MappingProxyType({s: Path(template.format(seed=s)) for s in SEEDS}),
                          pins=None if registered is None else registered.export_sha256)


def reference(key: str, pins: ModelPins, root: Union[str, Path] = CHECKPOINTS, one_microbatch: bool = True) -> Union[Reference, OfficialFlock]:
    """A reference system by key (``trix``, ``ultra3g``, ``flockoff``) with its pinned checkpoint below ``root``.

    Raises:
        SystemCheckError: For an unknown key.
    """
    if key not in REFERENCE_KEYS or key not in pins.references:
        raise SystemCheckError(f'unknown reference {key!r}; choose from {REFERENCE_KEYS}')
    pin = pins.references[key]
    path = Path(root) / str(pin.path)
    if key == 'flockoff':
        return OfficialFlock(path=path, sha256=pin.sha256, one_microbatch=one_microbatch)
    return Reference(key=key, path=path, sha256=pin.sha256)


def build_models(systems: Sequence[System], graph: GraphData, setting: Optional[Mapping[str, Any]]) -> dict[Label, GraphKGE]:
    """Every model of the systems, frozen on the CPU, in system order (a family's seeds in order)."""
    models: dict[Label, GraphKGE] = {}
    for system in systems:
        if isinstance(system, ProofPFNFamily):
            models.update(system.build())
        elif isinstance(system, OfficialFlock):
            if setting is None:
                raise SystemCheckError('official Flock needs the graph setting')
            models[system.key, None] = system.build_model(graph.num_entities, graph.num_relations, setting)
        else:
            models[system.key, None] = system.build_model(graph.num_entities, graph.num_relations)
    return models
