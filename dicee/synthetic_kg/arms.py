"""The seven ported ProofPFN arms as presets, and their initial weights.

An arm is a prior, two ULTRA options and a training target. Every arm trains ULTRA (dimension 64, 6 relation and 6
entity layers) from the initial weights of the stream ``capacity-prior-v1:<seed>:initial:d64``:

====  =======================================================  ==============  =============  ======
Arm   Prior                                                    Weighting       Pair channels  Target
====  =======================================================  ==============  =============  ======
S     the synthetic mixture (plain, structured, statistical,  normalized      on             rbw
      categorical2, rec1, attr2, list1), order ``REC10A-XLE``
R     real subgraph worlds only, order ``RO2``                 normalized      on             rbw
M     S's families at half their shares and real at 0.5,       normalized      on             rbw
      order ``M2-REC10A-XLE``
N     S's, the four record families with the structural null   normalized      on             rbw
HT    S's                                                      normalized      on             hard
CU    S's                                                      normalized      off            rbw
UL    S's                                                      binary          off            rbw
====  =======================================================  ==============  =============  ======

The mixture strings are the research ``--mix`` values, verbatim. S, N, HT, CU and UL have the same world order per
seed; R's order is the same for every seed. The presets pin the research order digests of seeds 42, 43 and 44.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import torch

from ..models.ultra import ULTRA
from .priors import FamilySpec, PriorConfig
from .streams import initial_seed

SEEDS = (42, 43, 44)
"""The seeds of the research runs."""

ULTRA_DIM = 64
ULTRA_LAYERS = 6


@dataclass(frozen=True)
class ArmPreset:
    """One arm: what it trains on, which model options it uses and what its target is.

    Attributes:
        name: The arm label (``'S'``, ``'R'``, ``'M'``, ``'N'``, ``'HT'``, ``'CU'`` or ``'UL'``).
        research_arm: The ``--arm`` name of the research runs (the ``arm`` field of their run records).
        prior: The prior.
        relation_weighting: ULTRA's ``ultra_relation_weighting``.
        pair_channels: ULTRA's ``ultra_pair_channels``.
        target: ``'rbw'`` (uniform over the true answers outside the context) or ``'hard'`` (one answer drawn per query).
        order_sha256: The research digests of the 24,000-slot order, as ``(seed, digest)`` pairs.
    """

    name: str
    research_arm: str
    prior: PriorConfig
    relation_weighting: str
    pair_channels: str
    target: str
    order_sha256: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if self.target not in ('rbw', 'hard'):
            raise ValueError(f'unknown target {self.target!r}')

    def model_config(self) -> dict[str, Any]:
        """The ULTRA options of the arm (the ``model_config`` of its checkpoints)."""
        return dict(ultra_dim=ULTRA_DIM, ultra_num_layers=ULTRA_LAYERS, ultra_relation_weighting=self.relation_weighting,
                    ultra_pair_channels=self.pair_channels)

    def pinned_order(self, seed: int) -> Optional[str]:
        """The research digest of the arm's order for ``seed``, or ``None`` for an unpinned seed."""
        return dict(self.order_sha256).get(seed)


def _records(null_swap_budget: Optional[int] = None) -> tuple[FamilySpec, ...]:
    """S's record families (categorical2 starts at the research default first index 200,000)."""
    return (FamilySpec('attr2', 'attr2', 7_000_000, null_swap_budget=null_swap_budget),
            FamilySpec('categorical2', 'categorical2', 200_000, null_swap_budget=null_swap_budget),
            FamilySpec('list1', 'list1', 11_000_000, null_swap_budget=null_swap_budget),
            FamilySpec('rec1', 'rec1', 8_100_000, (('max_entities', 30_000), ('rows', (2500, 15_000))), null_swap_budget))


SYNTHETIC_MIX = 'plain=0.2925,structured=0.07583337,statistical=0.08166663,rec1=0.09,categorical2=0.27,attr2=0.09,list1=0.1'
REAL_MIX = 'real=1.0'
MIXED_MIX = 'plain=0.14625,structured=0.0379167,statistical=0.0408333,rec1=0.045,categorical2=0.135,attr2=0.045,list1=0.05,real=0.5'
REAL = FamilySpec('real', 'real', 400_000)

SYNTHETIC = PriorConfig(SYNTHETIC_MIX, 'REC10A-XLE', _records())
SYNTHETIC_NULL = PriorConfig(SYNTHETIC_MIX, 'REC10A-XLE', _records(null_swap_budget=3))
REAL_ONLY = PriorConfig(REAL_MIX, 'RO2', (REAL,))
MIXED = PriorConfig(MIXED_MIX, 'M2-REC10A-XLE', (*_records(), REAL))

SYNTHETIC_ORDERS = ((42, '80e25cf47082065814665a395ce55808c66b504bf28fb4108b2ab42c662d61bf'),
                    (43, '0ecf3c7712d11fb7acea6a6a645ec171a51cdb98991122829a611f2553fa4c02'),
                    (44, 'da4b9878d359d8455a47c8189f2bd100c68a97c49e9ba49829ad0f3bd25e8346'))
REAL_ORDERS = tuple((seed, 'd0638d30f0a5e611552932488a778c5e173dd1e686313a62cf75f921d0a65f4a') for seed in SEEDS)
MIXED_ORDERS = ((42, 'ec176dac00d953c111c02846b338cd54688613ab9e046e63c787550ee937b81b'),
                (43, 'b86f64285eaa0ece1f0bf902250b5487874c43a86f0e824427bc3a21a5c14303'),
                (44, 'b7615d8385c095d28d880eeec41df2a6b3b95c7061d78ef46ec2589027a35ae3'))

ARMS: Mapping[str, ArmPreset] = MappingProxyType({preset.name: preset for preset in (
    ArmPreset('S', 'REC10A-XLE-PCU-noA', SYNTHETIC, 'normalized', 'on', 'rbw', SYNTHETIC_ORDERS),
    ArmPreset('R', 'PCU-R', REAL_ONLY, 'normalized', 'on', 'rbw', REAL_ORDERS),
    ArmPreset('M', 'PCU-M', MIXED, 'normalized', 'on', 'rbw', MIXED_ORDERS),
    ArmPreset('N', 'PCU-N', SYNTHETIC_NULL, 'normalized', 'on', 'rbw', SYNTHETIC_ORDERS),
    ArmPreset('HT', 'PCU-HT', SYNTHETIC, 'normalized', 'on', 'hard', SYNTHETIC_ORDERS),
    ArmPreset('CU', 'REC10A-XLE-CU-noA', SYNTHETIC, 'normalized', 'off', 'rbw', SYNTHETIC_ORDERS),
    ArmPreset('UL', 'REC10A-XLE-UL-noA', SYNTHETIC, 'binary', 'off', 'rbw', SYNTHETIC_ORDERS),
)})
"""The seven arms by name."""


def arm_preset(arm: Union[str, ArmPreset]) -> ArmPreset:
    """The preset of an arm name (a preset passes through).

    Raises:
        KeyError: For an unknown arm name.
    """
    if isinstance(arm, ArmPreset):
        return arm
    if arm not in ARMS:
        raise KeyError(f'unknown arm {arm!r}; expected one of {sorted(ARMS)}')
    return ARMS[arm]


def initial_model(arm: Union[str, ArmPreset], seed: int) -> ULTRA:
    """The arm's ULTRA with the research initial weights of ``seed``, on the CPU, in training mode.

    The CPU generator is seeded with ``streams.initial_seed(seed)`` and ULTRA is built in its fixed draw order (with
    pair channels, the six 6-row relation tables after all of ULTRA's tensors). The global random state is restored
    afterwards.

    Args:
        arm: The arm or its name.
        seed: The run seed.

    Returns:
        The initialized model without a graph.
    """
    preset = arm_preset(arm)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(initial_seed(seed))
        model = ULTRA(preset.model_config())
    model.train()
    return model


# Research hook of each generator (module, function); the structural null ran as s5_null:<generator>.
_RESEARCH_HOOKS = {'categorical2': ('pf_prior2f', 'sample'), 'rec1': ('pf_rec1', 'sample'), 'attr2': ('pf_attr2', 'sample'),
                   'list1': ('pf_list1', 'sample'), 'real': ('pt_real', 'sample')}


def research_hook(spec: FamilySpec) -> dict[str, Any]:
    """The research hook record of a family (``pt_common.parse_families``; tuples as JSON lists).

    Args:
        spec: The family.

    Returns:
        ``{'module', 'function', 'params', 'first'}`` as the research run records store it.
    """
    module, function = _RESEARCH_HOOKS[spec.generator]
    params = {k: list(v) if isinstance(v, tuple) else v for k, v in spec.params}
    if spec.null_swap_budget is not None:
        module, function = 's5_null', spec.generator
        params['swap_budget'] = spec.null_swap_budget
    return dict(module=module, function=function, params=params, first=spec.first)


def research_record(arm: Union[str, ArmPreset]) -> dict[str, Any]:
    """The arm's options as the research trainer recorded them (``pt_common.Plan.record``).

    The run record of a research export contains these entries; the checkpoint converter checks them.

    Args:
        arm: The arm or its name.

    Returns:
        The record: mixture shares, order name, hooks, the frozen adapters and the options that differ from the
        research defaults (target, backbone, pair channels).
    """
    preset = arm_preset(arm)
    record: dict[str, Any] = dict(mix=preset.prior.shares, order_name=preset.prior.order_name, order_file=None, adapter='none',
                                  hooks={spec.name: research_hook(spec) for spec in sorted(preset.prior.families, key=lambda s: s.name)},
                                  query_weights=None)
    if preset.target != 'rbw':
        record['target'] = preset.target
    if preset.relation_weighting == 'binary':   # plain ULTRA: the research backbone option, without adapters
        record['backbone'] = 'ultra'
    else:
        record['context_adapter'] = 'none'
        if preset.pair_channels == 'off':
            record['pair_channels'] = 'off'
    return record
