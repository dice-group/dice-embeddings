"""The eleven model variants of ProofPFN as presets, and their initial weights.

A variant is a prior, two ULTRA options and a training target. Its name is the model and a suffix for the training
worlds: S (synthetic), R (real), M (mixed), N (synthetic with the structural null) and HT (synthetic with hard
targets). Every variant trains ULTRA (dimension 64, 6 relation and 6 entity layers) from the initial weights of the
stream ``capacity-prior-v1:<seed>:initial:d64``:

============  ===================================================  ==========  =============  ======
Variant       Prior                                                Weighting   Pair channels  Target
============  ===================================================  ==========  =============  ======
PCUltra-S     the synthetic mixture of seven world families,       normalized  on             rbw
              order ``REC10A-XLE``
PCUltra-R     real subgraph worlds only, order ``RO2``             normalized  on             rbw
PCUltra-M     the synthetic families at half their shares and      normalized  on             rbw
              real worlds at 0.5, order ``M2-REC10A-XLE``
PCUltra-N     synthetic, the four record families with the         normalized  on             rbw
              structural null
PCUltra-HT    synthetic                                            normalized  on             hard
CountULTRA-S  synthetic                                            normalized  off            rbw
CountULTRA-R  real                                                 normalized  off            rbw
CountULTRA-M  mixed                                                normalized  off            rbw
ULTRA-S       synthetic                                            binary      off            rbw
ULTRA-R       real                                                 binary      off            rbw
ULTRA-M       mixed                                                binary      off            rbw
============  ===================================================  ==========  =============  ======

The world families have the code names ``plain``, ``structured``, ``statistical``, ``categorical2``, ``attr2``, ``rec1``
and ``list1`` (``priors.FAMILY_TITLES`` gives their titles). The mixture strings, the order names and the code names
fix the worlds and their order, so they stay as the thesis runs had them. The variants of one prior have the same
world order per seed; the order of the real prior is the same for every seed. The presets pin the order digests of
seeds 42, 43 and 44.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import torch

from ..models.ultra import ULTRA
from .priors import FamilySpec, PriorConfig
from .streams import initial_seed

SEEDS = (42, 43, 44)
"""The seeds of the thesis runs."""

ULTRA_DIM = 64
ULTRA_LAYERS = 6


@dataclass(frozen=True)
class Variant:
    """One model variant: what it trains on, which model options it uses and what its target is.

    Attributes:
        name: The variant name (a key of ``VARIANTS``, for example ``'PCUltra-S'``).
        prior: The prior.
        relation_weighting: ULTRA's ``ultra_relation_weighting``.
        pair_channels: ULTRA's ``ultra_pair_channels``.
        target: ``'rbw'`` (uniform over the true answers outside the context) or ``'hard'`` (one answer drawn per query).
        order_sha256: The digests of the 24,000-slot world order, as ``(seed, digest)`` pairs.
    """

    name: str
    prior: PriorConfig
    relation_weighting: str
    pair_channels: str
    target: str
    order_sha256: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if self.target not in ('rbw', 'hard'):
            raise ValueError(f'unknown target {self.target!r}')

    def model_config(self) -> dict[str, Any]:
        """The ULTRA options of the variant.

        Returns:
            The ``model_config`` of the variant's checkpoints.
        """
        return dict(ultra_dim=ULTRA_DIM, ultra_num_layers=ULTRA_LAYERS, ultra_relation_weighting=self.relation_weighting,
                    ultra_pair_channels=self.pair_channels)

    def pinned_order(self, seed: int) -> Optional[str]:
        """The pinned digest of the variant's world order.

        Args:
            seed: The run seed.

        Returns:
            The digest, or ``None`` for an unpinned seed.
        """
        return dict(self.order_sha256).get(seed)


def _records(null_swap_budget: Optional[int] = None) -> tuple[FamilySpec, ...]:
    """The record families of the synthetic prior (categorical2 starts at its default first index 200,000)."""
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

VARIANTS: Mapping[str, Variant] = MappingProxyType({preset.name: preset for preset in (
    Variant('PCUltra-S', SYNTHETIC, 'normalized', 'on', 'rbw', SYNTHETIC_ORDERS),
    Variant('PCUltra-R', REAL_ONLY, 'normalized', 'on', 'rbw', REAL_ORDERS),
    Variant('PCUltra-M', MIXED, 'normalized', 'on', 'rbw', MIXED_ORDERS),
    Variant('PCUltra-N', SYNTHETIC_NULL, 'normalized', 'on', 'rbw', SYNTHETIC_ORDERS),
    Variant('PCUltra-HT', SYNTHETIC, 'normalized', 'on', 'hard', SYNTHETIC_ORDERS),
    Variant('CountULTRA-S', SYNTHETIC, 'normalized', 'off', 'rbw', SYNTHETIC_ORDERS),
    Variant('CountULTRA-R', REAL_ONLY, 'normalized', 'off', 'rbw', REAL_ORDERS),
    Variant('CountULTRA-M', MIXED, 'normalized', 'off', 'rbw', MIXED_ORDERS),
    Variant('ULTRA-S', SYNTHETIC, 'binary', 'off', 'rbw', SYNTHETIC_ORDERS),
    Variant('ULTRA-R', REAL_ONLY, 'binary', 'off', 'rbw', REAL_ORDERS),
    Variant('ULTRA-M', MIXED, 'binary', 'off', 'rbw', MIXED_ORDERS),
)})
"""The eleven variants by name."""


def variant_of(variant: Union[str, Variant]) -> Variant:
    """The preset of a variant.

    Args:
        variant: The variant name; a preset passes through.

    Returns:
        The preset.

    Raises:
        KeyError: For an unknown variant name.
    """
    if isinstance(variant, Variant):
        return variant
    if variant not in VARIANTS:
        raise KeyError(f'unknown variant {variant!r}; expected one of {sorted(VARIANTS)}')
    return VARIANTS[variant]


def initial_model(variant: Union[str, Variant], seed: int) -> ULTRA:
    """The variant's ULTRA with the initial weights of ``seed``, on the CPU, in training mode.

    The CPU generator is seeded with ``streams.initial_seed(seed)`` and ULTRA is built in its fixed draw order (with
    pair channels, the six 6-row relation tables after all of ULTRA's tensors). The global random state is restored
    afterwards.

    Args:
        variant: The variant or its name.
        seed: The run seed.

    Returns:
        The initialized model without a graph.
    """
    preset = variant_of(variant)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(initial_seed(seed))
        model = ULTRA(preset.model_config())
    model.train()
    return model
