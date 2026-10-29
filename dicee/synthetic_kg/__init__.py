"""ProofPFN: pretraining ULTRA on synthetic knowledge graphs drawn from a fixed prior.

Modules:
    recipe: fixed values of the prior (the resolved research recipe, stream layout, family parameters).
    streams: the seeded SHA-256 streams with their research version strings.
    batch: the world batch of an update, its checks and its content digest.
    digests: the digest rules of the golden references.
    priors: the prior configuration, the world order of a run and the world of an order slot.
    arms: the seven arm presets and their initial weights.
    checkpoint: the checkpoint format, its loader and the converter of research exports.
    determinism: deterministic execution for pretraining and evaluation.

The package imports no submodule here, so that ``import dicee.synthetic_kg.<module>`` loads only what it needs.
"""
