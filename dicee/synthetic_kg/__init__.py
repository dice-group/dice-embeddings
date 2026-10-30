"""ProofPFN: pretraining ULTRA on synthetic knowledge graphs drawn from a fixed prior.

Modules:
    recipe: fixed values of the prior (the resolved research recipe, stream layout, family parameters).
    streams: the seeded SHA-256 streams with their research version strings.
    batch: the world batch of an update, its checks and its content digest.
    digests: the digest rules of the golden references.
    priors: the prior configuration, the world order of a run and the world of an order slot.
    arms: the seven arm presets and their initial weights.
    prepare: the worker preparation of a world for an update (graph, relation graph, targets, context masks).
    pretraining: the pretraining loop, its update, exports, training states and resume.
    checkpoint: the checkpoint format, its loader and the converter of research exports.
    determinism: deterministic execution for pretraining and evaluation.

The package imports no submodule here. A submodule import still imports the package ``dicee`` first, with all its
models and trainers: on the reference machine a spawned pretraining worker spends about 2 s and 0.6 GB on it before its
first world.
"""
