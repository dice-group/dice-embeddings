"""The ProofPFN benchmark: zero-shot link prediction on frozen graph sets (``PROTOCOL.md``).

Modules: ``manifests`` (graph sets, without PyTorch), ``data`` (loading and query structure), ``systems`` (checkpoints
of this package and the released reference models), ``evaluate`` (one graph x one system -> one result file),
``prior`` (checkpoints on held-out worlds of the synthetic prior) and ``cli``.
Run ``python -m benchmarks.proofpfn --help``.
"""
