"""The ProofPFN benchmark: zero-shot link prediction on frozen graph sets, by the research protocol (``PROTOCOL.md``).

Modules: ``manifests`` (graph sets and model pins, without PyTorch), ``data`` (loading and query structure),
``systems`` (ProofPFN families and the native references), ``evaluate`` (one graph x systems -> one row file),
``cells`` (the query cells of a graph set), ``gate`` (the N validity gate on held-out worlds of the prior) and ``cli``.
Run ``python -m benchmarks.proofpfn --help``.
"""
