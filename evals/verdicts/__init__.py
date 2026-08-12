"""Harness-side verdict modules (thin-dispatcher refactor).

A verdict module judges one finished gate run purely from on-disk artifacts:
it reads the ref side (``<run_dir>/ref/``) and the ours side
(``<run_dir>/ours/``) plus the gate's rendered products (thresholds /
shape), and returns the standard run-result dict. It never launches
subprocesses — execution belongs to the side scripts, judgment belongs here.

Contract every module implements::

    def run(*, suite_key, run_dir, repo_root, workload_config) -> dict

Routed by the ``verdict = "<module>"`` key in the suite registry
(``[evals.<gate>]``), imported as ``evals.verdicts.<module>`` by the
dispatcher's generic executor.
"""
