"""argus_ml.profiling ÔÇö Lightweight profiling decorators for hardware graph pipelines.

Provides two decorators:
- ``profile_function``: Measures cumulative wall-clock time across repeated calls.
- ``profile_graph_extraction``: Profiles a graph-extraction function and logs
  per-graph statistics (name, node count, edge count, time).

Both decorators write results to text files named after the wrapped function.
"""

from __future__ import annotations

import cProfile
import functools
import io
import os
import pstats
import sys
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def profile_function(func: F) -> F:
    """Decorator that profiles *func* and appends cumulative timing to ``{func.__name__}.txt``.

    Each call appends the incremental time to a running total. The file format is::

        <call_count> <cumulative_seconds>

    Example::

        @profile_function
        def train_epoch(model, data):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_file = f"{func.__name__}.txt"
        last_time = 0.0
        count = 0

        if os.path.exists(data_file):
            with open(data_file) as fh:
                parts = fh.readline().split()
                if len(parts) >= 2:
                    count = int(parts[0])
                    last_time = float(parts[1])

        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
        stats.print_stats()

        # Extract total time from the stats header line
        header = stream.getvalue().split("\n")[0]
        tokens = header.split()
        elapsed = float(tokens[-2]) if len(tokens) >= 2 else 0.0

        with open(data_file, "w") as fh:
            fh.write(f"{count + 1} {last_time + elapsed}")

        return result

    wrapper.unwrapped = func  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def profile_graph_extraction(func: F) -> F:
    """Decorator that profiles a graph-extraction function and logs per-graph stats.

    The wrapped function **must** return a ``networkx.Graph`` (or ``DiGraph``)
    with ``.name``, ``.nodes()``, and ``.edges()`` attributes.

    Results are appended to ``{func.__name__}.txt`` in the format::

        <graph_name> <node_count> <edge_count> <seconds>

    Example::

        @profile_graph_extraction
        def extract_dfg(verilog_path: str) -> nx.DiGraph:
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_file = f"{func.__name__}.txt"

        # Suppress stdout during extraction to avoid pyverilog noise
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            graph = func(*args, **kwargs)
        finally:
            profiler.disable()
            sys.stdout.close()
            sys.stdout = original_stdout

        hw_name = getattr(graph, "name", "unnamed")
        node_count = len(graph.nodes())
        edge_count = len(graph.edges())

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
        stats.print_stats()

        header = stream.getvalue().split("\n")[0]
        tokens = header.split()
        elapsed = float(tokens[-2]) if len(tokens) >= 2 else 0.0

        with open(data_file, "a") as fh:
            fh.write(f"{hw_name} {node_count} {edge_count} {elapsed}\n")

        return graph

    wrapper.unwrapped = func  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
