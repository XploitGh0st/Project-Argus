"""Run pyverilog's analysis passes directly on an AST object.

The legacy code re-parsed the Verilog *file* inside ``VerilogDataflowAnalyzer``.
Here the passes (module -> signal -> bind visitors, constant resolution) run on an
already-built AST, which is what lets this module work from a stub AST.

NOTE: this block is intentionally duplicated in ``argus_dfg``. Modules share
contracts, never code.
"""

from __future__ import annotations

from argus_ast.patch_pyverilog import apply_pyverilog_patches
apply_pyverilog_patches()

import contextlib
import io
import sys
from dataclasses import dataclass
from typing import Any

from pyverilog.dataflow.bindvisitor import BindVisitor
from pyverilog.dataflow.modulevisitor import ModuleVisitor
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.signalvisitor import SignalVisitor

__all__ = ["Analysis", "analyze_ast"]


@dataclass(frozen=True)
class Analysis:
    """Everything pyverilog's analysers need, derived from one AST."""

    terms: Any
    binddict: Any
    resolved_terms: Any
    resolved_binddict: Any
    constlist: Any


def analyze_ast(ast: Any, top_module: str, *, quiet: bool = True) -> Analysis:
    """Bind signals and resolve constants for ``top_module``.

    Args:
        ast: a pyverilog ``Source`` node.
        top_module: name of the top module inside ``ast``.
        quiet: swallow pyverilog's chatty stdout.
    """
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 16 * 1024))
    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        module_visitor = ModuleVisitor()
        module_visitor.visit(ast)
        info = module_visitor.get_moduleinfotable()

        signal_visitor = SignalVisitor(info, top_module)
        signal_visitor.start_visit()

        frametable = signal_visitor.getFrameTable()
        bind_visitor = BindVisitor(info, top_module, frametable, noreorder=False)
        bind_visitor.start_visit()
        dataflow = bind_visitor.getDataflows()
        terms, binddict = dataflow.getTerms(), dataflow.getBinddict()

        optimizer = VerilogDataflowOptimizer(terms, binddict)
        optimizer.resolveConstant()
        return Analysis(
            terms=terms,
            binddict=binddict,
            resolved_terms=optimizer.getResolvedTerms(),
            resolved_binddict=optimizer.getResolvedBinddict(),
            constlist=optimizer.getConstlist(),
        )
