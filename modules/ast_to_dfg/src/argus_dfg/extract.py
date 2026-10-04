"""AST -> DFG extraction.

Walks the dataflow tree pyverilog builds for every assigned signal and records it
as plain nodes/edges. No pygraphviz, no PDF/dot files, and no re-parsing.

Graph conventions (see CONTRACT.md):

* one ``signal`` node per net/register, id = hierarchical name (``top.out``);
* every operator / branch / constant / ... is its own node with a generated id;
* edges follow the DATA FLOW: operand -> operator -> assigned signal;
* branch inputs are labelled ``COND`` / ``TRUE`` / ``FALSE``; operator, concat and
  syscall inputs are labelled by operand position (``0``, ``1``, ...).
"""

from __future__ import annotations

from typing import Any

from pyverilog.dataflow.dataflow import (
    DFBranch,
    DFConcat,
    DFConstant,
    DFDelay,
    DFEvalValue,
    DFHighImpedance,
    DFOperator,
    DFPartselect,
    DFPointer,
    DFSyscall,
    DFTerminal,
    DFUndefined,
)

from argus_dfg.binding import analyze_ast
from argus_dfg.inputs import load_ast_input
from argus_dfg.result import DFGResult, Edge, Node

__all__ = ["build_dfg"]

_PREFIX = {
    "operator": "op",
    "branch": "br",
    "const": "const",
    "concat": "cat",
    "partselect": "ps",
    "pointer": "ptr",
    "delay": "dly",
    "syscall": "sys",
    "undefined": "undef",
    "highz": "hz",
    "unknown": "unk",
}


def _short(name: str) -> str:
    """Last hierarchical component: ``top.u1.out`` -> ``out``."""
    return name.rsplit(".", 1)[-1]


class _Builder:
    def __init__(self, terms: Any) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._counter = 0
        for key in sorted(terms, key=str):
            name = str(key)
            termtype = tuple(sorted(str(t) for t in getattr(terms[key], "termtype", ())))
            self.nodes[name] = Node(name, "signal", _short(name), termtype)

    # -- node / edge creation -------------------------------------------------

    def signal(self, name: str) -> str:
        if name not in self.nodes:  # referenced but not declared (e.g. hierarchical port)
            self.nodes[name] = Node(name, "signal", _short(name), ())
        return name

    def new(self, kind: str, label: str) -> str:
        self._counter += 1
        node_id = f"{_PREFIX[kind]}_{self._counter}"
        self.nodes[node_id] = Node(node_id, kind, label)
        return node_id

    def link(self, src: str, dst: str, label: str, sequential: bool = False) -> None:
        self.edges.append(Edge(src, dst, label, sequential))

    # -- tree walk ------------------------------------------------------------

    def walk(self, node: Any) -> str:  # noqa: PLR0911, C901 - one branch per pyverilog node type
        if isinstance(node, DFTerminal):
            return self.signal(str(node.name))

        if isinstance(node, (DFConstant, DFEvalValue)):
            return self.new("const", str(node.value))

        if isinstance(node, DFOperator):
            op = self.new("operator", str(node.operator))
            for i, child in enumerate(node.nextnodes):
                self.link(self.walk(child), op, str(i))
            return op

        if isinstance(node, DFBranch):
            br = self.new("branch", "Branch")
            self.link(self.walk(node.condnode), br, "COND")
            if node.truenode is not None:
                self.link(self.walk(node.truenode), br, "TRUE")
            if node.falsenode is not None:
                self.link(self.walk(node.falsenode), br, "FALSE")
            return br

        if isinstance(node, DFConcat):
            cat = self.new("concat", "Concat")
            for i, child in enumerate(node.nextnodes):
                self.link(self.walk(child), cat, str(i))
            return cat

        if isinstance(node, DFPartselect):
            ps = self.new("partselect", "Partselect")
            self.link(self.walk(node.var), ps, "VAR")
            self.link(self.walk(node.msb), ps, "MSB")
            self.link(self.walk(node.lsb), ps, "LSB")
            return ps

        if isinstance(node, DFPointer):
            ptr = self.new("pointer", "Pointer")
            self.link(self.walk(node.var), ptr, "VAR")
            self.link(self.walk(node.ptr), ptr, "PTR")
            return ptr

        if isinstance(node, DFDelay):
            dly = self.new("delay", "Delay")
            self.link(self.walk(node.nextnode), dly, "0")
            return dly

        if isinstance(node, DFSyscall):
            sc = self.new("syscall", str(node.syscall))
            for i, child in enumerate(node.nextnodes):
                self.link(self.walk(child), sc, str(i))
            return sc

        if isinstance(node, DFUndefined):
            return self.new("undefined", "Undefined")
        if isinstance(node, DFHighImpedance):
            return self.new("highz", "HighZ")
        return self.new("unknown", type(node).__name__)


def build_dfg(
    ast: Any,
    top_module: str | None = None,
    *,
    resolve_constants: bool = False,
) -> DFGResult:
    """Build the Data Flow Graph of a design.

    Args:
        ast: pyverilog ``Source`` node, or path to an ``ast.pkl`` (see AST contract).
        top_module: top module name. Required for in-memory ASTs; defaults to the
            name recorded in a pickle.
        resolve_constants: fold parameters/constants (pyverilog's optimiser) before
            building the graph. Off by default to keep the graph faithful to the source.

    Raises:
        ValueError: when the top module cannot be determined.
    """
    loaded = load_ast_input(ast, top_module)
    if loaded.top_module is None:
        raise ValueError("top_module is required when passing an in-memory AST")

    analysis = analyze_ast(loaded.ast, loaded.top_module)
    terms = analysis.resolved_terms if resolve_constants else analysis.terms
    binddict = analysis.resolved_binddict if resolve_constants else analysis.binddict

    builder = _Builder(terms)
    for key in sorted(binddict, key=str):
        for bind in binddict[key]:
            dest = builder.signal(str(bind.dest))
            root = builder.walk(bind.tree)
            builder.link(root, dest, "assign", sequential=bool(bind.isClockEdge()))

    return DFGResult(
        top_module=loaded.top_module,
        nodes=tuple(builder.nodes.values()),
        edges=tuple(builder.edges),
    )
