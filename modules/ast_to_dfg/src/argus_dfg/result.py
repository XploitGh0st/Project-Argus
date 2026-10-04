"""DFG result types and the DFG output contract (schema 1.0).

Edge direction is the direction of DATA FLOW: operand -> operator -> destination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

SCHEMA_VERSION = "1.0"
KIND = "dfg"

NODE_TYPES = (
    "signal",
    "const",
    "operator",
    "branch",
    "concat",
    "partselect",
    "pointer",
    "delay",
    "syscall",
    "undefined",
    "highz",
    "unknown",
)

__all__ = [
    "KIND",
    "NODE_TYPES",
    "SCHEMA_VERSION",
    "ContractError",
    "DFGResult",
    "Edge",
    "Node",
    "validate_dfg_document",
]


class ContractError(ValueError):
    """A document violates the DFG contract."""


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    label: str
    termtype: tuple[str, ...] = ()  # only for signals: e.g. ("Input",) or ("Reg", "Output")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str
    sequential: bool = False  # True on the edge into a register assigned at a clock edge


@dataclass(frozen=True)
class DFGResult:
    top_module: str
    nodes: tuple[Node, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Contract-shaped document (see CONTRACT.md)."""
        nodes = []
        for n in self.nodes:
            item: dict[str, Any] = {"id": n.id, "type": n.type, "label": n.label}
            if n.type == "signal":
                item["termtype"] = list(n.termtype)
            nodes.append(item)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "top_module": self.top_module,
            "nodes": nodes,
            "edges": [
                {"src": e.src, "dst": e.dst, "label": e.label, "sequential": e.sequential}
                for e in self.edges
            ],
        }

    def to_networkx(self) -> nx.MultiDiGraph:
        """MultiDiGraph (an operator may take the same operand twice)."""
        g = nx.MultiDiGraph(name=self.top_module, kind=KIND)
        for n in self.nodes:
            g.add_node(n.id, type=n.type, label=n.label, termtype=n.termtype)
        for e in self.edges:
            g.add_edge(e.src, e.dst, label=e.label, sequential=e.sequential)
        return g

    def signals(self) -> list[str]:
        return [n.id for n in self.nodes if n.type == "signal"]

    def save(self, out_dir: str | Path, name: str = "dfg.json") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def validate_dfg_document(doc: dict[str, Any]) -> None:
    """Raise :class:`ContractError` unless ``doc`` conforms to DFG schema 1.0."""

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ContractError(msg)

    need(isinstance(doc, dict), "document must be an object")
    need(doc.get("kind") == KIND, f"kind must be {KIND!r}, got {doc.get('kind')!r}")
    need(
        doc.get("schema_version") == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION!r}, got {doc.get('schema_version')!r}",
    )
    need(isinstance(doc.get("top_module"), str), "top_module must be a string")
    need(isinstance(doc.get("nodes"), list), "nodes must be a list")
    need(isinstance(doc.get("edges"), list), "edges must be a list")

    ids: set[str] = set()
    for i, n in enumerate(doc["nodes"]):
        need(isinstance(n.get("id"), str), f"nodes[{i}].id must be a string")
        need(n["id"] not in ids, f"nodes[{i}].id {n['id']!r} is duplicated")
        ids.add(n["id"])
        need(n.get("type") in NODE_TYPES, f"nodes[{i}].type {n.get('type')!r} is not allowed")
        need(isinstance(n.get("label"), str), f"nodes[{i}].label must be a string")
        if n["type"] == "signal":
            tt = n.get("termtype")
            need(
                isinstance(tt, list) and all(isinstance(t, str) for t in tt),
                f"nodes[{i}].termtype must be a list of strings",
            )

    for j, e in enumerate(doc["edges"]):
        need(e.get("src") in ids, f"edges[{j}].src {e.get('src')!r} is not a node")
        need(e.get("dst") in ids, f"edges[{j}].dst {e.get('dst')!r} is not a node")
        need(isinstance(e.get("label"), str), f"edges[{j}].label must be a string")
        need(isinstance(e.get("sequential"), bool), f"edges[{j}].sequential must be a bool")
