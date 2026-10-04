"""Unified Digraph result types."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
KIND = "digraph"


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    label: str
    termtype: tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str
    kind: str  # "dataflow" or "control_dep"


@dataclass(frozen=True)
class DigraphResult:
    top_module: str
    nodes: tuple[Node, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
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
                {"src": e.src, "dst": e.dst, "label": e.label, "kind": e.kind}
                for e in self.edges
            ],
        }

    def save(self, out_dir: str | Path, name: str = "digraph.json") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
