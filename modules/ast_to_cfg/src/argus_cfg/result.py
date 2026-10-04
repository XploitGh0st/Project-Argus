"""CFG result types and the CFG output contract (schema 1.0)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

SCHEMA_VERSION = "1.0"
KIND = "cfg"
ANY_STATE = "*"  # transitions that fire from every state

__all__ = [
    "ANY_STATE",
    "KIND",
    "SCHEMA_VERSION",
    "CFGResult",
    "ContractError",
    "FSM",
    "Transition",
    "state_node_id",
    "validate_cfg_document",
]


class ContractError(ValueError):
    """A document violates the CFG contract."""


def state_node_id(signal: str, state: str) -> str:
    """Namespaced graph node id for one FSM state."""
    return f"{signal}::state::{state}"


@dataclass(frozen=True)
class Transition:
    src: str  # state label, or ANY_STATE
    dst: str
    condition: str | None  # None == unconditional


@dataclass(frozen=True)
class FSM:
    signal: str
    delay_cnt: int
    states: tuple[str, ...]
    transitions: tuple[Transition, ...]


@dataclass(frozen=True)
class CFGResult:
    top_module: str
    fsms: tuple[FSM, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Contract-shaped document (see CONTRACT.md)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "top_module": self.top_module,
            "fsms": [
                {
                    "signal": f.signal,
                    "delay_cnt": f.delay_cnt,
                    "states": list(f.states),
                    "transitions": [
                        {"src": t.src, "dst": t.dst, "condition": t.condition}
                        for t in f.transitions
                    ],
                }
                for f in self.fsms
            ],
        }

    def to_networkx(self) -> nx.DiGraph:
        """One DiGraph holding every FSM; node ids are ``{signal}::state::{s}``."""
        g = nx.DiGraph(name=self.top_module, kind=KIND)
        for fsm in self.fsms:
            for s in fsm.states:
                g.add_node(state_node_id(fsm.signal, s), kind="fsm_state", fsm=fsm.signal, state=s)
            for t in fsm.transitions:
                sources = fsm.states if t.src == ANY_STATE else (t.src,)
                for src in sources:
                    g.add_edge(
                        state_node_id(fsm.signal, src),
                        state_node_id(fsm.signal, t.dst),
                        kind="control_flow",
                        condition=t.condition,
                    )
        return g

    def save(self, out_dir: str | Path, name: str = "cfg.json") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def validate_cfg_document(doc: dict[str, Any]) -> None:
    """Raise :class:`ContractError` unless ``doc`` conforms to CFG schema 1.0."""

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
    need(isinstance(doc.get("fsms"), list), "fsms must be a list")

    for i, fsm in enumerate(doc["fsms"]):
        where = f"fsms[{i}]"
        need(isinstance(fsm.get("signal"), str), f"{where}.signal must be a string")
        need(isinstance(fsm.get("delay_cnt"), int), f"{where}.delay_cnt must be an int")
        states = fsm.get("states")
        need(
            isinstance(states, list) and all(isinstance(s, str) for s in states),
            f"{where}.states must be a list of strings",
        )
        known = set(states)
        need(isinstance(fsm.get("transitions"), list), f"{where}.transitions must be a list")
        for j, t in enumerate(fsm["transitions"]):
            tw = f"{where}.transitions[{j}]"
            need(t.get("dst") in known, f"{tw}.dst {t.get('dst')!r} is not a declared state")
            need(
                t.get("src") == ANY_STATE or t.get("src") in known,
                f"{tw}.src {t.get('src')!r} is not a declared state",
            )
            cond = t.get("condition")
            need(cond is None or isinstance(cond, str), f"{tw}.condition must be string or null")
