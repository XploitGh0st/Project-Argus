"""AST -> CFG extraction.

Runs pyverilog's control-flow analyser on an AST and converts the discovered finite
state machines straight into plain data structures (no pygraphviz, no ``file.dot``).
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from typing import Any

from pyverilog.controlflow.controlflow_analyzer import VerilogControlflowAnalyzer

from argus_cfg.binding import analyze_ast
from argus_cfg.inputs import load_ast_input
from argus_cfg.result import ANY_STATE, FSM, CFGResult, Transition

__all__ = ["DEFAULT_FSM_HINTS", "build_cfg"]

# Same hints as the legacy implementation: a register whose name contains one of
# these (case-insensitive regex) is considered a candidate FSM state register.
DEFAULT_FSM_HINTS: tuple[str, ...] = ("fsm", "state", "count", "cnt", "step", "mode")


def _condition_text(cond: Any) -> str | None:
    """Readable transition condition; ``None`` means unconditional."""
    if cond is None:
        return None
    for attr in ("tocode", "tostr"):
        fn = getattr(cond, attr, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:  # noqa: BLE001 - fall through to the next renderer
                continue
    return str(cond)


def _state_sort_key(label: str) -> tuple[int, Any]:
    return (0, int(label)) if label.lstrip("-").isdigit() else (1, label)


def _convert_fsm(signal: Any, fsm: Any) -> FSM:
    transitions: list[Transition] = []
    states: set[str] = set()

    for src, dst_by_cond in fsm.fsm.items():
        for cond, dst in dst_by_cond.items():
            transitions.append(Transition(str(src), str(dst), _condition_text(cond)))
            states.update((str(src), str(dst)))
    for cond, dst in fsm.any.items():
        transitions.append(Transition(ANY_STATE, str(dst), _condition_text(cond)))
        states.add(str(dst))

    transitions.sort(key=lambda t: (_state_sort_key(t.src) if t.src != ANY_STATE else (2, ""),
                                    _state_sort_key(t.dst), t.condition or ""))
    return FSM(
        signal=str(signal),
        delay_cnt=int(fsm.delaycnt),
        states=tuple(sorted(states, key=_state_sort_key)),
        transitions=tuple(transitions),
    )


def build_cfg(
    ast: Any,
    top_module: str | None = None,
    *,
    fsm_hints: Sequence[str] = DEFAULT_FSM_HINTS,
) -> CFGResult:
    """Build the Control Flow Graph of a design.

    Args:
        ast: pyverilog ``Source`` node, or path to an ``ast.pkl`` (see AST contract).
        top_module: top module name. Required for in-memory ASTs; defaults to the
            name recorded in a pickle.
        fsm_hints: name fragments that mark FSM state registers.

    Returns:
        A :class:`CFGResult`. Designs without FSMs yield an empty ``fsms`` tuple.

    Raises:
        ValueError: when the top module cannot be determined.
    """
    loaded = load_ast_input(ast, top_module)
    if loaded.top_module is None:
        raise ValueError("top_module is required when passing an in-memory AST")

    analysis = analyze_ast(loaded.ast, loaded.top_module)
    analyzer = VerilogControlflowAnalyzer(
        loaded.top_module,
        analysis.terms,
        analysis.binddict,
        analysis.resolved_terms,
        analysis.resolved_binddict,
        analysis.constlist,
        tuple(fsm_hints),
    )
    # pyverilog prints progress for every candidate signal
    with contextlib.redirect_stdout(io.StringIO()):
        machines = analyzer.getFiniteStateMachines()

    fsms = tuple(
        _convert_fsm(signal, fsm)
        for signal, fsm in sorted(machines.items(), key=lambda kv: str(kv[0]))
    )
    return CFGResult(top_module=loaded.top_module, fsms=fsms)
