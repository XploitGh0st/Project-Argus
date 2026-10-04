"""Stub AST builders - mock input for ``argus_cfg`` WITHOUT any Verilog parser.

Each ``build_*`` function returns ``(ast, top_module)`` where ``ast`` is a real
pyverilog ``Source`` node, i.e. exactly what ``argus_ast`` would hand over. Use them
to develop and test the AST->CFG conversion in isolation:

    from ast_builders import build_traffic_light
    ast, top = build_traffic_light()
    build_cfg(ast, top)

The module deliberately has no import from any other Argus package.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyverilog.vparser import ast as v

# --------------------------------------------------------------------------- helpers


def _w(msb: int, lsb: int = 0) -> v.Width:
    return v.Width(v.IntConst(str(msb)), v.IntConst(str(lsb)))


def _const(value: int, width: int) -> v.IntConst:
    return v.IntConst(f"{width}'d{value}")


def _nb(signal: str, value: v.Node) -> v.NonblockingSubstitution:
    return v.NonblockingSubstitution(v.Lvalue(v.Identifier(signal)), v.Rvalue(value))


def _input(name: str) -> v.Ioport:
    return v.Ioport(v.Input(name))


def _module(name: str, ports: Sequence[v.Ioport], items: Sequence[v.Node]) -> v.Source:
    mod = v.ModuleDef(name, v.Paramlist(()), v.Portlist(tuple(ports)), tuple(items))
    return v.Source("", v.Description((mod,)))


def fsm_always(
    signal: str,
    width: int,
    transitions: Sequence[tuple[int, str | None, int]],
    *,
    clk: str = "clk",
    rst: str = "rst",
) -> v.Always:
    """``always @(posedge clk)`` block implementing a reset + case-based FSM.

    ``transitions`` are ``(src_state, guard_signal_or_None, dst_state)``.
    """
    by_src: dict[int, list[tuple[str | None, int]]] = {}
    for src, guard, dst in transitions:
        by_src.setdefault(src, []).append((guard, dst))

    cases = []
    for src, outs in by_src.items():
        stmt: v.Node | None = None
        for guard, dst in reversed(outs):
            nb = _nb(signal, _const(dst, width))
            stmt = nb if guard is None else v.IfStatement(v.Identifier(guard), nb, stmt)
        cases.append(v.Case((_const(src, width),), stmt))

    body = v.IfStatement(
        v.Identifier(rst),
        _nb(signal, _const(0, width)),
        v.Block((v.CaseStatement(v.Identifier(signal), tuple(cases)),)),
    )
    return v.Always(v.SensList((v.Sens(v.Identifier(clk), "posedge"),)), v.Block((body,)))


# --------------------------------------------------------------------------- designs


def build_traffic_light() -> tuple[v.Source, str]:
    """Same design as ``verilog_ast_parser/fixtures/verilog/fsm_traffic_light.v``.

    FSM: ``state`` 0 --start--> 1 --> 2 --> 0.
    """
    ports = [
        _input("clk"),
        _input("rst"),
        _input("start"),
        v.Ioport(v.Output("light", _w(1)), v.Reg("light", _w(1))),
    ]
    comb = v.Always(
        v.SensList((v.Sens(None, "all"),)),
        v.Block(
            (
                v.IfStatement(
                    v.Eq(v.Identifier("state"), _const(2, 2)),
                    v.BlockingSubstitution(
                        v.Lvalue(v.Identifier("light")), v.Rvalue(_const(3, 2))
                    ),
                    v.BlockingSubstitution(
                        v.Lvalue(v.Identifier("light")), v.Rvalue(v.Identifier("state"))
                    ),
                ),
            )
        ),
    )
    items = [
        v.Decl((v.Reg("state", _w(1)),)),
        fsm_always("state", 2, [(0, "start", 1), (1, None, 2), (2, None, 0)]),
        comb,
    ]
    return _module("traffic", ports, items), "traffic"


def build_two_fsms() -> tuple[v.Source, str]:
    """Two independent FSMs: ``state`` (0<->1 on ``go``) and ``mode`` (0->1->2->0)."""
    ports = [_input("clk"), _input("rst"), _input("go")]
    items = [
        v.Decl((v.Reg("state", _w(1)),)),
        v.Decl((v.Reg("mode", _w(1)),)),
        fsm_always("state", 2, [(0, "go", 1), (1, "go", 0)]),
        fsm_always("mode", 2, [(0, None, 1), (1, None, 2), (2, None, 0)]),
    ]
    return _module("dual", ports, items), "dual"


def build_no_fsm() -> tuple[v.Source, str]:
    """Purely combinational design: ``y = a & b`` (no FSM, empty CFG)."""
    ports = [
        _input("a"),
        _input("b"),
        v.Ioport(v.Output("y")),
    ]
    items = [
        v.Assign(
            v.Lvalue(v.Identifier("y")),
            v.Rvalue(v.And(v.Identifier("a"), v.Identifier("b"))),
        )
    ]
    return _module("comb", ports, items), "comb"


ALL_BUILDERS = {
    "traffic_light": build_traffic_light,
    "two_fsms": build_two_fsms,
    "no_fsm": build_no_fsm,
}
