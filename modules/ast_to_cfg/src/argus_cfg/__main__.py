"""Command line: ``python -m argus_cfg out/ast.pkl --out out/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from argus_cfg import __version__, build_cfg


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus-cfg", description=__doc__)
    ap.add_argument("ast", help="path to ast.pkl")
    ap.add_argument("--top", default=None, help="top module (default: from the pickle)")
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    ap.add_argument("--fsm-hint", action="append", help="extra FSM register name fragment")
    ap.add_argument("--version", action="version", version=f"argus-cfg {__version__}")
    args = ap.parse_args(argv)

    kwargs = {}
    if args.fsm_hint:
        from argus_cfg import DEFAULT_FSM_HINTS

        kwargs["fsm_hints"] = (*DEFAULT_FSM_HINTS, *args.fsm_hint)

    try:
        result = build_cfg(args.ast, args.top, **kwargs)
    except (ValueError, TypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    path = result.save(args.out)
    print(f"top module : {result.top_module}")
    print(f"FSMs       : {len(result.fsms)}")
    for fsm in result.fsms:
        print(f"  {fsm.signal}: {len(fsm.states)} states, {len(fsm.transitions)} transitions")
    print(f"written    : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
