"""Command line: ``python -m argus_ast design.v [more.v] --top top --out out/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from argus_ast import __version__, parse_verilog
from argus_ast.preprocess import TopModuleError


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus-ast", description=__doc__)
    ap.add_argument("files", nargs="+", help="Verilog source file(s)")
    ap.add_argument("--top", default=None, help="top module (auto-detected if omitted)")
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    ap.add_argument("--rename-top", default=None, help="rename top module to this name")
    ap.add_argument("--no-preprocess", action="store_true", help="do not strip comments")
    ap.add_argument("--version", action="version", version=f"argus-ast {__version__}")
    args = ap.parse_args(argv)

    try:
        artifact = parse_verilog(
            args.files,
            args.top,
            preprocess=not args.no_preprocess,
            rename_top_to=args.rename_top,
        )
    except (TopModuleError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    paths = artifact.save(args.out)
    print(f"top module : {artifact.top_module}")
    print(f"modules    : {', '.join(artifact.module_names())}")
    for kind, path in paths.items():
        print(f"{kind:<10} : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
