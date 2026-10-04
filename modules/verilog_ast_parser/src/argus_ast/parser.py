"""Verilog -> pyverilog AST, without Icarus Verilog."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from pyverilog.vparser.parser import VerilogParser

from argus_ast.artifact import ASTArtifact
from argus_ast.preprocess import (
    TopModuleError,
    find_modules,
    find_top_module,
    flatten,
    rename_module,
    strip_comments,
)

__all__ = ["parse_verilog", "parse_verilog_text"]

# LALR tables are cached here so only the first parse in a process pays for them.
_TABLE_DIR = Path(tempfile.gettempdir()) / "argus_ast_parsetab"


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def parse_verilog_text(
    text: str,
    top_module: str | None = None,
    *,
    preprocess: bool = True,
    rename_top_to: str | None = None,
    source_files: tuple[str, ...] = ("<text>",),
) -> ASTArtifact:
    """Parse Verilog source text into an :class:`ASTArtifact`.

    Args:
        text: Verilog source.
        top_module: top module name; auto-detected when ``None``.
        preprocess: strip comments before parsing.
        rename_top_to: if given, rename the top module (e.g. ``"top"``).
        source_files: provenance recorded in the artifact.

    Raises:
        TopModuleError: top module missing or ambiguous.
    """
    if preprocess:
        text = strip_comments(text)

    modules = find_modules(text)
    if top_module is None:
        top_module = find_top_module(text)
    elif top_module not in modules:
        raise TopModuleError(f"top module {top_module!r} not defined (found: {modules})")

    if rename_top_to and rename_top_to != top_module:
        text = rename_module(text, top_module, rename_top_to)
        top_module = rename_top_to

    parser = VerilogParser(outputdir=str(_TABLE_DIR), debug=False)
    ast = parser.parse(text)
    return ASTArtifact(top_module=top_module, source_files=source_files, ast=ast)


def parse_verilog(
    paths: str | Path | Iterable[str | Path],
    top_module: str | None = None,
    *,
    preprocess: bool = True,
    rename_top_to: str | None = None,
) -> ASTArtifact:
    """Parse one or more Verilog files into a single :class:`ASTArtifact`.

    Multiple files are concatenated into one translation unit, so a design split
    across files parses as a whole.
    """
    files = _as_paths(paths)
    if not files:
        raise ValueError("no input files")
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Verilog file(s) not found: {', '.join(missing)}")

    return parse_verilog_text(
        flatten(files),
        top_module,
        preprocess=preprocess,
        rename_top_to=rename_top_to,
        source_files=tuple(str(p) for p in files),
    )
