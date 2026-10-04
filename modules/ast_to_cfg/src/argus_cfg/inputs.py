"""Loading the AST input contract (``ast.pkl`` or an in-memory ``Source`` node)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AST_KIND = "ast"
AST_SCHEMA = "1.0"

__all__ = ["AST_KIND", "AST_SCHEMA", "AstInput", "load_ast_input"]


@dataclass(frozen=True)
class AstInput:
    ast: Any
    top_module: str | None


def load_ast_input(source: Any, top_module: str | None = None) -> AstInput:
    """Accept an AST in any contract-allowed form.

    ``source`` may be a pyverilog ``Source`` node or a path to an ``ast.pkl`` file
    (see the AST contract). When a pickle is given, ``top_module`` defaults to the
    one recorded in it.

    Warning:
        Pickle files are executable content; load only trusted files.
    """
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - trusted-input contract
        if not isinstance(payload, dict) or payload.get("kind") != AST_KIND:
            raise ValueError(f"{source}: not an AST artifact")
        if payload.get("schema_version") != AST_SCHEMA:
            raise ValueError(
                f"{source}: unsupported AST schema {payload.get('schema_version')!r}, "
                f"expected {AST_SCHEMA}"
            )
        return AstInput(payload["ast"], top_module or payload.get("top_module"))

    if type(source).__name__ != "Source":
        raise TypeError(
            f"expected pyverilog Source node or ast.pkl path, got {type(source).__name__}"
        )
    return AstInput(source, top_module)
