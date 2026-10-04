"""The AST artifact: the contract object handed to downstream modules."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
KIND = "ast"

__all__ = ["KIND", "SCHEMA_VERSION", "ASTArtifact", "ast_to_dict"]


def ast_to_dict(node: Any) -> dict[str, Any]:
    """Convert any pyverilog AST node to a plain, JSON-serialisable dict.

    Unlike the legacy whitelist-based exporter this handles *every* node type and
    keeps all attributes and line numbers. It is a *readable* view only; the
    lossless interchange format is the pickle written by :meth:`ASTArtifact.save`.
    """
    attrs = {name: getattr(node, name) for name in (node.attr_names or ())}
    return {
        "type": type(node).__name__,
        "attrs": attrs,
        "lineno": getattr(node, "lineno", 0),
        "children": [ast_to_dict(child) for child in node.children()],
    }


@dataclass(frozen=True)
class ASTArtifact:
    """Result of :func:`argus_ast.parse_verilog`.

    Attributes:
        top_module: name of the top-level module.
        source_files: files the AST was built from.
        ast: the pyverilog ``Source`` root node.
    """

    top_module: str
    source_files: tuple[str, ...]
    ast: Any

    def module_names(self) -> list[str]:
        """Names of all modules defined in the AST."""
        return [
            d.name
            for d in self.ast.description.definitions
            if type(d).__name__ == "ModuleDef"
        ]

    def to_dict(self) -> dict[str, Any]:
        """Readable JSON view including the contract header."""
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "top_module": self.top_module,
            "source_files": list(self.source_files),
            "ast": ast_to_dict(self.ast),
        }

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``ast.pkl`` (lossless) and ``ast.json`` (readable) into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

        pkl_path = out / "ast.pkl"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "top_module": self.top_module,
            "source_files": list(self.source_files),
            "ast": self.ast,
        }
        with pkl_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        json_path = out / "ast.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return {"pickle": pkl_path, "json": json_path}

    @classmethod
    def load(cls, pkl_path: str | Path) -> ASTArtifact:
        """Load an artifact written by :meth:`save`.

        Warning:
            Uses ``pickle``; only load files you produced or trust.
        """
        with Path(pkl_path).open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - documented, trusted-input contract
        if payload.get("kind") != KIND:
            raise ValueError(f"not an AST artifact: kind={payload.get('kind')!r}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported AST schema {payload.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION}"
            )
        return cls(
            top_module=payload["top_module"],
            source_files=tuple(payload["source_files"]),
            ast=payload["ast"],
        )
