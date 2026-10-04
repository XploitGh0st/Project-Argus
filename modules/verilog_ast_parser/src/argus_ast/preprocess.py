"""Source-level helpers for Verilog files.

These replace the old ``PreprocessVerilog`` class. They are pure functions on
strings, so they are trivial to test and have no dependency on pyverilog.

Fixes over the legacy implementation:

* comments are removed correctly (``//`` and ``/* */``) without touching string
  literals, and line numbers are preserved;
* the top module is found by usage analysis instead of "name occurs once";
* renaming uses word boundaries, so ``alu`` no longer clobbers ``alu_core``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "TopModuleError",
    "find_modules",
    "find_top_module",
    "flatten",
    "rename_module",
    "strip_comments",
]

_COMMENT_RE = re.compile(
    r'(?P<string>"(?:\\.|[^"\\\n])*")'  # keep string literals untouched
    r"|(?P<line>//[^\n]*)"
    r"|(?P<block>/\*.*?\*/)",
    re.DOTALL,
)
_MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_][\w$]*)")


class TopModuleError(ValueError):
    """The top module could not be determined unambiguously."""


def strip_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments, preserving line numbers."""

    def repl(match: re.Match[str]) -> str:
        if match.group("string") is not None:
            return match.group("string")
        if match.group("block") is not None:
            return "\n" * match.group("block").count("\n")
        return ""

    return _COMMENT_RE.sub(repl, text)


def find_modules(text: str) -> list[str]:
    """Return module names defined in ``text`` in order of appearance."""
    return _MODULE_RE.findall(strip_comments(text))


def find_top_module(text: str) -> str:
    """Return the single module that is defined but never instantiated.

    Raises:
        TopModuleError: if there is no such module or more than one candidate.
    """
    clean = strip_comments(text)
    defined = _MODULE_RE.findall(clean)
    if not defined:
        raise TopModuleError("no module definitions found")

    candidates = []
    for name in dict.fromkeys(defined):
        # one occurrence == its own `module <name>` header, i.e. never used elsewhere
        uses = len(re.findall(rf"\b{re.escape(name)}\b", clean))
        definitions = defined.count(name)
        if uses <= definitions:
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise TopModuleError("every module is instantiated (cyclic hierarchy?)")
    raise TopModuleError(
        f"ambiguous top module, candidates: {', '.join(candidates)}; pass top_module explicitly"
    )


def rename_module(text: str, old: str, new: str) -> str:
    """Rename identifier ``old`` to ``new`` using whole-word matching."""
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def flatten(paths: Iterable[str | Path]) -> str:
    """Concatenate Verilog sources into one translation unit."""
    chunks = [Path(p).read_text(encoding="utf-8", errors="replace") for p in paths]
    return "\n".join(chunks)
