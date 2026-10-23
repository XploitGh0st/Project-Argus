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
    "normalize_verilog",
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


# ---------------------------------------------------------------------------
# normalize_verilog: semantics-preserving Verilog preprocessing
# ---------------------------------------------------------------------------
_SIZED_LITERAL_SPACE = re.compile(r"(\d+)'([bBdDhHoO])\s+")
_INCLUDE_RE = re.compile(r'^\s*`include\s+"([^"]+)"', re.MULTILINE)
_DEFINE_SIMPLE = re.compile(r'^\s*`define\s+(\w+)\s+(.*?)(?:\s*//.*)?$', re.MULTILINE)
_DEFINE_FUNC = re.compile(
    r'^\s*`define\s+(\w+)\(([^)]*)\)\s+(.*?)(?:\s*//.*)?$', re.MULTILINE
)
_UNDEF_RE = re.compile(r'^\s*`undef\s+(\w+)', re.MULTILINE)
_TIMESCALE_RE = re.compile(r'^\s*`timescale\b[^\n]*', re.MULTILINE)
_MACRO_USE = re.compile(r'`(\w+)(?:\(([^)]*)\))?')
_IFDEF_RE = re.compile(
    r'^\s*`(ifdef|ifndef|elsif)\s+(\w+)', re.MULTILINE
)
_ELSE_RE = re.compile(r'^\s*`else\b', re.MULTILINE)
_ENDIF_RE = re.compile(r'^\s*`endif\b', re.MULTILINE)
_EMPTY_PORT = re.compile(r',\s*,')
_EMPTY_PORT_FIRST = re.compile(r'\(\s*,')
_EMPTY_PORT_LAST = re.compile(r',\s*\)')
_MODULE_BODY_RE = re.compile(
    r'\bmodule\s+(\w+)\s*(?:#\s*\([^)]*\)\s*)?\(([^)]*)\)\s*;',
    re.DOTALL,
)


def normalize_verilog(
    text_by_file: dict[str, str],
    include_dirs: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Semantics-preserving Verilog normalization.

    Applies the same fixes to both clean and infected versions:
    1. Expand ``include``, ``define``, ``ifdef`` etc.
    2. Fix sized-literal spaces  (34'b 0011 -> 34'b0011).
    3. Replace empty positional port slots with dummy wires.
    4. Drop ``timescale``.

    Returns (normalised_text, log_lines).
    """
    log: list[str] = []
    inc_dirs: list[Path] = [Path(d) for d in (include_dirs or [])]

    # ---- helpers ----
    resolved_files: dict[str, str] = {}  # cache for include resolution

    def _resolve_include(inc_name: str, from_dir: Path) -> str:
        """Resolve an include file, searching from_dir then inc_dirs."""
        base_name = Path(inc_name).name
        search = [from_dir] + inc_dirs
        for d in search:
            try:
                p = d / inc_name
                if p.is_file():
                    return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
            p2 = d / base_name
            if p2.is_file():
                return p2.read_text(encoding="utf-8", errors="replace")
        raise FileNotFoundError(f"`include \"{inc_name}\" not found in {search}")

    # ---- Phase 1: expand includes ----
    # Build a single text with includes expanded
    parts: list[str] = []
    file_dirs: list[Path] = []
    for fname, txt in text_by_file.items():
        file_dirs.append(Path(fname).parent)
        parts.append(txt)
    combined = "\n".join(parts)

    # Iteratively expand includes (handles nested includes)
    max_rounds = 10
    for _round in range(max_rounds):
        matches = list(_INCLUDE_RE.finditer(combined))
        if not matches:
            break
        new_text = combined
        for m in reversed(matches):
            inc_name = m.group(1)
            from_dir = file_dirs[0] if file_dirs else Path(".")
            # Try to find the right source dir
            for fd in file_dirs:
                if (fd / inc_name).is_file():
                    from_dir = fd
                    break
            try:
                if inc_name not in resolved_files:
                    resolved_files[inc_name] = _resolve_include(inc_name, from_dir)
                inc_text = resolved_files[inc_name]
                new_text = new_text[:m.start()] + inc_text + new_text[m.end():]
                log.append(f"include: expanded `include \"{inc_name}\"")
            except FileNotFoundError as e:
                log.append(f"include: WARNING {e}")
                new_text = new_text[:m.start()] + f"// FAILED: {m.group(0)}" + new_text[m.end():]
        combined = new_text

    # ---- Phase 2: drop timescale ----
    combined, n = _TIMESCALE_RE.subn("", combined)
    if n:
        log.append(f"timescale: dropped {n}")

    # ---- Phase 3: expand defines and ifdefs ----
    combined = _expand_macros(combined, log)

    # ---- Phase 4: fix sized-literal spaces ----
    combined, n = _SIZED_LITERAL_SPACE.subn(r"\1'\2", combined)
    if n:
        log.append(f"sized_literal: fixed {n} space(s)")

    # ---- Phase 5: fix empty positional ports ----
    combined = _fix_empty_ports(combined, log)

    # ---- Phase 6: add implicit wires for undeclared nets in port connections ----
    combined = _add_implicit_wires(combined, log)

    # ---- Phase 7: fix trailing commas in port lists (e.g. , );) ----
    combined, n = re.subn(r",\s*\)", ")", combined)
    if n:
        log.append(f"trailing_comma: stripped {n} trailing comma(s)")

    return combined, log


def _expand_macros(text: str, log: list[str]) -> str:
    """Expand `define, `undef, `ifdef/ifndef/elsif/else/endif."""
    defines: dict[str, tuple[str, list[str] | None]] = {}  # name -> (body, params or None)

    lines = text.split("\n")
    out_lines: list[str] = []
    cond_stack: list[bool] = []  # True = currently emitting
    active_stack: list[bool] = []  # True = some branch in this level was True

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # `define
        m = re.match(r'\s*`define\s+(\w+)\(([^)]*)\)\s+(.*)', line)
        if m and _is_active(cond_stack):
            name, params, body = m.group(1), m.group(2).split(","), m.group(3).rstrip()
            params = [p.strip() for p in params]
            defines[name] = (body, params)
            log.append(f"define: `{name}({','.join(params)})")
            i += 1
            continue

        m = re.match(r'\s*`define\s+(\w+)\s+(.*)', line)
        if m and _is_active(cond_stack):
            name, body = m.group(1), m.group(2).rstrip()
            # Handle line continuation
            while body.endswith("\\") and i + 1 < len(lines):
                i += 1
                body = body[:-1] + lines[i].rstrip()
            # Strip trailing inline comment
            body = re.sub(r'\s*//.*$', '', body)
            defines[name] = (body, None)
            log.append(f"define: `{name} = {body[:60]}")
            i += 1
            continue

        m = re.match(r'\s*`define\s+(\w+)\s*$', line)
        if m and _is_active(cond_stack):
            defines[m.group(1)] = ("", None)
            i += 1
            continue

        # `undef
        m = re.match(r'\s*`undef\s+(\w+)', line)
        if m and _is_active(cond_stack):
            defines.pop(m.group(1), None)
            i += 1
            continue

        # `ifdef / `ifndef
        m = re.match(r'\s*`(ifdef|ifndef)\s+(\w+)', line)
        if m:
            directive, macro = m.group(1), m.group(2)
            defined = macro in defines
            active = (defined if directive == "ifdef" else not defined) and _is_active(cond_stack)
            cond_stack.append(active)
            active_stack.append(active)
            i += 1
            continue

        # `elsif
        m = re.match(r'\s*`elsif\s+(\w+)', line)
        if m:
            macro = m.group(1)
            if cond_stack:
                prev_any_active = active_stack[-1]
                parent_active = _is_active(cond_stack[:-1])
                defined = macro in defines
                new_active = defined and parent_active and not prev_any_active
                cond_stack[-1] = new_active
                active_stack[-1] = active_stack[-1] or new_active
            i += 1
            continue

        # `else
        if stripped.startswith("`else"):
            if cond_stack:
                prev_any_active = active_stack[-1]
                parent_active = _is_active(cond_stack[:-1])
                cond_stack[-1] = parent_active and not prev_any_active
                active_stack[-1] = True
            i += 1
            continue

        # `endif
        if stripped.startswith("`endif"):
            if cond_stack:
                cond_stack.pop()
                active_stack.pop()
            i += 1
            continue

        # Active emission
        if _is_active(cond_stack):
            # Expand macro usages in this line
            line = _substitute_macros(line, defines)
            out_lines.append(line)
        # else: ifdef'd out, skip line
        i += 1

    if len(out_lines) != len(lines):
        diff = len(lines) - len(out_lines)
        if diff > 0:
            log.append(f"ifdef: removed {diff} lines via ifdef/ifndef")

    return "\n".join(out_lines)


def _is_active(cond_stack: list[bool]) -> bool:
    return all(cond_stack) if cond_stack else True


def _substitute_macros(line: str, defines: dict[str, tuple[str, list[str] | None]]) -> str:
    """Replace `MACRO and `MACRO(args) in a single line."""
    max_iter = 20
    for _ in range(max_iter):
        m = _MACRO_USE.search(line)
        if not m:
            break
        name = m.group(1)
        # Skip known Verilog compiler directives that are not user macros
        if name in ("include", "define", "undef", "ifdef", "ifndef", "else",
                     "elsif", "endif", "timescale", "celldefine", "endcelldefine",
                     "default_nettype", "resetall"):
            break
        if name not in defines:
            # Unknown macro — leave it (will cause parse error, caught later)
            break
        body, params = defines[name]
        if params is not None and m.group(2) is not None:
            # Function-like macro
            args = m.group(2).split(",")
            expanded = body
            for p, a in zip(params, args):
                expanded = re.sub(rf'\b{re.escape(p.strip())}\b', a.strip(), expanded)
            line = line[:m.start()] + expanded + line[m.end():]
        else:
            line = line[:m.start()] + body + line[m.end():]
    return line


def _fix_empty_ports(text: str, log: list[str]) -> str:
    """Replace empty positional ports with dummy wires."""
    counter = [0]
    modules_text = text

    def _process_module(m: re.Match) -> str:
        mod_name = m.group(0)  # full match
        # Find the module body up to endmodule
        return mod_name

    # Find all instance port lists with empty slots
    # Strategy: find patterns like (, ,) or (x, , y) in instantiations
    # We need to be inside a module body. Simple approach: iterate lines,
    # track module context, and fix empty ports adding wire declarations.

    lines = text.split("\n")
    out_lines: list[str] = []
    current_module: str | None = None
    wire_decls: list[str] = []
    module_start_idx: int = -1
    insert_after_idx: int = -1

    for idx, line in enumerate(lines):
        # Detect module declaration start
        mm = re.match(r'\s*module\s+(\w+)', line)
        if mm:
            # Flush any pending wire decls from previous module
            if wire_decls and insert_after_idx >= 0:
                for wd in wire_decls:
                    out_lines.insert(insert_after_idx + 1, wd)
                    insert_after_idx += 1
                wire_decls = []
            current_module = mm.group(1)
            module_start_idx = len(out_lines)
            insert_after_idx = -1

        # Detect end of module port list (the ;)
        if current_module and insert_after_idx < 0 and ";" in line:
            insert_after_idx = len(out_lines)

        # Fix empty ports in this line
        new_line = line
        while _EMPTY_PORT.search(new_line) or _EMPTY_PORT_FIRST.search(new_line):
            # Replace first empty port
            m = _EMPTY_PORT.search(new_line)
            if m:
                wire_name = f"__unconn_{counter[0]}"
                counter[0] += 1
                new_line = new_line[:m.start()] + f", {wire_name}," + new_line[m.end():]
                wire_decls.append(f"  wire {wire_name};")
                log.append(f"empty_port: inserted {wire_name} in {current_module or '?'}")
            m = _EMPTY_PORT_FIRST.search(new_line)
            if m:
                wire_name = f"__unconn_{counter[0]}"
                counter[0] += 1
                new_line = new_line[:m.start()] + f"({wire_name}," + new_line[m.end():]
                wire_decls.append(f"  wire {wire_name};")
                log.append(f"empty_port: inserted {wire_name} in {current_module or '?'}")

        # Detect endmodule
        if re.match(r'\s*endmodule\b', line):
            # Insert wire decls before endmodule
            if wire_decls and insert_after_idx >= 0:
                for wd in wire_decls:
                    out_lines.insert(insert_after_idx + 1, wd)
                    insert_after_idx += 1
                wire_decls = []
            current_module = None
            insert_after_idx = -1

        out_lines.append(new_line)

    # Handle any remaining wire decls
    if wire_decls and insert_after_idx >= 0:
        for wd in wire_decls:
            out_lines.insert(insert_after_idx + 1, wd)

    return "\n".join(out_lines)


def _add_implicit_wires(text: str, log: list[str]) -> str:
    """
    IEEE 1364-2001 §19.2: When a signal is used in port connection of an instance
    without declaration, declare implicit wire.
    """
    decl_pats = [
        r'\b(?:input|output|inout|wire|reg|integer|genvar|real)\s+(?:(?:signed|reg)\s+)?(?:\[[^\]]+\]\s+)?([^;]+);',
        r'\bparameter\s+(?:\[[^\]]+\]\s+)?(\w+)',
        r'\blocalparam\s+(?:\[[^\]]+\]\s+)?(\w+)',
    ]
    mod_pattern = re.compile(r'\bmodule\s+(\w+)\s*(?:\(([^)]*)\))?\s*;(.*?)endmodule', re.DOTALL)

    def repl_mod(m):
        mod_name = m.group(1)
        ports_hdr = m.group(2) or ""
        body = m.group(3)

        declared = set()
        for p in re.findall(r'\b([a-zA-Z_]\w*)\b', ports_hdr):
            declared.add(p)
        for pat in decl_pats:
            for match in re.finditer(pat, body):
                for ident in re.findall(r'\b([a-zA-Z_]\w*)\b', match.group(1)):
                    declared.add(ident)

        # Find instances in body
        inst_pat = re.compile(r'\b([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)\s*\(([^;]+)\)\s*;')
        new_wires = []
        for inst_m in inst_pat.finditer(body):
            mod_type, inst_name, ports_str = inst_m.groups()
            if mod_type in ('module', 'function', 'task', 'begin', 'fork', 'case', 'always', 'initial', 'assign'):
                continue
            for arg in re.findall(r'(?:\.\w+\s*\(\s*)?([a-zA-Z_]\w*)', ports_str):
                if (
                    arg not in declared
                    and arg not in new_wires
                    and not re.match(r'^(b|d|h|o)[0-9a-fA-F]+$', arg, re.I)
                    and not arg.startswith('__unconn_')
                ):
                    new_wires.append(arg)

        if new_wires:
            decl_lines = "\n" + "\n".join(f"  wire {w};" for w in new_wires) + "\n"
            for w in new_wires:
                log.append(f"implicit_wire: added 'wire {w};' in module {mod_name}")
            return f"module {mod_name} ({ports_hdr});{decl_lines}{body}endmodule"
        return m.group(0)

    return mod_pattern.sub(repl_mod, text)
