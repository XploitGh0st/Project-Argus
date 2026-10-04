"""argus_ast - Verilog ingestion and AST generation (Project Argus).

Public API::

    from argus_ast import parse_verilog
    artifact = parse_verilog(["design.v"], top_module="top")
    artifact.save("out/")      # ast.pkl + ast.json
"""

from argus_ast.artifact import KIND, SCHEMA_VERSION, ASTArtifact, ast_to_dict
from argus_ast.parser import parse_verilog, parse_verilog_text
from argus_ast.preprocess import (
    TopModuleError,
    find_modules,
    find_top_module,
    flatten,
    rename_module,
    strip_comments,
)

__version__ = "0.1.0"

__all__ = [
    "KIND",
    "SCHEMA_VERSION",
    "ASTArtifact",
    "TopModuleError",
    "__version__",
    "ast_to_dict",
    "find_modules",
    "find_top_module",
    "flatten",
    "parse_verilog",
    "parse_verilog_text",
    "rename_module",
    "strip_comments",
]
