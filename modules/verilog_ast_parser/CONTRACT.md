# CONTRACT: AST (schema 1.0)

This module **produces** the AST artifact. It consumes only raw Verilog.

## Output

`parse_verilog(...) -> ASTArtifact`

| Field | Type | Meaning |
|---|---|---|
| `top_module` | `str` | name of the top module (after optional rename) |
| `source_files` | `tuple[str, ...]` | provenance |
| `ast` | `pyverilog.vparser.ast.Source` | the lossless AST root |

### On disk (`artifact.save(dir)`)

* `ast.pkl` - **the interchange format**. A pickled dict:
  `{"schema_version": "1.0", "kind": "ast", "top_module": str, "source_files": [str], "ast": Source}`.
  Downstream modules load it with `pickle.load` (trusted input only) and must check
  `kind == "ast"` and `schema_version == "1.0"`.
* `ast.json` - human-readable view, **not** an interchange format:
  `{"schema_version", "kind", "top_module", "source_files", "ast": {"type", "attrs", "lineno", "children": [...]}}`.

## Versioning

Any breaking change to the pickle payload bumps `schema_version` (major). Consumers
must reject unknown versions.
