# argus-ast - Verilog ingestion & AST generation

Branch: `feature/verilog-ast-parser` &nbsp;|&nbsp; Package: `argus_ast`

Turns Verilog source into a pyverilog AST and a versioned artifact that the
CFG/DFG modules consume. **No other Argus module is needed** to build or test it,
and it does **not** require Icarus Verilog (it uses pyverilog's grammar directly).

## Install & test (standalone)

```powershell
pip install -e ".[test]"
pytest
```

## Use

```python
from argus_ast import parse_verilog

art = parse_verilog(["design.v", "alu.v"])         # top module auto-detected
art = parse_verilog("design.v", top_module="soc", rename_top_to="top")
art.save("out/")                                    # ast.pkl (lossless) + ast.json (readable)
```

```powershell
python -m argus_ast design.v alu.v --top soc --out out/
```

## What it replaces (from the legacy monolith)

| Legacy | Now | Fix |
|---|---|---|
| `PreprocessVerilog.remove_comments` | `strip_comments` | handled `//` only via first-`/` cut, destroying `a / b`; now real `//` and `/* */`, strings kept |
| `PreprocessVerilog.rename_top_module` | `find_top_module` / `rename_module` | substring replace clobbered `alu_core`; now word-boundary |
| `PreprocessVerilog.flatten` | `flatten` | wrote next to `__file__`; now returns text |
| `AstGenerator` / `_generate_ast_dict` | `ASTArtifact.to_dict` | whitelist raised on unknown nodes; now every node type |

## Mock / stub data (this branch needs none upstream)

Inputs are plain Verilog in [`fixtures/verilog/`](fixtures/verilog):
`counter.v`, `fsm_traffic_light.v`, `trojan_trigger.v`, `multi_file/`.

## Contract

See [`CONTRACT.md`](CONTRACT.md).
