# argus-cfg - AST to Control Flow Graph

Branch: `feature/ast-to-cfg` &nbsp;|&nbsp; Package: `argus_cfg`

Finds the finite state machines in a design and returns their states and transitions.
**It does not need `verilog-ast-parser`**: feed it a stub AST (below) and test it alone.

## Install & test (standalone)

```powershell
pip install -e ".[test]"
pytest
```

## Use

```python
from argus_cfg import build_cfg

result = build_cfg("out/ast.pkl")                 # ast.pkl from argus_ast ...
result = build_cfg(ast_node, top_module="top")    # ... or any pyverilog Source node
result.save("out/")                                # cfg.json
graph = result.to_networkx()
```

```powershell
python -m argus_cfg out/ast.pkl --out out/
```

## Testing without the parser (mock inputs)

You have three ways to get an AST, none of which touch `argus_ast`:

1. **Python stub builders** - [`fixtures/ast_builders.py`](fixtures/ast_builders.py) builds real
   pyverilog `Source` nodes directly:

   ```python
   import sys; sys.path.insert(0, "fixtures")
   from ast_builders import build_traffic_light, build_two_fsms, build_no_fsm
   from argus_cfg import build_cfg

   ast, top = build_traffic_light()
   print(build_cfg(ast, top).to_dict())
   ```

2. **Pre-pickled mock `ast.pkl`** - [`fixtures/pkl/`](fixtures/pkl) holds ready-made files:
   `python -m argus_cfg fixtures/pkl/traffic_light.ast.pkl`
3. **Your own stub** - build any AST with `pyverilog.vparser.ast` classes. `fsm_always(...)` in
   the builders module generates a reset+case FSM from a plain transition list:

   ```python
   fsm_always("state", 2, [(0, "go", 1), (1, None, 0)])   # (src, guard_signal|None, dst)
   ```

Golden outputs live in [`fixtures/expected/`](fixtures/expected); regenerate everything with
`python fixtures/make_golden.py`.

## Fixed versus the legacy `CfgGenerator`

| Legacy | Now |
|---|---|
| analyser hard-coded to module name `"top"` | uses the real top module |
| exported the edge **label** as the destination | destination is always a state |
| needed pygraphviz, wrote `./file.dot`, `top_state.png` | direct data conversion; no files, no pygraphviz |
| re-parsed the Verilog file | consumes the AST |

## Contract

See [`CONTRACT.md`](CONTRACT.md) (input AST, output CFG).
