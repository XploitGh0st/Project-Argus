# argus_dfg

Converts a pyverilog AST into a structural Data Flow Graph (DFG).

This module strictly adheres to the zero cross-dependency rule. It requires only standard PyVerilog structures (pickled AST) to operate and depends on no other Project Argus sub-modules. It replaces the legacy PyGraphviz and dot-file output approach by generating clean, robust JSON graphs.

## Quickstart

```bash
# Extract DFG from a Pickled AST
python -m argus_dfg path/to/ast.pkl --out ./out/
```

This will produce `./out/dfg.json`.

## Testing & Development

Run the test suite via hatchling and pytest:

```bash
# Run tests
pytest modules/ast_to_dfg/tests

# Lint and fix
ruff check modules/ast_to_dfg --fix
```

## Contracts

Read `CONTRACT.md` for the I/O specification.
