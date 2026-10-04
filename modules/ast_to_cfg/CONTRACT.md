# CONTRACTS: AST (input) and CFG (output), schema 1.0

## Input - AST (`ast.pkl`, produced by `argus_ast`)

A pickled dict (trusted input only):

```text
{"schema_version": "1.0", "kind": "ast", "top_module": str,
 "source_files": [str], "ast": <pyverilog.vparser.ast.Source>}
```

`build_cfg` also accepts a bare pyverilog `Source` node plus `top_module`. It rejects
any other `kind` or `schema_version`. This module re-implements the check locally; it
does **not** import `argus_ast`.

## Output - CFG document

```json
{
  "schema_version": "1.0",
  "kind": "cfg",
  "top_module": "traffic",
  "fsms": [
    {
      "signal": "traffic.state",
      "delay_cnt": 0,
      "states": ["0", "1", "2"],
      "transitions": [
        {"src": "0", "dst": "1", "condition": "(traffic_start>'d0)"},
        {"src": "1", "dst": "2", "condition": null}
      ]
    }
  ]
}
```

| Field | Rule |
|---|---|
| `fsms[].signal` | hierarchical register name, e.g. `traffic.state`. **Join key** used by the digraph module to find the same signal in the DFG |
| `fsms[].states` | state labels as strings, numerically sorted |
| `transitions[].src` | a declared state, or `"*"` meaning "from every state" |
| `transitions[].dst` | always a declared state (never an edge label) |
| `transitions[].condition` | `null` = unconditional, else a readable expression (flattened names, `.` becomes `_`) |

Designs with no detected FSM produce `"fsms": []` (valid).

`argus_cfg.validate_cfg_document(doc)` enforces all of the above; consumers should call
an equivalent check at their boundary.

## Graph view

`CFGResult.to_networkx()` -> `DiGraph`. Node id `"{signal}::state::{state}"`
(`kind="fsm_state"`); edges `kind="control_flow"` with `condition`.

## Versioning

A breaking change bumps `schema_version`'s major number; consumers must reject
versions they do not know.
