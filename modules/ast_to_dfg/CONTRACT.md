# DFG Contract (Schema 1.0)

## Input Contract

This module accepts the Argus AST artifact (`ast.pkl`). It reads the `Source` node and generates a Data Flow Graph by walking the PyVerilog dataflow structures.

## Output Contract

The DFG is emitted as `dfg.json`. It encodes a directed multi-graph tracking the data flow of the design.

### Graph Rules
* **Nodes**: Represent either signals (nets, registers, ports) or operators/constants.
* **Edges**: Directed from operand (source) to operator (destination), matching the flow of data.
* **`sequential` attribute**: True if the edge terminates into a register updated on a clock edge (non-blocking assignments).

### Node Types
* `signal`: Verilog nets/registers.
* `const`: Constants and eval values.
* `operator`: Unary and binary operations (e.g. `+`, `&`, `~`).
* `branch`: Conditionals. `COND` -> branch, `TRUE` -> branch, `FALSE` -> branch.
* `concat`: Concatenation of signals.
* `partselect`: Selecting bits from a signal.
* `pointer`: Pointer dereference into an array.
* `delay`: Delay operator.
* `syscall`: System calls like `$display`.
* `undefined`: Undefined nodes.
* `highz`: High impedance nodes.
* `unknown`: Unknown/Unsupported PyVerilog nodes.

### Schema: `dfg.json`

```json
{
  "schema_version": "1.0",
  "kind": "dfg",
  "top_module": "top",
  "nodes": [
    {
      "id": "top.clk",
      "type": "signal",
      "label": "clk",
      "termtype": ["Input"]
    },
    {
      "id": "op_1",
      "type": "operator",
      "label": "And"
    }
  ],
  "edges": [
    {
      "src": "top.clk",
      "dst": "op_1",
      "label": "0",
      "sequential": false
    },
    {
      "src": "op_1",
      "dst": "top.out",
      "label": "assign",
      "sequential": true
    }
  ]
}
```
