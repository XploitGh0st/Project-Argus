# Dependency Digraph Contract (Schema 1.0)

## Input Contract

Accepts `cfg.json` and `dfg.json` conforming to their respective 1.0 schemas. Both must correspond to the same `top_module`.

## Output Contract

Emits `digraph.json`. This merges data flow and control flow dependencies into a single networkx-compatible graph structure.

### Graph Rules
* Merges all nodes from the DFG.
* Adds `fsm_state` nodes representing FSM states.
* Adds control dependency edges (`control_dep`) connecting:
  * FSM state registers to signal assignments that occur in conditionals relying on them.
  * FSM state registers to their specific FSM states.
* Adds FSM transitions (`fsm_trans`).

### Edge Kinds
* `dataflow`: pure dataflow (inherited from DFG).
* `control_dep`: Control dependency from FSM logic.
* `fsm_trans`: State transition between FSM states.
