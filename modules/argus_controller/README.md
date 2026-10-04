# Argus Controller

This module is the master integration branch orchestrating the entire Argus pipeline.

## Integration Specification

The pipeline relies on strictly decoupled feature branches. Once all branches are merged into `core/argus-controller`, the pipeline can be executed end-to-end.

### Dependencies

To run the pipeline, the master environment must have the following dependencies installed from the respective sub-modules:
- `pyverilog` (from `feature/verilog-ast-parser`)
- `networkx` (from `feature/ast-to-cfg`, `feature/ast-to-dfg`, `feature/dependency-digraph`)
- `torch`, `torch_geometric` (from `feature/argus-ml`)

### Orchestration Strategy

1. **AST Extraction**: The controller invokes `argus_ast.parser.parse_file` to ingest raw Verilog RTL. This produces an `ast.pkl` (in-memory `AstResult`).
2. **CFG & DFG Construction**: The controller passes the identical in-memory AST sequentially to `argus_cfg.extract.build_cfg` and `argus_dfg.extract.build_dfg`.
3. **Dependency Digraph Generation**: The controller pipes the output JSON dictionaries of the CFG and DFG extractions into `argus_digraph.merge.merge_graphs`.
4. **Machine Learning Pipeline**: The unified `digraph.json` is handed off to `argus_ml.models.GraphEncoder` as a `torch_geometric.data.Data` object via custom Dataset wrappers, kicking off Graph Neural Network inference.

### Zero Cross-Dependency Testing

The pipeline guarantees that any module can be modified independently:
- **CFG/DFG** use stub Python scripts (`fixtures/ast_builders.py`) to construct PyVerilog nodes purely in memory for testing, simulating the output of `argus_ast` without parsing any Verilog.
- **Digraph** merges JSON inputs directly, easily testable by providing mock JSON graphs.
- **ML** expects JSON-to-Data objects, agnostic to how the graphs were actually generated.
