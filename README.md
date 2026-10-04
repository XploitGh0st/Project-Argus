# Project Argus

Hardware-security analysis pipeline: **Verilog → AST → CFG + DFG → dependency digraph → GNN**
(Hardware Trojan detection and IP-piracy detection).

The code base is split into **five fully decoupled feature modules** and one
**integration controller**. Every module can be built, run and tested on its own
using bundled mock inputs.

## Branch map

| Branch | Folder | Python package | Responsibility |
|---|---|---|---|
| `feature/verilog-ast-parser` | `modules/verilog_ast_parser/` | `argus_ast` | Verilog ingestion, preprocessing, AST generation |
| `feature/ast-to-cfg` | `modules/ast_to_cfg/` | `argus_cfg` | AST → Control Flow Graph (FSM state machines) |
| `feature/ast-to-dfg` | `modules/ast_to_dfg/` | `argus_dfg` | AST → Data Flow Graph |
| `feature/dependency-digraph` | `modules/dependency_digraph/` | `argus_digraph` | CFG + DFG → unified digraph with control-dependency edges |
| `feature/argus-ml` | `modules/argus_ml/` | `argus_ml` | GNN models, training, inference |
| `core/argus-controller` | `controller/` | `argus_controller` | Master integration branch / orchestrator |

### Rules

1. **Zero cross-dependency.** A `feature/*` branch only ever touches its own
   `modules/<name>/` folder and never imports another module. The only thing
   modules share is a *written contract* (`CONTRACT.md` + JSON schema), never code.
2. **Mock-first.** Each module ships `fixtures/` (stub inputs + golden outputs).
   `pytest` for a module needs **nothing** from any other module.
3. **Only the controller integrates.** `core/argus-controller` is the single place
   where modules are imported together.

### Naming conventions

- Branches: lowercase kebab-case with a `feature/` or `core/` prefix.
- Python packages: `snake_case`, `argus_` prefix. Module folders: `snake_case`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(ast-to-cfg): ...`, `fix(...)`, `chore: ...`, `docs: ...`, `test: ...`).

## Tooling (every module)

`pyproject.toml` (PEP 621, hatchling) · `src/` layout · full type hints (`py.typed`)
· `ruff` lint/format · `pytest` · Python ≥ 3.10. CI tests each module in isolation
(`.github/workflows/ci.yml`).

```powershell
# work on a single module, with nothing else installed
pip install -e "modules/ast_to_cfg[test]"
pytest modules/ast_to_cfg/tests
```

## Data contracts

| Edge | Producer → Consumer | Format |
|---|---|---|
| AST | `argus_ast` → `argus_cfg`, `argus_dfg` | `ast.pkl` (pickled pyverilog `Source`) + readable `ast.json` |
| CFG | `argus_cfg` → `argus_digraph` | JSON, `schema_version: "1.0"`, `kind: "cfg"` |
| DFG | `argus_dfg` → `argus_digraph` | JSON, `schema_version: "1.0"`, `kind: "dfg"` |
| Digraph | `argus_digraph` → `argus_ml` | networkx node-link JSON, `kind: "digraph"` |

Each module's `CONTRACT.md` is the authoritative schema.

## Controller integration specification

`core/argus-controller` imports, builds and wires all modules into one pipeline.
See [`controller/README.md`](controller/README.md) once that branch is merged.

```text
.v files ─► ParseStage ─┬─► CFGStage ─┐
                        │             ├─► DigraphStage ─► MLStage ─► report.json
                        └─► DFGStage ─┘
```

- **Install:** `pip install -e controller/` installs all five modules (editable).
- **Stages** are thin adapters that call each module's single public function
  (`parse_verilog`, `build_cfg`, `build_dfg`, `merge_graphs`, `predict`).
- **CFG and DFG run in parallel** (they are independent).
- **Contract checks** validate `schema_version` at every stage boundary.
- **Caching** by content hash lets you resume (`--from dfg --until digraph`).
- **Batch mode** isolates failures per design and records them in the report.

## Legacy

The original monolith (`DFG_CFG_Computation.py`) and the vendored `hw2vec`
package were replaced by this structure. They remain in git history
(commit `1060914` and earlier).
