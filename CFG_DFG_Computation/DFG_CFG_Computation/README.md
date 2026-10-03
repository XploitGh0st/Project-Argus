# 🔬 DFG / CFG / AST Computation — Project Argus

> **Part of Project Argus** — A hardware security analysis framework that converts RTL (Register Transfer Level) Verilog designs into structured graphs for downstream machine learning tasks (e.g., Hardware Trojan detection via GNNs).

---

## 📖 Table of Contents

- [Overview](#overview)
- [What Are DFG, CFG, and AST?](#what-are-dfg-cfg-and-ast)
- [Architecture](#architecture)
- [Class Reference](#class-reference)
  - [PreprocessVerilog](#1-preprocessverilog)
  - [VerilogParser](#2-verilogparser)
  - [DfgGenerator](#3-dfggenerator)
  - [CfgGenerator](#4-cfggenerator)
  - [AstGenerator](#5-astgenerator)
  - [JsonGraphParser](#6-jsongraphparser)
- [Pipeline Walkthrough (End-to-End)](#pipeline-walkthrough-end-to-end)
- [Output File Formats](#output-file-formats)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)

---

## Overview

This module takes raw **Verilog hardware description files** (`.v`) and extracts three types of structural graphs:

| Graph Type | Full Name | What it captures |
|---|---|---|
| **DFG** | Data Flow Graph | How data/signals flow between hardware components |
| **CFG** | Control Flow Graph | Which states a finite state machine (FSM) can transition between |
| **AST** | Abstract Syntax Tree | The full syntactic parse tree of the Verilog source |

These graphs are serialized to **JSON** and can then be fed into a Graph Neural Network (GNN) pipeline for hardware security analysis.

---

## What Are DFG, CFG, and AST?

### 🔀 Data Flow Graph (DFG)

Think of a DFG as a map that answers: **"where does data come from and where does it go?"**

```
Input A ──┐
           ├──► Adder ──► Output Z
Input B ──┘
```

In hardware, every signal (wire, register) is a node. Edges show how signals depend on each other.

### 🔁 Control Flow Graph (CFG)

A CFG answers: **"what are the possible execution paths / state transitions?"**

It is built from **Finite State Machines (FSMs)** inside the Verilog design:

```
IDLE ──► FETCH ──► DECODE ──► EXECUTE ──► WRITEBACK
          ↑                                    │
          └────────────────────────────────────┘
```

### 🌲 Abstract Syntax Tree (AST)

An AST is the **parse tree of the source code itself**. Every keyword, operator, and identifier becomes a node:

```
ModuleDef
  ├── Portlist
  │     ├── Input(clk)
  │     └── Output(out)
  └── Always
        └── Substitution
              ├── Lvalue(out)
              └── Rvalue(in + 1)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DFG_CFG_Computation.py                        │
│                                                                     │
│  ┌──────────────────┐     ┌────────────────────────────────────┐   │
│  │ PreprocessVerilog│     │          VerilogParser              │   │
│  │                  │     │  (wraps PyVerilog internals)        │   │
│  │ - remove_comments│     │                                    │   │
│  │ - rename_top_mod │     │  - dataflow analysis               │   │
│  │ - flatten        │────►│  - controlflow analysis            │   │
│  └──────────────────┘     │  - AST parsing                     │   │
│                            └───────┬────────────────────────────┘   │
│                                    │                                 │
│              ┌─────────────────────┼──────────────────┐             │
│              ▼                     ▼                  ▼             │
│       ┌─────────────┐     ┌──────────────┐    ┌─────────────┐      │
│       │ DfgGenerator│     │ CfgGenerator │    │ AstGenerator│      │
│       │             │     │              │    │             │      │
│       │ topModule   │     │ topModule    │    │ ast.json    │      │
│       │   .json     │     │   .json      │    │             │      │
│       └─────────────┘     └──────────────┘    └─────────────┘      │
│                                    │                                 │
│                                    ▼                                 │
│                          ┌──────────────────┐                       │
│                          │  JsonGraphParser  │                       │
│                          │  (reads JSON →    │                       │
│                          │   nx.DiGraph for  │                       │
│                          │   GNN training)   │                       │
│                          └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Class Reference

### 1. `PreprocessVerilog`

**Purpose:** Cleans and normalizes raw Verilog files before parsing. This step is important because PyVerilog is strict about file format.

| Method | What it does |
|---|---|
| `remove_comments(input_path, target_path)` | Strips all inline comments (everything after `/`) |
| `rename_top_module(input_path, target_path)` | Finds the top-level module (the one only defined, never instantiated) and renames it to `"top"` |
| `flatten(input_path, target_path, top_module)` | Merges all `.v` files in a directory into a single `topModule.v` |

#### Example — Preprocessing a multi-file design

```python
from DFG_CFG_Computation import PreprocessVerilog

pre = PreprocessVerilog(code_language="verilog")

# Step 1: Flatten all .v files into one
pre.flatten(
    input_path="./my_design/",
    output_path="./cleaned/",
    top_module="my_top"
)

# Step 2: Rename the top module to "top" (required by PyVerilog)
PreprocessVerilog.rename_top_module(
    input_path="./cleaned/topModule.v",
    target_path="./cleaned/topModule_renamed.v"
)
```

---

### 2. `VerilogParser`

**Purpose:** The core engine. Directly wraps PyVerilog to perform dataflow analysis, controlflow analysis, or AST parsing. All three generators (`DfgGenerator`, `CfgGenerator`, `AstGenerator`) use this class internally — you rarely need to instantiate it directly.

**Constructor signature:**
```python
VerilogParser(
    verilog_file,     # path to .v file
    output_directory, # where to save results
    top_module,       # name of the top module (usually "top")
    generate_cfg,     # True → set up for CFG extraction
    generate_ast      # True → set up for AST extraction
    # (if both are False → set up for DFG extraction)
)
```

**Key internal methods:**

| Method | Description |
|---|---|
| `graph_separate_modules()` | Generates one sub-graph per signal in the DFG |
| `merge_graphs()` | Merges duplicate nodes across sub-graphs (e.g. `top_clk` → `top.clk`) |
| `export_dfg_graph(output)` | Exports DFG as JSON (`'graph'`, `'nodes'`, `'edges'`, `'roots'`) |
| `generate_dot_file()` | Generates a `.dot` file for each FSM in the design |
| `export_cfg_graph(output)` | Exports CFG from the dot file as JSON |
| `_generate_ast_dict(ast_node)` | Recursively converts the PyVerilog AST to a nested Python dict |
| `cleanup_files()` | Removes temporary files (`file.dot`, `parser.out`, etc.) |

---

### 3. `DfgGenerator`

**Purpose:** High-level, easy-to-use API for generating a **Data Flow Graph** from a Verilog file.

**Constructor:**
```python
DfgGenerator(
    verilog_file,              # path to .v file
    output_path,               # output directory (will be created if missing)
    code_language='verilog',   # currently only 'verilog' is supported
    top_module='top',          # top module name
    draw_graph=False           # if True, also save a PDF of the graph
)
```

When constructed, it **automatically** runs the full DFG generation pipeline:
1. Analyzes dataflow with PyVerilog
2. Generates per-signal subgraphs
3. Merges duplicate/related nodes
4. Exports `topModule.json`

#### Example — Generate a DFG

```python
from DFG_CFG_Computation import DfgGenerator

gen = DfgGenerator(
    verilog_file="./cleaned/topModule.v",
    output_path="./output/dfg/",
    top_module="top"
)

# The file ./output/dfg/topModule.json is now created automatically.

# You can also export subsets:
gen.export_nodes()       # → all_nodes.json
gen.export_edges()       # → all_edges.json
gen.export_root_nodes()  # → root_nodes.json
```

**What `topModule.json` looks like:**
```json
{
    "top.out": [
        ["\\N", "top.a"],
        ["\\N", "top.b"]
    ],
    "top.a": [],
    "top.b": []
}
```
Each key is a **node** (signal name). Its value is a list of `[edge_label, destination_node]` pairs — i.e. its **outgoing edges**.

---

### 4. `CfgGenerator`

**Purpose:** High-level API for generating a **Control Flow Graph** from a Verilog file. Specifically targets FSMs in the design.

**Constructor:**
```python
CfgGenerator(
    verilog_file,
    output_path,
    code_language="verilog",
    top_module="top",
    draw_graph=False
)
```

**Pipeline (runs automatically on construction):**
1. Analyzes controlflow with PyVerilog
2. Finds all FSMs (signals named `fsm`, `state`, `count`, `cnt`, `step`, `mode`)
3. Generates `.dot` files for each FSM
4. Converts `.dot` → `topModule.json`
5. Cleans up temporary files

#### Example — Generate a CFG

```python
from DFG_CFG_Computation import CfgGenerator

cfg_gen = CfgGenerator(
    verilog_file="./cleaned/topModule.v",
    output_path="./output/cfg/",
    top_module="top"
)

# topModule.json is created automatically.
# Export subsets if needed:
cfg_gen.export_nodes()
cfg_gen.export_edges()
```

**What `topModule.json` looks like for a CFG:**
```json
{
    "IDLE":   ["FETCH"],
    "FETCH":  ["DECODE"],
    "DECODE": ["EXECUTE"],
    "EXECUTE":["WRITEBACK"],
    "WRITEBACK":["FETCH"]
}
```
Each key is a **state node**. Its value is the list of states it can transition to.

---

### 5. `AstGenerator`

**Purpose:** Parses a Verilog file and exports its **Abstract Syntax Tree** as a JSON file.

**Constructor:**
```python
AstGenerator(
    verilog_file,  # path to .v file
    output         # output directory
)
```

#### Example — Generate an AST

```python
from DFG_CFG_Computation import AstGenerator

ast_gen = AstGenerator(
    verilog_file="./cleaned/topModule.v",
    output="./output/ast/"
)

ast_gen.generate_ast_json()  # → saves ast.json
```

**What `ast.json` looks like:**
```json
{
  "Source": {
    "Description": {
      "ModuleDef": [
        { "Identifier": "top" },
        { "Portlist": [
            { "Input": [ { "Identifier": "clk" } ] },
            { "Output": [ { "Identifier": "out" } ] }
        ]},
        { "Always": [
            { "Substitution": [
                { "Lvalue": { "Identifier": "out" } },
                { "Rvalue": { "Plus": [
                    { "Identifier": "in" },
                    { "IntConst": "1" }
                ]}}
            ]}
        ]}
      ]
    }
  }
}
```

---

### 6. `JsonGraphParser`

**Purpose:** Reads the generated `topModule.json` files back in as **NetworkX directed graphs** (`nx.DiGraph`) for use in GNN training/evaluation pipelines.

This class is used **after** the graph files have been generated. It handles:
- Loading and organizing graphs into `train` / `test` splits
- Encoding node labels as integer indices for GNN input
- Pairing graphs for similarity-based tasks (e.g. GNN4IP)

#### Key methods

| Method | Description |
|---|---|
| `read_node_labels(key)` | Scans all `topModule.json` files to build the `label→index` vocabulary |
| `get_graph_from_json(json_path)` | Loads a single JSON file into an `nx.DiGraph` |
| `get_graphs()` | Returns `(train_graphs, test_graphs)` split |
| `get_pairs()` | Returns graph-pair tuples for pairwise training |
| `split_dataset(ratio, seed, dataset)` | Stratified train/test split |

#### Example — Load graphs for GNN training

```python
from DFG_CFG_Computation import JsonGraphParser
from pathlib import Path

# Assume cfg object has attributes: raw_dataset_path, ratio, seed, etc.
parser = JsonGraphParser(cfg)

# Build the node label vocabulary by scanning the dataset
parser.read_node_labels(key="train")

# Load a single graph
import json
with open("./output/dfg/topModule.json") as f:
    graph = parser.get_graph_from_json("./output/dfg/topModule.json")

print(graph.nodes(data=True))
# NodeDataView({'top.out': {'x': 3, 'label': 'out'}, ...})

print(graph.edges())
# OutEdgeView([('top.out', 'top.a'), ('top.out', 'top.b')])
```

---

## Pipeline Walkthrough (End-to-End)

Here is the full sequence from raw Verilog to GNN-ready graphs:

```
Raw .v files
     │
     ▼
┌────────────────────────────────────┐
│  Step 1: PreprocessVerilog         │
│  - flatten() → merge all .v files  │
│  - rename_top_module()             │
│  - remove_comments()               │
└────────────────────┬───────────────┘
                     │
                     ▼
              topModule.v (clean)
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │   DFG    │ │   CFG    │ │   AST    │
  │Generator │ │Generator │ │Generator │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       │             │             │
       ▼             ▼             ▼
  topModule    topModule       ast.json
    .json        .json
       │             │
       └──────┬──────┘
              ▼
     ┌─────────────────┐
     │ JsonGraphParser  │
     │  nx.DiGraph ×N   │
     └────────┬─────────┘
              ▼
     GNN Training Pipeline
```

---

## Output File Formats

| File | Generator | Description |
|---|---|---|
| `topModule.json` | DFG / CFG | Adjacency list: `{node: [[label, neighbor], ...]}` |
| `all_nodes.json` | DFG / CFG | Flat list of all node names |
| `all_edges.json` | DFG / CFG | Flat list of `[src, dst, label]` triples |
| `root_nodes.json` | DFG / CFG | Nodes with no incoming edges (graph roots) |
| `ast.json` | AST | Nested dictionary mirroring the Verilog parse tree |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/XploitGh0st/Project-Argus.git
cd Project-Argus/CFG_DFG_Computation/DFG_CFG_Computation
```

### 2. Create and activate a virtual environment

```bash
python -m venv env

# On Windows:
env\Scripts\activate

# On Linux/macOS:
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE]
> PyVerilog requires **Icarus Verilog** (`iverilog`) to be installed and available on your system `PATH` for some parsing features. Download it from [http://iverilog.icarus.com](http://iverilog.icarus.com).

> [!TIP]
> The `argus_ml` GNN engine is included in the `argus_ml/` directory — a purpose-built replacement for the abandoned `hw2vec` package. No external GNN library install is needed beyond `torch` and `torch_geometric`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pyverilog` | latest | Verilog parsing, DFG/CFG extraction |
| `networkx` | ≥ 3.3 | In-memory graph representation |
| `pydot` | latest | Reading `.dot` files for CFG |
| `scikit-learn` | ≥ 1.5.0 | Train/test splitting, metrics |
| `torch` | ≥ 2.0.0 | Tensor operations, neural network backbone |
| `torch_geometric` | ≥ 2.4.0 | Graph convolution layers (GCN, GIN), pooling |
| `matplotlib` | ≥ 3.9.0 | Graph visualization |
| `numpy` | ≥ 2.0.0 | Numerical operations |
| `tqdm` | ≥ 4.66.0 | Progress bars |
| `pyyaml` | ≥ 6.0.1 | YAML config loading |
| `scipy` | ≥ 1.14.0 | Scientific computing utilities |

---

## Project Structure

```
DFG_CFG_Computation/
├── DFG_CFG_Computation.py     ← Main module (Verilog → Graph pipeline)
├── requirements.txt            ← Python dependencies
├── DFG_CFG_Computation.pyproj  ← Visual Studio project file
├── argus_ml/                   ← GNN engine for hardware security analysis
│   ├── __init__.py             ← Public API: GraphEncoder, TrojanTrainer, ...
│   ├── models.py               ← GraphEncoder (GCN/GIN, SAGPool, embeddings)
│   ├── trainers.py             ← TrojanTrainer, SimilarityTrainer
│   ├── metrics.py              ← MetricsResult, compute_binary_metrics
│   └── profiling.py            ← @profile_function, @profile_graph_extraction
├── env/                        ← Virtual environment (not committed)
└── README.md                   ← This file
```

---

## `argus_ml` API Reference

### `GraphEncoder` — GNN Model

Configurable graph neural network that converts hardware circuit graphs into
fixed-dimensional vector embeddings.

```python
from argus_ml import GraphEncoder

config = {
    "num_features": 37,          # Number of distinct DFG node types
    "hidden_dim": 64,            # Internal feature dimension
    "output_dim": 2,             # Classes: Trojan-Free / Trojan-Infected
    "num_layers": 3,             # Graph convolution depth
    "conv_type": "gin",          # "gcn" or "gin" (GIN recommended for Trojan detection)
    "pool_type": "sagpool",      # "sagpool" or "topkpool"
    "pool_ratio": 0.5,
    "readout": "max",            # "max", "mean", or "add"
    "dropout": 0.3,
    "node_encoding": "embedding",  # "embedding" (learnable) or "one_hot"
    "device": "cuda",
}
model = GraphEncoder(config).to("cuda")

# Save / Load
model.save("checkpoints/my_model.pt")
model = GraphEncoder.load("checkpoints/my_model.pt", device="cuda")
```

### `TrojanTrainer` — Hardware Trojan Detection

Binary graph classification: labels circuit designs as Trojan-Free or Trojan-Infected.

```python
from argus_ml import TrojanTrainer

trainer = TrojanTrainer(model, {
    "device": "cuda",
    "epochs": 200,
    "learning_rate": 1e-3,
    "eval_every": 10,
    "patience": 30,           # Early stopping
    "save_dir": "checkpoints/",
})

best_metrics = trainer.fit(train_loader, val_loader)
labels, preds, attn_maps = trainer.predict(test_loader)
```

### `SimilarityTrainer` — IP Piracy Detection

Pairwise graph similarity via cosine embedding loss.

```python
from argus_ml import SimilarityTrainer

trainer = SimilarityTrainer(model, {
    "device": "cuda",
    "epochs": 200,
    "similarity_margin": 0.5,
})

best_metrics = trainer.fit(pair_train_loader, pair_val_loader)
labels, scores, preds = trainer.predict(pair_test_loader)
```

---

> **Part of Project Argus** — Hardware security analysis using Graph Neural Networks.
