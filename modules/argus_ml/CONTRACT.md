# Argus ML Contract

## Input Contract

This module accepts `digraph.json` (the Dependency Digraph) and transforms it into a PyTorch Geometric `Data` object to feed into the graph neural networks.
It does not depend on any specific parsing engine (e.g., PyVerilog or Yosys).

## Output Contract

Emits classification probabilities (Trojan likelihood) or node embeddings for hardware intellectual property analysis.
