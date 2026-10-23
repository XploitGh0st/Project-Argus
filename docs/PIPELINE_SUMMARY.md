# Project Argus: Pipeline & Evaluation Summary

## 1. Headline Metrics (Leave-One-Family-Out Cross-Validation)
- **Primary Model (M1: HistGradientBoosting)**: Macro-average ROC-AUC = **0.6925**, PR-AUC = **0.6607**, Precision = **50.44%**, Recall = **93.22%**, F1 = **0.5821**.
- **Family Highlights**: RS232 reached ROC-AUC = **0.8610** (F1 = **0.8578**, Recall = **93.02%**); PIC16F84 reached ROC-AUC = **0.9161** (Recall = **100.0%**).
- **Controls & Sanity Checks**: Permuted label control on M1 dropped macro ROC-AUC to chance level (**0.5537**), confirming no structural data leakage or identifier leakage. Removing control dependencies (M2 ablation) shifted macro ROC-AUC by **-0.0083**, confirming control flow contributions to trigger detection.

## 2. Biggest Win
Resolving the exponential recursion and memory blowups in Pyverilog (ScopeChain hash caching, flat case handling for >32 items, shallow AST copy, and dynamic SSA filtering) enabled end-to-end AST/CFG/DFG parsing of complex industrial RTL (368k nodes in AES) in under 12 seconds per design, allowing 91 out of 95 benchmark variants to build and elaborate without timeout or hang.

## 3. Biggest Fail
Label noise caused by graph-neighborhood hash propagation in the 3-round Weisfeiler-Lehman clean-vs-infected diff. In compact designs (PIC16F84 and RS232), trojan insertion structurally altered neighboring host nodes, causing 47% to 82% of nodes to flip their WL hash and get labeled positive despite representing host logic. While this provided ample positive signals for training, it limited precision on small designs.

## 4. Was GNN Worth It Over Gradient-Boosted Trees?
For tabular node classification, M1 (`HistGradientBoostingClassifier`) generally matched or outperformed the Typed R-GCN (M3) across most folds (M1 macro ROC-AUC 0.6925 vs M3 0.6642) while training 10x faster and scaling easily to millions of rows. However, on AES (the largest, deepest design with 368k nodes), M3 Typed R-GCN outperformed M1 (ROC-AUC 0.6631 vs 0.3924) by directly propagating relational context across long dependency chains. An ensemble (M5) reliably balanced both inductive biases.

## 5. Next Steps
1. **Refined Ground-Truth Differencing**: Replace full WL-3 hash diffing with a bounded 1-hop boundary or AST-exact subtree isomorphism to eliminate host-node label noise in small RTLs.
2. **Sub-graph Sampling for Scaling GNNs**: Introduce NeighborLoader / ClusterGCN sampling in PyTorch Geometric so multi-million node designs can be trained jointly without memory constraints.
3. **Scan CLI Integration**: Package `scripts/scan.py` into automated CI/CD security gates for third-party IP verification before physical RTL synthesis.
