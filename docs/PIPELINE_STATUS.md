# Pipeline Status

## Phase 0 — Verify ✅
- Branch: version1-trojan-pipeline
- Modules found: verilog_ast_parser, ast_to_cfg, ast_to_dfg, dependency_digraph, argus_ml, argus_controller
- merge.py fixed: generic control_dep logic (was FSM-only). trojan_trigger.v now gives 3 control_dep edges (spec said 4; decision: 3 is correct — trojan_active is a wire traced through, not a root)
- Inventory: 62 total, 27 AES ok, 14 RS232 ok, 4 PIC16F84 ok, 2 wb_conmax ok, 1 memctrl no_clean, 14 VHDL excluded
- .gitignore updated for TrustHub/ and .cache/

## Phase 1 — Preprocessing & Build ✅
- Pyverilog patches: fast ScopeChain/ScopeLabel hashing, shallow copy, flat case >32, searchTerminal scope slice, SSA/wide reg filter in isFsmVar.
- Normalization: `include resolution, implicit net declarations, empty port dummies, timescale/trailing comma stripping.
- Audit: 91/95 runs complete (24/27 AES infected + 2 clean, 4/4 PIC16F84, 14/14 RS232, 2/2 wb_conmax). 4 failed: AES-T400, T1600, T1700 (Bind dest empty), memctrl-T100 (DFEvalValue, no clean).
- Trojan elaboration: AES-T100 elaborated 2457 trojan-only signals across TSC, lfsr, top modules.
- Unit tests: 62 tests across argus_ast, argus_cfg, argus_dfg, dependency_digraph all PASS.
