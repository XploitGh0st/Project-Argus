# Pipeline Status

## Phase 0 — Verify ✅
- Branch: version1-trojan-pipeline
- Modules found: verilog_ast_parser, ast_to_cfg, ast_to_dfg, dependency_digraph, argus_ml, argus_controller
- merge.py fixed: generic control_dep logic (was FSM-only). trojan_trigger.v now gives 3 control_dep edges (spec said 4; decision: 3 is correct — trojan_active is a wire traced through, not a root)
- Inventory: 62 total, 27 AES ok, 14 RS232 ok, 4 PIC16F84 ok, 2 wb_conmax ok, 1 memctrl no_clean, 14 VHDL excluded
- .gitignore updated for TrustHub/ and .cache/

## Phase 1 — Preprocessing & Build
- Status: IN PROGRESS
