#!/usr/bin/env python3
"""Audit: run the full pipeline on every usable design. Cache digraphs."""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules" / "verilog_ast_parser" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "ast_to_cfg" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "ast_to_dfg" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "dependency_digraph" / "src"))

TRUSTHUB = ROOT / "TrustHub - Dataset"
MANIFEST = ROOT / "benchmarks" / "manifest.json"
CACHE = ROOT / ".cache" / "digraphs"
RESULTS = ROOT / "results" / "audit.json"

TB_RE = re.compile(r"(^test|^tb|_tb$|_test$)", re.IGNORECASE)


def load_design_files(design_dir: Path) -> dict[str, str]:
    """Load .v files (excluding testbenches) from a directory."""
    text_by_file = {}
    for f in sorted(design_dir.rglob("*.v")):
        if not TB_RE.search(f.stem):
            text_by_file[str(f)] = f.read_text(encoding="utf-8", errors="replace")
    return text_by_file


def run_one(bench_id: str, variant: str, design_dir: Path, timeout: int = 300) -> dict:
    """Run full pipeline on one design variant."""
    from argus_ast.patch_pyverilog import apply_pyverilog_patches
    apply_pyverilog_patches()

    from argus_ast.preprocess import normalize_verilog
    from argus_ast import parse_verilog_text
    from argus_cfg import build_cfg
    from argus_dfg import build_dfg
    from argus_digraph.merge import merge_graphs

    result = {
        "id": bench_id,
        "variant": variant,
        "stage": "init",
        "nodes": 0,
        "edges": 0,
        "control_dep": 0,
        "seconds": 0.0,
        "error": "",
    }
    t0 = time.time()

    # Check cache
    cache_dir = CACHE / bench_id
    cache_file = cache_dir / f"{variant}.json"
    if cache_file.exists():
        try:
            g = json.loads(cache_file.read_text())
            result["stage"] = "cached"
            result["nodes"] = len(g["nodes"])
            result["edges"] = len(g["edges"])
            result["control_dep"] = sum(1 for e in g["edges"] if e.get("kind") == "control_dep")
            result["seconds"] = time.time() - t0
            return result
        except Exception:
            pass

    try:
        # Load files
        text_by_file = load_design_files(design_dir)
        if not text_by_file:
            result["error"] = "no .v files found"
            return result
        result["stage"] = "loaded"

        # Normalize
        norm, log = normalize_verilog(text_by_file, [str(design_dir)])
        result["stage"] = "normalized"

        # Parse
        ast = parse_verilog_text(norm)
        result["stage"] = "parsed"

        # CFG
        cfg = build_cfg(ast.ast, ast.top_module)
        result["stage"] = "cfg"

        # DFG
        dfg = build_dfg(ast.ast, ast.top_module)
        result["stage"] = "dfg"

        # Merge
        g = merge_graphs(cfg.to_dict(), dfg.to_dict())
        result["stage"] = "complete"
        result["nodes"] = len(g.nodes)
        result["edges"] = len(g.edges)
        result["control_dep"] = sum(1 for e in g.edges if e.kind == "control_dep")

        # Cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        gd = g.to_dict()
        cache_file.write_text(json.dumps(gd), encoding="utf-8")

    except Exception as e:
        result["error"] = str(e)[:200]
        result["seconds"] = time.time() - t0
        return result

    result["seconds"] = time.time() - t0
    return result


def main():
    manifest = json.loads(MANIFEST.read_text())

    results = []
    for b in manifest:
        if b["language"] != "Verilog":
            continue

        bench_id = b["id"]

        # Clean variant
        if b["clean_dir"]:
            clean_dir = TRUSTHUB / b["clean_dir"]
            r = run_one(bench_id, "clean", clean_dir)
            results.append(r)
            print(f"{bench_id:20s} clean   -> {r['stage']:12s} N={r['nodes']:5d} E={r['edges']:5d} ctrl={r['control_dep']:3d} {r['seconds']:.1f}s {r['error'][:60]}", flush=True)

        # Infected variant
        if b["infected_dir"]:
            inf_dir = TRUSTHUB / b["infected_dir"]
            r = run_one(bench_id, "infected", inf_dir)
            results.append(r)
            print(f"{bench_id:20s} infected-> {r['stage']:12s} N={r['nodes']:5d} E={r['edges']:5d} ctrl={r['control_dep']:3d} {r['seconds']:.1f}s {r['error'][:60]}", flush=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    complete = sum(1 for r in results if r["stage"] in ("complete", "cached"))
    failed = [r for r in results if r["stage"] not in ("complete", "cached")]
    print(f"\n=== SUMMARY ===")
    print(f"Total runs: {len(results)}, Complete: {complete}, Failed: {len(failed)}")
    for r in failed:
        print(f"  FAIL: {r['id']} {r['variant']} at {r['stage']}: {r['error'][:100]}")

    # Check trojan submodules for AES-T100
    aes_cache = CACHE / "AES-T100" / "infected.json"
    if aes_cache.exists():
        g = json.loads(aes_cache.read_text())
        # Files only in TjIn but not TjFree
        tjin_files = set()
        tjfree_files = set()
        tjin_dir = TRUSTHUB / "AES-T100" / "src" / "TjIn"
        tjfree_dir = TRUSTHUB / "AES-T100" / "src" / "TjFree"
        for f in tjin_dir.glob("*.v"):
            if not TB_RE.match(f.stem):
                tjin_files.add(f.stem)
        for f in tjfree_dir.glob("*.v"):
            if not TB_RE.match(f.stem):
                tjfree_files.add(f.stem)
        trojan_modules = tjin_files - tjfree_files
        print(f"\nAES-T100 trojan-only modules: {trojan_modules}")
        # Count signals whose id mentions a trojan module name
        trojan_signals = [n for n in g["nodes"]
                         if n["type"] == "signal"
                         and any(tm.lower() in n["id"].lower() for tm in trojan_modules)]
        print(f"AES-T100 signals from trojan modules: {len(trojan_signals)}")
        if trojan_signals:
            for s in trojan_signals[:10]:
                print(f"  {s['id']}")
        else:
            print("  WARNING: trojan submodules not elaborated by pyverilog")


if __name__ == "__main__":
    main()
