#!/usr/bin/env python3
"""Build benchmarks/manifest.json from TrustHub dataset.

Handles: TjFree/TjIn layout, AES-T600 TnIn typo, flat RS232 T100-T901,
VHDL exclusions, memctrl no-clean, testbench exclusion.
"""

import json
import os
import re
import sys
from pathlib import Path

TRUSTHUB = Path(__file__).resolve().parent.parent / "TrustHub - Dataset"
OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "manifest.json"

# Families
VHDL_FAMILIES = {"BasicRSA", "b19", "MC8051"}
TB_RE = re.compile(r"(^test|^tb|_tb$|_test$)", re.IGNORECASE)

FLAT_RS232 = {f"RS232-T{n}" for n in [100,200,300,400,500,600,700,800,900,901]}
RS232_CLEAN_BASE = "RS232-T2100"


def list_verilog(d: Path) -> list[str]:
    """List .v files excluding testbenches."""
    if not d.exists():
        return []
    out = []
    for f in sorted(d.rglob("*.v")):
        if not TB_RE.search(f.stem):
            out.append(str(f.relative_to(TRUSTHUB)))
    return out


def family_of(bench_id: str) -> str:
    m = re.match(r"(.+?)-T\d+$", bench_id)
    if m:
        return m.group(1)
    return bench_id


def build_manifest():
    if not TRUSTHUB.exists():
        print(f"ERROR: TrustHub folder not found at {TRUSTHUB}", file=sys.stderr)
        sys.exit(1)

    benchmarks = []
    dirs = sorted([d for d in TRUSTHUB.iterdir() if d.is_dir()])

    for d in dirs:
        bench_id = d.name
        fam = family_of(bench_id)
        entry = {
            "id": bench_id,
            "family": fam,
            "language": "VHDL" if fam in VHDL_FAMILIES else "Verilog",
            "clean_dir": None,
            "infected_dir": None,
            "design_files_clean": [],
            "design_files_infected": [],
            "status": "ok",
            "reason": "",
        }

        # VHDL exclusion
        if fam in VHDL_FAMILIES:
            entry["status"] = "excluded"
            entry["reason"] = "VHDL out of scope"
            benchmarks.append(entry)
            continue

        # Determine clean and infected dirs
        src = d / "src"
        rtl = d / "rtl" / "verilog"

        # Check for TjFree / TjIn (or TnIn for AES-T600)
        tjfree = src / "TjFree" if (src / "TjFree").exists() else None
        tjin = src / "TjIn" if (src / "TjIn").exists() else None
        if tjin is None and (src / "TnIn").exists():
            tjin = src / "TnIn"  # AES-T600 typo

        if tjfree and tjin:
            # Standard paired layout
            entry["clean_dir"] = str(tjfree.relative_to(TRUSTHUB))
            entry["infected_dir"] = str(tjin.relative_to(TRUSTHUB))
            entry["design_files_clean"] = list_verilog(tjfree)
            entry["design_files_infected"] = list_verilog(tjin)
        elif bench_id in FLAT_RS232:
            # Flat RS232 (infected only), use RS232-T2100/src/TjFree as clean base
            clean_base = TRUSTHUB / RS232_CLEAN_BASE / "src" / "TjFree"
            entry["clean_dir"] = str(clean_base.relative_to(TRUSTHUB)) if clean_base.exists() else None
            entry["infected_dir"] = str(src.relative_to(TRUSTHUB))
            entry["design_files_clean"] = list_verilog(clean_base) if clean_base.exists() else []
            entry["design_files_infected"] = list_verilog(src)
        elif rtl.exists():
            # memctrl style: rtl/verilog, no clean
            entry["clean_dir"] = None
            entry["infected_dir"] = str(rtl.relative_to(TRUSTHUB))
            entry["design_files_infected"] = list_verilog(rtl)
            entry["status"] = "no_clean"
            entry["reason"] = "no clean reference available"
        elif src.exists() and not tjfree and not tjin:
            # Flat src with no TjFree/TjIn - infected only
            entry["infected_dir"] = str(src.relative_to(TRUSTHUB))
            entry["design_files_infected"] = list_verilog(src)
            entry["status"] = "no_clean"
            entry["reason"] = "flat layout, no clean reference identified"
        else:
            entry["status"] = "error"
            entry["reason"] = f"unknown layout: {[x.name for x in d.iterdir()]}"

        # Validate: must have at least one infected file
        if not entry["design_files_infected"] and entry["status"] == "ok":
            entry["status"] = "error"
            entry["reason"] = "no Verilog design files found in infected dir"

        benchmarks.append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(benchmarks, f, indent=2)

    # Print summary
    families = {}
    for b in benchmarks:
        fam = b["family"]
        if fam not in families:
            families[fam] = {"total": 0, "statuses": {}}
        families[fam]["total"] += 1
        s = b["status"]
        families[fam]["statuses"][s] = families[fam]["statuses"].get(s, 0) + 1

    print(f"Total benchmarks: {len(benchmarks)}")
    print(f"{'Family':<15} {'Total':>5}  Statuses")
    for fam in sorted(families):
        f = families[fam]
        sts = ", ".join(f"{k}:{v}" for k, v in sorted(f["statuses"].items()))
        print(f"{fam:<15} {f['total']:>5}  {sts}")

    verilog_ok = sum(1 for b in benchmarks if b["language"] == "Verilog" and b["status"] in ("ok", "no_clean"))
    vhdl_ex = sum(1 for b in benchmarks if b["status"] == "excluded")
    no_clean = sum(1 for b in benchmarks if b["status"] == "no_clean")
    print(f"\nVerilog usable (ok+no_clean): {verilog_ok}, VHDL excluded: {vhdl_ex}, no_clean: {no_clean}")


if __name__ == "__main__":
    build_manifest()
