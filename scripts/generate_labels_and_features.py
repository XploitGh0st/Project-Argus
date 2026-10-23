#!/usr/bin/env python3
"""
Generate and cache:
1. Node labels via Weisfeiler-Lehman graph diffing (.cache/labels/<bench>.json)
2. Engineered node features (.cache/features/<bench>/<variant>.npz)
Validates clean-vs-itself (0 positives) and prints required verification summaries.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules" / "argus_ml" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "verilog_ast_parser" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "dependency_digraph" / "src"))

from argus_ml.labeling import diff_graphs, validate_labeling
from argus_ml.nodefeat import FEATURE_COLUMNS, node_features

MANIFEST = ROOT / "benchmarks" / "manifest.json"
CACHE_DIGRAPHS = ROOT / ".cache" / "digraphs"
CACHE_LABELS = ROOT / ".cache" / "labels"
CACHE_FEATURES = ROOT / ".cache" / "features"


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CACHE_LABELS.mkdir(parents=True, exist_ok=True)
    CACHE_FEATURES.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: VALIDATION OF LABELING REQUIREMENTS ===")
    
    # Check AES-T100 clean vs itself
    aes_clean_file = CACHE_DIGRAPHS / "AES-T100" / "clean.json"
    if aes_clean_file.exists():
        aes_clean = json.loads(aes_clean_file.read_text(encoding="utf-8"))
        self_metrics = validate_labeling(aes_clean, aes_clean)
        print(f"Validation: AES-T100 clean vs itself positives = {self_metrics['self_positives']} (expected 0)")
        assert self_metrics["self_positives"] == 0, "Clean vs itself must have 0 positives!"

    validation_benchmarks = ["AES-T100", "RS232-T2100", "PIC16F84-T100"]
    validation_reports = {}

    for vb in validation_benchmarks:
        c_path = CACHE_DIGRAPHS / vb / "clean.json"
        i_path = CACHE_DIGRAPHS / vb / "infected.json"
        if c_path.exists() and i_path.exists():
            cg = json.loads(c_path.read_text(encoding="utf-8"))
            ig = json.loads(i_path.read_text(encoding="utf-8"))
            v_met = validate_labeling(cg, ig)
            validation_reports[vb] = v_met
            print(f"\nBenchmark {vb}:")
            print(f"  Total nodes: {v_met['total_nodes']}, Positives: {v_met['total_positives']} ({v_met['positive_fraction']*100:.2f}%)")
            print(f"  Agreement with new signal names: {v_met['agreement_with_new_signals']*100:.1f}%")
            print("  Sample positive nodes:")
            for sample in v_met["sample_positive_nodes"][:15]:
                print(f"    - [{sample['type']}] {sample['id']} ({sample['label']})")

    print("\n=== STEP 2: PROCESSING ALL BENCHMARKS (LABELS + FEATURES) ===")
    
    rs232_clean_base = CACHE_DIGRAPHS / "RS232-T2100" / "clean.json"
    loaded_clean: dict[str, dict] = {}
    if rs232_clean_base.exists():
        rs232_clean_g = json.loads(rs232_clean_base.read_text(encoding="utf-8"))
        loaded_clean[str(rs232_clean_base)] = rs232_clean_g
    else:
        rs232_clean_g = None

    processed_count = 0
    flagged_high_noise = []

    for b in manifest:
        if b["language"] != "Verilog":
            continue

        bench_id = b["id"]
        family = b["family"]
        bench_dir = CACHE_DIGRAPHS / bench_id
        inf_file = bench_dir / "infected.json"

        if not inf_file.exists():
            continue

        clean_file = bench_dir / "clean.json"
        if clean_file.exists():
            c_key = str(clean_file)
            if c_key not in loaded_clean:
                loaded_clean[c_key] = json.loads(clean_file.read_text(encoding="utf-8"))
            clean_g = loaded_clean[c_key]
        elif family == "RS232" and rs232_clean_g is not None:
            clean_g = rs232_clean_g
        else:
            print(f"Skipping labels for {bench_id}: no clean reference (e.g. memctrl)")
            clean_g = None

        inf_g = json.loads(inf_file.read_text(encoding="utf-8"))

        # 1. Labels
        label_file = CACHE_LABELS / f"{bench_id}.json"
        if clean_g is not None:
            if not label_file.exists():
                labels = diff_graphs(clean_g, inf_g)
                label_file.write_text(json.dumps(labels), encoding="utf-8")
            else:
                labels = json.loads(label_file.read_text(encoding="utf-8"))

            pos_count = sum(labels.values())
            tot_count = len(labels)
            pos_rate = pos_count / tot_count if tot_count > 0 else 0.0
            if pos_rate > 0.10:
                flagged_high_noise.append((bench_id, pos_rate))
        else:
            labels = {n["id"]: 0 for n in inf_g["nodes"]}

        # 2. Features
        feat_dir = CACHE_FEATURES / bench_id
        feat_dir.mkdir(parents=True, exist_ok=True)
        feat_file = feat_dir / "infected.npz"

        if not feat_file.exists() or feat_file.stat().st_size == 0:
            X, cols, nids = node_features(inf_g)
            y = np.array([labels.get(nid, 0) for nid in nids], dtype=np.int32)
            tmp_feat = feat_dir / "infected.tmp.npz"
            np.savez_compressed(tmp_feat, X=X, y=y, node_ids=np.array(nids, dtype=object), columns=np.array(cols, dtype=object))
            tmp_feat.replace(feat_file)
        else:
            try:
                data = np.load(feat_file, allow_pickle=True)
                X, y, nids = data["X"], data["y"], data["node_ids"]
            except Exception:
                X, cols, nids = node_features(inf_g)
                y = np.array([labels.get(nid, 0) for nid in nids], dtype=np.int32)
                np.savez_compressed(feat_file, X=X, y=y, node_ids=np.array(nids, dtype=object), columns=np.array(cols, dtype=object))

        processed_count += 1
        pos_str = f"pos={sum(y)} ({sum(y)/len(y)*100:.2f}%)" if len(y) > 0 else "empty"
        print(f"[{processed_count:2d}] {bench_id:18s} ({family:8s}) -> N={len(y):6d} {pos_str} feats={X.shape[1]}")

    print(f"\n=== SUMMARY ===")
    print(f"Total designs labeled and featurized: {processed_count}")
    if flagged_high_noise:
        print(f"Flagged designs (>10% positive rate, likely label noise):")
        for bid, pr in flagged_high_noise:
            print(f"  - {bid}: {pr*100:.2f}% positives")


if __name__ == "__main__":
    main()
