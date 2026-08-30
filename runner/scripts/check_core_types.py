#!/usr/bin/env python3
"""
check_core_types.py — does the core dataset list agree with the data on disk?

`runner/datasets/core.yaml` groups the 45 TALENT-tiny members under comment
headers ("binary classification (18)", "multi-class classification (12)",
"regression (15)"), taken from the per-row Tiny markers of the benchmark's
Table 1. Every unit, however, is loaded with the `task_type` recorded in
`$TALENT_DATA_ROOT/<dataset>/info.json`, and it is that value which decides the
metric, the split policy and whether a classification-only model is skipped.

The stratified matched_hpo subsample tallied 14 binary / 12 multiclass /
19 regression, which matches neither the markers (18/12/15) nor the benchmark's
prose (15/12/18). This script names the disagreeing datasets exactly, so the
manuscript states the composition that was actually run.

Usage (on the experiment box, TALENT_DATA_ROOT set):
    python runner/scripts/check_core_types.py [--core-yaml runner/datasets/core.yaml]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path

HEADER = re.compile(r"^\s*#\s*(binary classification|multi-class classification|"
                    r"multiclass classification|regression)\s*\((\d+)\)", re.I)
ITEM = re.compile(r"^\s*-\s+(\S.*?)\s*$")
DECLARED = {"binary classification": "binclass",
            "multi-class classification": "multiclass",
            "multiclass classification": "multiclass",
            "regression": "regression"}


def parse_declared(core_yaml: Path):
    """Read the group each dataset is listed under, plus each header's own count."""
    declared, counts, group = {}, {}, None
    in_list = False
    for line in core_yaml.read_text().splitlines():
        m = HEADER.match(line)
        if m:
            group = DECLARED[m.group(1).lower()]
            counts[group] = int(m.group(2))
            in_list = True
            continue
        if line.strip().startswith("datasets:"):
            in_list = True
            continue
        m = ITEM.match(line)
        if in_list and m and group:
            declared[m.group(1)] = group
    return declared, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core-yaml", default="runner/datasets/core.yaml")
    a = ap.parse_args()

    core_path = Path(a.core_yaml)
    declared, header_counts = parse_declared(core_path)
    root = Path(os.environ["TALENT_DATA_ROOT"])
    # The dataset yaml may correct a shipped task type (task_type_override);
    # load_split honours it, so this checker must judge against the EFFECTIVE
    # type, not against info.json alone, or it reports phantom mismatches.
    override = {}
    try:
        import yaml
        override = (yaml.safe_load(core_path.read_text()) or {}).get("task_type_override") or {}
    except ImportError:
        block = re.search(r"^task_type_override:\s*$(.*?)^\S", t if False else
                          core_path.read_text(), re.M | re.S)
        if block:
            for line in block.group(1).splitlines():
                m = re.match(r"\s+(\S+):\s*(\S+)", line)
                if m:
                    override[m.group(1)] = m.group(2)

    shipped, effective, missing, label_check = {}, {}, [], {}
    for name in declared:
        info = root / name / "info.json"
        if not info.exists():
            missing.append(name)
            continue
        d = json.loads(info.read_text())
        tt = d.get("task_type", "?")
        n_cls = d.get("n_classes")
        if tt == "classification":            # older layouts: split by n_classes
            tt = "binclass" if n_cls == 2 else "multiclass"
        shipped[name] = tt
        eff = override.get(name, tt)
        effective[name] = eff
        if name in override:                  # an override must be justified by the labels
            import numpy as np
            y = np.load(root / name / "y_train.npy", allow_pickle=True).reshape(-1)
            u = np.unique(y)
            label_check[name] = (str(y.dtype), len(u))

    cnt = lambda d: dict(sorted(collections.Counter(d.values()).items()))
    print(f"datasets listed in core.yaml: {len(declared)}"
          f" | info.json read: {len(shipped)}"
          + (f" | MISSING FOLDERS: {missing}" if missing else ""))
    print(f"header counts declared:   {dict(sorted(header_counts.items()))}")
    print(f"shipped in info.json:     {cnt(shipped)}")
    print(f"EFFECTIVE (what will run):{cnt(effective)}"
          + (f"   [{len(override)} override(s) applied]" if override else ""))

    if override:
        print("\ntask_type_override, checked against the shipped labels:")
        print("| dataset | info.json | override | y dtype | distinct y |")
        print("|---|---|---|---|---|")
        for n in sorted(override):
            dt, nu = label_check.get(n, ("?", "?"))
            warn = "" if (override[n] != "binclass" or nu == 2) else "  <-- NOT BINARY"
            print(f"| {n} | {shipped.get(n, '?')} | {override[n]} | {dt} | {nu} |{warn}")

    bad = [(n, declared[n], effective[n]) for n in sorted(effective)
           if declared[n] != effective[n]]
    if not bad:
        print("\nevery dataset's declared group matches the effective task type"
              " — composition is consistent")
        return
    print(f"\n{len(bad)} dataset(s) whose declared group differs from the EFFECTIVE type:")
    print("| dataset | declared in core.yaml | effective |")
    print("|---|---|---|")
    for n, d, x in bad:
        print(f"| {n} | {d} | {x} |")
    print("\nThe loader follows the effective type; classification-only systems are"
          "\nskipped on anything that is effectively regression.")


if __name__ == "__main__":
    main()
