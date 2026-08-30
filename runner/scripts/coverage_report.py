#!/usr/bin/env python3
"""
coverage_report.py — per-system coverage of the matrix and a named list of the
cells that produced no result.

The manifest only records that the runner finished: every line in it reads `ok`.
The statuses that decide what may be claimed, applicability skips and errors,
live inside the per-unit records. This script reads the records and prints three
things:

  1. a system x family x protocol table split into ok / skipped / errors, which
     is also the draft of the applicability table for the appendix;
  2. a named list of the cells that returned no result, with the reason and the
     seeds, from which the documented-limit entries are assembled;
  3. a reconciliation against the declared grid: how many cells never reached
     the manifest at all.

The output is compact enough (tens of lines) to be attached in full.

    python runner/scripts/coverage_report.py runs/final
    python runner/scripts/coverage_report.py runs/final --csv coverage.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path


def family_of(ds: str) -> str:
    return (ds or "?").split(":")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("--csv", help="also write the coverage table to a CSV file")
    a = ap.parse_args()

    out = Path(a.outdir)
    recs, unreadable = [], 0
    for f in out.glob("*__*.json"):
        try:
            recs.append(json.loads(f.read_text()))
        except Exception:
            unreadable += 1
    if not recs:
        raise SystemExit(f"no unit records under {out}")

    # The declared grid: the union of what the legs announced at launch.
    declared = set()
    for p in out.glob("run_meta*.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        declared |= set(m.get("units") or [])
        # For a leg sealed as interrupted, the original announcement is wider
        # than the list of units it completed; both numbers are kept in the meta.
    manifest_units = set()
    mp = out / "manifest.jsonl"
    if mp.exists():
        for line in mp.open():
            try:
                manifest_units.add(json.loads(line)["unit"])
            except Exception:
                continue

    print(f"records: {len(recs)}" + (f" (unreadable: {unreadable})" if unreadable else ""))
    print(f"manifest lines: {len(manifest_units)}\n")

    # --- 1. coverage table -------------------------------------------------
    cell = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        key = (r.get("model"), family_of(r.get("dataset")), r.get("protocol"))
        st = str(r.get("status", "?"))
        bucket = ("ok" if st == "ok"
                  else "error" if st.startswith("error:")
                  else "skip")
        cell[key][bucket] += 1

    rows = []
    print("| system | family | protocol | ok | skipped | errors | coverage |")
    print("|---|---|---|---:|---:|---:|---:|")
    for key in sorted(cell, key=lambda k: (str(k[0]), str(k[1]), str(k[2]))):
        m, fam, proto = key
        c = cell[key]
        tot = c["ok"] + c["skip"] + c["error"]
        pct = 100.0 * c["ok"] / tot if tot else 0.0
        flag = "" if pct == 100 else "  <- INCOMPLETE"
        print(f"| {m} | {fam} | {proto} | {c['ok']} | {c['skip']} | {c['error']} | {pct:.0f}%{flag} |")
        rows.append([m, fam, proto, c["ok"], c["skip"], c["error"], round(pct, 1)])

    # --- 2. incomplete cells, named ---------------------------------------
    print("\n" + "=" * 70)
    print("CELLS WITHOUT A RESULT (documented limits and the limitations section)")
    print("=" * 70)
    bad = collections.defaultdict(list)
    for r in recs:
        st = str(r.get("status", ""))
        if st != "ok":
            bad[(r.get("dataset"), r.get("model"), r.get("protocol"), st)].append(r.get("seed"))
    errors = {k: v for k, v in bad.items() if k[3].startswith("error:")}
    skips = {k: v for k, v in bad.items() if not k[3].startswith("error:")}

    if errors:
        print(f"\nERRORS ({sum(len(v) for v in errors.values())} units). The manifest hides"
              " these; every one must match a documented-limit entry:")
        print("| dataset | model | protocol | status | seeds |")
        print("|---|---|---|---|---|")
        for (ds, m, p, st), seeds in sorted(errors.items()):
            print(f"| {ds} | {m} | {p} | {st} | {sorted(x for x in seeds if x is not None)} |")
    else:
        print("\nno errors")

    if skips:
        print(f"\nAPPLICABILITY SKIPS ({sum(len(v) for v in skips.values())} units),"
              " grouped by reason:")
        by_reason = collections.defaultdict(set)
        for (ds, m, p, st), seeds in skips.items():
            by_reason[st].add((m, ds))
        for st in sorted(by_reason):
            pairs = sorted(by_reason[st])
            models = sorted({m for m, _ in pairs})
            print(f"  {st}: {len(pairs)} model x dataset pairs, models: {', '.join(models)}")

    # --- 3. cells that never arrived --------------------------------------
    print("\n" + "=" * 70)
    missing = sorted(declared - manifest_units)
    if missing:
        print(f"CELLS THAT NEVER REACHED THE MANIFEST: {len(missing)}")
        for u in missing[:20]:
            print(f"  {u}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("every declared cell reached the manifest")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "family", "protocol", "ok", "skipped", "errors", "coverage_pct"])
            w.writerows(rows)
        print(f"\ncoverage table written to {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
