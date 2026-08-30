#!/usr/bin/env python3
"""Deterministic, stratified subsample of the core datasets for `matched_hpo`.

Why: a matched 100-trial nested HPO over all 45 core datasets x 3 GBDT x 5 seeds
(675 units) does not fit the compute budget of the final clean run. Rather than
cut the budget (which IS the object under measurement), we cut the dataset axis
and publish exactly which datasets were kept and how they were chosen.

Selection rule (fully determined by the inputs, no hidden state):
  1. read the core dataset list from `runner/datasets/core.yaml`;
  2. read each dataset's task_type and n_train from $TALENT_DATA_ROOT/<name>/info.json
     (fallback: shape of y_train.npy / N_train.npy);
  3. stratify by task_type x size tercile (terciles computed within task_type);
  4. allocate the quota across strata proportionally to stratum size (largest
     remainder), so the subsample mirrors the composition of the core;
  5. within a stratum, sort names lexicographically and draw with a fixed seed.

Usage:
  python runner/scripts/select_hpo_subset.py --k 20 --seed 20260709 \
      --out runner/datasets/core_hpo20.yaml

The emitted yaml is a drop-in dataset family: point the final matched_hpo pass at
`--datasets core_hpo20`. The markdown table it prints goes into the appendix.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except ImportError:  # minimal fallback: `datasets:` block of `- name` lines
        out, in_block = [], False
        for line in path.read_text().splitlines():
            if line.startswith("datasets:"):
                in_block = True
                continue
            if in_block:
                s = line.strip()
                if s.startswith("- "):
                    out.append(s[2:].strip().strip("'\""))
                elif s and not s.startswith("#"):
                    break
        return {"datasets": out}


def dataset_meta(name: str, root: Path, override: dict | None = None) -> tuple[str, int]:
    """(task_type, n_train) for one TALENT-format dataset folder.

    `override` is the dataset yaml's `task_type_override`. load_split honours it,
    so the strata must be built on the same effective type — otherwise the
    subsample is stratified by a composition that will not be the one that runs."""
    d = root / name
    info = json.loads((d / "info.json").read_text())
    task = str((override or {}).get(name)
               or info.get("task_type") or info.get("task") or "unknown").lower()
    if task.startswith("bin"):
        task = "binclass"
    elif task.startswith("multi"):
        task = "multiclass"
    elif task.startswith("reg"):
        task = "regression"

    n = info.get("n_train")
    if n is None:
        import numpy as np
        for cand in ("y_train.npy", "N_train.npy", "C_train.npy"):
            p = d / cand
            if p.exists():
                n = int(np.load(p, allow_pickle=True).shape[0])
                break
    return task, int(n)


def terciles(values: list[int]) -> list[int]:
    """Tercile boundaries (two cut points) of a value list."""
    s = sorted(values)
    if len(s) < 3:
        return [s[0], s[-1]]
    return [s[len(s) // 3], s[2 * len(s) // 3]]


def stratum_of(n: int, cuts: list[int]) -> str:
    return "small" if n <= cuts[0] else ("medium" if n <= cuts[1] else "large")


def largest_remainder(shares: dict[str, float], k: int) -> dict[str, int]:
    """Allocate k slots proportionally, resolving fractions by largest remainder."""
    raw = {s: v * k for s, v in shares.items()}
    alloc = {s: int(v) for s, v in raw.items()}
    left = k - sum(alloc.values())
    for s, _ in sorted(raw.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0])):
        if left <= 0:
            break
        alloc[s] += 1
        left -= 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-yaml", default="runner/datasets/core.yaml")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--out", default="runner/datasets/core_hpo20.yaml")
    a = ap.parse_args()

    root = Path(os.environ["TALENT_DATA_ROOT"])  # loud failure if unset, by design
    cfg = _load_yaml(Path(a.core_yaml))
    names = sorted(cfg["datasets"])
    override = cfg.get("task_type_override") or {}
    if override:
        print(f"[select_hpo_subset] task_type_override applied to {len(override)} "
              f"dataset(s): {', '.join(sorted(override))}\n")
    meta = {n: dataset_meta(n, root, override) for n in names}

    # terciles are computed WITHIN task_type so "large" means large for its own task
    by_task: dict[str, list[str]] = {}
    for n, (task, _) in meta.items():
        by_task.setdefault(task, []).append(n)

    strata: dict[str, list[str]] = {}
    for task, members in by_task.items():
        cuts = terciles([meta[m][1] for m in members])
        for m in members:
            strata.setdefault(f"{task}/{stratum_of(meta[m][1], cuts)}", []).append(m)

    total = len(names)
    shares = {s: len(v) / total for s, v in strata.items()}
    quota = largest_remainder(shares, a.k)

    rng = random.Random(a.seed)
    chosen: list[str] = []
    for s in sorted(strata):
        pool = sorted(strata[s])                      # lexicographic -> reproducible
        take = min(quota.get(s, 0), len(pool))
        chosen.extend(rng.sample(pool, take) if take else [])
    # if rounding left us short (a stratum was smaller than its quota), top up
    if len(chosen) < a.k:
        rest = [n for n in names if n not in chosen]
        chosen.extend(rng.sample(sorted(rest), a.k - len(chosen)))
    chosen = sorted(chosen)

    out = Path(a.out)
    out.write_text(
        "# Stratified subsample of core for the matched_hpo protocol.\n"
        f"# Generated by select_hpo_subset.py --k {a.k} --seed {a.seed}\n"
        "# Stratification: task_type x size tercile (terciles within task_type),\n"
        "# proportional quota (largest remainder), lexicographic pool + fixed seed.\n"
        "# emit_as makes every unit a plain core:<name> record: loaders,\n"
        "# VALID_FAMILY_PROTOCOL and the analysis stage see the parent family;\n"
        "# this file only narrows the CLI dataset axis for the matched_hpo leg.\n"
        f"family: core_hpo20\nemit_as: core\n"
        f"protocol: matched_hpo\nsource_family: core\nk: {a.k}\nseed: {a.seed}\n"
        "datasets:\n" + "".join(f"  - {n}\n" for n in chosen)
    )

    print(f"wrote {out}  ({len(chosen)} datasets)\n")
    print("| dataset | task | n_train | stratum |")
    print("|---|---|---|---|")
    inv = {m: s for s, ms in strata.items() for m in ms}
    for n in chosen:
        task, nt = meta[n]
        print(f"| {n} | {task} | {nt} | {inv[n].split('/')[1]} |")
    print("\nStratum coverage (selected/total):")
    for s in sorted(strata):
        got = sum(1 for c in chosen if inv[c] == s)
        print(f"  {s:26} {got}/{len(strata[s])}")


if __name__ == "__main__":
    main()
