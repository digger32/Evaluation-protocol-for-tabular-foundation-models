#!/usr/bin/env python3
"""Freeze the OpenML-CC18 suite into a static snapshot yaml.

Why: `expand_family("cc18")` resolves the suite live via `openml.study.get_suite`.
When the call fails (no network, or `openml` missing from the interpreter that
actually ran), the runner falls back to a bare `cc18` stub unit and the whole
independent leg silently disappears from the grid — this happened once already.
The final `--no-resume` pass must not depend on a network round trip, so the task
ids are captured once, with a date, and read from disk thereafter.

Usage (once, with network + openml available):
  python runner/scripts/snapshot_cc18.py --out runner/datasets/cc18_snapshot.yaml

Then point `cc18.yaml` at the snapshot:
  resolve: static_snapshot
  snapshot: cc18_snapshot.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def core_openml_ids(datasets_dir: Path) -> set[int]:
    """Dataset ids of the core family, so the independent leg stays disjoint."""
    import yaml
    core = yaml.safe_load((datasets_dir / "core.yaml").read_text()) or {}
    ids = core.get("openml_dataset_ids") or []
    return {int(i) for i in ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default="runner/datasets")
    ap.add_argument("--suite", default="OpenML-CC18")
    ap.add_argument("--out", default="runner/datasets/cc18_snapshot.yaml")
    ap.add_argument("--allow-drift", dest="allow_drift", action="store_true",
                    help="write the snapshot even if it differs from cc18_resolved.json")
    a = ap.parse_args()

    import openml

    suite = openml.study.get_suite(a.suite)
    exclude = core_openml_ids(Path(a.datasets_dir))

    rows: list[tuple[int, int, str]] = []
    for tid in suite.tasks:
        t = openml.tasks.get_task(tid, download_data=False,
                                  download_qualities=False, download_splits=False)
        did = int(t.dataset_id)
        if did in exclude:
            continue
        rows.append((int(tid), did, t.get_dataset().name))

    rows.sort()

    # Drift check: the dev matrix ran on datasets/cc18_resolved.json. A final
    # grid that silently differs from it is a paper decision, not a side
    # effect of an upstream suite edit — abort loudly unless --allow-drift.
    resolved = Path(a.datasets_dir) / "cc18_resolved.json"
    if resolved.exists():
        frozen = {(int(r["task_id"]), int(r["dataset_id"]))
                  for r in json.loads(resolved.read_text())["kept"]}
        live = {(tid, did) for tid, did, _ in rows}
        if live != frozen and not a.allow_drift:
            raise SystemExit(
                "[snapshot_cc18] DRIFT vs datasets/cc18_resolved.json (the dev grid):\n"
                f"  live-only:   {sorted(live - frozen)}\n"
                f"  frozen-only: {sorted(frozen - live)}\n"
                "Decide deliberately, then re-run with --allow-drift if that is the call.")
        print(f"[snapshot_cc18] live suite matches the dev resolution ({len(frozen)} kept).")

    today = dt.date.today().isoformat()
    body = [
        f"# Static snapshot of {a.suite}, taken {today}.",
        "# Resolved once with `openml.study.get_suite`; the final clean run reads this",
        "# file so that a network failure cannot silently shrink the grid.",
        f"suite: {a.suite}",
        f"snapshot_date: '{today}'",
        f"openml_package: '{openml.__version__}'",
        f"n_tasks_in_suite: {len(suite.tasks)}",
        f"n_tasks_after_core_dedup: {len(rows)}",
        "tasks:",
    ]
    for tid, did, name in rows:
        body.append(f"  - task_id: {tid}    # dataset_id={did}  {name}")

    Path(a.out).write_text("\n".join(body) + "\n")
    print(f"wrote {a.out}: {len(rows)} tasks "
          f"({len(suite.tasks)} in suite, {len(suite.tasks) - len(rows)} dropped as core overlap)")


if __name__ == "__main__":
    main()
