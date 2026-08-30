#!/usr/bin/env python3
"""
Review-proofing gate.

Reads the runner's outputs (run_meta.json + manifest.jsonl + per-unit JSONs) and
a gate_config.yaml, then asserts the conditions that keep dirty numbers out of
figures. Exits NON-ZERO on any failure so it can block a finalisation step in a
pipeline (e.g. `python review_gate.py runs/final && python make_figures.py`).

Built-in assertions:
  A1  clean final run      final pass had resume DISABLED, no unit skipped,
                           every grid cell exactly once (no duplicate manifest rows)
  B1  external validity    every comparative claim has >=1 independent-dataset run
  F1  env integrity        no skipped_missing_backend:* records (environment != capability)
  F2  env homogeneity      one interpreter prefix AND one hostname across all records
  F3  per-system coverage  each (model x family x protocol) group has >0 ok record
                           or is a pure capability skip (coverage.require_ok_per_system)
  H1  context integrity    no unit was fitted on a silently reduced in-context
                           support set (AutoGluon/Mitra halves it under VRAM pressure)
  G1  documented limits    every timeout/fail in the manifest matches an allowlist
                           entry (model x dataset x protocol + reason); new ones fail
Optional (enable in config):
  C1  calibration present  ECE/coverage recorded for UQ contributions
  D1  optimism gap         inner-CV vs held-out gap recorded for tuned configs
  E1  stats present        omnibus + post-hoc outputs exist for the main comparison

Usage:
    python review_gate.py <outdir> [--config gate_config.yaml]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[gate] PyYAML required: pip install pyyaml --break-system-packages")
    sys.exit(2)


def load_manifest(outdir: Path):
    mf = outdir / "manifest.jsonl"
    if not mf.exists():
        return []
    return [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]


def load_units(outdir: Path):
    units = []
    for p in outdir.glob("*__*__*__seed*.json"):
        try:
            units.append(json.loads(p.read_text()))
        except Exception:
            pass
    return units


KNOWN_KEYS = {"require_calibration", "require_optimism_gap", "require_stats",
              "stats_artifacts", "comparative_claims", "coverage", "allowlist"}
KNOWN_COVERAGE = {"require_ok_per_system", "shards_cover_grid_exactly_once",
                  "single_hostname"}
# These two are enforced unconditionally by A1 and F2; the config may restate
# them for documentation but may not switch them off.
ALWAYS_ON_COVERAGE = {"shards_cover_grid_exactly_once", "single_hostname"}


def check_config(cfg):
    """A key the gate does not read is a silent hole: a merged-but-misspelt
    allowlist would leave the run ungated while the file looks complete.
    Fail on anything unrecognised rather than ignoring it."""
    bad = sorted(set(cfg) - KNOWN_KEYS)
    if bad:
        return False, f"unknown top-level key(s) in gate_config: {bad}"
    cov = cfg.get("coverage") or {}
    bad = sorted(set(cov) - KNOWN_COVERAGE)
    if bad:
        return False, f"unknown key(s) under coverage: {bad}"
    off = sorted(k for k in ALWAYS_ON_COVERAGE if cov.get(k) is False)
    if off:
        return False, (f"coverage.{off} set to false, but A1/F2 enforce these "
                       "unconditionally — the config would be lying")
    for i, a in enumerate(cfg.get("allowlist") or []):
        bad = sorted(set(a) - {"model", "dataset", "protocol", "reason", "note"})
        if bad:
            return False, f"allowlist entry {i}: unknown key(s) {bad}"
        if not a.get("reason"):
            return False, f"allowlist entry {i} has no reason: {a}"
    n = len(cfg.get("allowlist") or [])
    return True, f"config keys recognised; {n} documented-limit entr{'y' if n == 1 else 'ies'}"


def check_A1(outdir, manifest, cfg):
    """The final matrix is one clean sweep: resume disabled, one host, and every
    grid cell present exactly once.

    Ownership of a cell comes from the MANIFEST (`started` names the leg that
    produced the row), not from the legs' launch declarations. That distinction
    matters once a cell has been recomputed: a leg that first computed a cell
    still declares it, while the row now belongs to the later leg. Comparing
    declarations would call that an overlap, which is wrong — the cell is in the
    matrix once, computed by a known leg, and the earlier attempt was discarded
    on purpose. What must NOT happen is a cell declared by a leg, absent from the
    manifest, and picked up by nobody; that is a silent hole and is reported.
    """
    import collections
    metas = []
    for p in sorted(outdir.glob("run_meta*.json")):
        if p.name.endswith((".removed", ".preseal")):
            continue
        try:
            metas.append((p.name, json.loads(p.read_text())))
        except Exception:
            return False, f"unreadable {p.name}"
    if not metas:
        return False, "no run_meta*.json — cannot verify the final pass"

    not_final = [n for n, m in metas if not m.get("no_resume", False)]
    if not_final:
        return False, f"{len(not_final)} invocation(s) ran WITHOUT --no-resume: {not_final[:3]}"
    hosts = {m.get("hostname") for _, m in metas if m.get("hostname")}
    if len(hosts) > 1:
        return False, f"final legs ran on {len(hosts)} hosts: {sorted(hosts)}"
    if any(r.get("status") == "skip" for r in manifest):
        return False, "manifest shows skipped units in a no-resume pass"

    seen = collections.Counter(r.get("unit") for r in manifest)
    dupes = [u for u, c in seen.items() if c > 1]
    if dupes:
        return False, (f"{len(dupes)} unit(s) appear more than once in the manifest "
                       f"(a cell may be recomputed, but only one row may survive): "
                       f"{dupes[:3]}...")
    started_ok = {m.get("run_started") for _, m in metas}
    stale = [r["unit"] for r in manifest if r.get("started") not in started_ok]
    if stale:
        return False, (f"{len(stale)} unit(s) carry a run_started not declared by any "
                       f"final invocation (carry-over): {stale[:3]}...")

    owner = {r["unit"]: r.get("started") for r in manifest}
    superseded, orphaned = 0, []
    for name, m in metas:
        decl = m.get("units")
        if decl is None:
            continue
        for u in decl:
            if u not in owner:
                orphaned.append(u)
            elif owner[u] != m.get("run_started"):
                superseded += 1          # cell recomputed by a later leg
    if orphaned:
        allow = cfg.get("allowlist", [])
        def covered(unit):
            ds, _, rest = unit.partition("__")
            parts = rest.split("__")
            proto = parts[0] if parts else None
            model = parts[1] if len(parts) > 1 else None
            return any(a.get("model") in (None, model) and
                       a.get("dataset") in (None, ds) and
                       a.get("protocol") in (None, proto) for a in allow)
        uncovered = sorted({u for u in orphaned if not covered(u)})
        if uncovered:
            return False, (f"{len(uncovered)} declared cell(s) reached no leg and are not "
                           f"in the allowlist: {uncovered[:3]}...")

    interrupted = [n for n, m in metas if m.get("interrupted")]
    desc = (f"final matrix clean: {len(metas)} no-resume leg(s), one host, "
            f"{len(seen)} cells each exactly once")
    if superseded:
        desc += f"; {superseded} cell(s) recomputed by a later leg (earlier attempt discarded)"
    if orphaned:
        desc += f"; {len(set(orphaned))} declared cell(s) absent, all covered by the allowlist"
    if interrupted:
        desc += f"; {len(interrupted)} leg(s) sealed as interrupted"
    return True, desc


def check_B1(outdir, units, cfg):
    """Every comparative claim has at least one independent-dataset run.
    The runner's dataset tokens are '<family>' or '<family>:<name>', so the
    independent_datasets in the config are matched at the FAMILY level."""
    def fam(tok):
        return (tok or "").split(":")[0]
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared in config — declare them"
    present_families = {fam(u.get("dataset")) for u in units}
    failures = []
    for c in claims:
        needed = set(c.get("independent_datasets", []))
        if not needed:
            failures.append(f"claim '{c.get('id')}' lists no independent_datasets")
        elif not (needed & present_families):
            failures.append(f"claim '{c.get('id')}' has no run on any of {sorted(needed)}")
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(claims)} comparative claim(s) each have an independent-dataset run"


def check_metric_present(outdir, units, cfg, key, label):
    """Generic: assert some unit recorded a non-null metric `key`, looking both
    under the per-unit 'metrics' dict and at the record top level (the pipeline
    nests point/proper-scoring/calibration under 'metrics' but keeps
    'optimism_gap' at the top level). Skipped/error units carry neither and are
    ignored here."""
    def _val(u):
        m = (u.get("metrics") or {})
        return m[key] if key in m else u.get(key)
    have = [u for u in units if _val(u) is not None]
    if not have:
        return False, f"no unit recorded '{key}' ({label})"
    return True, f"'{key}' present in {len(have)} unit(s) ({label})"


def check_F1_env(outdir, units, cfg):
    """No unit may be skipped because a backend was missing from the environment.

    `skipped_missing_backend:<pkg>` looks like a capability skip (clean record,
    worker exit 0, orchestrator logs [ok]) but means the leg never ran. Left
    unchecked, a whole protocol can evaporate while D1/C1 still pass on stale
    records from an earlier run.
    """
    import collections
    bad = collections.Counter()
    for u in units:
        st = str(u.get("status", ""))
        if st.startswith("skipped_missing_backend"):
            bad[(u.get("model"), u.get("protocol"), st)] += 1
    if bad:
        top = ", ".join(f"{m}/{p} {s} x{n}" for (m, p, s), n in bad.most_common(3))
        return False, (f"{sum(bad.values())} unit(s) skipped for a MISSING BACKEND "
                       f"(environment, not capability): {top}")
    return True, "no environment-induced skips (all skips are capability limits)"


def check_F2_env_homogeneity(outdir, units, cfg):
    """Every record must come from ONE interpreter with one set of library versions.

    A unit computed under another project's virtualenv looks perfectly valid: a
    clean record, real metrics, different library builds. Records stamped with
    `env` make the mixture visible; unstamped legacy records are reported too.
    """
    import collections
    prefixes = collections.Counter()
    hosts = collections.Counter()
    unstamped = 0
    for u in units:
        env = u.get("env")
        if not env:
            unstamped += 1
            continue
        prefixes[env.get("prefix", "?")] += 1
        if env.get("hostname"):
            hosts[env["hostname"]] += 1
    if len(hosts) > 1:
        detail = ", ".join(f"{h} x{n}" for h, n in hosts.most_common())
        return False, (f"records come from {len(hosts)} different hosts "
                       f"(same-host-shards violated): {detail}")
    # Code provenance. A record without a fingerprint is NOT evidence of
    # sameness — the field was introduced partway through this matrix, so the
    # earlier legs carry nothing to compare. Silently ignoring the blanks would
    # report homogeneity that was never checked, which is the failure mode this
    # gate exists to prevent, so the count of unstamped records is stated
    # explicitly and travels into the report.
    envs = [u.get("env") or {} for u in units]
    codes = collections.Counter(e["code_sha256"] for e in envs if e.get("code_sha256"))
    unstamped_code = sum(1 for e in envs if not e.get("code_sha256"))
    if len(codes) > 1:
        detail = ", ".join(f"{c} x{n}" for c, n in codes.most_common())
        return False, (f"records were produced by {len(codes)} different versions of the "
                       f"runner sources (code_sha256): {detail}")
    if len(prefixes) > 1:
        detail = ", ".join(f"{p} x{n}" for p, n in prefixes.most_common())
        return False, f"records come from {len(prefixes)} different environments: {detail}"
    if unstamped:
        return False, (f"{unstamped} record(s) carry no environment fingerprint "
                       f"(computed before stamping was added — recompute before freezing)")
    only = next(iter(prefixes)) if prefixes else "none"
    msg = f"all {sum(prefixes.values())} records from one environment ({only})"
    if unstamped_code:
        pct = 100.0 * unstamped_code / max(len(units), 1)
        msg += (f"; NOTE: {unstamped_code} record(s) ({pct:.0f}%) predate the code "
                f"fingerprint — one interpreter and one library set are proven for them, "
                f"the exact source version is not, and the manuscript must say so")
    return True, msg


def check_F3_coverage(outdir, units, cfg):
    """Every (model x family x protocol) group present in the run must have
    >0 record-level ok, unless the whole group is capability skips
    (skipped_*). A group with only error records means a system silently
    evaporated from a leg while the aggregate counters still looked healthy
    (the June 'GBDT cv was never finished' failure mode)."""
    import collections
    groups = collections.defaultdict(list)
    for u in units:
        fam = (u.get("dataset") or "").split(":")[0]
        groups[(u.get("model"), fam, u.get("protocol"))].append(str(u.get("status", "")))
    bad = []
    for key, sts in sorted(groups.items()):
        if any(s == "ok" for s in sts):
            continue
        if all(s.startswith("skipped_") for s in sts):
            continue   # pure capability group (e.g. a cls-only TFM on regression)
        bad.append(f"{' x '.join(str(k) for k in key)} ({len(sts)} records, no ok)")
    if bad:
        return False, f"{len(bad)} system-leg group(s) have no ok record: {bad[:3]}..."
    return True, f"all {len(groups)} system-leg groups carry >0 ok or are capability-skipped"


def check_H1_context_integrity(outdir, units, cfg):
    """No unit REPORTED AS OK may have been fitted on a reduced in-context set.

    AutoGluon's Mitra halves its support set under VRAM pressure and still
    reports success; a unit fitted on 1024 rows instead of 8192 is noise wearing
    an ok. The pipeline now turns a detected reduction into
    `skipped_context_reduced` — a documented limit, exactly on the units where
    it happened — so this assertion should pass by construction. It stays as the
    backstop for any path that returns an ok while carrying the flag (an older
    record, or a wrapper version whose message we do not parse)."""
    bad = [u for u in units if u.get("context_reduced") and u.get("status") == "ok"]
    if bad:
        det = ", ".join(f"{u.get('dataset')}x{u.get('model')}"
                        f"[{u.get('context_from')}->{u.get('context_to')}]" for u in bad[:3])
        return False, (f"{len(bad)} OK unit(s) fitted on a REDUCED in-context support "
                       f"set (silent degradation): {det}...")
    skipped = [u for u in units if str(u.get("status", "")) == "skipped_context_reduced"]
    if skipped:
        ds = sorted({u.get("dataset") for u in skipped})
        return True, (f"no degraded ok unit; {len(skipped)} unit(s) on {len(ds)} dataset(s) "
                      f"recorded as documented context-reduction limits")
    return True, "no unit reports a reduced in-context support set"


def check_G1_allowlist(outdir, manifest, cfg, units=None):
    """Every incomplete cell must match a documented-limits allowlist entry.

    Two kinds of incompleteness, and the second is the treacherous one:
      * the MANIFEST shows timeout / fail — the unit never produced a record;
      * the RECORD shows `error:*` — the runner exited zero and the manifest
        says ok, so the failure is invisible above the record level. The
        analysis then quietly drops the cell and no reader ever learns it was
        attempted. This is exactly the silent gap the paper's own standard
        forbids, so both kinds are checked here.

    Capability skips (skipped_*) are documented non-results by construction and
    are not required to appear in the allowlist. An allowlist entry never masks
    a success: a listed unit that completed ok simply needs no forgiveness."""
    allow = cfg.get("allowlist", [])
    def covered(rec):
        return any(a.get("model") in (None, rec.get("model")) and
                   a.get("dataset") in (None, rec.get("dataset")) and
                   a.get("protocol") in (None, rec.get("protocol"))
                   for a in allow)
    bad = [r for r in manifest if r.get("status") != "ok" and not covered(r)]
    errored = [u for u in (units or []) if str(u.get("status", "")).startswith("error:")]
    bad_rec = [u for u in errored if not covered(u)]
    if bad or bad_rec:
        parts = []
        if bad:
            parts.append(f"{len(bad)} manifest-level: "
                         + ", ".join(f"{r.get('unit')}[{r.get('status')}]" for r in bad[:3]))
        if bad_rec:
            byd = collections.Counter(f"{u.get('dataset')} x {u.get('model')} "
                                      f"[{u.get('status')}]" for u in bad_rec)
            parts.append(f"{len(bad_rec)} record-level (manifest says ok!): "
                         + "; ".join(f"{k} x{n}" for k, n in byd.most_common(4)))
        return False, "incomplete cell(s) not in the documented-limits allowlist — " + " | ".join(parts)
    n_forgiven = sum(1 for r in manifest if r.get("status") != "ok") + len(errored)
    return True, (f"all {n_forgiven} incomplete cell(s) covered by the allowlist"
                  if n_forgiven else "no incomplete cells")


def check_E1_stats(outdir, cfg):
    art = cfg.get("stats_artifacts", ["stats/omnibus.json", "stats/posthoc.json"])
    missing = [a for a in art if not (outdir / a).exists()
               and not Path(a).exists()]
    if missing:
        return False, f"missing stats artifacts: {missing}"
    return True, f"stats artifacts present: {art}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--config", default="gate_config.yaml")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        cfg_path = outdir / a.config
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    manifest = load_manifest(outdir)
    units = load_units(outdir)

    results = []
    results.append(("A0 config-schema", *check_config(cfg)))
    results.append(("A1 clean-final-run", *check_A1(outdir, manifest, cfg)))
    results.append(("B1 external-validity", *check_B1(outdir, units, cfg)))
    if cfg.get("require_calibration"):
        results.append(("C1 calibration", *check_metric_present(
            outdir, units, cfg, "ece", "calibration present")))
    if cfg.get("require_optimism_gap"):
        results.append(("D1 optimism-gap", *check_metric_present(
            outdir, units, cfg, "optimism_gap", "HPO honesty")))
    if cfg.get("require_stats"):
        results.append(("E1 stats", *check_E1_stats(outdir, cfg)))
    results.append(("F1 env-integrity", *check_F1_env(outdir, units, cfg)))
    results.append(("F2 env-homogeneity", *check_F2_env_homogeneity(outdir, units, cfg)))
    if (cfg.get("coverage") or {}).get("require_ok_per_system"):
        results.append(("F3 per-system-coverage", *check_F3_coverage(outdir, units, cfg)))
    results.append(("G1 documented-limits", *check_G1_allowlist(outdir, manifest, cfg, units)))
    results.append(("H1 context-integrity", *check_H1_context_integrity(outdir, units, cfg)))

    print("=" * 64)
    print(f"REVIEW-PROOFING GATE  | outdir={outdir}")
    print("=" * 64)
    ok = True
    for name, passed, msg in results:
        flag = "PASS" if passed else "FAIL"
        print(f"[{flag}] {name:24s} {msg}")
        ok = ok and passed
    print("=" * 64)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures.")
        sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
