#!/usr/bin/env python3
"""
Job-based benchmark runner — tabular-eval-protocol (significance-, stability- and calibration-
aware TFM re-benchmark). Adapted from the paper-build base.

Pattern (unchanged from the base): the batch is a set of INDEPENDENT units =
dataset x protocol x model x seed. Each unit runs in its OWN subprocess so a
hang / OOM / segfault in one unit can never take down the batch. The
orchestrator skips units whose output exists (RESUME), enforces a per-unit
wall-clock timeout (HARD TIMEOUT), writes one manifest record per completed unit
(for the gate), and keeps going past timeouts and failures.

What is adapted for this topic (and ONLY this):
  * run_unit()        — fit/infer one system, save per-prediction outputs, and
                        compute point + proper-scoring + calibration metrics
                        and the HPO optimism gap.
  * unit_is_valid()   — topic hook: skip invalid (dataset-family, protocol)
                        combinations (e.g. tabred only under `temporal`) so the
                        cartesian product does not create meaningless units.
  * argparse defaults — this topic's datasets / protocols / models.
The orchestration (resume, timeout, shard, manifest, no-resume invariant) is the
part the review-proofing gate depends on and is left faithful to the base.

Launch wrapped in tmux (built into the command, per house convention):

    # DEV / HPO sweeps shard across both boxes:
    # box 1 (A100-40):  tmux new -s arbench ; python scripts/bench_runner.py \
    #     --datasets core,cc18,tabred --protocols cv_standard,temporal,matched_hpo \
    #     --models tabpfn25,tabicl,mitra,tabpfnv2,catboost_tuned,xgboost_tuned,lightgbm_tuned,realmlp,tabm \
    #     --seeds 0,1,2,3,4 --outdir runs/dev --shard 0/2
    # box 2 (A800-80):  tmux new -s arbench ; python scripts/bench_runner.py ... --outdir runs/dev --shard 1/2
    # detach: Ctrl-b d ; reattach: tmux attach -t arbench

    # FINAL pass — ONE box (A800-80), single shard, resume DISABLED, fresh outdir:
    # tmux new -s arfinal ; python scripts/bench_runner.py ... --outdir runs/final --no-resume --shard 0/1

For the FINAL results pass use --no-resume and a fresh --outdir so the gate's
A1 (clean final run) assertion passes.
"""
import argparse
import json
import os
import subprocess
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# TOPIC HOOK 1 — valid (dataset-family, protocol) pairs.                       #
# The dataset families use different split policies, so most of the cartesian  #
# product is meaningless. Skip the invalid pairs at axis-build time.           #
# --------------------------------------------------------------------------- #
VALID_FAMILY_PROTOCOL = {
    "core":   {"cv_standard", "matched_hpo"},  # host benchmark (TALENT-tiny)
    "cc18":   {"cv_standard"},                  # public independent leg
    "tabred": {"temporal"},                     # leakage-aware leg
}


def family_of(token: str) -> str:
    """Token is '<family>' (stub) or '<family>:<dataset_name>' (expanded)."""
    return token.split(":")[0]


def unit_is_valid(dataset_token: str, protocol: str) -> bool:
    return protocol in VALID_FAMILY_PROTOCOL.get(family_of(dataset_token), set())


def expand_families(families, datasets_dir: Path):
    """Expand each family token into '<family>:<dataset_name>' tokens.

    Delegates to topic_pipeline.expand_family so the expansion matches the data
    layer exactly: core/tabred read their yaml `datasets:` list, while cc18 is
    resolved LIVE from the OpenML-CC18 suite (minus the core overlap) on the
    experiment box. Keeping a single source of truth avoids a stale dataset list.

    Fallbacks keep the build runnable off the experiment box: if the pipeline
    cannot be imported, or a family cannot be resolved (e.g. cc18 with no OpenML
    egress), the bare family token is kept, which the runner then treats as a
    single fallback unit. The expansion is an AXIS adaptation, not an
    orchestration change.
    """
    try:
        import topic_pipeline
    except Exception as e:
        print(f"[runner] topic_pipeline import failed ({e!r}); "
              f"running families as single fallback units", flush=True)
        return list(families)
    out = []
    for fam in families:
        try:
            out += topic_pipeline.expand_family(fam, str(datasets_dir))
        except Exception as e:
            print(f"[runner] expand_family('{fam}') failed ({e!r}); "
                  f"using bare '{fam}' fallback unit", flush=True)
            out.append(fam)
    return out


# --------------------------------------------------------------------------- #
# TOPIC HOOK 2 — per-unit work. Fit/infer one system on one (family, protocol, #
# seed) split, save per-prediction outputs (so proper-scoring, calibration and #
# the bootstrap are all computable post-hoc), and record the metrics.          #
# Deterministic given (dataset, protocol, model, seed). Heavy imports stay     #
# inside the function so a unit's import failure is contained to that unit.     #
# --------------------------------------------------------------------------- #
def run_unit(dataset: str, protocol: str, model: str, seed: int, out_path: Path) -> dict:
    """One unit = (dataset_token, protocol, model, seed). Delegates the data and
    model work to topic_pipeline (the data/model layer), which owns the
    TALENT/Gorishniy preprocessing, the model wrappers, the capability guards and
    the metric computation. This hook only wires the three entry points together
    and writes the record; it holds no topic logic of its own so the contract has
    a single source of truth.

    topic_pipeline contract:
        split = load_split(dataset_token, protocol, seed)        -> Split
        pred  = get_predictions(model, split, seed, protocol)    -> dict (status)
        rec   = compute_unit_metrics(pred, split)                -> dict (+metrics)
    A model that cannot run a unit (regression-only TFM on a regression task,
    feature-cap exceeded, missing backend) returns a status string rather than
    raising, so the record is a clean 'skipped_*' / 'error:*' marker and the
    batch continues. The saved record carries the per-prediction arrays so E1-E4
    (proper scoring, calibration, bootstrap rank-stability) are post-hoc.
    """
    import json as _json
    try:
        import topic_pipeline as tp
    except Exception as e:
        out_path.write_text(_json.dumps({
            "status": f"error:topic_pipeline_import:{type(e).__name__}",
            "dataset": dataset, "protocol": protocol, "model": model, "seed": seed,
            "msg": str(e)[:300],
        }, indent=2))
        return {"status": "error"}

    split = tp.load_split(dataset, protocol, seed)
    pred = tp.get_predictions(model, split, seed, protocol)
    rec = tp.compute_unit_metrics(pred, split)
    # carry the unit axes onto the record so the gate/manifest can key on them
    rec.setdefault("dataset", dataset)
    rec.setdefault("family", getattr(split, "family", dataset.split(":")[0]))
    rec.setdefault("protocol", protocol)
    rec.setdefault("model", model)
    rec.setdefault("seed", seed)
    rec["env"] = env_fingerprint()
    out_path.write_text(_json.dumps(rec))
    return rec


def env_fingerprint() -> dict:
    """Interpreter + library versions that produced this record.

    A unit computed under a different virtualenv is a valid-looking record with
    incomparable numbers. Stamping every record lets the gate assert that the
    whole matrix came from one environment, and lets the paper publish it.
    """
    import importlib.metadata as md
    vers = {}
    for pkg in ("scikit-learn", "catboost", "xgboost", "lightgbm", "torch",
                "tabpfn", "tabicl", "autogluon.tabular", "pytabkit", "optuna",
                "numpy", "openml"):
        try:
            vers[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001 - package simply absent for this unit
            pass
    return {"python": sys.executable, "prefix": sys.prefix,
            "hostname": socket.gethostname(), "versions": vers,
            "code_sha256": _code_fingerprint()}


def _code_fingerprint() -> str:
    """Hash of the sources that decide what a unit means.

    Library versions are not enough: this project's own code changed between
    legs of the same matrix (model caps, metric normalisation, skip semantics).
    A matrix assembled from several legs is only comparable if the legs ran the
    same code, so the hash travels in every record and the gate can assert it."""
    import hashlib
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("bench_runner.py", "topic_pipeline.py"):
        f = here / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Orchestration — faithful to the base. (Only the unit-list filter on          #
# unit_is_valid is added; resume/timeout/shard/manifest/no-resume are intact.) #
# --------------------------------------------------------------------------- #
def unit_id(dataset, protocol, model, seed):
    return f"{dataset}__{protocol}__{model}__seed{seed}"


def unit_out_path(outdir: Path, dataset, protocol, model, seed) -> Path:
    return outdir / f"{unit_id(dataset, protocol, model, seed)}.json"


def append_manifest(outdir: Path, record: dict):
    with (outdir / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_worker(args):
    outdir = Path(args.outdir)
    out_path = unit_out_path(outdir, args.dataset, args.protocol, args.model, args.seed)
    run_unit(args.dataset, args.protocol, args.model, args.seed, out_path)


# Backends a model/protocol needs at import time. A missing backend used to be
# swallowed as `skipped_missing_backend:<pkg>` per unit: the worker exits 0, the
# orchestrator logs [ok], and an entire protocol leg silently evaporates while the
# gate still passes on stale records. Fail the whole run instead, before unit one.
_MODEL_BACKENDS = {
    "tabpfn3": ["tabpfn"], "tabpfn25": ["tabpfn"], "tabpfnv2": ["tabpfn"],
    "tabicl": ["tabicl"], "mitra": ["autogluon.tabular"],
    "realmlp": ["pytabkit"], "tabm": ["pytabkit"],
    "catboost_tuned": ["catboost"], "xgboost_tuned": ["xgboost"],
    "lightgbm_tuned": ["lightgbm"],
}
_PROTOCOL_BACKENDS = {"matched_hpo": ["optuna"]}
_FAMILY_BACKENDS = {"cc18": ["openml"]}
_FAMILY_ROOTS = {"core": "TALENT_DATA_ROOT", "core_hpo20": "TALENT_DATA_ROOT",
                 "tabred": "TABRED_DATA_ROOT"}


_GPU_MODELS = {"tabpfnv2", "tabpfn25", "tabpfn3", "tabicl", "mitra", "realmlp", "tabm"}


def preflight_gpu_exclusive(models, allow_shared: bool) -> None:
    """Refuse to start a GPU leg while another process already holds the card.

    Two legs sharing one card do not queue politely. Measured on this project:
    RealMLP and TabM raise AcceleratorError outright (157 units lost in one
    night), TabICL raises CUDA OOM, and AutoGluon's Mitra does something worse —
    it silently halves its in-context support set and returns a perfectly
    ordinary success. The failures are indistinguishable from genuine model
    limits after the fact, so the only reliable remedy is not to start.

    `--allow-shared-gpu` exists for deliberate exceptions (e.g. a quick smoke on
    a card that is only lightly used), never for production legs."""
    if allow_shared or not (set(models) & _GPU_MODELS):
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        return  # no nvidia-smi: nothing to assert, the run may be CPU-only anyway
    busy = [l for l in out.stdout.splitlines() if l.strip()]
    mine = str(os.getpid())
    busy = [l for l in busy if l.split(",")[0].strip() != mine]
    if busy:
        sys.exit("[runner] PREFLIGHT FAILED — the GPU is already in use:\n  - "
                 + "\n  - ".join(busy)
                 + "\n[runner] a second consumer causes AcceleratorError, CUDA OOM and "
                   "(for Mitra) a SILENT reduction of the in-context support set that "
                   "still reports success.\n[runner] wait for the running leg to print "
                   "'[runner] done', or pass --allow-shared-gpu if this is deliberate.")


def _make_worker_killable_first() -> None:
    """Make the kernel prefer this worker over the orchestrator under memory pressure.

    A single unit can ask for more RAM than the box has (TabICL on a 224k x 299
    industrial split wanted >128 GB). The OOM killer then picks a victim by score,
    and it has twice picked the orchestrator: the whole leg died silently, without
    even a [FAIL] line, losing hours of completed work that were only recoverable
    because the per-unit records were already on disk. Raising the worker's
    oom_score_adj to the maximum makes it the obvious victim, so the orchestrator
    survives, records the failure and carries on with the next unit — which is the
    documented behaviour we rely on for the allowlisted host-OOM cell.

    Best effort: the file is absent or unwritable on some systems, and failing to
    set it must not stop the run.
    """
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("1000")
    except OSError:
        pass


def preflight_disk(outdir, min_free_gb: float = 20.0) -> None:
    """Refuse to start without room to write the results.

    A full disk does not announce itself: units fail with OSError errno 28
    inside write_text, which the runner records as an ordinary fail(rc=1), and
    the orchestrator itself dies when the manifest append fails. The cause then
    looks like a model problem on whichever dataset happened to be next. Check
    once, up front, and say so plainly."""
    import shutil as _sh
    free_gb = _sh.disk_usage(outdir).free / 2**30
    if free_gb < min_free_gb:
        sys.exit(f"[runner] PREFLIGHT FAILED — only {free_gb:.1f} GB free on the "
                 f"volume holding {outdir} (need >= {min_free_gb:.0f} GB).\n"
                 "[runner] a full disk surfaces as unit failures and a dead "
                 "orchestrator, not as a disk error.\n"
                 "[runner] check AutogluonModels/ first: a TabularPredictor without "
                 "an explicit path leaves one directory per fit and never cleans up.")


def preflight(families, protocols, models) -> None:
    """Abort loudly if a backend or a data root is missing."""
    import importlib
    need: dict[str, str] = {}
    for m in models:
        for pkg in _MODEL_BACKENDS.get(m, []):
            need[pkg] = f"model {m}"
    for p in protocols:
        for pkg in _PROTOCOL_BACKENDS.get(p, []):
            need[pkg] = f"protocol {p}"
    for f in families:
        for pkg in _FAMILY_BACKENDS.get(f, []):
            need[pkg] = f"dataset family {f}"

    missing = []
    for pkg, why in need.items():
        try:
            importlib.import_module(pkg)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{pkg} (required by {why}): {type(e).__name__}")

    for f in families:
        var = _FAMILY_ROOTS.get(f)
        if var and not os.environ.get(var):
            missing.append(f"${var} unset (required by dataset family {f})")

    if missing:
        sys.exit("[runner] PREFLIGHT FAILED — refusing to start:\n  - "
                 + "\n  - ".join(missing)
                 + "\n[runner] fix the environment; a missing backend would otherwise be "
                   "recorded as skipped_missing_backend on every unit.")


def run_orchestrator(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preflight(args.datasets.split(","), args.protocols.split(","),
              args.models.split(","))
    if not args.worker:      # workers inherit the parent's already-checked card
        preflight_gpu_exclusive(args.models.split(","), args.allow_shared_gpu)
        preflight_disk(outdir, float(os.environ.get("TEP_MIN_FREE_GB", "20")))

    datasets = expand_families(args.datasets.split(","),
                               Path(args.datasets_dir))
    protocols = args.protocols.split(",")
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    run_started = datetime.now(timezone.utc).isoformat()

    # axis build: cartesian product, then drop invalid (family, protocol) pairs.
    # `datasets` tokens are dataset-family names here; the per-unit pipeline
    # expands a family into its member datasets via runner/datasets/<fam>.yaml.
    units = [(d, p, m, s) for d in datasets for p in protocols
             for m in models for s in seeds if unit_is_valid(d, p)]
    if args.complete_run:
        seen = set()
        man = outdir / "manifest.jsonl"
        if man.exists():
            for line in man.open():
                try:
                    seen.add(json.loads(line)["unit"])
                except Exception:      # torn tail from a killed leg
                    continue
        before = len(units)
        units = [u for u in units if unit_id(*u) not in seen]
        print(f"[runner] --complete-run: {before} cells in the grid, {len(seen)} already "
              f"in this manifest, {len(units)} to compute", flush=True)
        if not units:
            sys.exit("[runner] nothing left to compute — the matrix is already complete")

    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        if args.no_resume and not args.same_host_shards:
            sys.exit("[runner] refuse: --no-resume across shards risks a split-brain "
                     "manifest between boxes. If every shard runs on THIS host, pass "
                     "--same-host-shards; the gate then asserts the shards cover the "
                     "grid exactly once.")
        units = [u for i, u in enumerate(units) if i % shard_n == shard_k]
        print(f"[runner] shard {shard_k}/{shard_n}: {len(units)} units this box", flush=True)

    # One run_meta PER INVOCATION. A --same-host-shards final is several
    # invocations (legs) into one outdir; a single run_meta.json would be
    # clobbered by each leg and the gate would flag every earlier leg's records
    # as stale. Each leg therefore writes run_meta_<stamp>_shard<k>of<n>.json
    # with its OWN unit list, and the gate asserts the legs are pairwise
    # disjoint and cover the manifest exactly once, all on one hostname.
    meta = {
        "run_started": run_started,
        "no_resume": args.no_resume,
        "shard": args.shard,
        "hostname": socket.gethostname(),
        "same_host_shards": bool(getattr(args, "same_host_shards", False)),
        "env": env_fingerprint(),
        "axes": {"datasets": datasets, "protocols": protocols,
                 "models": models, "seeds": seeds},
        "units": [unit_id(*u) for u in units],
        "timeout_s": args.timeout_s,
    }
    if args.same_host_shards:
        stamp = run_started.replace(":", "").replace("-", "").split(".")[0]
        meta_name = f"run_meta_{stamp}_shard{shard_k}of{shard_n}.json"
    else:
        meta_name = "run_meta.json"
    (outdir / meta_name).write_text(json.dumps(meta, indent=2))
    print(f"[runner] {len(units)} units | outdir={outdir} | "
          f"no_resume={args.no_resume} | per-unit timeout={args.timeout_s}s",
          flush=True)

    n_done = n_skip = n_fail = n_timeout = 0
    for d, p, m, s in units:
        out_path = unit_out_path(outdir, d, p, m, s)
        uid = unit_id(d, p, m, s)

        if out_path.exists() and not args.no_resume:
            n_skip += 1
            print(f"[skip] {uid} (output exists)", flush=True)
            continue
        if out_path.exists() and args.no_resume:
            out_path.unlink()

        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--dataset", d, "--protocol", p, "--model", m,
               "--seed", str(s), "--outdir", str(outdir)]
        t0 = time.time()
        status = "ok"
        try:
            subprocess.run(cmd, timeout=args.timeout_s, check=True,
                           preexec_fn=_make_worker_killable_first)
        except subprocess.TimeoutExpired:
            status = "timeout"; n_timeout += 1
            print(f"[TIMEOUT] {uid} > {args.timeout_s}s — unit killed, batch continues",
                  flush=True)
        except subprocess.CalledProcessError as e:
            status = f"fail(rc={e.returncode})"; n_fail += 1
            print(f"[FAIL] {uid} rc={e.returncode} — batch continues", flush=True)
        else:
            n_done += 1
            print(f"[ok] {uid} ({time.time()-t0:.1f}s)", flush=True)

        append_manifest(outdir, {
            "unit": uid, "dataset": d, "protocol": p, "model": m, "seed": s,
            "status": status, "started": run_started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "wall_s": round(time.time() - t0, 1), "no_resume": args.no_resume,
        })

    print(f"[runner] done | ok={n_done} skip={n_skip} fail={n_fail} timeout={n_timeout}",
          flush=True)
    if n_fail or n_timeout:
        print("[runner] some units did not complete — inspect manifest before freezing.",
              flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="internal: run one unit")
    ap.add_argument("--datasets", default="core,cc18,tabred",
                    help="dataset FAMILIES; expanded to member datasets by the pipeline")
    ap.add_argument("--datasets-dir", dest="datasets_dir", default="datasets",
                    help="dir holding <family>.yaml dataset lists (default: runner/datasets)")
    ap.add_argument("--protocols", default="cv_standard,temporal,matched_hpo")
    ap.add_argument("--models",
                    default="tabpfn25,tabicl,mitra,tabpfnv2,"
                            "catboost_tuned,xgboost_tuned,lightgbm_tuned,realmlp,tabm")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--outdir", default="runs/dev")
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=1800,
                    help="per-unit hard wall-clock timeout in seconds "
                         "(raise to 3600 for matched_hpo units)")
    ap.add_argument("--shard", default="0/1",
                    help="k/n: run only units with index%%n==k. Use for parallel "
                         "dev/HPO sweeps across both HPC boxes (A100 --shard 0/2, "
                         "A800 --shard 1/2). The FINAL --no-resume pass must run on ONE "
                         "box (single shard 0/1) so the gate's A1 single-run invariant holds.")
    ap.add_argument("--no-resume", dest="no_resume", action="store_true",
                    help="FINAL pass: recompute every unit into a fresh outdir")
    ap.add_argument("--complete-run", dest="complete_run", action="store_true",
                    help="compute only the cells of the grid that this outdir's "
                         "manifest does not already contain. For finishing a final "
                         "matrix whose earlier leg was interrupted: it is NOT resume "
                         "(nothing is skipped by file existence, and the gate still "
                         "asserts one code hash, one host and exactly-once coverage) "
                         "— the leg simply declares the remaining cells as its grid.")
    ap.add_argument("--allow-shared-gpu", dest="allow_shared_gpu", action="store_true",
                    help="start a GPU leg even though another process holds the card "
                         "(causes AcceleratorError / CUDA OOM / silent context reduction; "
                         "for deliberate smokes only)")
    ap.add_argument("--same-host-shards", dest="same_host_shards", action="store_true",
                    help="permit --no-resume together with --shard i/N when EVERY shard "
                         "runs on this same host; each record is stamped with the hostname "
                         "and the gate asserts the shards cover the grid exactly once")
    ap.add_argument("--dataset"); ap.add_argument("--protocol")
    ap.add_argument("--model"); ap.add_argument("--seed", type=int)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    if a.worker:
        run_worker(a)
    else:
        run_orchestrator(a)
