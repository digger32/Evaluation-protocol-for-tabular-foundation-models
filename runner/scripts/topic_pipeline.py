#!/usr/bin/env python3
"""
topic_pipeline.py — per-unit data + model layer for the tabular-eval-protocol re-benchmark.

The runner (bench_runner.py) calls into exactly three entry points and stays
topic-agnostic:

  expand_family(family, datasets_dir) -> ["family:member", ...]
      Turn a family token (core | cc18 | tabred) into concrete dataset tokens
      by reading runner/datasets/<family>.yaml. cc18 is resolved from the
      OpenML-CC18 suite at run time, minus any dataset that overlaps core.

  load_split(dataset_token, protocol, seed) -> Split
      Load one dataset and produce a protocol-correct, seed-keyed split with the
      TALENT / Gorishniy-2021 preprocessing applied.

  get_predictions(model_name, split, seed, protocol) -> dict
      Fit/inference one model on the split and return PER-PREDICTION outputs
      (probabilities for classification, values + optional predictive sigma for
      regression) plus the held-out labels. E1-E4 (proper scoring, calibration,
      bootstrap rank-stability) are all post-hoc over these saved arrays, so a
      single forward pass per (dataset, model, seed) feeds four of the six
      claims. For matched_hpo, tunable baselines also return the inner-vs-held-
      out optimism gap (gate D1).

The runner's run_unit() is expected to do:

    split = load_split(dataset_token, protocol, seed)
    pred  = get_predictions(model, split, seed, protocol)
    out   = compute_unit_metrics(pred, split)      # scalar metrics + arrays
    out_path.write_text(json.dumps(out))

Heavy backends are imported lazily inside get_predictions so a missing backend
or a single unit's failure is contained to that unit (the runner already isolates
each unit in its own subprocess). A model that cannot run a unit returns a
status string instead of raising, so the batch never stops.

House conventions: deterministic given (dataset, protocol, model, seed); no
global state; safe as an isolated subprocess.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# LightGBM 4.6 + sklearn: fitting with an eval_set makes the sklearn wrapper
# believe the model was fitted "with feature names", and every later
# predict/predict_proba on a plain ndarray then emits a UserWarning. Purely
# cosmetic (numbers identical), but it floods multi-day sweep logs and buries
# the [ok]/[TIMEOUT] lines monitoring depends on — silence exactly this message.
import warnings

# Log hygiene. Each entry below is either FIXED at the source or deliberately
# silenced with a reason; nothing is suppressed merely because it is noisy,
# because a warning that turns out to matter (sklearn no longer renormalises
# log-loss input) must stay visible until it is dealt with.
#
#   FIXED, not silenced:
#     - "y_prob values do not sum to one" -> rows are renormalised before scoring
#       (measured impact on the existing records: log-loss shift < 1e-6, i.e.
#       below the reported digits, so no leg needed recomputing for it)
#     - Mitra ag.max_classes=10        -> declared in MODEL_CAPS, unit skipped first
#
#   SILENCED (cosmetic, cannot be fixed here, floods multi-day sweep logs and
#   buries the [ok]/[FAIL] lines that monitoring depends on):
warnings.filterwarnings(  # lightgbm+sklearn: eval_set makes the wrapper expect names
    "ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings(  # sklearn imputer on all-NaN columns; the column is dropped
    "ignore", message="Skipping features without any observed values", category=UserWarning)
warnings.filterwarnings(  # openml dataset_format="array": deprecated but kept on
    # purpose mid-matrix (see _load_openml_task) — the dataframe form encodes
    # categoricals differently and would break comparability with the legs
    # already computed. Migrate between complete runs, not inside one.
    "ignore", message=".*dataset_format.*", category=FutureWarning)
# torch>=2.8 renamed the variable and warns once per worker process; it is set
# inside a dependency, not by us, so the only lever on our side is the filter.
os.environ.setdefault("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
if not os.environ["PYTORCH_ALLOC_CONF"]:
    os.environ.pop("PYTORCH_ALLOC_CONF", None)

# --------------------------------------------------------------------------- #
# Model capability registry.                                                  #
# Used to (a) skip invalid units cleanly (e.g. a classification-only TFM on a  #
# regression task, or a TFM on a dataset whose feature count exceeds its cap), #
# and (b) decide which models are tuned under matched_hpo.                     #
# Caps are conservative defaults; override via env if a checkpoint relaxes one.#
# --------------------------------------------------------------------------- #
MODEL_CAPS: dict[str, dict[str, Any]] = {
    # tabular foundation models (inference-only; no per-dataset HPO)
    # max_classes: documented checkpoint limits (TabPFN v2/2.5 <= 10 classes,
    # TabPFN-3 <= 160). max_rows on tabicl: in-context attention on >50k support
    # rows requested 74-177 GiB on an A100-40GB; subsample like the TabPFNs.
    "tabpfn3":       dict(kind="tfm", regression=True,  max_features=2000, max_rows=50000, max_classes=160, tuned=False),
    "tabpfn25":      dict(kind="tfm", regression=True,  max_features=2000, max_rows=50000, max_classes=10,  tuned=False),
    "tabpfnv2":      dict(kind="tfm", regression=True,  max_features=500,  max_rows=10000, max_classes=10,  tuned=False),
    "tabicl":        dict(kind="tfm", regression=False, max_features=None, max_rows=50000, max_classes=None, tuned=False),
    # mitra: AutoGluon's memory-safety estimator scales with rows x one-hot width and
    # pre-emptively skips the model past ~90% of host RAM (observed: 7.9k x 381 one-hot
    # -> 139 GB estimate on a 128 GB box). 5k support rows is within Mitra's native
    # in-context regime (pretrained at <=640 rows); residual estimator refusals become
    # skipped_ag_memory (see the adapter), not errors.
    # max_classes=10: AutoGluon enforces ag.max_classes=10 for Mitra and, past
    # it, declines to train the model — leaving the predictor with no model at
    # all, which surfaces as the same "No models were trained successfully" as a
    # memory refusal. Without this row the unit is recorded as
    # skipped_ag_memory, i.e. a class limit mislabelled as a memory limit
    # (observed on 15 units: core:ASP-POTASSCO, cc18:6, cc18:3022, all 11-class).
    # Declaring the cap here makes it a skipped_class_cap like TabPFN's, which is
    # what it is, and the unit is skipped before the fit rather than after it.
    "mitra":         dict(kind="tfm", regression=True,  max_features=None, max_rows=5000,  max_classes=10, tuned=False,
                          # max_cells is an OPTIONAL pre-filter on rows x encoded
                          # width. The primary mechanism is different: an actual
                          # context reduction is caught from AutoGluon's own report
                          # and the unit becomes skipped_context_reduced (see
                          # _run_tfm). This threshold only saves time on hopeless
                          # datasets and is DISABLED by default, so that a limit is
                          # declared only where it was observed.
                          max_cells=None),
    # tunable baselines (HPO under matched_hpo; defaults under cv_standard/temporal)
    "catboost_tuned":  dict(kind="gbdt", regression=True, max_features=None, max_rows=None, max_classes=None, tuned=True),
    "xgboost_tuned":   dict(kind="gbdt", regression=True, max_features=None, max_rows=None, max_classes=None, tuned=True),
    "lightgbm_tuned":  dict(kind="gbdt", regression=True, max_features=None, max_rows=None, max_classes=None, tuned=True),
    "realmlp":         dict(kind="nn",   regression=True, max_features=None, max_rows=None, max_classes=None, tuned=True),
    "tabm":            dict(kind="nn",   regression=True, max_features=None, max_rows=None, max_classes=None, tuned=True),
}

# Number of inner-CV folds for the matched_hpo optimism-gap measurement.
HPO_INNER_FOLDS = 5
HPO_TRIALS = int(os.environ.get("TEP_HPO_TRIALS", "100"))

# --------------------------------------------------------------------------- #
# GBDT iteration budgets + early stopping (v6 boosting rework).                #
# Search trials run under a moderate tree ceiling with early stopping; only    #
# the final refit of the selected configuration gets the full ceiling. The     #
# number of trees is chosen by ES on a held-out eval set and is part of the    #
# model (exactly as pytabkit fits RealMLP/TabM against X_val), not a leak:     #
# during matched_hpo the ES eval fold is carved from the INNER-train portion,  #
# so the inner-CV score fold stays untouched and optimism_gap keeps its        #
# semantics (inner held-out minus test).                                       #
# --------------------------------------------------------------------------- #
GBDT_SEARCH_ITER = int(os.environ.get("TEP_GBDT_SEARCH_ITER", "500"))
GBDT_FULL_ITER = int(os.environ.get("TEP_GBDT_FULL_ITER", "1000"))
GBDT_ES_ROUNDS = int(os.environ.get("TEP_GBDT_ES_ROUNDS", "50"))
GBDT_ES_VAL_FRAC = float(os.environ.get("TEP_GBDT_ES_VAL_FRAC", "0.15"))


# --------------------------------------------------------------------------- #
# Split container.                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    dataset: str                       # "core:jasmine"
    family: str                        # core | cc18 | tabred
    task_type: str                     # "classification" | "regression"
    n_classes: int                     # 1 for regression
    # design matrices, already preprocessed to a numeric float array unless a
    # model wants the raw categoricals (see cat_idx / raw_cat for CatBoost)
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_idx: np.ndarray                # column indices of numeric features (post one-hot: all)
    cat_idx: np.ndarray                # column indices of categorical features in raw_* matrices
    raw_train: np.ndarray | None = None  # pre-one-hot (ordinal-coded) for CatBoost native cats
    raw_val: np.ndarray | None = None
    raw_test: np.ndarray | None = None
    reg_mean: float = 0.0              # regression target standardisation (undo for reporting)
    reg_std: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


# --------------------------------------------------------------------------- #
# Family expansion.                                                           #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def _core_openml_ids(datasets_dir: Path) -> set[int]:
    """OpenML dataset ids of the TALENT-tiny core members, for cc18 dedup.
    Reads each core/<name>/info.json 'openml_id'; falls back to the verified
    name->openml_id map (datasets/core_openml_ids.json, produced by
    build_openml_id_map.py) when info.json does not carry the id. Without either,
    a member contributes no id and is simply not deduplicated against cc18 -- the
    caller should ensure the map exists so the independent leg is truly disjoint."""
    cfg = _load_yaml(datasets_dir / "core.yaml")
    root = Path(os.environ.get(cfg.get("data_root_env", "TALENT_DATA_ROOT"), ""))
    id_map = {}
    map_path = datasets_dir / "core_openml_ids.json"
    if map_path.exists():
        try:
            id_map = {k: v for k, v in json.loads(map_path.read_text()).items()
                      if v is not None}
        except Exception:
            id_map = {}
    ids: set[int] = set()
    for name in cfg.get("datasets", []):
        oid = None
        info = root / name / "info.json"
        if info.exists():
            try:
                oid = json.loads(info.read_text()).get("openml_id")
            except Exception:
                oid = None
        if oid is None:
            oid = id_map.get(name)
        if oid:
            ids.add(int(oid))
    return ids


def expand_family(family: str, datasets_dir: str | Path) -> list[str]:
    """Expand a family token into concrete "family:member" dataset tokens.

    core / tabred : read the explicit `datasets:` list from the yaml.
    cc18          : resolve the OpenML-CC18 suite live, drop tasks whose dataset
                    overlaps core, return "cc18:<task_id>" tokens.
    Empty / unfilled list falls back to a single family unit, mirroring the
    runner's stub behaviour.
    """
    datasets_dir = Path(datasets_dir)
    cfg = _load_yaml(datasets_dir / f"{family}.yaml")

    if cfg.get("resolve") == "static_snapshot":
        # Final-pass mode: read the frozen resolution (task ids already
        # deduplicated against core at snapshot time) instead of the network.
        # A network outage can then never silently shrink the grid. Two frozen
        # formats are accepted: cc18_resolved.json (the dev-era resolution,
        # key "kept") and the yaml written by snapshot_cc18.py (key "tasks").
        snap_path = datasets_dir / cfg["snapshot"]
        if snap_path.suffix == ".json":
            snap = json.loads(snap_path.read_text())
            ids = [int(r["task_id"]) for r in snap["kept"]]
        else:
            snap = _load_yaml(snap_path)
            ids = [int(r["task_id"]) for r in snap["tasks"]]
        emit = cfg.get("emit_as", family)
        tokens = [f"{emit}:{tid}" for tid in ids]
        cap = cfg.get("max_datasets")
        return tokens[:cap] if cap else tokens

    if cfg.get("resolve") == "openml_suite":
        import openml  # lazy; only needed on the experiment box
        suite = openml.study.get_suite(cfg["openml_suite"])
        exclude = _core_openml_ids(datasets_dir) if cfg.get("dedup") else set()
        tokens: list[str] = []
        for tid in suite.tasks:
            t = openml.tasks.get_task(tid, download_data=False,
                                      download_qualities=False, download_splits=False)
            if int(t.dataset_id) in exclude:
                continue
            tokens.append(f"{family}:{tid}")
        cap = cfg.get("max_datasets")
        return tokens[:cap] if cap else tokens

    members = cfg.get("datasets") or []
    if not members:
        return [family]  # fallback single unit (stub behaviour)
    # `emit_as` lets a SUBSET yaml (e.g. core_hpo20) emit tokens of its parent
    # family, so loaders, VALID_FAMILY_PROTOCOL and the analysis all see plain
    # "core:<name>" units; the subset file only narrows the CLI axis.
    emit = cfg.get("emit_as", family)
    return [f"{emit}:{m}" for m in members]


# --------------------------------------------------------------------------- #
# Preprocessing — TALENT / Gorishniy et al. (2021).                           #
# --------------------------------------------------------------------------- #
_CAT_MISSING_TOKEN = "-1"


def _fit_preprocess(X_num, X_cat, X_num_v, X_cat_v, X_num_t, X_cat_t, model_wants_raw_cat):
    """Fit on train only; transform val/test. Returns (one-hot float matrices,
    ordinal-coded raw matrices, num_idx, cat_idx).

    numeric : column-mean impute (train means) + standardise (train stats)
    categorical: ordinal encode (train categories; unseen / missing -> "-1");
                 one-hot for the float matrix, ordinal-coded kept in raw matrix
                 for native-categorical models (CatBoost).
    """
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    parts_float, parts_raw = [], []
    num_idx, cat_idx = [], []
    col = 0

    if X_num is not None and X_num.shape[1] > 0:
        imp = SimpleImputer(strategy="mean").fit(X_num)
        sc = StandardScaler().fit(imp.transform(X_num))
        def _num(a): return sc.transform(imp.transform(a))
        tr, va, te = _num(X_num), _num(X_num_v), _num(X_num_t)
        parts_float.append((tr, va, te)); parts_raw.append((tr, va, te))
        num_idx = list(range(col, col + tr.shape[1])); col += tr.shape[1]

    if X_cat is not None and X_cat.shape[1] > 0:
        # Cast to str (not object): OpenML's array format can mix str categories
        # and non-null float codes in one column, which makes OrdinalEncoder's
        # internal sort raise on str-vs-float. A uniform string column is a no-op
        # for already-string categoricals; for numeric ones it only relabels the
        # ordinal codes, which one-hot (order-invariant) and CatBoost native
        # categoricals are insensitive to.
        Xc = np.where(_isnull(X_cat), _CAT_MISSING_TOKEN, X_cat).astype(str)
        Xcv = np.where(_isnull(X_cat_v), _CAT_MISSING_TOKEN, X_cat_v).astype(str)
        Xct = np.where(_isnull(X_cat_t), _CAT_MISSING_TOKEN, X_cat_t).astype(str)
        ore = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1).fit(Xc)
        ord_tr, ord_va, ord_te = ore.transform(Xc), ore.transform(Xcv), ore.transform(Xct)
        # raw ordinal matrix (for CatBoost native categoricals)
        raw_start = sum(p[0].shape[1] for p in parts_raw)
        parts_raw.append((ord_tr, ord_va, ord_te))
        cat_idx = list(range(raw_start, raw_start + ord_tr.shape[1]))
        # one-hot for the float matrix
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xc)
        oh_tr, oh_va, oh_te = ohe.transform(Xc), ohe.transform(Xcv), ohe.transform(Xct)
        parts_float.append((oh_tr, oh_va, oh_te))

    def _stack(parts, i): return np.hstack([p[i] for p in parts]) if parts else np.empty((0, 0))
    Xf = [_stack(parts_float, i) for i in range(3)]
    Xr = [_stack(parts_raw, i) for i in range(3)]
    return Xf, Xr, np.array(num_idx), np.array(cat_idx)


def _isnull(a):
    try:
        return np.equal(a, None) | (a != a)  # None or NaN
    except Exception:
        return np.zeros(a.shape, dtype=bool)


# --------------------------------------------------------------------------- #
# Loaders.                                                                    #
# --------------------------------------------------------------------------- #
def _load_talent_folder(name: str):
    """Read one TALENT dataset folder -> (N, C, y) concatenated over the shipped
    train/val/test, plus info. We re-split ourselves so cv_standard can vary the
    split by seed; the shipped split is only the default partition."""
    root = Path(os.environ["TALENT_DATA_ROOT"]) / name
    info = json.loads((root / "info.json").read_text())

    def _cat(stub):
        f = root / f"{stub}.npy"
        return np.load(f, allow_pickle=True) if f.exists() else None

    N = [_cat(f"N_{s}") for s in ("train", "val", "test")]
    C = [_cat(f"C_{s}") for s in ("train", "val", "test")]
    y = [_cat(f"y_{s}") for s in ("train", "val", "test")]
    Nall = np.vstack([a for a in N if a is not None]) if any(a is not None for a in N) else None
    Call = np.vstack([a for a in C if a is not None]) if any(a is not None for a in C) else None
    # Some TALENT folders ship the target as a column vector (n, 1); concatenate
    # preserves the trailing axis, which then breaks pd.qcut stratification (and
    # downstream label-encoding). Force 1-D. No-op for already-1-D targets.
    yall = np.concatenate([a for a in y if a is not None]).reshape(-1)
    return Nall, Call, yall, info


def _stratify_bins(y, task_type, n_bins=10):
    """Stratification labels: y for classification; quantile bins for regression.
    Targets are normalised to 1-D by the loaders, so qcut receives a flat array."""
    if task_type == "classification":
        return y
    import pandas as pd
    return pd.qcut(y.astype(float), q=min(n_bins, len(np.unique(y))),
                  labels=False, duplicates="drop")


def _holdout_split(Nall, Call, yall, task_type, seed, fracs):
    """Seed-keyed stratified 64/16/20 split."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(yall))
    strat = _stratify_bins(yall, task_type)
    tr_f, va_f, te_f = fracs
    idx_tr, idx_tmp = train_test_split(idx, test_size=(va_f + te_f),
                                       random_state=seed, stratify=strat)
    rel = _stratify_bins(yall[idx_tmp], task_type)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=te_f / (va_f + te_f),
                                      random_state=seed, stratify=rel)
    def sl(a, i): return None if a is None else a[i]
    return (sl(Nall, idx_tr), sl(Call, idx_tr), yall[idx_tr],
            sl(Nall, idx_va), sl(Call, idx_va), yall[idx_va],
            sl(Nall, idx_te), sl(Call, idx_te), yall[idx_te])


def _load_tabred_folder(name: str):
    """TabReD ships an explicit temporal split. Layout mirrors TALENT but with a
    time-ordered split; we honour the shipped indices."""
    root = Path(os.environ["TABRED_DATA_ROOT"]) / name
    info = json.loads((root / "info.json").read_text())
    def _opt(p):
        f = root / p
        return np.load(f, allow_pickle=True) if f.exists() else None
    def _opty(p):  # target may ship as a column vector; force 1-D (see core path)
        a = _opt(p)
        return None if a is None else np.asarray(a).reshape(-1)
    parts = {}
    for s in ("train", "val", "test"):
        parts[s] = (_opt(f"N_{s}.npy"), _opt(f"C_{s}.npy"), _opty(f"y_{s}.npy"))
    return parts, info


_TASKTYPE_OVERRIDE_CACHE: dict = {}


def _task_type_override(family: str, member: str, datasets_dir=None):
    """Return an overriding task type for one member, or None.

    TALENT-tiny ships four datasets whose Table-1 marker is binary but whose
    info.json says task_type=regression; their y arrays are uint8 with exactly
    the values {0, 1}. A harness that trusts the metadata silently regresses
    binary labels and drops every classification-only system on them. The
    correction lives in the dataset yaml (`task_type_override`), never in code,
    and every affected unit records where its task type came from."""
    if datasets_dir is None:
        datasets_dir = os.environ.get("TEP_DATASETS_DIR", "runner/datasets")
    key = (family, str(datasets_dir))
    if key not in _TASKTYPE_OVERRIDE_CACHE:
        path = Path(datasets_dir) / f"{family}.yaml"
        cfg = _load_yaml(path) if path.exists() else {}
        _TASKTYPE_OVERRIDE_CACHE[key] = cfg.get("task_type_override") or {}
    return _TASKTYPE_OVERRIDE_CACHE[key].get(member)


def load_split(dataset_token: str, protocol: str, seed: int) -> Split:
    """Load + preprocess one dataset under one protocol/seed."""
    family, _, member = dataset_token.partition(":")
    member = member or dataset_token

    if family == "tabred":
        parts, info = _load_tabred_folder(member)
        task_type = "regression" if info.get("task_type") == "regression" else "classification"
        n_classes = int(info.get("n_classes", 1)) if task_type == "classification" else 1
        (Ntr, Ctr, ytr), (Nva, Cva, yva), (Nte, Cte, yte) = (
            parts["train"], parts["val"], parts["test"])
    else:
        if family == "cc18":
            Nall, Call, yall, info = _load_openml_task(member)
        else:  # core
            Nall, Call, yall, info = _load_talent_folder(member)
        task_type = "regression" if info.get("task_type") == "regression" else "classification"
        n_classes = int(info.get("n_classes", 1)) if task_type == "classification" else 1
        shipped = task_type
        over = _task_type_override(family, member)
        if over:
            uniq = np.unique(np.asarray(yall).reshape(-1))
            if over == "binclass" and len(uniq) != 2:
                raise ValueError(
                    f"{dataset_token}: task_type_override says binclass but y has "
                    f"{len(uniq)} distinct values — refusing to relabel")
            task_type = "regression" if over == "regression" else "classification"
            n_classes = len(uniq) if task_type == "classification" else 1
        fracs = (0.64, 0.16, 0.20)
        (Ntr, Ctr, ytr, Nva, Cva, yva, Nte, Cte, yte) = _holdout_split(
            Nall, Call, yall, task_type, seed, fracs)

    Xf, Xr, num_idx, cat_idx = _fit_preprocess(
        Ntr, Ctr, Nva, Cva, Nte, Cte, model_wants_raw_cat=True)

    # encode classification labels to 0..K-1; standardise regression target
    reg_mean, reg_std = 0.0, 1.0
    if task_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(ytr)
        ytr, yva, yte = le.transform(ytr), le.transform(yva), le.transform(yte)
        n_classes = len(le.classes_)
    else:
        ytr = ytr.astype(float); yva = yva.astype(float); yte = yte.astype(float)
        reg_mean, reg_std = float(ytr.mean()), float(ytr.std() + 1e-12)
        ytr = (ytr - reg_mean) / reg_std
        yva = (yva - reg_mean) / reg_std
        yte = (yte - reg_mean) / reg_std

    return Split(
        dataset=dataset_token, family=family, task_type=task_type, n_classes=n_classes,
        X_train=Xf[0], y_train=ytr, X_val=Xf[1], y_val=yva, X_test=Xf[2], y_test=yte,
        raw_train=Xr[0], raw_val=Xr[1], raw_test=Xr[2],
        num_idx=num_idx, cat_idx=cat_idx, reg_mean=reg_mean, reg_std=reg_std,
        meta={"member": member, "protocol": protocol, "seed": seed,
              "n_train": len(ytr), "n_test": len(yte),
              "openml_id": info.get("openml_id"),
              "task_type_shipped": locals().get("shipped", task_type),
              "task_type_source": ("dataset_yaml_override"
                                   if locals().get("shipped", task_type) != task_type
                                   else "info_json")},
    )


def _load_openml_task(task_id: str):
    """cc18: pull an OpenML task's (X, y) and return TALENT-style arrays + info.
    Resolution happens on the experiment box (OpenML egress)."""
    import openml
    task = openml.tasks.get_task(int(task_id), download_splits=False)
    ds = task.get_dataset()
    # NOTE: staying on the deprecated `dataset_format="array"` ON PURPOSE until
    # the final matrix is complete. The dataframe form returns categorical
    # columns as their original labels, whereas the array form returns integer
    # codes; one-hot encoding then orders the resulting columns differently, so
    # the two loaders can hand a model different design matrices for the same
    # dataset. The gradient-boosting legs already in `runs/final` were loaded
    # with the array form, and a leg loaded the other way would not be strictly
    # comparable with them — the exact class of silent incomparability this
    # paper is about. The FutureWarning is filtered at the top of this module;
    # migrate deliberately, between complete runs, verifying the arrays match.
    X, y, cat_mask, _ = ds.get_data(target=task.target_name,
                                    dataset_format="array")
    X = np.asarray(X, dtype=object)
    cat_mask = np.asarray(cat_mask, dtype=bool)
    Nall = X[:, ~cat_mask].astype(float) if (~cat_mask).any() else None
    Call = X[:, cat_mask].astype(object) if cat_mask.any() else None
    info = {"task_type": "classification", "n_classes": int(len(np.unique(y))),
            "openml_id": int(ds.dataset_id)}
    return Nall, Call, np.asarray(y).reshape(-1), info


# --------------------------------------------------------------------------- #
# get_predictions — the per-unit forward pass.                                #
# --------------------------------------------------------------------------- #
def _skip(model_name, split, reason, **extra) -> dict:
    """A documented, comparable non-result. `extra` carries the evidence behind
    the skip (e.g. the observed context reduction), so a reader of the released
    records can check the limit rather than take it on trust."""
    return {"status": f"skipped_{reason}", "model": model_name,
            "dataset": split.dataset, "task_type": split.task_type, **extra}


def get_predictions(model_name: str, split: Split, seed: int, protocol: str) -> dict:
    """Run one model on one split. Returns per-prediction outputs + (optional)
    optimism gap, never raises on an unrunnable unit.

    Classification -> 'proba' (n_test x n_classes); Regression -> 'pred' (n_test,)
    plus optional 'pred_sigma' when the model exposes a predictive distribution
    (TFMs do; GBDTs do not -> CRPS degenerates to |error| and is flagged).
    """
    caps = MODEL_CAPS.get(model_name)
    if caps is None:
        return _skip(model_name, split, "unknown_model")

    # capability guards -> clean skip, batch continues
    if split.task_type == "regression" and not caps["regression"]:
        return _skip(model_name, split, "no_regression")
    if caps["max_features"] is not None and split.n_features > caps["max_features"]:
        return _skip(model_name, split, "feature_cap")
    if (caps.get("max_classes") is not None and split.task_type == "classification"
            and split.n_classes > caps["max_classes"]):
        # documented model limit (e.g. TabPFN v2/2.5 support <=10 classes) ->
        # capability skip, not an error
        return _skip(model_name, split, "class_cap")
    if caps["max_rows"] is not None and split.meta["n_train"] > caps["max_rows"]:
        # TFM row caps: subsample the in-context/support set deterministically
        split = _subsample_support(split, caps["max_rows"], seed)
    # Cell budget (rows x encoded width) AFTER the row cap. Mitra's VRAM use is
    # driven by the product, not by rows alone: one-hot encoding turns 17 raw
    # columns into 381 for online_shoppers, and AutoGluon then silently halves
    # the in-context support set instead of failing. A dataset past the budget
    # is skipped as a documented limit, which is comparable; a unit fitted on a
    # quietly reduced context is not. The threshold is measured, not guessed
    # (scripts/measure_mitra_width.py); H1 in the gate remains the backstop for
    # residual cases, because the reduction is not perfectly deterministic.
    max_cells = caps.get("max_cells")
    if max_cells is not None:
        cells = split.meta["n_train"] * split.n_features
        if cells > max_cells:
            return _skip(model_name, split, "cell_cap")

    # pareto_hpo (accuracy-vs-calibration front) is tunable-classification only.
    if protocol == "pareto_hpo" and not caps["tuned"]:
        return _skip(model_name, split, "pareto_tunable_only")
    do_pareto = caps["tuned"] and (protocol == "pareto_hpo")
    if do_pareto and split.task_type == "regression":
        return _skip(model_name, split, "no_pareto_for_regression")

    out = {"status": "ok", "model": model_name, "dataset": split.dataset,
           "family": split.family, "task_type": split.task_type,
           "n_classes": split.n_classes, "seed": seed, "protocol": protocol,
           "y_true": split.y_test.tolist(),
           "reg_mean": split.reg_mean, "reg_std": split.reg_std,
           "optimism_gap": None, "inner_score": None, "test_score": None}

    tuned = caps["tuned"] and (protocol == "matched_hpo")
    try:
        if caps["kind"] == "gbdt":
            pred = _run_gbdt(model_name, split, seed, tuned, out, do_pareto)
        elif caps["kind"] == "nn":
            pred = _run_nn(model_name, split, seed, tuned, out)
        else:  # tfm
            pred = _run_tfm(model_name, split, seed)
    except ImportError as e:
        return _skip(model_name, split, f"missing_backend:{e.name}")
    except Exception as e:  # noqa: BLE001 - one unit must not take down the batch
        return {"status": f"error:{type(e).__name__}", "model": model_name,
                "dataset": split.dataset, "msg": str(e)[:300]}

    out.update(pred)
    return out


def _subsample_support(split: Split, max_rows: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(split.y_train), size=max_rows, replace=False)
    import copy
    s = copy.copy(split)
    s.X_train = split.X_train[idx]; s.y_train = split.y_train[idx]
    if split.raw_train is not None:
        s.raw_train = split.raw_train[idx]
    s.meta = {**split.meta, "n_train": max_rows, "subsampled": True}
    return s


# ---- GBDT baselines (xgboost / lightgbm / catboost) ------------------------ #
def _catboost_frame(X, cat_idx):
    """CatBoost rejects float ndarrays with nonempty cat_features (ordinal codes
    come out of OrdinalEncoder as float64). Convert to a DataFrame with string
    column names, casting categorical columns to int-strings; numeric columns
    stay float. Returns (frame, cat_feature_names)."""
    import pandas as pd
    df = pd.DataFrame(np.asarray(X))
    df.columns = [f"c{j}" for j in range(df.shape[1])]
    names = [f"c{j}" for j in cat_idx]
    for n in names:
        df[n] = df[n].astype(np.int64).astype(str)
    return df, names


def _run_gbdt(model_name, split, seed, tuned, out, do_pareto=False):
    is_cls = split.task_type == "classification"
    if do_pareto:
        return _run_gbdt_pareto(model_name, split, seed, out)
    if tuned:
        best, inner = _tune_gbdt(model_name, split, seed, is_cls)
        out["inner_score"] = inner
    else:
        best = {}
    # Full ceiling + ES on the protocol-reserved validation split. cv_standard's
    # declared TALENT protocol is "early stopping on the validation metric", and
    # pytabkit already fits RealMLP/TabM against X_val — this puts the GBDTs on
    # the same footing instead of a blind 1000-tree fit.
    est = _make_gbdt(model_name, split, seed, is_cls, best,
                     iterations=GBDT_FULL_ITER, es=True)
    # CatBoost uses raw ordinal cats natively; others use the one-hot float matrix
    cat_names = None
    if model_name == "catboost_tuned" and split.cat_idx.size:
        Xtr, cat_names = _catboost_frame(split.raw_train, split.cat_idx)
        Xva, _ = _catboost_frame(split.raw_val, split.cat_idx)
        Xte, _ = _catboost_frame(split.raw_test, split.cat_idx)
    elif model_name == "catboost_tuned":
        Xtr, Xva, Xte = split.raw_train, split.raw_val, split.raw_test
    else:
        Xtr, Xva, Xte = split.X_train, split.X_val, split.X_test
    best_iter = _fit_gbdt(est, model_name, Xtr, split.y_train,
                          eval_set=(Xva, split.y_val), cat_names=cat_names)
    out["gbdt_es"] = {"search_iter": GBDT_SEARCH_ITER, "full_iter": GBDT_FULL_ITER,
                      "es_rounds": GBDT_ES_ROUNDS, "es_val_frac": GBDT_ES_VAL_FRAC,
                      "best_iter": best_iter, "threads": _gbdt_threads(model_name)}
    if is_cls:
        proba = est.predict_proba(Xte)
        out["proba"] = proba.tolist()
        out["test_score"] = _auc(split.y_test, proba, split.n_classes)
    else:
        pred = est.predict(Xte).astype(float)
        out["pred"] = pred.tolist()
        out["test_score"] = -_rmse(split.y_test, pred)  # higher is better
    if tuned and out["inner_score"] is not None and out["test_score"] is not None:
        out["optimism_gap"] = out["inner_score"] - out["test_score"]
    return {}


def _run_gbdt_pareto(model_name, split, seed, out):
    """Trace this system's accuracy-vs-calibration Pareto front (classification).
    Tune multi-objective (NSGA-II), refit each non-dominated config on the full
    train split, score held-out (accuracy, ECE, AUC), and attach the front to the
    record. The primary prediction is the max-accuracy front member, so the record
    still carries proba and the standard scalar metrics for the cross-system view."""
    front = _tune_gbdt_pareto(model_name, split, seed)
    y = split.y_test
    cat_names = None
    if model_name == "catboost_tuned" and split.cat_idx.size:
        Xtr, cat_names = _catboost_frame(split.raw_train, split.cat_idx)
        Xva, _ = _catboost_frame(split.raw_val, split.cat_idx)
        Xte, _ = _catboost_frame(split.raw_test, split.cat_idx)
    elif model_name == "catboost_tuned":
        Xtr, Xva, Xte = split.raw_train, split.raw_val, split.raw_test
    else:
        Xtr, Xva, Xte = split.X_train, split.X_val, split.X_test
    evald, best_proba, best_acc = [], None, -1.0
    for cfg in front:
        est = _make_gbdt(model_name, split, seed, True, cfg["params"],
                         iterations=GBDT_FULL_ITER, es=True)
        _fit_gbdt(est, model_name, Xtr, split.y_train,
                  eval_set=(Xva, split.y_val), cat_names=cat_names)
        proba = est.predict_proba(Xte)
        acc = float((proba.argmax(1) == y).mean())
        evald.append({"params": cfg["params"],
                      "inner_acc": cfg["inner_acc"], "inner_ece": cfg["inner_ece"],
                      "test_acc": acc, "test_ece": _ece(y, proba),
                      "test_auc": _auc(y, proba, split.n_classes)})
        if acc > best_acc:
            best_acc, best_proba = acc, proba
    out["pareto_front"] = evald
    if best_proba is not None:
        out["proba"] = best_proba.tolist()
        out["test_score"] = _auc(y, best_proba, split.n_classes)
    return {}


def _gbdt_threads(model_name):
    """Per-library thread cap, falling back to the shared TEP_GBDT_THREADS.
    xgb/lgb pay heavy OpenMP sync costs on small data and must be pinned tight;
    CatBoost self-throttles and can be given more cores (TEP_CB_THREADS)."""
    shared = os.environ.get("TEP_GBDT_THREADS", "-1") or "-1"
    key = {"xgboost_tuned": "TEP_XGB_THREADS",
           "lightgbm_tuned": "TEP_LGB_THREADS",
           "catboost_tuned": "TEP_CB_THREADS"}[model_name]
    return int(os.environ.get(key, shared) or shared)


def _make_gbdt(model_name, split, seed, is_cls, params, iterations=None, es=False):
    # Thread cap: three concurrent shards each grabbing every core (n_jobs=-1)
    # thrash the OpenMP pools; on tiny multiclass datasets (wine-quality-red,
    # 1.6k rows) a default LightGBM fit then takes >1 h instead of minutes.
    # `iterations`: tree ceiling (GBDT_SEARCH_ITER during HPO trials,
    # GBDT_FULL_ITER for the final refit). `es=True` arms early stopping;
    # the caller must then fit through _fit_gbdt with an eval set.
    n_threads = _gbdt_threads(model_name)
    n_iter = iterations or GBDT_FULL_ITER
    if model_name == "xgboost_tuned":
        import xgboost as xgb
        cls = xgb.XGBClassifier if is_cls else xgb.XGBRegressor
        base = dict(n_estimators=n_iter, tree_method="hist", random_state=seed,
                    n_jobs=n_threads, **(params or {}))
        if es:  # xgboost>=2 accepts ES only at construction time
            base.setdefault("early_stopping_rounds", GBDT_ES_ROUNDS)
        if is_cls:
            base.setdefault("objective",
                            "multi:softprob" if split.n_classes > 2 else "binary:logistic")
        return cls(**base)
    if model_name == "lightgbm_tuned":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if is_cls else lgb.LGBMRegressor
        return cls(n_estimators=n_iter, random_state=seed, n_jobs=n_threads,
                   verbosity=-1, **(params or {}))  # ES via callback in _fit_gbdt
    if model_name == "catboost_tuned":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if is_cls else CatBoostRegressor
        extra = {"thread_count": n_threads} if n_threads > 0 else {}
        if es:
            extra.update(early_stopping_rounds=GBDT_ES_ROUNDS, use_best_model=True)
        return cls(iterations=n_iter, random_seed=seed, verbose=False,
                   **extra, **(params or {}))
    raise ValueError(model_name)


def _fit_gbdt(est, model_name, Xtr, ytr, eval_set=None, cat_names=None):
    """Fit one GBDT, with per-library early stopping when eval_set=(Xva, yva)
    is given (the estimator must have been built with es=True). Returns the
    best iteration (None when ES was not armed or the library reports none).
    ES monitors each library's native loss (logloss / RMSE) — a proper score."""
    kw = {"cat_features": cat_names} if cat_names else {}
    if eval_set is None:
        est.fit(Xtr, ytr, **kw)
        return None
    Xva, yva = eval_set
    if model_name == "lightgbm_tuned":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(GBDT_ES_ROUNDS, verbose=False)])
        return getattr(est, "best_iteration_", None)
    if model_name == "xgboost_tuned":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return getattr(est, "best_iteration", None)
    est.fit(Xtr, ytr, eval_set=(Xva, yva), **kw)          # catboost
    bi = est.get_best_iteration()
    return int(bi) if bi is not None else None


def _es_carve(itr, y, is_cls, seed):
    """Split an inner-train index set into (fit, es_eval) for early stopping.
    Stratified when possible; a rare class with <2 members falls back to a
    plain shuffle split rather than crashing the unit."""
    from sklearn.model_selection import train_test_split
    strat = y[itr] if is_cls else None
    try:
        return train_test_split(itr, test_size=GBDT_ES_VAL_FRAC,
                                random_state=seed, stratify=strat)
    except ValueError:
        return train_test_split(itr, test_size=GBDT_ES_VAL_FRAC,
                                random_state=seed)


def _tune_gbdt(model_name, split, seed, is_cls):
    """Nested Optuna search on inner CV of the TRAIN split. Returns (best_params,
    best_inner_score). The inner score is the matched metric so the optimism gap
    = inner_score - test_score is directly comparable (gate D1)."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, KFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = (split.raw_train if model_name == "catboost_tuned" else split.X_train), split.y_train
    splitter = (StratifiedKFold(HPO_INNER_FOLDS, shuffle=True, random_state=seed)
                if is_cls else KFold(HPO_INNER_FOLDS, shuffle=True, random_state=seed))

    def space(t):
        if model_name == "xgboost_tuned":
            return dict(max_depth=t.suggest_int("max_depth", 3, 10),
                        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        subsample=t.suggest_float("subsample", 0.5, 1.0),
                        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        if model_name == "lightgbm_tuned":
            return dict(num_leaves=t.suggest_int("num_leaves", 15, 255),
                        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        feature_fraction=t.suggest_float("feature_fraction", 0.5, 1.0),
                        min_child_samples=t.suggest_int("min_child_samples", 5, 100))
        return dict(depth=t.suggest_int("depth", 4, 10),
                    learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True))

    def objective(t):
        p = space(t); scores = []
        use_cb_frame = model_name == "catboost_tuned" and split.cat_idx.size
        for itr, iva in splitter.split(X, y if is_cls else None):
            # ES eval fold is carved from the INNER-train portion; the scored
            # fold iva never feeds early stopping, so the inner score (and the
            # optimism gap built on it) keeps its held-out semantics.
            ifit, ies = _es_carve(itr, y, is_cls, seed)
            est = _make_gbdt(model_name, split, seed, is_cls, p,
                             iterations=GBDT_SEARCH_ITER, es=True)
            cat_names = None
            if use_cb_frame:
                Xfit, cat_names = _catboost_frame(X[ifit], split.cat_idx)
                Xes, _ = _catboost_frame(X[ies], split.cat_idx)
                Xsc, _ = _catboost_frame(X[iva], split.cat_idx)
            else:
                Xfit, Xes, Xsc = X[ifit], X[ies], X[iva]
            _fit_gbdt(est, model_name, Xfit, y[ifit],
                      eval_set=(Xes, y[ies]), cat_names=cat_names)
            if is_cls:
                scores.append(_auc(y[iva], est.predict_proba(Xsc), split.n_classes))
            else:
                scores.append(-_rmse(y[iva], est.predict(Xsc).astype(float)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def _tune_gbdt_pareto(model_name, split, seed):
    """Multi-objective NSGA-II search (Optuna) over the SAME GBDT space, tracing
    the accuracy-vs-calibration Pareto front of CONFIGURATIONS for one tunable
    system on one classification dataset. Objectives on inner CV of the TRAIN
    split: maximise accuracy, minimise ECE. Returns the Pareto set of trials as
    [{params, inner_acc, inner_ece}, ...].

    NSGA-II is the standard multi-objective evolutionary algorithm and is used
    here via Optuna's NSGAIISampler so no second dependency is introduced; a
    pymoo backend can be swapped in behind TEP_PARETO_BACKEND=pymoo for an
    independent cross-check of the front (see scripts/pareto_hpo.py)."""
    import optuna
    from sklearn.model_selection import StratifiedKFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = (split.raw_train if model_name == "catboost_tuned" else split.X_train), split.y_train
    splitter = StratifiedKFold(HPO_INNER_FOLDS, shuffle=True, random_state=seed)

    def space(t):
        if model_name == "xgboost_tuned":
            return dict(max_depth=t.suggest_int("max_depth", 3, 10),
                        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        subsample=t.suggest_float("subsample", 0.5, 1.0),
                        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        if model_name == "lightgbm_tuned":
            return dict(num_leaves=t.suggest_int("num_leaves", 15, 255),
                        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        feature_fraction=t.suggest_float("feature_fraction", 0.5, 1.0),
                        min_child_samples=t.suggest_int("min_child_samples", 5, 100))
        return dict(depth=t.suggest_int("depth", 4, 10),
                    learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True))

    def objective(t):
        p = space(t); accs, eces = [], []
        use_cb_frame = model_name == "catboost_tuned" and split.cat_idx.size
        for itr, iva in splitter.split(X, y):
            ifit, ies = _es_carve(itr, y, True, seed)
            est = _make_gbdt(model_name, split, seed, True, p,
                             iterations=GBDT_SEARCH_ITER, es=True)
            cat_names = None
            if use_cb_frame:
                # same fix as bug C in _tune_gbdt: CatBoost rejects float
                # ndarrays with nonempty cat_features — go through the frame
                Xfit, cat_names = _catboost_frame(X[ifit], split.cat_idx)
                Xes, _ = _catboost_frame(X[ies], split.cat_idx)
                Xsc, _ = _catboost_frame(X[iva], split.cat_idx)
            else:
                Xfit, Xes, Xsc = X[ifit], X[ies], X[iva]
            _fit_gbdt(est, model_name, Xfit, y[ifit],
                      eval_set=(Xes, y[ies]), cat_names=cat_names)
            proba = est.predict_proba(Xsc)
            accs.append(float((proba.argmax(1) == y[iva]).mean()))
            eces.append(_ece(y[iva], proba))
        return float(np.mean(accs)), float(np.mean(eces))

    study = optuna.create_study(directions=["maximize", "minimize"],
                                sampler=optuna.samplers.NSGAIISampler(seed=seed))
    study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=False)
    front = []
    for tr in study.best_trials:           # the non-dominated set
        front.append({"params": tr.params,
                      "inner_acc": float(tr.values[0]),
                      "inner_ece": float(tr.values[1])})
    return front


# ---- neural baselines (realmlp / tabm) ------------------------------------- #
def _run_nn(model_name, split, seed, tuned, out):
    """RealMLP / TabM via pytabkit if installed. Falls back to a clean
    missing-backend skip otherwise (handled by the caller)."""
    is_cls = split.task_type == "classification"
    if model_name == "realmlp":
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
        Est = RealMLP_TD_Classifier if is_cls else RealMLP_TD_Regressor
    else:  # tabm
        from pytabkit import TabM_D_Classifier, TabM_D_Regressor
        Est = TabM_D_Classifier if is_cls else TabM_D_Regressor
    est = Est(random_state=seed)
    est.fit(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)
    if is_cls:
        proba = est.predict_proba(split.X_test)
        out["proba"] = proba.tolist()
        out["test_score"] = _auc(split.y_test, proba, split.n_classes)
    else:
        pred = est.predict(split.X_test).astype(float).ravel()
        out["pred"] = pred.tolist()
        out["test_score"] = -_rmse(split.y_test, pred)
    return {}


# ---- tabular foundation models --------------------------------------------- #
# Each TabPFN token is pinned to a specific checkpoint. The local `tabpfn`
# package (>=6) selects the version with create_default_for_version(ModelVersion
# .V2 | .V2_5 | .V2_6), and V3 is the package default in the 8.x line. Pinning per
# token is what makes tabpfnv2 / tabpfn25 / tabpfn3 genuinely distinct systems
# rather than three aliases of whatever checkpoint the installed package defaults
# to. The version that actually ran is written to the record (model_version_used)
# so any silent drift is auditable.
_TABPFN_VERSION = {"tabpfnv2": "V2", "tabpfn25": "V2_5", "tabpfn3": "V3"}


def _make_tabpfn(Est, model_name, seed):
    vname = _TABPFN_VERSION.get(model_name, "V3")
    try:
        from tabpfn.constants import ModelVersion
        ver = getattr(ModelVersion, vname, None)
        if ver is not None:
            est = Est.create_default_for_version(ver)
            ver_used = vname
        else:
            # constant absent (e.g. V3 requested on a 6.x/7.x package): use the
            # package default and flag it so the record is not mislabelled.
            est = Est(random_state=seed)
            ver_used = f"{vname}_unavailable_used_pkg_default"
        try:
            est.random_state = seed
        except Exception:
            pass
    except Exception:
        est = Est(random_state=seed)
        ver_used = f"{vname}_fallback_pkg_default"
    return est, ver_used


def _run_tfm(model_name, split, seed):
    """In-context inference for the TFMs. Each adapter returns proba (cls) or
    pred + pred_sigma (reg, where the model exposes a predictive distribution).
    Imports are lazy so a missing checkpoint is a clean skip."""
    is_cls = split.task_type == "classification"
    res: dict = {}
    if model_name in ("tabpfnv2", "tabpfn25", "tabpfn3"):
        if is_cls:
            from tabpfn import TabPFNClassifier
            clf, ver = _make_tabpfn(TabPFNClassifier, model_name, seed)
            clf.fit(split.X_train, split.y_train)
            proba = clf.predict_proba(split.X_test)
            res["proba"] = proba.tolist()
            res["test_score"] = _auc(split.y_test, proba, split.n_classes)
        else:
            from tabpfn import TabPFNRegressor
            reg, ver = _make_tabpfn(TabPFNRegressor, model_name, seed)
            reg.fit(split.X_train, split.y_train)
            try:  # full output exposes a predictive std -> CRPS for all versions
                r = reg.predict(split.X_test, output_type="full")
                mean = np.asarray(r["mean"]).astype(float)
                if "std" in r:
                    res["pred_sigma"] = np.asarray(r["std"]).astype(float).tolist()
            except TypeError:
                mean = np.asarray(reg.predict(split.X_test)).astype(float)
            res["pred"] = mean.tolist()
            res["test_score"] = -_rmse(split.y_test, mean)
        res["model_version_used"] = ver
    elif model_name == "tabicl":
        from tabicl import TabICLClassifier  # classification only (guarded above)
        clf = TabICLClassifier(random_state=seed); clf.fit(split.X_train, split.y_train)
        proba = clf.predict_proba(split.X_test)
        res["proba"] = proba.tolist()
        res["test_score"] = _auc(split.y_test, proba, split.n_classes)
    elif model_name == "mitra":
        # AutoGluon mitra-classifier / mitra-regressor checkpoints.
        #
        # Under VRAM pressure AutoGluon does NOT fail: it halves the in-context
        # support set repeatedly ("Reducing max_samples_support from 8192 to
        # 4096 due to OOM error", down to 16 in the worst case observed) and
        # returns a perfectly ordinary success. A unit fitted on 1024 support
        # rows is not comparable with one fitted on 8192, and nothing in the
        # status, the metrics or the timings reveals the difference — it is
        # visible only in the fit's stdout. We therefore capture that stream and
        # record the reduction in the unit, so the gate can refuse to average
        # degraded units into a figure. This is the level-3 invalidity of the
        # paper's own reporting standard, caught in our own pipeline.
        import contextlib
        import io
        import re as _re
        import shutil
        import tempfile
        from autogluon.tabular import TabularPredictor  # adapter; see BUILD note
        import pandas as pd
        label = "__y__"
        dtr = pd.DataFrame(split.X_train); dtr[label] = split.y_train
        dte = pd.DataFrame(split.X_test)
        problem = ("binary" if (is_cls and split.n_classes == 2)
                   else "multiclass" if is_cls else "regression")
        # A TabularPredictor with no `path` writes its fitted artefacts into
        # ./AutogluonModels/ag-<timestamp>/ and NEVER removes them. One directory
        # per unit, each holding a copy of the Mitra checkpoint: across the sweeps
        # this silently grew to 430 GB and filled the disk, which then surfaced as
        # unrelated unit failures (OSError errno 28 inside write_text). We give the
        # predictor a temporary directory and delete it once the predictions are
        # out. Nothing downstream needs the fitted model: the unit record keeps the
        # predictions, and the final pass recomputes from scratch by design.
        ag_dir = tempfile.mkdtemp(prefix="ag_mitra_",
                                  dir=os.environ.get("TEP_AG_TMPDIR") or None)
        pred_kw = dict(label=label, problem_type=problem, verbosity=0, path=ag_dir)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                predictor = TabularPredictor(**pred_kw).fit(
                    dtr, hyperparameters={"MITRA": {}}, presets="medium_quality")
        except RuntimeError as e:
            shutil.rmtree(ag_dir, ignore_errors=True)
            if "No models were trained successfully" in str(e):
                # AutoGluon's memory-safety estimator declined to fit Mitra on this
                # dataset (rows x one-hot width past the host-RAM guard). Treat as a
                # documented capability limit, mirroring the class/feature caps.
                return _skip(model_name, split, "ag_memory")
            raise
        finally:
            log = buf.getvalue()
            # Only max_samples_SUPPORT invalidates the unit: it is the in-context
            # training set, so cutting it changes the predictions. A reduction of
            # max_samples_QUERY only shrinks the inference batch — the same rows
            # are predicted and the outputs are identical — so it is recorded as
            # information, not as a defect.
            steps = _re.findall(
                r"Reducing max_samples_support from\s*(\d+)\s*to\s*(\d+)", log)
            qsteps = _re.findall(
                r"Reducing max_samples_query from\s*(\d+)\s*to\s*(\d+)", log)
        if steps:
            # The unit ran, but not at the context it claims. Rather than
            # predicting which datasets will overflow (a static threshold is a
            # guess, and the overflow point is not even deterministic: the same
            # dataset reduced on three seeds and not on the other two), we take
            # the model's own report of what happened and record the unit as a
            # documented limit exactly where the reduction occurred. The
            # evidence travels with the record, so the Limitations section can
            # state which units hit the limit and by how much, verifiably.
            shutil.rmtree(ag_dir, ignore_errors=True)
            return _skip(model_name, split, "context_reduced",
                         context_from=int(steps[0][0]), context_to=int(steps[-1][1]),
                         context_reduction_steps=len(steps))
        if qsteps:
            res["query_batch_reduced_to"] = int(qsteps[-1][1])
        try:
            if is_cls:
                proba = predictor.predict_proba(dte).to_numpy()
                res["proba"] = proba.tolist()
                res["test_score"] = _auc(split.y_test, proba, split.n_classes)
            else:
                pred = predictor.predict(dte).to_numpy().astype(float)
                res["pred"] = pred.tolist()
                res["test_score"] = -_rmse(split.y_test, pred)
        finally:
            del predictor
            shutil.rmtree(ag_dir, ignore_errors=True)
    else:
        raise ValueError(model_name)
    return res


# --------------------------------------------------------------------------- #
# Metrics helpers (scalar metrics; per-prediction arrays already saved).      #
# The full proper-scoring / calibration / bootstrap suite is post-hoc in the  #
# figures step over the saved 'proba'/'pred' arrays; these are the cheap       #
# in-unit scalars that the gate and the dev sweep read directly.              #
# --------------------------------------------------------------------------- #
def _auc(y, proba, n_classes):
    from sklearn.metrics import roc_auc_score
    proba = np.asarray(proba)
    try:
        if n_classes == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovo", average="macro"))
    except Exception:
        return float("nan")


def _rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


def compute_unit_metrics(pred: dict, split: Split) -> dict:
    """Turn a get_predictions() result into the per-unit JSON the runner saves.
    Keeps the per-prediction arrays (E1-E4 read them post-hoc) and adds the cheap
    scalar metrics + the rare-event / calibration scalars the checklist wants."""
    if not str(pred.get("status", "")).startswith("ok"):
        return pred  # skipped / error record passes through unchanged

    rec = dict(pred)  # already carries y_true, proba/pred, optimism_gap, etc.
    # Provenance of the task type: 'info_json' (as shipped) or
    # 'dataset_yaml_override' (metadata corrected in the dataset yaml). Carried
    # per unit so the correction is auditable from the released records rather
    # than only from the manuscript.
    rec["task_type_source"] = split.meta.get("task_type_source", "info_json")
    rec["task_type_shipped"] = split.meta.get("task_type_shipped", split.task_type)
    if split.task_type == "classification":
        proba = np.asarray(pred["proba"]); y = np.asarray(pred["y_true"])
        # A model may not observe every class in its training fold (rare class),
        # returning proba with fewer columns than n_classes. Right-pad the missing
        # trailing classes with probability 0 so every metric sees an (n, n_classes)
        # matrix. If proba came back mapped to specific class ids, `proba_classes`
        # (set by the adapter) tells us where each column belongs.
        if proba.ndim == 2 and proba.shape[1] != split.n_classes:
            full = np.zeros((proba.shape[0], split.n_classes), dtype=float)
            cols = pred.get("proba_classes")
            if cols is not None and len(cols) == proba.shape[1]:
                full[:, np.asarray(cols, dtype=int)] = proba
            else:
                w = min(proba.shape[1], split.n_classes)
                full[:, :w] = proba[:, :w]
            proba = full
            rec["proba"] = proba.tolist()
            rec.setdefault("notes", []).append(
                f"proba padded {np.asarray(pred['proba']).shape[1]}->{split.n_classes} (unobserved class)")
        # Row-normalise before scoring. The models do not all emit rows that sum
        # to one (observed on 1539 units across five systems and 67 datasets, at
        # deviations far above float32 noise), and scikit-learn 1.5+ no longer
        # renormalises inside log_loss — it warns and scores the raw values, so a
        # row summing to s inflates the log loss by exactly -ln(s). That would
        # penalise systems for a numerical artefact of their output layer rather
        # than for their predictions, on the very proper-scoring rule the paper's
        # C3 claim rests on. We renormalise, record the largest deviation seen and
        # the emitted-probability convention, and leave `proba` in the record as
        # the model emitted it so the correction is auditable.
        if proba.ndim == 2 and proba.shape[0]:
            sums = proba.sum(axis=1)
            dev = float(np.abs(sums - 1.0).max())
            rec["proba_max_sum_deviation"] = dev
            if dev > 1e-9:
                proba = proba / np.where(sums[:, None] == 0, 1.0, sums[:, None])
                rec["proba_renormalised"] = True
        rec["metrics"] = {
            "auc": _auc(y, proba, split.n_classes),
            "acc": float((proba.argmax(1) == y).mean()),
            "log_loss": _log_loss(y, proba),
            "ece": _ece(y, proba),
            "brier": _brier(y, proba, split.n_classes),
            "pr_auc": _pr_auc(y, proba, split.n_classes),
        }
    else:
        p = np.asarray(pred["pred"], float); y = np.asarray(pred["y_true"], float)
        sig = np.asarray(pred["pred_sigma"], float) if pred.get("pred_sigma") else None
        rec["metrics"] = {
            "rmse": _rmse(y, p),
            "mae": float(np.mean(np.abs(y - p))),
            "r2": _r2(y, p),
            "crps_gaussian": _crps_gaussian(y, p, sig) if sig is not None else None,
            "has_predictive_sigma": sig is not None,
        }
    rec["optimism_gap"] = pred.get("optimism_gap")
    return rec


def _log_loss(y, proba):
    from sklearn.metrics import log_loss
    try:
        return float(log_loss(y, proba, labels=list(range(proba.shape[1]))))
    except Exception:
        return float("nan")


def _brier(y, proba, n_classes):
    oh = np.eye(n_classes)[np.asarray(y)]
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))


def _pr_auc(y, proba, n_classes):
    from sklearn.metrics import average_precision_score
    try:
        if n_classes == 2:
            return float(average_precision_score(y, proba[:, 1]))
        oh = np.eye(n_classes)[np.asarray(y)]
        return float(average_precision_score(oh, proba, average="macro"))
    except Exception:
        return float("nan")


def _ece(y, proba, n_bins=15):
    """Expected calibration error (top-label, equal-width bins)."""
    proba = np.asarray(proba); conf = proba.max(1); pred = proba.argmax(1)
    correct = (pred == np.asarray(y)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0; n = len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def _crps_gaussian(y, mu, sigma):
    """Closed-form CRPS for a Gaussian predictive distribution."""
    from math import pi
    sigma = np.clip(np.asarray(sigma, float), 1e-9, None)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    from scipy.stats import norm
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(pi))
    return float(np.mean(crps))


# --------------------------------------------------------------------------- #
# Self-test: synthetic TALENT-style folder + xgboost, exercising the contract. #
# Run:  python topic_pipeline.py --selftest                                    #
# --------------------------------------------------------------------------- #
def _selftest():
    import tempfile
    from sklearn.datasets import make_classification, make_regression
    root = Path(tempfile.mkdtemp())
    os.environ["TALENT_DATA_ROOT"] = str(root)

    def _write(name, task):
        d = root / name; d.mkdir()
        if task == "classification":
            X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                                       n_classes=3, random_state=0)
            info = {"name": name, "task_type": "multiclass", "n_classes": 3,
                    "n_num_features": 8, "n_cat_features": 0, "openml_id": 111}
        else:
            X, y = make_regression(n_samples=400, n_features=8, noise=0.3, random_state=0)
            info = {"name": name, "task_type": "regression", "n_classes": 1,
                    "n_num_features": 8, "n_cat_features": 0, "openml_id": 222}
        # ship a default 64/16/20 partition (we re-split anyway)
        n = len(y); a, b = int(.64 * n), int(.80 * n)
        for s, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
            np.save(d / f"N_{s}.npy", X[sl]); np.save(d / f"y_{s}.npy", y[sl])
        (d / "info.json").write_text(json.dumps(info))

    _write("toy_cls", "classification")
    _write("toy_reg", "regression")
    # categorical dataset -> exercises the CatBoost native-cat frame + ES path
    d = root / "toy_cat"; d.mkdir()
    rng = np.random.default_rng(0)
    Xn = rng.normal(size=(400, 4)); Xc = rng.integers(0, 5, size=(400, 2)).astype(str)
    yc = (Xn[:, 0] + (Xc[:, 0] == "3") + rng.normal(scale=.5, size=400) > .5).astype(int)
    n = 400; a, b = int(.64 * n), int(.80 * n)
    for s, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
        np.save(d / f"N_{s}.npy", Xn[sl]); np.save(d / f"C_{s}.npy", Xc[sl])
        np.save(d / f"y_{s}.npy", yc[sl])
    (d / "info.json").write_text(json.dumps(
        {"name": "toy_cat", "task_type": "binclass", "n_classes": 2,
         "n_num_features": 4, "n_cat_features": 2, "openml_id": 333}))

    gbdts = ("xgboost_tuned", "lightgbm_tuned", "catboost_tuned")
    for token, exp in (("core:toy_cls", "classification"),
                       ("core:toy_cat", "classification"),
                       ("core:toy_reg", "regression")):
        for proto in ("cv_standard", "matched_hpo"):
            sp = load_split(token, proto, seed=0)
            assert sp.task_type == exp and len(sp.y_test) > 0
            for model in gbdts:
                pr = get_predictions(model, sp, seed=0, protocol=proto)
                assert pr["status"] == "ok", pr
                rec = compute_unit_metrics(pr, sp)
                assert "gbdt_es" in rec, "ES metadata missing from the record"
                if proto == "matched_hpo":
                    assert rec["optimism_gap"] is not None, rec
                tag = "auc" if exp == "classification" else "rmse"
                print(f"[ok] {token:12} {proto:11} {model:15} "
                      f"{tag}={rec['metrics'][tag]:.4f} "
                      f"best_iter={rec['gbdt_es']['best_iter']} "
                      f"optimism_gap={rec['optimism_gap']}")
        # capability guard: tabicl must skip the regression unit cleanly
        if exp == "regression":
            sk = get_predictions("tabicl", load_split(token, "cv_standard", 0), 0, "cv_standard")
            assert sk["status"] == "skipped_no_regression", sk
            print(f"[ok] tabicl correctly skipped regression -> {sk['status']}")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
