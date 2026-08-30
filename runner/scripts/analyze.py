#!/usr/bin/env python3
"""
analyze.py — analysis stage for the tabular-eval-protocol re-benchmark.

Reads the runner's per-unit JSON records under <outdir> (the schema written by
topic_pipeline.compute_unit_metrics) and produces the statistics that the six
Results claims and the review-proofing gate depend on. Everything here is
post-hoc and CPU-cheap; no model is fitted and no GPU is touched.

Outputs (all under <outdir>/stats/):
  friedman_nemenyi.json    omnibus Friedman + Nemenyi critical difference and
                           Holm-corrected pairwise tests           [gate E1, C1]
  wilcoxon_survival.json   paired Wilcoxon signed-rank claim-survival audit with
                           bootstrap CIs on the metric delta        [gate E1, C1]
  rank_stability.json      bootstrap rank-stability bands per system           [C2]
  metric_reorder.json      Kendall-tau between point and proper-scoring rankings [C3]
  calibration_rank.json    ranking under ECE next to ranking under accuracy     [C4]
  temporal_reorder.json    ranking under temporal vs random split               [C5a]
  optimism_gap.json        per-system optimism gap from matched_hpo             [C5b]
  reporting_standard.json  audited-claim flags under the minimum standard       [C6]
  summary.json             headline survival / reorder rates for the abstract

The per-unit record schema this reads (ok records only):
  family, protocol, model, seed, dataset, task_type
  metrics = {auc, acc, log_loss, ece, brier, pr_auc}            (classification)
          | {rmse, mae, r2, crps_gaussian, has_predictive_sigma} (regression)
  optimism_gap                                                   (top level)
  y_true, proba | pred[, pred_sigma]                            (per-prediction)

Skipped / error records (status not starting "ok") carry no metrics and are
ignored by every aggregate.

Usage:
    python analyze.py <outdir> [--config gate_config.yaml] [--alpha 0.05] \
                       [--bootstrap 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# Metric orientation. Higher oriented score is always better, so deltas and    #
# ranks share one convention across point, proper-scoring and calibration.     #
# --------------------------------------------------------------------------- #
HIGHER_BETTER = {"auc": True, "acc": True, "pr_auc": True, "r2": True,
                 "log_loss": False, "brier": False, "ece": False,
                 "rmse": False, "mae": False, "crps_gaussian": False}

PRIMARY_POINT = {"classification": "auc", "regression": "rmse"}
PROPER_SCORE = {"classification": "log_loss", "regression": "crps_gaussian"}
# A point predictor has no predictive distribution, so crps_gaussian is null for
# every GBDT and neural regression unit (has_predictive_sigma=False). Dropping
# those systems would leave the regression proper-scoring comparison with only
# the TFMs in it. The continuous ranked probability score of a deterministic
# forecast is exactly the absolute error, so MAE is the same quantity for a
# point predictor and the two are directly comparable. The fallback is applied
# per unit and reported, never silently.
PROPER_SCORE_FALLBACK = {"crps_gaussian": "mae"}
FALLBACK_USED = defaultdict(int)   # (model, task, metric) -> units scored via fallback
CALIB_METRIC = "ece"  # classification only

# Nemenyi critical values q_alpha (alpha=0.05), indexed by number of systems k.
# q_alpha = studentised range / sqrt(2); standard Demsar (2006) table.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
        15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}
_Q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589, 7: 2.693, 8: 2.780,
        9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
        15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319}


# --------------------------------------------------------------------------- #
# Loading and aggregation.                                                     #
# --------------------------------------------------------------------------- #
def load_ok_records(outdir: Path) -> list[dict]:
    recs = []
    for p in sorted(outdir.glob("*__*__*__seed*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if str(r.get("status", "")).startswith("ok") and r.get("metrics"):
            recs.append(r)
    return recs


def _metric(rec: dict, key: str):
    v = (rec.get("metrics") or {}).get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def aggregate_over_seeds(recs, family, protocol, metric_for_task):
    """Mean over seeds -> {model: {dataset: oriented_score}} for one
    (family, protocol). metric_for_task(task_type) picks the metric key; the
    score is oriented so higher is always better. Datasets whose chosen metric
    is unavailable for a model are simply absent (handled by the block builder).
    Returns (scores, datasets_seen, task_of_dataset)."""
    bucket = defaultdict(lambda: defaultdict(list))   # model -> dataset -> [scores]
    task_of = {}
    for r in recs:
        if r.get("family") != family or r.get("protocol") != protocol:
            continue
        task = r.get("task_type")
        key = metric_for_task(task)
        if key is None:
            continue
        v = _metric(r, key)
        if v is None:
            alt = PROPER_SCORE_FALLBACK.get(key)
            if alt is None:
                continue
            v = _metric(r, alt)
            if v is None:
                continue
            key = alt          # same quantity for a deterministic forecast
            FALLBACK_USED[(r["model"], task, alt)] += 1
        oriented = v if HIGHER_BETTER[key] else -v
        ds = r.get("dataset")
        bucket[r["model"]][ds].append(oriented)
        task_of[ds] = task
    scores = {m: {d: float(np.mean(vs)) for d, vs in dd.items()}
              for m, dd in bucket.items()}
    datasets = sorted({d for dd in scores.values() for d in dd})
    return scores, datasets, task_of


def build_complete_block(scores, datasets):
    """Largest practical complete model x dataset block. Greedy: drop the model
    or dataset carrying the most gaps until the block is full, preferring to keep
    at least three models. Returns (models, datasets, matrix[m,d])."""
    models = sorted(scores)
    datasets = list(datasets)
    def matrix(ms, ds):
        return np.array([[scores[m].get(d, np.nan) for d in ds] for m in ms])
    while models and datasets:
        M = matrix(models, datasets)
        if not np.isnan(M).any():
            return models, datasets, M
        gaps_by_model = np.isnan(M).sum(axis=1)
        gaps_by_ds = np.isnan(M).sum(axis=0)
        drop_model = gaps_by_model.max() >= gaps_by_ds.max() and len(models) > 3
        if drop_model:
            models.pop(int(np.argmax(gaps_by_model)))
        else:
            datasets.pop(int(np.argmax(gaps_by_ds)))
    return models, datasets, matrix(models, datasets)


def ranks_per_dataset(M):
    """M[m,d] oriented score (higher better) -> rank[m,d] with 1 = best.
    Average ranks on ties."""
    R = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        R[:, j] = stats.rankdata(-M[:, j], method="average")
    return R


# --------------------------------------------------------------------------- #
# C1 — Friedman + Nemenyi, and the Wilcoxon claim-survival audit.              #
# --------------------------------------------------------------------------- #
def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def friedman_nemenyi(scores, datasets, alpha=0.05):
    models, ds, M = build_complete_block(scores, datasets)
    out = {"models": models, "n_datasets": len(ds), "datasets": ds,
           "dropped_for_completeness": sorted(set(scores) - set(models))}
    if len(models) < 3 or len(ds) < 3:
        out["status"] = "insufficient_block"
        out["note"] = "need >=3 models on >=3 shared datasets for Friedman"
        return out
    R = ranks_per_dataset(M)            # [models, datasets], 1 = best
    mean_rank = R.mean(axis=1)
    chi2, p = stats.friedmanchisquare(*[M[i, :] for i in range(len(models))])
    k, N = len(models), len(ds)
    q = _Q05.get(k, _Q05[20]) if alpha == 0.05 else _Q10.get(k, _Q10[20])
    cd = q * math.sqrt(k * (k + 1) / (6.0 * N))
    # Holm-corrected pairwise Wilcoxon across the per-dataset oriented scores.
    pairs, pv = [], []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = M[i, :], M[j, :]
            try:
                _, pij = stats.wilcoxon(a, b, zero_method="wilcox",
                                        alternative="two-sided")
            except ValueError:
                pij = 1.0
            pairs.append((models[i], models[j])); pv.append(pij)
    adj = holm(np.array(pv)) if pv else np.array([])
    out.update({
        "status": "ok",
        "friedman_chi2": float(chi2), "friedman_p": float(p),
        "mean_rank": {m: float(r) for m, r in zip(models, mean_rank)},
        "critical_difference": float(cd), "alpha": alpha,
        "best_system": models[int(np.argmin(mean_rank))],
        "pairwise": [{"a": a, "b": b, "wilcoxon_p": float(p0),
                      "holm_p": float(pa),
                      "rank_gap": float(abs(mean_rank[models.index(a)]
                                            - mean_rank[models.index(b)])),
                      "separated_by_cd": bool(abs(mean_rank[models.index(a)]
                                                  - mean_rank[models.index(b)]) > cd),
                      "significant_holm": bool(pa < alpha)}
                     for (a, b), p0, pa in zip(pairs, pv, adj)],
    })
    return out


def _boot_delta_ci(delta, B, rng, alpha=0.05):
    """Bootstrap CI of the mean paired delta over datasets."""
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(delta), size=(B, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_survival(scores, datasets, claims, B, rng, alpha=0.05):
    """For each audited A-beats-B claim, pair the per-dataset oriented scores of
    A and B over their shared datasets, run a one-sided Wilcoxon in the claimed
    direction, bootstrap the mean-delta CI, and decide survival (Holm-significant
    AND CI strictly positive). If no claim list is supplied, derive candidate
    claims from the data: every ordered pair where the observed mean oriented
    score of A exceeds B, on the complete block."""
    models, ds, M = build_complete_block(scores, datasets)
    by_model = {m: dict(zip(ds, M[i, :])) for i, m in enumerate(models)}
    if not claims:
        mean_oriented = {m: float(np.mean(list(by_model[m].values()))) for m in models}
        order = sorted(models, key=lambda m: -mean_oriented[m])
        claims = [{"id": f"{a}_beats_{b}", "a": a, "b": b}
                  for ii, a in enumerate(order) for b in order[ii + 1:]]
        derived = True
    else:
        derived = False
    results, pvals = [], []
    for c in claims:
        a, b = c["a"], c["b"]
        # A system dropped by the complete block (TabICL and Mitra fall out
        # through capability skips and memory limits) can still be compared by a
        # PAIRED test: that needs only the datasets the TWO systems share.
        # Otherwise the claim is recorded as model_absent and the frontier
        # systems quietly vanish from the study's own claim audit.
        src_a = by_model.get(a) or scores.get(a) or {}
        src_b = by_model.get(b) or scores.get(b) or {}
        outside_block = a not in by_model or b not in by_model
        shared = sorted(set(src_a) & set(src_b))
        if len(shared) < 3:
            results.append({**c, "status": "insufficient_shared_datasets",
                            "n_datasets": len(shared)})
            pvals.append(1.0)
            continue
        da = np.array([src_a[d] for d in shared])
        db = np.array([src_b[d] for d in shared])
        delta = da - db                       # >0 means A better (oriented)
        try:
            _, p = stats.wilcoxon(da, db, alternative="greater")
        except ValueError:
            p = 1.0
        lo, hi = _boot_delta_ci(delta, B, rng, alpha)
        results.append({"id": c.get("id", f"{a}_beats_{b}"), "a": a, "b": b,
                        "n_datasets": len(shared), "mean_delta": float(delta.mean()),
                        "wilcoxon_p_onesided": float(p),
                        "delta_ci": [lo, hi],
                        "outside_complete_block": outside_block})
        pvals.append(p)
    adj = holm(np.array(pvals)) if pvals else np.array([])
    survived = 0; total = 0
    for r, pa in zip(results, adj):
        if r.get("status"):          # insufficient_shared_datasets
            r["survives"] = None; continue
        total += 1
        r["holm_p"] = float(pa)
        r["survives"] = bool(pa < alpha and r["delta_ci"][0] > 0)
        survived += int(r["survives"])
    return {"status": "ok", "derived_claims": derived, "alpha": alpha,
            "n_claims": total, "n_survived": survived,
            "survival_rate": (survived / total) if total else None,
            "claims": results}


# --------------------------------------------------------------------------- #
# C2 — bootstrap rank-stability bands.                                         #
# --------------------------------------------------------------------------- #
def rank_stability(scores, datasets, B, rng):
    models, ds, M = build_complete_block(scores, datasets)
    if len(models) < 2 or len(ds) < 2:
        return {"status": "insufficient_block", "models": models}
    N = len(ds)
    rank_draws = np.empty((B, len(models)))
    for bi in range(B):
        cols = rng.integers(0, N, size=N)
        mean_rank = ranks_per_dataset(M[:, cols]).mean(axis=1)
        rank_draws[bi, :] = stats.rankdata(mean_rank, method="average")
    bands = {}
    for i, m in enumerate(models):
        lo, med, hi = np.percentile(rank_draws[:, i], [2.5, 50, 97.5])
        bands[m] = {"median_rank": float(med), "lo": float(lo), "hi": float(hi),
                    "point_mean_rank": float(ranks_per_dataset(M).mean(axis=1)[i])}
    order = sorted(models, key=lambda m: bands[m]["median_rank"])
    top = order[:3]
    overlap = any(bands[top[0]]["hi"] >= bands[t]["lo"] for t in top[1:]) if len(top) > 1 else False
    return {"status": "ok", "B": B, "models": models, "bands": bands,
            "top_cluster": top, "top_cluster_overlaps": bool(overlap)}


# --------------------------------------------------------------------------- #
# C3 — point vs proper-scoring reorder (Kendall-tau).                          #
# --------------------------------------------------------------------------- #
def metric_reorder(recs, family, protocol, B, rng):
    pt, _, _ = aggregate_over_seeds(recs, family, protocol,
                                    lambda t: PRIMARY_POINT.get(t))
    pr, _, _ = aggregate_over_seeds(recs, family, protocol,
                                    lambda t: PROPER_SCORE.get(t))
    common_ds = sorted(set(d for dd in pt.values() for d in dd)
                       & set(d for dd in pr.values() for d in dd))
    models = sorted(set(pt) & set(pr))
    models = [m for m in models if all(d in pt[m] for d in common_ds)
              and all(d in pr[m] for d in common_ds)]
    if len(models) < 3 or len(common_ds) < 3:
        return {"status": "insufficient_block", "models": models}
    Mp = np.array([[pt[m][d] for d in common_ds] for m in models])
    Mq = np.array([[pr[m][d] for d in common_ds] for m in models])
    rank_pt = ranks_per_dataset(Mp).mean(axis=1)
    rank_pr = ranks_per_dataset(Mq).mean(axis=1)
    tau, p = stats.kendalltau(rank_pt, rank_pr)
    taus = []
    for _ in range(B):
        cols = rng.integers(0, len(common_ds), size=len(common_ds))
        rp = ranks_per_dataset(Mp[:, cols]).mean(axis=1)
        rq = ranks_per_dataset(Mq[:, cols]).mean(axis=1)
        t, _ = stats.kendalltau(rp, rq)
        if not math.isnan(t):
            taus.append(t)
    lo, hi = (np.percentile(taus, [2.5, 97.5]) if taus else (float("nan"),) * 2)
    shifts = {m: {"point_rank": float(rank_pt[i]), "proper_rank": float(rank_pr[i]),
                  "shift": float(rank_pr[i] - rank_pt[i])}
              for i, m in enumerate(models)}
    movers = sorted(models, key=lambda m: -abs(shifts[m]["shift"]))[:3]
    return {"status": "ok", "kendall_tau": float(tau), "tau_p": float(p),
            "tau_ci": [float(lo), float(hi)], "n_datasets": len(common_ds),
            "shifts": shifts, "largest_movers": movers}


# --------------------------------------------------------------------------- #
# C4 — calibration as a distinct axis.                                         #
# --------------------------------------------------------------------------- #
def calibration_rank(recs, family, protocol, B, rng):
    acc, _, _ = aggregate_over_seeds(recs, family, protocol,
                                     lambda t: "acc" if t == "classification" else None)
    ece, _, _ = aggregate_over_seeds(recs, family, protocol,
                                     lambda t: CALIB_METRIC if t == "classification" else None)
    ds = sorted(set(d for dd in acc.values() for d in dd)
                & set(d for dd in ece.values() for d in dd))
    models = [m for m in sorted(set(acc) & set(ece))
              if all(d in acc[m] for d in ds) and all(d in ece[m] for d in ds)]
    if len(models) < 3 or len(ds) < 2:
        return {"status": "insufficient_block", "models": models}
    Ma = np.array([[acc[m][d] for d in ds] for m in models])       # higher better
    Me = np.array([[ece[m][d] for d in ds] for m in models])       # already oriented (-ece) -> higher better
    rank_acc = ranks_per_dataset(Ma).mean(axis=1)
    rank_ece = ranks_per_dataset(Me).mean(axis=1)
    acc_leader = models[int(np.argmin(rank_acc))]
    ece_leader = models[int(np.argmin(rank_ece))]
    tau, _ = stats.kendalltau(rank_acc, rank_ece)
    return {"status": "ok", "n_datasets": len(ds), "models": models,
            "rank_under_accuracy": {m: float(rank_acc[i]) for i, m in enumerate(models)},
            "rank_under_ece": {m: float(rank_ece[i]) for i, m in enumerate(models)},
            "accuracy_leader": acc_leader, "calibration_leader": ece_leader,
            "leader_disagrees": bool(acc_leader != ece_leader),
            "kendall_tau_acc_vs_ece": float(tau)}


# --------------------------------------------------------------------------- #
# C5a — temporal vs random reorder.                                           #
# --------------------------------------------------------------------------- #
def temporal_reorder(recs, B, rng):
    """C5: does the ranking change when the split becomes time-based?

    Each side must be ranked on a COMMON set of datasets. Averaging a model's
    metric over whichever datasets it happens to cover, then ranking those
    averages, compares numbers computed on different data: on the temporal
    family the coverage runs from one dataset (TabICL is classification-only and
    meets a memory limit on the rest) to all eight (the gradient boosters), and a
    mean over two easy datasets outranks a mean over eight mixed ones for reasons
    that have nothing to do with the split policy. We build a complete block on
    each side, rank within each dataset and average the ranks, and report what
    the block cost in coverage.
    """
    rnd, ds_r, _ = aggregate_over_seeds(recs, "core", "cv_standard",
                                        lambda t: PRIMARY_POINT.get(t))
    tmp, ds_t, _ = aggregate_over_seeds(recs, "tabred", "temporal",
                                        lambda t: PRIMARY_POINT.get(t))
    common = sorted(m for m in set(rnd) & set(tmp) if rnd[m] and tmp[m])
    if len(common) < 3:
        return {"status": "insufficient_block", "models": common}

    r_models, r_ds, R = build_complete_block({m: rnd[m] for m in common}, ds_r)
    t_models, t_ds, T = build_complete_block({m: tmp[m] for m in common}, ds_t)
    models = sorted(set(r_models) & set(t_models))
    if len(models) < 3:
        return {"status": "insufficient_block", "models": models,
                "note": "no three systems share a complete block on both split policies"}

    def mean_ranks(block_models, M):
        idx = [block_models.index(m) for m in models]
        return ranks_per_dataset(M[idx, :]).mean(axis=1)

    mr_rnd, mr_tmp = mean_ranks(r_models, R), mean_ranks(t_models, T)
    tau, p = stats.kendalltau(mr_rnd, mr_tmp)
    movers = sorted(range(len(models)), key=lambda i: -abs(mr_tmp[i] - mr_rnd[i]))[:3]
    return {"status": "ok", "models": models,
            "n_datasets_random": len(r_ds), "n_datasets_temporal": len(t_ds),
            "dropped_for_completeness": sorted(set(common) - set(models)),
            "coverage_note": ("ranked only on systems holding a complete block under BOTH "
                              "split policies; systems dropped here appear with their "
                              "coverage in the applicability table"),
            "mean_rank_random_split": {m: float(mr_rnd[i]) for i, m in enumerate(models)},
            "mean_rank_temporal_split": {m: float(mr_tmp[i]) for i, m in enumerate(models)},
            # Legacy key names, read by make_figures. Renaming them once broke
            # Figure 5: the contract with the consumer outweighs tidy naming.
            "rank_random_split": {m: float(mr_rnd[i]) for i, m in enumerate(models)},
            "rank_temporal_split": {m: float(mr_tmp[i]) for i, m in enumerate(models)},
            "kendall_tau": float(tau), "tau_p": float(p),
            "largest_movers": [models[i] for i in movers]}


def pairwise_axis_flips(recs, family, protocol, axis_a, axis_b, name_a, name_b, B, rng):
    """Compare two scoring axes PAIRWISE, so systems the complete block excludes
    still take part.

    A complete block needs every system on every dataset, and on this suite that
    requirement quietly removes the systems the study is about: the calibration
    block holds no TabPFN variant at all, the proper-scoring block holds no
    foundation model at all. A paired comparison needs only the datasets the two
    systems share, so it keeps them in. For each pair we report the mean
    difference on both axes, the Wilcoxon p for each, and whether the sign flips
    — a flip is the claim "the system that wins on one axis loses on the other",
    stated with the number of datasets it rests on.
    """
    a_sc, _, _ = aggregate_over_seeds(recs, family, protocol, axis_a)
    b_sc, _, _ = aggregate_over_seeds(recs, family, protocol, axis_b)
    models = sorted(set(a_sc) & set(b_sc))
    pairs, flips, thin = [], 0, 0
    for i, A in enumerate(models):
        for Bm in models[i + 1:]:
            shared = sorted(set(a_sc[A]) & set(a_sc[Bm]) & set(b_sc[A]) & set(b_sc[Bm]))
            if len(shared) < 3:
                thin += 1
                pairs.append({"a": A, "b": Bm, "n_datasets": len(shared),
                              "status": "insufficient_shared_datasets"})
                continue
            da = np.array([a_sc[A][d] - a_sc[Bm][d] for d in shared])
            db = np.array([b_sc[A][d] - b_sc[Bm][d] for d in shared])

            def wp(v):
                try:
                    return float(stats.wilcoxon(v)[1])
                except ValueError:
                    return 1.0

            flip = bool(np.sign(da.mean()) != np.sign(db.mean())
                        and da.mean() != 0 and db.mean() != 0)
            flips += int(flip)
            lo_a, hi_a = _boot_delta_ci(da, B, rng, 0.05)
            lo_b, hi_b = _boot_delta_ci(db, B, rng, 0.05)
            pairs.append({"a": A, "b": Bm, "n_datasets": len(shared),
                          f"mean_delta_{name_a}": float(da.mean()),
                          f"ci_{name_a}": [lo_a, hi_a], f"p_{name_a}": wp(da),
                          f"mean_delta_{name_b}": float(db.mean()),
                          f"ci_{name_b}": [lo_b, hi_b], f"p_{name_b}": wp(db),
                          "sign_flip": flip})
    # A sign flip on its own proves NOTHING: the difference on the second axis
    # may be noise. The counter is therefore broken down by significance, and it
    # is the breakdown that reaches the summary — otherwise "20 flips out of 45"
    # travels into the text as a result while not one of them is significant on
    # both axes.
    def sig(pr, ax):
        return pr.get(f"p_{ax}", 1.0) < 0.05
    scored = [pr for pr in pairs if "sign_flip" in pr]
    fl = [pr for pr in scored if pr["sign_flip"]]
    both = [pr for pr in fl if sig(pr, name_a) and sig(pr, name_b)]
    one = [pr for pr in fl if sig(pr, name_a) != sig(pr, name_b)]
    return {"status": "ok" if pairs else "no_pairs", "axis_a": name_a, "axis_b": name_b,
            "family": family, "protocol": protocol, "models": models,
            "n_pairs": len(pairs), "n_sign_flips": flips,
            "n_sign_flips_both_significant": len(both),
            "n_sign_flips_one_significant": len(one),
            "n_sign_flips_neither_significant": len(fl) - len(both) - len(one),
            "flips_both_significant": [{"a": pr["a"], "b": pr["b"],
                                        "n_datasets": pr["n_datasets"]} for pr in both],
            "interpretation": ("a flip counts ONLY when both differences are "
                               "significant; the rest merely show that an advantage on "
                               "the first axis does not carry over to the second"),
            "n_pairs_insufficient": thin,
            "note": ("paired on the datasets each pair shares, so systems outside the "
                     "complete block are included; n_datasets differs per pair by design"),
            "pairs": pairs}


# --------------------------------------------------------------------------- #
# C5b — optimism gap from matched_hpo.                                         #
# --------------------------------------------------------------------------- #
def optimism_gap(recs, B, rng):
    """Optimism gap (inner held-out minus test), per system AND per task type.

    The two task families measure the gap in different units: classification
    scores the inner fold and the test set with ROC-AUC, regression with
    negative RMSE on a train-standardised target. Pooling them into one mean
    would average an AUC difference with a standard-deviation difference, so
    the per-task-type figures are the reportable ones (Table 2) and the pooled
    number is kept only for continuity with earlier runs. On the final
    matched_hpo subset the split matters: 11 of the 20 datasets are
    classification and 9 are regression."""
    def summarise(arr):
        arr = np.array(arr, float)
        idx = rng.integers(0, len(arr), size=(B, len(arr)))
        lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
        return {"n": len(arr), "mean_optimism_gap": float(arr.mean()),
                "ci": [float(lo), float(hi)]}

    by_model = defaultdict(list)
    by_model_task = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("protocol") != "matched_hpo":
            continue
        g = r.get("optimism_gap")
        if g is None:
            continue
        try:
            g = float(g)
        except (TypeError, ValueError):
            continue
        by_model[r["model"]].append(g)
        by_model_task[r["model"]][r.get("task_type", "unknown")].append(g)

    per_model = {m: summarise(gs) for m, gs in by_model.items()}
    per_model_task = {m: {t: summarise(gs) for t, gs in tasks.items()}
                      for m, tasks in by_model_task.items()}
    return {"status": "ok" if per_model else "no_matched_hpo_records",
            "units": {"classification": "ROC-AUC", "regression": "negative RMSE"
                                                                " on the standardised target"},
            "per_model_task_type": per_model_task,
            "per_model_pooled_mixed_units": per_model,
            "per_model": per_model}   # legacy key: gate D1 and older readers


# --------------------------------------------------------------------------- #
# C6 — reporting-standard flags over the audited claim list.                  #
# --------------------------------------------------------------------------- #
def reporting_standard(survival, audited_claims):
    """The minimum standard flags a claim that lacks any of: a paired
    significance test, a confidence interval, a proper-scoring or calibration
    score, or that rests on an uncalibrated default output. When an editorial
    audited-claim list is provided it is scored against that checklist; otherwise
    the survival audit is reused, treating every non-surviving claim as one the
    standard would have flagged."""
    checklist = ["paired_significance_test", "confidence_interval",
                 "proper_scoring_or_calibration", "calibrated_outputs"]
    if audited_claims:
        flagged = []
        for c in audited_claims:
            missing = [k for k in checklist if not c.get(k, False)]
            if missing:
                flagged.append({"id": c.get("id"), "missing": missing})
        return {"status": "ok", "source": "editorial_audited_claims",
                "checklist": checklist, "n_claims": len(audited_claims),
                "n_flagged": len(flagged), "flagged": flagged}
    claims = [c for c in survival.get("claims", []) if c.get("survives") is not None]
    flagged = [c["id"] for c in claims if not c["survives"]]
    return {"status": "ok", "source": "derived_from_survival_audit",
            "checklist": checklist, "n_claims": len(claims),
            "n_flagged": len(flagged), "flagged": flagged,
            "note": "replace with the editorial audited-claim list for the final Table 3"}


# --------------------------------------------------------------------------- #
# C7 — accuracy-vs-calibration Pareto front.                                   #
# --------------------------------------------------------------------------- #
def _pareto_nondominated(points):
    """points: list of (acc, ece). Non-dominated = maximise acc, minimise ece.
    Returns the indices of the Pareto-optimal points."""
    idx = []
    for i, (ai, ei) in enumerate(points):
        dominated = False
        for j, (aj, ej) in enumerate(points):
            if j == i:
                continue
            if aj >= ai and ej <= ei and (aj > ai or ej < ei):
                dominated = True; break
        if not dominated:
            idx.append(i)
    return idx


def _hypervolume_2d(points, ref):
    """2D dominated hypervolume for (acc maximise, ece minimise) against a
    reference (acc_ref low, ece_ref high). Area swept by the non-dominated set."""
    rx, ry = ref
    nd = [points[i] for i in _pareto_nondominated(points)]
    nd = [(a, e) for a, e in nd if a >= rx and e <= ry]
    if not nd:
        return 0.0
    nd.sort(key=lambda p: -p[0])          # by accuracy descending
    hv, prev_e = 0.0, ry
    for a, e in nd:
        if e < prev_e:
            hv += (a - rx) * (prev_e - e)
            prev_e = e
    return float(hv)


def pareto_analysis(recs, family="core", protocol="cv_standard"):
    # cross-system operating points from the point/calibration metrics
    acc, _, _ = aggregate_over_seeds(recs, family, protocol,
                                     lambda t: "acc" if t == "classification" else None)
    ece, _, _ = aggregate_over_seeds(recs, family, protocol,
                                     lambda t: CALIB_METRIC if t == "classification" else None)
    # ece was oriented to higher-better (i.e. -ece); flip back to raw ece for the plane
    ds = sorted(set(d for dd in acc.values() for d in dd)
                & set(d for dd in ece.values() for d in dd))
    models = [m for m in sorted(set(acc) & set(ece))
              if all(d in acc[m] for d in ds) and all(d in ece[m] for d in ds)]
    cross = {"status": "insufficient_block", "models": models}
    if len(models) >= 3 and len(ds) >= 2:
        op = {m: {"acc": float(np.mean([acc[m][d] for d in ds])),
                  "ece": float(np.mean([-ece[m][d] for d in ds]))} for m in models}
        # per-dataset front membership
        on_front = {m: 0 for m in models}
        for d in ds:
            pts = [(acc[m][d], -ece[m][d]) for m in models]
            for i in _pareto_nondominated(pts):
                on_front[models[i]] += 1
        on_front_rate = {m: on_front[m] / len(ds) for m in models}
        agg_pts = [(op[m]["acc"], op[m]["ece"]) for m in models]
        agg_front = [models[i] for i in _pareto_nondominated(agg_pts)]
        # is each TFM dominated by the front traced by the TUNED baselines only?
        tuned = [m for m in models if m.endswith("_tuned") or m in ("realmlp", "tabm")]
        tfms = [m for m in models if m not in tuned]
        tuned_pts = [(op[m]["acc"], op[m]["ece"]) for m in tuned]
        tfm_dom = {}
        for t in tfms:
            at, et = op[t]["acc"], op[t]["ece"]
            tfm_dom[t] = any(a >= at and e <= et and (a > at or e < et)
                             for a, e in tuned_pts)
        cross = {"status": "ok", "n_datasets": len(ds), "models": models,
                 "operating_points": op, "on_front_rate": on_front_rate,
                 "aggregate_front": agg_front,
                 "tfm_dominated_by_tuned_front": tfm_dom}

    # per-system multi-objective front from pareto_hpo records (if any)
    by_model = defaultdict(list)   # model -> list of (acc, ece) over datasets/seeds/members
    fronts_present = 0
    for r in recs:
        if r.get("protocol") != "pareto_hpo":
            continue
        pf = r.get("pareto_front")
        if not pf:
            continue
        fronts_present += 1
        for m in pf:
            if m.get("test_acc") is not None and m.get("test_ece") is not None:
                by_model[r["model"]].append((float(m["test_acc"]), float(m["test_ece"])))
    per_system = {"status": "no_pareto_records" if not fronts_present else "ok",
                  "n_records": fronts_present, "systems": {}}
    for m, pts in by_model.items():
        eces = [e for _, e in pts]
        ref = (0.0, max(eces) * 1.05 if eces else 1.0)
        per_system["systems"][m] = {
            "n_points": len(pts),
            "mean_acc": float(np.mean([a for a, _ in pts])),
            "mean_ece": float(np.mean(eces)),
            "front_hypervolume": _hypervolume_2d(pts, ref),
            "front_points": [{"acc": a, "ece": e}
                             for i, (a, e) in enumerate(pts)
                             if i in set(_pareto_nondominated(pts))][:50],
        }
    return {"cross_system": cross, "per_system_mo": per_system}


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--config", default="gate_config.yaml")
    ap.add_argument("--claims", default=None,
                    help="optional YAML with audited A-beats-B claims and the "
                         "reporting-standard checklist flags")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    statsdir = outdir / "stats"
    statsdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    recs = load_ok_records(outdir)
    if not recs:
        raise SystemExit(f"[analyze] no ok records under {outdir} — run the sweep first")

    claims_cfg = {}
    if a.claims and Path(a.claims).exists():
        import yaml
        claims_cfg = yaml.safe_load(Path(a.claims).read_text()) or {}
    audited = claims_cfg.get("audited_claims")
    explicit_claims = claims_cfg.get("comparative_claims")

    # primary family/protocol for the main comparison (C1, C2).
    primary_scores, primary_ds, _ = aggregate_over_seeds(
        recs, "core", "cv_standard", lambda t: PRIMARY_POINT.get(t))

    fn = friedman_nemenyi(primary_scores, primary_ds, a.alpha)
    surv = wilcoxon_survival(primary_scores, primary_ds, explicit_claims,
                             a.bootstrap, rng, a.alpha)
    rs = rank_stability(primary_scores, primary_ds, a.bootstrap, rng)
    mr = metric_reorder(recs, "core", "cv_standard", a.bootstrap, rng)
    # Pairwise layers: keep the systems that the complete block excludes (all
    # TabPFN variants are absent from the calibration block, every foundation
    # model from the proper-scoring block) in the comparison, each pair on the
    # datasets it shares.
    pw_proper = pairwise_axis_flips(recs, "core", "cv_standard",
                                    lambda t: PRIMARY_POINT.get(t),
                                    lambda t: PROPER_SCORE.get(t),
                                    "point", "proper", a.bootstrap, rng)
    pw_calib = pairwise_axis_flips(recs, "core", "cv_standard",
                                   lambda t: "acc" if t == "classification" else None,
                                   lambda t: CALIB_METRIC if t == "classification" else None,
                                   "accuracy", "ece", a.bootstrap, rng)
    if FALLBACK_USED:
        # Which systems were scored through the CRPS -> MAE identity, and on how
        # many units. Reported so the regression proper-scoring column cannot be
        # read as if every system had emitted a predictive distribution.
        mr["proper_score_fallback"] = {
            "rule": "CRPS of a deterministic forecast equals the absolute error;"
                    " point predictors are scored by MAE",
            "units": {f"{m}|{t}|{k}": n for (m, t, k), n in sorted(FALLBACK_USED.items())}}
    cal = calibration_rank(recs, "core", "cv_standard", a.bootstrap, rng)
    tr = temporal_reorder(recs, a.bootstrap, rng)
    og = optimism_gap(recs, a.bootstrap, rng)
    rep = reporting_standard(surv, audited)
    par = pareto_analysis(recs, "core", "cv_standard")

    def dump(name, obj):
        (statsdir / name).write_text(json.dumps(obj, indent=2))

    dump("pareto_front.json", par)
    dump("friedman_nemenyi.json", fn)
    dump("wilcoxon_survival.json", surv)
    dump("rank_stability.json", rs)
    dump("metric_reorder.json", mr)
    dump("pairwise_proper_scoring.json", pw_proper)
    dump("pairwise_calibration.json", pw_calib)
    dump("calibration_rank.json", cal)
    dump("temporal_reorder.json", tr)
    dump("optimism_gap.json", og)
    dump("reporting_standard.json", rep)

    summary = {
        "n_ok_units": len(recs),
        "families": sorted({r.get("family") for r in recs}),
        "protocols": sorted({r.get("protocol") for r in recs}),
        "best_system_by_mean_rank": fn.get("best_system"),
        "claim_survival_rate": surv.get("survival_rate"),
        "pairwise_proper_flips_significant":
            f"{pw_proper.get('n_sign_flips_both_significant')} of "
            f"{pw_proper.get('n_sign_flips')} sign flips significant on both axes "
            f"({pw_proper.get('n_pairs')} pairs in total)",
        "pairwise_calibration_flips_significant":
            f"{pw_calib.get('n_sign_flips_both_significant')} of "
            f"{pw_calib.get('n_sign_flips')} sign flips significant on both axes "
            f"({pw_calib.get('n_pairs')} pairs in total)",
        "n_claims": surv.get("n_claims"), "n_survived": surv.get("n_survived"),
        "top_cluster": rs.get("top_cluster"),
        "top_cluster_overlaps": rs.get("top_cluster_overlaps"),
        "point_vs_proper_kendall_tau": mr.get("kendall_tau"),
        "calibration_leader": cal.get("calibration_leader"),
        "accuracy_leader": cal.get("accuracy_leader"),
        "calibration_leader_disagrees": cal.get("leader_disagrees"),
        "temporal_vs_random_kendall_tau": tr.get("kendall_tau"),
        "reporting_standard_flagged": rep.get("n_flagged"),
        "pareto_aggregate_front": par.get("cross_system", {}).get("aggregate_front"),
        "pareto_tfm_dominated_by_tuned": par.get("cross_system", {})
            .get("tfm_dominated_by_tuned_front"),
        "pareto_mo_records": par.get("per_system_mo", {}).get("n_records"),
    }
    dump("summary.json", summary)

    print("=" * 64)
    print(f"ANALYSIS STAGE | {len(recs)} ok units -> {statsdir}")
    print("=" * 64)
    for k, v in summary.items():
        print(f"  {k:38s} {v}")
    print("-" * 64)
    print("wrote: " + ", ".join(sorted(p.name for p in statsdir.glob('*.json'))))


if __name__ == "__main__":
    main()
