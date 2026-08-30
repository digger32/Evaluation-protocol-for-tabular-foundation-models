#!/usr/bin/env python3
"""
make_figures.py — turn the analysis-stage stats (and the saved per-prediction
arrays) into the six Results display items. Runs only after the review-proofing
gate passes, so it never renders dirty numbers.

Produces, under <outdir>/figures/:
  fig1_cd_diagram.pdf          Friedman-Nemenyi critical-difference diagram   [C1]
  fig1_cd_classification.pdf   the same on the classification subset (so the
                               classification-only TFMs appear)
  table1_claim_survival.csv    paired Wilcoxon survival audit                 [C1]
  fig2_rank_stability.pdf      bootstrap rank-stability bands                 [C2]
  fig3_point_vs_proper.pdf     point vs proper-scoring rank slope chart       [C3]
  fig4_calibration.pdf         ECE bars + reliability curves + ECE/acc ranks  [C4]
  fig5_temporal_vs_random.pdf  ranking under temporal vs random split         [C5a]
  table2_optimism_gap.csv      per-system optimism gap under matched HPO      [C5b]
  table3_reporting_standard.csv minimum-standard checklist + flagged count    [C6]

Reads stats written by analyze.py; reliability curves read y_true/proba from the
ok classification records on core/cv_standard. Matplotlib only, Agg backend.

Usage:
    python make_figures.py <outdir>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Human-readable labels for the figures. The runner axis uses tokens; tabpfnv2 /
# tabpfn25 / tabpfn3 are three DISTINCT TabPFN checkpoints (v2 predecessor, v2.5
# under study, v3 newest), not a typo, so each gets its own proper name.
DISPLAY_NAMES = {
    "tabpfn3": "TabPFN-3", "tabpfn25": "TabPFN-2.5", "tabpfnv2": "TabPFN v2",
    "tabicl": "TabICL", "mitra": "Mitra",
    "catboost_tuned": "CatBoost (tuned)", "xgboost_tuned": "XGBoost (tuned)",
    "lightgbm_tuned": "LightGBM (tuned)", "realmlp": "RealMLP", "tabm": "TabM",
}


def disp(m):
    return DISPLAY_NAMES.get(m, m)


def _load(statsdir, name):
    p = statsdir / name
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------------------- #
# Fig 1 — critical-difference diagram (Demsar style).                          #
# --------------------------------------------------------------------------- #
def cd_diagram(mean_rank: dict, cd: float, title: str, path: Path):
    if not mean_rank:
        return
    items = sorted(mean_rank.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    ranks = [v for _, v in items]
    k = len(names)
    lo, hi = min(ranks), max(ranks)
    lo, hi = np.floor(lo - 0.3), np.ceil(hi + 0.3)

    fig, ax = plt.subplots(figsize=(8, 0.6 * k + 2.2))
    ax.set_xlim(lo, hi); ax.set_ylim(0, k + 2)
    ax.axhline(k + 1.3, color="black", lw=1)
    for x in np.arange(lo, hi + 1):
        ax.plot([x, x], [k + 1.25, k + 1.35], color="black", lw=1)
        ax.text(x, k + 1.55, f"{int(x)}", ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, k + 1.85, "mean rank (1 = best)", ha="center", fontsize=9)

    for i, (name, r) in enumerate(items):
        y = k - i
        ax.plot([r, r], [y, k + 1.25], color="0.4", lw=0.8)
        ax.plot(r, y, "o", color="#1f4e79", ms=6)
        ax.text(lo - 0.05, y, disp(name), ha="right", va="center", fontsize=9)
        ax.text(r + 0.04, y + 0.18, f"{r:.2f}", ha="left", va="center", fontsize=7, color="0.3")

    # CD bar
    ax.plot([lo + 0.1, lo + 0.1 + cd], [k + 0.6, k + 0.6], color="crimson", lw=2)
    ax.text(lo + 0.1 + cd / 2, k + 0.78, f"CD = {cd:.2f}", ha="center",
            color="crimson", fontsize=8)
    # connect systems whose rank gap <= CD (non-significant cliques)
    cliques, y0 = [], 0.45
    used = [False] * k
    for i in range(k):
        if used[i]:
            continue
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            cliques.append((ranks[i], ranks[j]))
            for t in range(i, j + 1):
                used[t] = True
    for c, (a, b) in enumerate(cliques):
        yy = y0 - 0.18 * c
        ax.plot([a - 0.03, b + 0.03], [yy, yy], color="black", lw=3, solid_capstyle="round")

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2 — rank-stability bands.                                                #
# --------------------------------------------------------------------------- #
def rank_stability_fig(rs, path):
    bands = rs.get("bands", {})
    if not bands:
        return
    order = sorted(bands, key=lambda m: bands[m]["median_rank"])
    y = np.arange(len(order))
    med = [bands[m]["median_rank"] for m in order]
    lo = [bands[m]["median_rank"] - bands[m]["lo"] for m in order]
    hi = [bands[m]["hi"] - bands[m]["median_rank"] for m in order]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.5))
    topset = set(rs.get("top_cluster", []))
    colors = ["#c0392b" if m in topset else "#1f4e79" for m in order]
    ax.errorbar(med, y, xerr=[lo, hi], fmt="o", capsize=4,
                ecolor="0.6", mfc="white", color="black", zorder=3)
    for yi, c in zip(y, colors):
        ax.plot(med[yi], yi, "o", color=c, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([disp(m) for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("bootstrap rank (1 = best), 95% band")
    ax.set_title("Rank stability under resampling"
                 + (" — top cluster overlaps" if rs.get("top_cluster_overlaps") else ""),
                 fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3 — point vs proper-scoring rank slope chart.                            #
# --------------------------------------------------------------------------- #
def reorder_fig(mr, path):
    shifts = mr.get("shifts", {})
    if not shifts:
        return
    models = list(shifts)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(models) + 1.5))
    for m in models:
        pr, qr = shifts[m]["point_rank"], shifts[m]["proper_rank"]
        moved = abs(qr - pr) >= 1.0
        ax.plot([0, 1], [pr, qr], "-o",
                color="#c0392b" if moved else "0.6",
                lw=2 if moved else 1, zorder=3 if moved else 1)
        ax.text(-0.04, pr, disp(m), ha="right", va="center", fontsize=8)
        ax.text(1.04, qr, disp(m), ha="left", va="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["point metric", "proper score"])
    ax.invert_yaxis(); ax.set_ylabel("mean rank (1 = best)")
    tau = mr.get("kendall_tau"); ci = mr.get("tau_ci", [None, None])
    ax.set_title(f"Rank reorder under proper scoring  (Kendall τ = {tau:.2f}, "
                 f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}])", fontsize=9)
    ax.set_xlim(-0.35, 1.35)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4 — calibration: ECE bars + reliability curves.                          #
# --------------------------------------------------------------------------- #
def _reliability(y, proba, n_bins=12):
    conf = proba.max(1); pred = proba.argmax(1)
    corr = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(corr[m].mean())
    return xs, ys


def calibration_fig(cal, records_by, path):
    if cal.get("status") != "ok":
        return
    models = cal["models"]
    ece_rank = cal["rank_under_ece"]; acc_rank = cal["rank_under_accuracy"]
    # mean ECE per model from records
    ece_vals = defaultdict(list)
    for r in records_by:
        if r.get("family") == "core" and r.get("protocol") == "cv_standard" \
                and r.get("task_type") == "classification":
            e = (r.get("metrics") or {}).get("ece")
            if e is not None:
                ece_vals[r["model"]].append(e)
    mean_ece = {m: float(np.mean(v)) for m, v in ece_vals.items() if m in models}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = sorted(mean_ece, key=mean_ece.get)
    axes[0].barh(range(len(order)), [mean_ece[m] for m in order],
                 color=["#c0392b" if m == cal["calibration_leader"] else "#1f4e79" for m in order])
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([disp(m) for m in order])
    axes[0].invert_yaxis(); axes[0].set_xlabel("mean ECE (lower better)")
    axes[0].set_title(f"Calibration error  (acc leader: {disp(cal['accuracy_leader'])}, "
                      f"ECE leader: {disp(cal['calibration_leader'])})", fontsize=9)

    axes[1].plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    show = [cal["accuracy_leader"], cal["calibration_leader"]]
    for m in dict.fromkeys(show):
        recs = [r for r in records_by if r.get("model") == m
                and r.get("family") == "core" and r.get("protocol") == "cv_standard"
                and r.get("task_type") == "classification" and r.get("proba")]
        if not recs:
            continue
        r0 = recs[0]
        xs, ys = _reliability(np.array(r0["y_true"]), np.array(r0["proba"]))
        axes[1].plot(xs, ys, "-o", ms=4, label=disp(m))
    axes[1].set_xlabel("confidence"); axes[1].set_ylabel("empirical accuracy")
    axes[1].set_title("Reliability (one core dataset)", fontsize=9)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 5 — temporal vs random ranking.                                          #
# --------------------------------------------------------------------------- #
def temporal_fig(tr, path):
    if tr.get("status") != "ok":
        return
    rnd = tr["rank_random_split"]; tmp = tr["rank_temporal_split"]
    models = list(rnd)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(models) + 1.5))
    movers = set(tr.get("largest_movers", []))
    for m in models:
        ax.plot([0, 1], [rnd[m], tmp[m]], "-o",
                color="#c0392b" if m in movers else "0.6",
                lw=2 if m in movers else 1)
        ax.text(-0.04, rnd[m], disp(m), ha="right", va="center", fontsize=8)
        ax.text(1.04, tmp[m], disp(m), ha="left", va="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["random split (core)", "temporal split (tabred)"])
    ax.invert_yaxis(); ax.set_ylabel("mean rank (1 = best)")
    ax.set_title(f"Ranking under split policy  (Kendall τ = {tr['kendall_tau']:.2f})", fontsize=9)
    ax.set_xlim(-0.4, 1.4)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables.                                                                       #
# --------------------------------------------------------------------------- #
def table1(surv, path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "A", "B", "n_datasets", "mean_delta",
                    "wilcoxon_p_onesided", "holm_p", "ci_lo", "ci_hi", "survives"])
        for c in surv.get("claims", []):
            if c.get("survives") is None:
                continue
            w.writerow([c["id"], c["a"], c["b"], c["n_datasets"],
                        f"{c['mean_delta']:.4f}", f"{c['wilcoxon_p_onesided']:.4g}",
                        f"{c.get('holm_p', float('nan')):.4g}",
                        f"{c['delta_ci'][0]:.4f}", f"{c['delta_ci'][1]:.4f}",
                        c["survives"]])


def table2(og, path):
    """Optimism gap, ONE ROW PER SYSTEM AND TASK TYPE.

    The two task families measure the gap in different units — ROC-AUC for
    classification, negative RMSE on the standardised target for regression —
    so a single pooled mean per system averages quantities that are not the
    same kind of thing. The pooled figure is still written, in its own clearly
    named rows, only so a reader can see it was not silently dropped.
    """
    units = og.get("units", {})
    by_task = og.get("per_model_task_type") or {}
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task_type", "unit", "n",
                    "mean_optimism_gap", "ci_lo", "ci_hi"])
        for m in sorted(by_task):
            for task in sorted(by_task[m]):
                d = by_task[m][task]
                w.writerow([m, task, units.get(task, "?"), d["n"],
                            f"{d['mean_optimism_gap']:.4f}",
                            f"{d['ci'][0]:.4f}", f"{d['ci'][1]:.4f}"])
        if not by_task:                      # older stats without the split
            for m, d in og.get("per_model", {}).items():
                w.writerow([m, "pooled_MIXED_UNITS", "mixed", d["n"],
                            f"{d['mean_optimism_gap']:.4f}",
                            f"{d['ci'][0]:.4f}", f"{d['ci'][1]:.4f}"])


def table3(rep, path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reporting_standard_item", "checklist"])
        for item in rep.get("checklist", []):
            w.writerow([item, "required"])
        w.writerow([])
        w.writerow(["n_claims_audited", rep.get("n_claims")])
        w.writerow(["n_flagged_by_standard", rep.get("n_flagged")])
        w.writerow(["source", rep.get("source")])
        for f in rep.get("flagged", []):
            w.writerow(["flagged", f if isinstance(f, str) else json.dumps(f)])


def pareto_fig(par, path):
    cs = par.get("cross_system", {})
    if cs.get("status") != "ok":
        return
    op = cs["operating_points"]
    front = set(cs.get("aggregate_front", []))
    dom = cs.get("tfm_dominated_by_tuned_front", {})
    tuned_like = lambda m: m.endswith("_tuned") or m in ("realmlp", "tabm")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # aggregate Pareto-front line (sorted by accuracy)
    fpts = sorted([(op[m]["acc"], op[m]["ece"]) for m in front], key=lambda p: p[0])
    if len(fpts) >= 2:
        ax.plot([a for a, _ in fpts], [e for _, e in fpts], "-", color="0.6",
                lw=1.5, zorder=1, label="accuracy–calibration Pareto front")
    for m, v in op.items():
        on = m in front
        is_tfm = not tuned_like(m)
        marker = "s" if is_tfm else "o"
        color = "#c0392b" if on else ("#e67e22" if (is_tfm and dom.get(m)) else "#1f4e79")
        ax.scatter(v["acc"], v["ece"], marker=marker, s=70 if on else 45,
                   color=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(disp(m), (v["acc"], v["ece"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("accuracy (higher better)")
    ax.set_ylabel("ECE (lower better)")
    ax.invert_yaxis()   # better calibration upward
    ax.set_title("Accuracy vs calibration operating points (core)\n"
                 "red = on Pareto front; orange square = TFM dominated by the tuned-baseline front",
                 fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    statsdir = outdir / "stats"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fn = _load(statsdir, "friedman_nemenyi.json")
    surv = _load(statsdir, "wilcoxon_survival.json")
    rs = _load(statsdir, "rank_stability.json")
    mr = _load(statsdir, "metric_reorder.json")
    cal = _load(statsdir, "calibration_rank.json")
    tr = _load(statsdir, "temporal_reorder.json")
    og = _load(statsdir, "optimism_gap.json")
    rep = _load(statsdir, "reporting_standard.json")
    par = _load(statsdir, "pareto_front.json")

    # Fig 1 (mixed-core CD) + a classification-only CD so class-only TFMs appear.
    if fn.get("status") == "ok":
        cd_diagram(fn["mean_rank"], fn["critical_difference"],
                   "Critical-difference diagram (core, complete block)",
                   figdir / "fig1_cd_diagram.pdf")

    records = []
    for p in outdir.glob("*__*__*__seed*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if str(r.get("status", "")).startswith("ok"):
            records.append(r)

    # classification-subset CD: re-rank on core classification datasets only.
    cls_scores = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("family") == "core" and r.get("protocol") == "cv_standard" \
                and r.get("task_type") == "classification":
            auc = (r.get("metrics") or {}).get("auc")
            if auc is not None:
                cls_scores[r["model"]][r["dataset"]].append(auc)
    if cls_scores:
        from scipy import stats as _st
        ms = sorted(cls_scores)
        ds = sorted({d for dd in cls_scores.values() for d in dd})
        ds = [d for d in ds if all(d in cls_scores[m] for m in ms)]
        if len(ms) >= 3 and len(ds) >= 3:
            M = np.array([[np.mean(cls_scores[m][d]) for d in ds] for m in ms])
            R = np.vstack([_st.rankdata(-M[:, j], method="average")
                           for j in range(M.shape[1])]).T
            mean_rank = {m: float(R[i].mean()) for i, m in enumerate(ms)}
            import math as _m
            k, N = len(ms), len(ds)
            q = {2: 1.96, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
                 8: 3.031, 9: 3.102, 10: 3.164}.get(k, 3.164)
            cd = q * _m.sqrt(k * (k + 1) / (6.0 * N))
            cd_diagram(mean_rank, cd,
                       "Critical-difference diagram (core classification subset)",
                       figdir / "fig1_cd_classification.pdf")

    table1(surv, figdir / "table1_claim_survival.csv")
    rank_stability_fig(rs, figdir / "fig2_rank_stability.pdf")
    reorder_fig(mr, figdir / "fig3_point_vs_proper.pdf")
    calibration_fig(cal, records, figdir / "fig4_calibration.pdf")
    temporal_fig(tr, figdir / "fig5_temporal_vs_random.pdf")
    pareto_fig(par, figdir / "fig6_accuracy_vs_calibration.pdf")
    table2(og, figdir / "table2_optimism_gap.csv")
    table3(rep, figdir / "table3_reporting_standard.csv")

    made = sorted(p.name for p in figdir.iterdir())
    print(f"[figures] wrote {len(made)} items to {figdir}:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
