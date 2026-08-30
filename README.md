# Evaluation protocol for tabular foundation models

Reference implementation of the evaluation protocol described in the
accompanying article: a benchmark runner that records one file per
(dataset x protocol x model x seed) cell, an analysis stage that scores point
metrics, proper scoring rules and calibration together with paired tests and
resampling-based rank stability, and a review-proofing gate that refuses to
release numbers whose provenance cannot be checked.

The point of the release is not a leaderboard. It is that every claim in the
article can be traced to a per-cell record, and that the checks which decide
whether a number may be published are code rather than prose.

## Layout

```
runner/
  scripts/
    bench_runner.py      job-based runner: one process per cell, resume, timeout,
                         shards on a single host, environment fingerprint
    topic_pipeline.py    data and model layer: dataset families, splits, model
                         adapters, capability caps, per-unit metrics
    analyze.py           analysis stage -> runs/<n>/stats/*.json
    review_gate.py       the gate: eleven assertions over a finished run
    coverage_report.py   per-system applicability table and incomplete cells
    make_figures.py      figures and tables from the analysis output
    select_hpo_subset.py stratified subsample for the budget-matched HPO leg
    snapshot_cc18.py     freezes the OpenML-CC18 resolution before the final run
    check_core_types.py  compares the declared task types against the shipped data
  gate_config.yaml       coverage requirements and documented limits
  audited_claims.yaml    the published claims put under audit, and what each
                         source supplied for them
  datasets/
    core.yaml            TALENT-tiny reproducible core, 45 datasets
    cc18.yaml            OpenML-CC18 leg, resolved from the frozen snapshot
    cc18_resolved.json   the frozen resolution: 72 tasks, 3 dropped as core
                         overlaps, 69 kept
    tabred.yaml          leakage-aware temporal leg, 8 datasets
    core_hpo20.yaml      the stratified 20-dataset subsample of the core used
                         for the budget-matched leg
    core_openml_ids.json        name -> OpenML dataset id map used for the dedup
    core_name_aliases.json      release names that differ from the OpenML names
    core_openml_ids_VERIFY.csv  how each id was resolved, and which were not
requirements.txt         the environment recorded in the published records
```

## Reproducing a run

Set the data roots and run the matrix into a fresh output directory. The final
pass disables resume; several invocations may share one output directory on a
single host, and the gate checks that they cover the grid exactly once.

```bash
export TALENT_DATA_ROOT=/path/to/talent-tiny
export TABRED_DATA_ROOT=/path/to/tabred

python runner/scripts/bench_runner.py \
  --datasets core,cc18,tabred --protocols cv_standard,temporal,matched_hpo \
  --models tabpfnv2,tabpfn25,tabpfn3,tabicl,mitra,catboost_tuned,xgboost_tuned,lightgbm_tuned,realmlp,tabm \
  --seeds 0,1,2,3,4 --datasets-dir runner/datasets --outdir runs/final \
  --no-resume --shard 0/1 --timeout-s 3600

python runner/scripts/analyze.py runs/final --claims runner/audited_claims.yaml
python runner/scripts/review_gate.py runs/final --config runner/gate_config.yaml \
  && python runner/scripts/make_figures.py runs/final
```

Figures are produced only if the gate passes. `--claims` is what turns the
generic pairwise machinery into the claim-survival audit of Table 1 and the
reporting-standard audit of Table 3; without it the analysis derives candidate
claims from the data instead, which is a different and weaker exercise.

The budget-matched leg is run separately on the stratified subsample, because a
100-trial nested search over the whole core does not fit a reasonable compute
budget and the budget itself is what the leg measures:

```bash
python runner/scripts/select_hpo_subset.py --core-yaml runner/datasets/core.yaml \
  --k 20 --seed 20260709 --out runner/datasets/core_hpo20.yaml

python runner/scripts/bench_runner.py \
  --datasets core_hpo20 --protocols matched_hpo \
  --models catboost_tuned,xgboost_tuned,lightgbm_tuned \
  --seeds 0,1,2,3,4 --datasets-dir runner/datasets --outdir runs/final \
  --no-resume --shard 0/1 --timeout-s 14400
```

## Settings

Behaviour is configured through the dataset yaml files and through environment
variables, all prefixed `TEP_`. The defaults are the values used for the
published numbers, so nothing below has to be set in order to reproduce them.

| variable | default | effect |
|---|---|---|
| `TEP_HPO_TRIALS` | 100 | trials of the nested search in the budget-matched leg; this budget is the object under measurement and should not be cut |
| `TEP_GBDT_SEARCH_ITER` | 500 | ceiling on the number of trees during the search |
| `TEP_GBDT_FULL_ITER` | 1000 | ceiling on the number of trees at the final refit |
| `TEP_GBDT_ES_ROUNDS` | 50 | early-stopping patience |
| `TEP_GBDT_ES_VAL_FRAC` | 0.15 | fraction carved out of the inner training split for early stopping, so the scoring fold is untouched and the optimism gap keeps its meaning |
| `TEP_GBDT_THREADS` | -1 | shared thread cap for the gradient-boosted libraries |
| `TEP_XGB_THREADS`, `TEP_LGB_THREADS`, `TEP_CB_THREADS` | fall back to `TEP_GBDT_THREADS` | per-library thread caps; CatBoost self-throttles and can be given more cores |
| `TEP_MIN_FREE_GB` | 20 | preflight refuses to start with less free disk than this |
| `TEP_DATASETS_DIR` | `runner/datasets` | where the dataset definitions are read from |
| `TEP_AG_TMPDIR` | system temporary directory | scratch directory for the AutoGluon-backed model |

## What the gate asserts

| id | assertion |
|---|---|
| A0 | every key in the configuration is one the gate reads |
| A1 | resume disabled, one host, each cell present exactly once, recomputed cells accounted for |
| B1 | every comparative claim has a run on an independent dataset family |
| C1 | calibration recorded |
| D1 | optimism gap recorded for the budget-matched leg |
| E1 | omnibus and post-hoc statistics present |
| F1 | no skip caused by a missing backend |
| F2 | one interpreter, one library set, one host; unstamped records reported |
| F3 | every system-family-protocol group holds a result or is a capability skip |
| G1 | every incomplete cell matches a documented-limit entry |
| H1 | no unit was fitted on a silently reduced in-context support set |

A cell may be absent, but not unmentioned: documented limits live in
`gate_config.yaml` and are published with the results.

## Per-unit records

Each record carries the predictions, the metrics, the environment fingerprint
(interpreter, host, library versions, source hash) and the provenance of its
task type. Records also carry the automatic adaptations a system applied to
itself, such as a reduced in-context support set, so that a result which is not
comparable can be recognised as such from the data rather than from a log.

## Relation to the published numbers

This release differs from the copy that produced the published numbers in two
respects, both cosmetic: the project name, which appears in module docstrings
and in the `TEP_` environment-variable prefix, and the language of the source
comments, which were translated from the working language of the project. No
default, threshold, statistical procedure or gate assertion differs. The source
hash recorded under `env.code_sha256` in the released records was computed over
the working copy and will therefore not match a hash computed over this
directory; the records, the statistics and the figures are unaffected.

## Requirements

Python 3.12; see `requirements.txt`. The twelve pinned versions there are the
ones recorded in every unit record, so the environment can be checked against
the data rather than taken on trust.

## Licence

To be added before release.
