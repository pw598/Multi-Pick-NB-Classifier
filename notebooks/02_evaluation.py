# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Evaluation (Cross-Validation and Grid Search)
# MAGIC
# MAGIC * `GRID_SEARCH = False`: one cross-validation of the hyperparameters below (every value must be a
# MAGIC   scalar or a one-element list).
# MAGIC * `GRID_SEARCH = True`: any hyperparameter may be a list; every combination is cross-validated.
# MAGIC
# MAGIC Every (candidate, fold) result is written to `outputs/evaluation_<timestamp>/` as soon as it is
# MAGIC done. If the session dies, set `RESUME_FROM` to that folder and run all cells again: finished
# MAGIC work is skipped, and the data is re-read from the folder rather than re-pulled.
# MAGIC
# MAGIC The best candidate's out-of-fold predictions are exported in the ensemble's format to
# MAGIC `outputs/evaluation_<timestamp>/best/`. Bucket tables and plots can be re-run with a different
# MAGIC `BUCKET_WIDTH` at any time without re-fitting.
# MAGIC
# MAGIC **Calibration.** The last section fits calibrated probabilities for the dropdown interface from
# MAGIC out-of-fold predictions of one candidate (by default the best one): per-level temperatures, then
# MAGIC lookup tables for every scenario (target level x levels fixed by the user x whether the user
# MAGIC confirmed or corrected the model's pick), over all top-k candidates, not only rank 1. The result is
# MAGIC written to `outputs/evaluation_<timestamp>/calibration/candidate_<k>/`; notebook 03 copies it into
# MAGIC the model folder (`CALIBRATION = "evaluation_<timestamp>"`). Calibrate the hyperparameters you will
# MAGIC deploy, including `prediction_mode` (normally `"conditional"`).

# COMMAND ----------

# ---- MLflow -------------------------------------------------------------------
USE_MLFLOW = True                     # False = no tracking at all
MLFLOW_EXPERIMENT = "/Users/patrick.white@vallen.com/nb-classifier"     # None = this notebook's experiment; or e.g. "/Users/patrick.white@vallen.com/nb-classifier"

# ---- evaluation ---------------------------------------------------------------
GRID_SEARCH = True                    # True: lists below are expanded into every combination
CV_TYPE = "oof"                       # "oof" (k-fold out-of-fold) | "repeated_holdout"
N_FOLDS = 3                           # oof: must match the NN repo for the ensemble
FOLD_METHOD = "id_hash"               # oof: "id_hash" (matches the NN repo) | "stratified"
FOLD_KEY_COL = "ID"                   # oof id_hash: column whose value decides the fold
FOLD_SEED = 0                         # oof: must match the NN repo for the ensemble
N_REPEATS = 5                         # repeated_holdout: number of random splits
TEST_SIZE = 0.2                       # repeated_holdout: share of rows held out each time
HOLDOUT_SEED = 42                     # repeated_holdout
SELECTION_METRIC = "L4 Accuracy"      # how the best candidate is chosen ("L1 Accuracy" ... "L4 Accuracy")
RESUME_FROM = None                    # e.g. "evaluation_20261016_143000" to continue an interrupted run
SAVE_FOLD_MODELS = True               # keep each fold's fitted model (~15-20 MB per fold and vectorizer setting)
                                      # so the calibration section reuses it instead of refitting

# COMMAND ----------

# ---- prepared data ------------------------------------------------------------
PREPARED_DATA = None                  # a data/<timestamp> folder from notebook 01 ("latest" = newest);
                                      # None = pull and clean below with the settings in this cell

# ---- data source (RAW_DATA_FILE supersedes SQL_QUERY) -----------------------
RAW_DATA_FILE = None                  # e.g. "data/my_extract.txt"; read with dtype=str
RAW_DATA_DELIMITER = "|"              # delimiter of RAW_DATA_FILE
RAW_DATA_QUOTED = True                # False if RAW_DATA_FILE uses no quoting and descriptions may start with "
SQL_QUERY = "labeled_items"          # a file in data/queries/ (name without .sql); may be None only if RAW_DATA_FILE is set

# ---- items to exclude ---------------------------------------------------------
BLACKLIST_FILE = "data/blacklists/bl1.csv"                 # e.g. "data/blacklists/blacklist.txt" (one column)
BLACKLIST_COL = "ID"           # data column the blacklist values are matched against
BLACKLIST_HAS_HEADER = True           # first line of the blacklist file is a header

# ---- cleaning -----------------------------------------------------------------
CLEANING_METHOD = "cleaning_method_20261015"   # a folder in cleaning/ (code, term lists, settings in method.yaml)
APPEND_FIELDS = {}                    # e.g. {"Global Vendno": "VEND", "Prodline": "PLINE"} -> VEND_<value> tokens
APPEND_NULLS_AS_NONE = True           # True: null field -> VEND_None;  False: no marker for that row

# ---- columns ------------------------------------------------------------------
TEXT_COL = "FullDesc"                 # raw description field to clean
CLEANED_COL = "CleanDesc"             # header for the cleaned description
LEVEL_COLUMNS = ["Level 1 ID", "Level 2 ID", "Level 3 ID", "Level 4 ID"]      # labels the model learns
NAME_COLUMNS = ["Level 1 Name", "Level 2 Name", "Level 3 Name", "Level 4 Name"]  # carried for readability
ID_COLS = ["ID", "Vallen ID"]         # identifiers carried into every output ("ID" is the ensemble join key)

# COMMAND ----------

# ==== HYPERPARAMETERS ==========================================================
# Every hyperparameter is listed. With GRID_SEARCH = True give lists, e.g. "alpha": [0.001, 0.01].
# Tuples are single values (ngram_range=(1, 2)); write a list of tuples to search, e.g. [(1, 1), (1, 2)].

# vectorizer_kind      "tfidf" or "count"
# max_features         vocabulary cap (None = keep every feature that passes min_df/max_df)
# ngram_range          (min_n, max_n) as a TUPLE, e.g. (1, 2) = unigrams + bigrams
# min_df               ignore features in fewer documents than this (int) or proportion (float < 1)
# max_df               ignore features in more documents than this proportion (float) or count (int)
# stop_words           "english" or None
# sublinear_tf         tfidf only: use 1 + log(tf)
# binary               all non-zero counts set to 1 (before idf for tfidf)
# norm                 tfidf only: "l2", "l1" or None
# use_idf              tfidf only: inverse-document-frequency weighting
# ordered_bigrams      False = "A B" and "B A" are the same bigram (needs 2 in ngram_range)
# token_pattern        "whitespace" = each cleaned word is a token (<SIZE>IN stays whole);
#                      "sklearn" = sklearn default (2+ word characters: <SIZE>IN -> SIZE, IN); or a regex
# alpha                additive (Laplace/Lidstone) smoothing, > 0
# fit_prior            learn class priors (True) or use uniform priors (False)
# prediction_mode      "independent" | "conditional" | "flattened"  (see README)
# rare_path_threshold  drop training rows whose category path occurs fewer times than this

HYPERPARAMETERS = {
    "vectorizer_kind":     ["tfidf"],               # tfidf, count
    "max_features":        [None],
    "ngram_range":         [(1, 2)],
    "min_df":              [2],
    "max_df":              [0.95],
    "stop_words":          ["english"],
    "sublinear_tf":        [True],
    "binary":              [True],
    "norm":                ["l2"],
    "use_idf":             [False],
    "ordered_bigrams":     [True],
    "token_pattern":       ["whitespace"],
    "alpha":               [0.001],
    "fit_prior":           [True],
    "prediction_mode":     ["conditional"],         # independent, conditional, flattened
    "rare_path_threshold": [5],
}

# COMMAND ----------

# ---- setup (no settings below this line need changing) ----------------------
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from nbclassifier import config, crossval, data, evaluation, paths
from nbclassifier.tracking import Tracker

candidates = config.expand_grid(HYPERPARAMETERS, GRID_SEARCH)
cv = config.CVSettings(cv_type=CV_TYPE, n_folds=N_FOLDS, fold_method=FOLD_METHOD, fold_key_col=FOLD_KEY_COL,
                       fold_seed=FOLD_SEED, n_repeats=N_REPEATS, test_size=TEST_SIZE, holdout_seed=HOLDOUT_SEED)
tracker = Tracker(USE_MLFLOW, MLFLOW_EXPERIMENT, run_name=f"02_evaluation_{paths.timestamp()}",
                  tags={"notebook": "02_evaluation", "grid_search": GRID_SEARCH})
print(f"{len(candidates)} candidate(s) x {N_FOLDS if CV_TYPE == 'oof' else N_REPEATS} splits")
config.params_table(candidates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data

# COMMAND ----------

if RESUME_FROM:
    prepared = None
    print(f"Resuming {RESUME_FROM}: the data saved in that folder is used; data settings above are ignored.")
elif PREPARED_DATA:
    prepared = data.load_dataset(PREPARED_DATA)
    print(f"loaded {prepared.metadata['loaded_from']} (cleaning method {prepared.cleaner.method_name!r})")
    prepared = data.blacklist_prepared(prepared, BLACKLIST_FILE, BLACKLIST_COL, BLACKLIST_HAS_HEADER)
    data.print_summary(prepared.metadata["row_counts"], prepared.metadata["ngram_stats"])
    TEXT_COL, CLEANED_COL = prepared.metadata["text_col"], prepared.metadata["cleaned_col"]
else:
    prepared = data.prepare_dataset(
        raw_data_file=RAW_DATA_FILE, sql_query=SQL_QUERY, delimiter=RAW_DATA_DELIMITER,
        blacklist_file=BLACKLIST_FILE, blacklist_col=BLACKLIST_COL, blacklist_has_header=BLACKLIST_HAS_HEADER,
        cleaning_method=CLEANING_METHOD,
        append_fields=APPEND_FIELDS, append_nulls_as_none=APPEND_NULLS_AS_NONE,
        text_col=TEXT_COL, cleaned_col=CLEANED_COL, level_columns=LEVEL_COLUMNS,
        raw_data_quoted=RAW_DATA_QUOTED, spark=globals().get("spark"),
    )
if prepared is not None and data.duplicate_count(prepared.df, FOLD_KEY_COL):
    print(f"note: {data.duplicate_count(prepared.df, FOLD_KEY_COL):,} rows share a {FOLD_KEY_COL!r} value "
          "(they always land in the same fold)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross-validation / grid search

# COMMAND ----------

if RESUME_FROM:
    ev = crossval.Evaluation.open(RESUME_FROM, candidates=candidates, cv=cv)
else:
    columns = crossval.Columns(text_col=TEXT_COL, cleaned_col=CLEANED_COL, level_columns=LEVEL_COLUMNS,
                               name_columns=NAME_COLUMNS, id_cols=ID_COLS)
    ev = crossval.Evaluation.create(prepared, candidates, cv, columns, grid_search=GRID_SEARCH)
    del prepared  # the evaluation keeps its own copy on disk
LEVEL_COLUMNS = ev.cols.level_columns  # (as saved in the evaluation folder, relevant when resuming)
print("evaluation folder:", ev.folder)

results = ev.run(tracker=tracker, save_fold_models=SAVE_FOLD_MODELS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

acc_cols = [c for c in results.columns if c.endswith("Accuracy")]
best_id, best_hp, best_row = ev.best(SELECTION_METRIC)

if ev.grid_search:
    show = results.sort_values(SELECTION_METRIC, ascending=False)
    varying = [c for c in show.columns if c in best_hp.to_dict() and show[c].astype(str).nunique() > 1]
    display(show[varying + acc_cols + [c + " std" for c in acc_cols] + ["splits_done"]])
    print(f"best candidate: {best_id}  ({SELECTION_METRIC} = {best_row[SELECTION_METRIC]:.4f})")
    print("\nbest hyperparameters (paste into notebook 03):")
else:
    print("hyperparameters:")
print("HYPERPARAMETERS = {")
for k, v in best_hp.to_dict().items():
    v = tuple(v) if k == "ngram_range" else v
    print(f"    {k!r:<22}: {v!r},")
print("}")

print("\nheadline accuracy by level (best candidate):")
display(ev.headline(best_id))

tracker.log_params({"best_candidate": best_id, **{f"best.{k}": v for k, v in best_hp.to_dict().items()}})
tracker.log_metrics({f"best {c}": best_row[c] for c in acc_cols})

# COMMAND ----------

# Export the best candidate's out-of-fold predictions (ensemble input) and load them for the plots below
export_path = ev.export(best_id)
pred = ev.predictions(best_id, with_text=False)
tracker.log_artifact(export_path.with_suffix(".json"), "best")
pred.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results by confidence bucket (best candidate)
# MAGIC
# MAGIC Change `BUCKET_WIDTH` and re-run the cells below; no re-fitting is needed.

# COMMAND ----------

BUCKET_WIDTH = 0.05
TARGET_ACCURACY = 0.95                # dashed lines: target accuracy and the coverage that achieves it
SPLIT_COL = ev.split_col
PLOT_DIR = ev.folder / "best" / "plots"

buckets = evaluation.bucket_tables(pred, LEVEL_COLUMNS, width=BUCKET_WIDTH)
buckets.to_csv(ev.folder / "best" / f"buckets_width_{BUCKET_WIDTH}.csv", index=False)
tracker.log_table(buckets, f"best/buckets_width_{BUCKET_WIDTH}.csv")
ROUND_COLS = ["lower", "upper", "share_of_items", "accuracy", "mean_confidence",
              "cumulative_coverage", "cumulative_accuracy"]
display(buckets[buckets["level"] == f"L{len(LEVEL_COLUMNS)}"].round({c: 3 for c in ROUND_COLS}))

# COMMAND ----------

# Item count and accuracy by confidence bucket (deepest level)
fig = evaluation.plot_accuracy_by_bucket(pred, LEVEL_COLUMNS, width=BUCKET_WIDTH, depth=len(LEVEL_COLUMNS),
                                         target_accuracy=TARGET_ACCURACY)
evaluation.save_figure(fig, PLOT_DIR, f"accuracy_by_bucket_L{len(LEVEL_COLUMNS)}_w{BUCKET_WIDTH}", tracker)

# COMMAND ----------

# Violin plot of cross-validation results by confidence bucket (deepest level)
fig = evaluation.plot_violins(pred, LEVEL_COLUMNS, width=BUCKET_WIDTH, depth=len(LEVEL_COLUMNS), split_col=SPLIT_COL)
evaluation.save_figure(fig, PLOT_DIR, f"violins_L{len(LEVEL_COLUMNS)}_w{BUCKET_WIDTH}", tracker)

# COMMAND ----------

# Item count and accuracy by confidence bucket, by category level
fig = evaluation.plot_accuracy_by_bucket(pred, LEVEL_COLUMNS, width=BUCKET_WIDTH, target_accuracy=TARGET_ACCURACY)
evaluation.save_figure(fig, PLOT_DIR, f"accuracy_by_bucket_by_level_w{BUCKET_WIDTH}", tracker)

# COMMAND ----------

# Violin plots of cross-validation results by confidence bucket, by category level
fig = evaluation.plot_violins(pred, LEVEL_COLUMNS, width=BUCKET_WIDTH, split_col=SPLIT_COL)
evaluation.save_figure(fig, PLOT_DIR, f"violins_by_level_w{BUCKET_WIDTH}", tracker)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration (out-of-fold)
# MAGIC
# MAGIC For the calibrated candidate, the fold models from the cross-validation above are reused
# MAGIC (`SAVE_FOLD_MODELS = True`; folds without a saved model are refitted) and the held-out rows are scored
# MAGIC for every scenario the interface can produce:
# MAGIC
# MAGIC * **nothing fixed**: each level's top-k classes;
# MAGIC * **L1..Lc fixed**: the top-k descendants of the (true) selected node, split into **confirmed**
# MAGIC   (the selected node was the model's own top pick) and **corrected** (it was not).
# MAGIC
# MAGIC Both dropdown **rankings** are calibrated, each with its own tables (`CAL_RANKINGS`):
# MAGIC `"total"` ranks a category by the summed probability of every path below it ("most likely to be
# MAGIC correct"); `"best_path"` by its single most likely path ("on the most convincing complete paths").
# MAGIC Level 4 is the same under both.
# MAGIC
# MAGIC Step 1 fits one temperature per level (OOF log-loss of the true child given the true parent).
# MAGIC Step 2 builds a lookup table per scenario from tempered score to observed accuracy:
# MAGIC `CAL_SCHEME = "percent"` is the classic 1%-bucket table; `"logit"` merges fine log-odds bins until each
# MAGIC holds `CAL_MIN_BIN_COUNT` candidates (better resolution near 0% and 100%, where most naive Bayes
# MAGIC scores sit). Both are made monotone and clipped to [0.01%, 99.99%].
# MAGIC
# MAGIC The *calibrated* rows in the metrics are **cross-fitted** (tables fitted without the fold they score),
# MAGIC so they are an honest estimate. With saved fold models nothing is refitted: runtime is the scoring
# MAGIC pass, roughly 1-2 s per 1,000 rows per ranking.

# COMMAND ----------

# ---- calibration settings ------------------------------------------------------
CALIBRATE = True                      # False = skip this section
CALIBRATION_CANDIDATE = None          # None = the best candidate above; or a candidate number
CAL_TOP_K = 10                        # candidates recorded per row and scenario (>= dropdown size of 5)
CAL_SCHEME = "logit"                  # "logit" (adaptive log-odds bins) | "percent" (1% buckets)
CAL_MIN_BIN_COUNT = 200               # "logit": candidates per merged bin
CAL_MIN_SCENARIO_COUNT = 1000         # fewer -> fall back to the pooled (confirmed + corrected) table
CAL_CLIP = (0.0001, 0.9999)           # calibrated probabilities are kept inside this range
CAL_FIT_TEMPERATURE = True            # False = lookup tables on raw scores (T = 1)
CAL_TEMPERATURE_MAX_ROWS = 200_000    # OOF rows sampled to fit the temperatures
CAL_KEEP_FOLD_MODELS = True           # only without saved fold models: False = refit each fold for the
                                      # second pass (less memory)
CAL_RANKINGS = ["total", "best_path"] # dropdown rankings to calibrate (drop "best_path" to save time)

# COMMAND ----------

from nbclassifier import calibration

if CALIBRATE:
    cal_candidate = best_id if CALIBRATION_CANDIDATE is None else CALIBRATION_CANDIDATE
    if ev.candidates[cal_candidate].prediction_mode != "conditional":
        print(f"note: candidate {cal_candidate} uses prediction_mode="
              f"{ev.candidates[cal_candidate].prediction_mode!r}; calibrate the mode you deploy")
    cal_settings = calibration.CalibrationSettings(
        top_k=CAL_TOP_K, scheme=CAL_SCHEME, min_bin_count=CAL_MIN_BIN_COUNT,
        min_scenario_count=CAL_MIN_SCENARIO_COUNT, clip_low=CAL_CLIP[0], clip_high=CAL_CLIP[1],
        fit_temperature=CAL_FIT_TEMPERATURE, temperature_max_rows=CAL_TEMPERATURE_MAX_ROWS,
        keep_fold_models=CAL_KEEP_FOLD_MODELS, rankings=CAL_RANKINGS)
    cal_result = calibration.fit_from_evaluation(ev, cal_candidate, cal_settings)
    CAL_DIR = cal_result.folder
    display(cal_result.temperature_fit)
    for r in cal_result.calibrator.rankings:
        print(f"\n[{r} ranking] expected calibration error by scenario (lower is better):")
        display(calibration.metrics_wide(cal_result.metrics, "ece", ranking=r).round(4))
        print(f"[{r} ranking] log-loss by scenario (lower is better):")
        display(calibration.metrics_wide(cal_result.metrics, "log_loss", ranking=r).round(4))
    tracker.log_params({"calibration.candidate": cal_candidate, **{f"calibration.{k}": v for k, v
                        in cal_settings.to_dict().items()}})
    tracker.log_metrics({f"temperature L{r.depth}": r.temperature for r in cal_result.temperature_fit.itertuples()})
    tracker.log_artifacts(CAL_DIR, artifact_path=f"calibration/{CAL_DIR.name}")
else:
    print("CALIBRATE = False; skipped")

# COMMAND ----------

# Reliability diagrams per ranking: rows = target level, columns = which levels are fixed (and confirmed/corrected)
if CALIBRATE:
    for r in cal_result.calibrator.rankings:
        fig = calibration.plot_reliability(cal_result.reliability, min_count=50, coarse=5, ranking=r)
        evaluation.save_figure(fig, CAL_DIR / "plots", f"reliability_{r}", tracker)

# COMMAND ----------

# Observed accuracy by rank within each scenario (e.g. how often the 7th L4 option is right once L1-L3 are fixed)
if CALIBRATE:
    ra = cal_result.rank_accuracy
    display(ra.pivot_table(index=["ranking", "target_level", "n_fixed", "kind"], columns="rank",
                           values="accuracy", sort=False).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Re-build the tables without re-fitting
# MAGIC
# MAGIC The fine histograms are saved, so the table settings (`scheme`, `min_bin_count`, `min_scenario_count`,
# MAGIC clip range) can be changed in seconds. Temperatures and `top_k` stay as fitted.

# COMMAND ----------

# Example: switch to 1% buckets. Overwrites calibration.json and the diagnostics in CAL_DIR.
# cal_result = calibration.rebuild_from_histograms(
#     CAL_DIR, calibration.CalibrationSettings(scheme="percent", min_scenario_count=CAL_MIN_SCENARIO_COUNT))
# display(calibration.metrics_wide(cal_result.metrics, "ece").round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Free the disk space used by fold models (optional)
# MAGIC
# MAGIC Once the calibration you want is fitted, the saved fold models are no longer needed. Delete them to free
# MAGIC roughly 15-20 MB per fold and vectorizer setting. A later calibration of this evaluation will then refit.

# COMMAND ----------

# freed = ev.delete_fold_models()
# print(f"freed {freed / 1e6:,.0f} MB")

# COMMAND ----------

tracker.end()