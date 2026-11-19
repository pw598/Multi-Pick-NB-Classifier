# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Train to Deploy
# MAGIC
# MAGIC Trains one model on **all** labelled rows (SQL pull or a file, optional blacklist) and saves it to
# MAGIC `models/model_<timestamp>/` together with the exact cleaning recipe, the hyperparameters and a
# MAGIC description of the training data. Notebook 04 uses the newest model folder by default.
# MAGIC
# MAGIC With `CALIBRATION` set, the calibration fitted from out-of-fold predictions in notebook 02 is
# MAGIC checked against these hyperparameters and saved in the model folder as `calibration.json`. The
# MAGIC temperatures and lookup tables then apply to the model trained here on all rows.

# COMMAND ----------

# ---- MLflow -------------------------------------------------------------------
USE_MLFLOW = False                    # False = no tracking at all
MLFLOW_EXPERIMENT = None              # None = this notebook's experiment; or e.g. "/Users/patrick.white@vallen.com/nb-classifier"

NOTES = ""                            # free text stored in the model's manifest.json

# ---- calibration (fitted from OOF cross-validation in notebook 02) --------------
CALIBRATION = "latest"                    # None = no calibration; "latest" / "evaluation_<ts>" = that evaluation's
                                      # calibration; or a path to a calibration.json
CALIBRATION_CANDIDATE = None          # None = the calibrated candidate in that evaluation (most recent if several)

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
# Single values only. Notebook 02 prints the best combination in this exact format.

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
    "vectorizer_kind":     "tfidf",
    "max_features":        None,
    "ngram_range":         (1, 2),
    "min_df":              2,
    "max_df":              0.95,
    "stop_words":          "english",
    "sublinear_tf":        True,
    "binary":              True,
    "norm":                "l2",
    "use_idf":             False,
    "ordered_bigrams":     True,
    "token_pattern":       "whitespace",
    "alpha":               0.001,
    "fit_prior":           True,
    "prediction_mode":     "conditional",
    "rare_path_threshold": 5,
}

# COMMAND ----------

# ---- setup (no settings below this line need changing) ----------------------
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from nbclassifier import artifacts, calibration, config, data, paths
from nbclassifier.model import HierarchicalNB
from nbclassifier.tracking import Tracker

TS = paths.timestamp()
hp = config.expand_grid(HYPERPARAMETERS, grid_search=False)[0]
calibrator = None
if CALIBRATION:
    cal_path = calibration.find_calibration(CALIBRATION, CALIBRATION_CANDIDATE)
    calibrator = calibration.Calibrator.load(cal_path)
    print(f"calibration: {cal_path}\n  {calibrator}")
    probe = HierarchicalNB(LEVEL_COLUMNS, hp)
    calibrator.check_compatible(probe, strict=True)   # same hyperparameters as notebook 02's candidate
tracker = Tracker(USE_MLFLOW, MLFLOW_EXPERIMENT, run_name=f"03_train_{TS}", tags={"notebook": "03_train_to_deploy"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull and clean data

# COMMAND ----------

if PREPARED_DATA:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure model

# COMMAND ----------

train_df, rare = data.filter_rare_paths(prepared.df, LEVEL_COLUMNS, hp.rare_path_threshold)
print(f"rare-path filter (threshold {hp.rare_path_threshold}): {rare['rows_dropped']:,} rows dropped, "
      f"{rare['paths_before']:,} -> {rare['paths_after']:,} paths, {rare['rows_after']:,} training rows")
config.params_table([hp]).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model

# COMMAND ----------

t0 = time.time()
clf = HierarchicalNB(LEVEL_COLUMNS, hp).fit(train_df[CLEANED_COL], train_df[LEVEL_COLUMNS])
fit_seconds = time.time() - t0
print(f"fitted in {fit_seconds:.0f}s: {clf.n_features:,} features, {clf.n_paths:,} paths")
print("classes per level:", clf.classes_per_level)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save model

# COMMAND ----------

MODEL_DIR = paths.new_folder(paths.MODELS_DIR, "model_", TS)
lookup_cols = [c for c in LEVEL_COLUMNS + NAME_COLUMNS if c in train_df.columns]
training_metadata = {
    **{k: v for k, v in prepared.metadata.items() if k not in ("loaded_from",)},
    "dataset_folder": prepared.metadata.get("loaded_from"),
    "rare_path_filter": rare,
    "n_training_rows": int(len(train_df)),
    "fit_seconds": round(fit_seconds, 1),
}
files = artifacts.save_model(MODEL_DIR, clf, prepared.cleaner, lookup=train_df[lookup_cols],
                             training_metadata=training_metadata, notes=NOTES,
                             calibrator=calibrator)
for k, v in files.items():
    print(f"{k:<22} {v.name}")
print("\nmodel folder:", MODEL_DIR)

# reload check: the saved model must give identical predictions
check = artifacts.load_model(MODEL_DIR)
sample = train_df[CLEANED_COL].head(500)
assert clf.predict_frame(sample).equals(check.clf.predict_frame(sample)), "reloaded model differs!"
if calibrator is not None:
    assert check.calibrator is not None
    a = clf.predict_frame(sample, calibrator=calibrator)
    b = check.clf.predict_frame(sample, calibrator=check.calibrator)
    assert a.equals(b), "reloaded calibration differs!"
print("reload check passed" + (" (with calibration)" if calibrator is not None else ""))

# COMMAND ----------

tracker.log_params(hp.to_dict())
if calibrator is not None:
    tracker.log_params({"calibration": str(cal_path), "temperatures": calibrator.temperatures})
tracker.log_params({"model_folder": MODEL_DIR.name, "cleaning_method": prepared.cleaner.method_name,
                    "source": {k: v for k, v in prepared.metadata["source"].items() if k != "query_text"}})
tracker.log_metrics({"n_training_rows": len(train_df), "n_features": clf.n_features, "n_paths": clf.n_paths,
                     "fit_seconds": fit_seconds, **rare, **prepared.metadata["row_counts"],
                     **prepared.metadata["ngram_stats"]})
tracker.log_artifacts(MODEL_DIR, artifact_path="model")
tracker.end()