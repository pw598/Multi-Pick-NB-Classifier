# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Batch Predict
# MAGIC
# MAGIC Predicts every item from an SQL pull (default: uncategorized items) or a file, using a saved model
# MAGIC and **that model's own cleaning recipe**. Writes `outputs/predictions_<timestamp>/`:
# MAGIC
# MAGIC | file | contents |
# MAGIC |---|---|
# MAGIC | `predictions_<ts>.csv` | one row per input row, in the ensemble's format (comma-separated) |
# MAGIC | `metadata_<ts>.json` | model folder, hyperparameters, cleaning configuration, data source, row counts |
# MAGIC | `evaluation/` | only when the input carries true labels: accuracy by level and by confidence bucket |
# MAGIC
# MAGIC Confidences are raw model scores. When the model folder holds a `calibration.json` and
# MAGIC `CALIBRATED_COLUMNS = True`, `L1 Calibrated` ... `L4 Calibrated` are added after the margins: the
# MAGIC calibrated probability that the predicted prefix is correct with nothing selected by a user.
# MAGIC `TOP_K_RECOMMENDATIONS > 0` also writes `recommendations_<ts>.csv`, the k options per level that the
# MAGIC dropdowns would show (one row per item, level and rank), ranked by `RECOMMENDATION_RANKING`.

# COMMAND ----------

# ---- MLflow -------------------------------------------------------------------
USE_MLFLOW = False                    # False = no tracking at all
MLFLOW_EXPERIMENT = None              # None = this notebook's experiment; or e.g. "/Users/patrick.white@vallen.com/nb-classifier"

# ---- model --------------------------------------------------------------------
MODEL = None                          # None = newest folder in models/; or "model_20261016_120000" / "20261016_120000"

# ---- data source (RAW_DATA_FILE supersedes SQL_QUERY) -----------------------
RAW_DATA_FILE = None                  # e.g. "data/my_extract.txt"; read with dtype=str
RAW_DATA_DELIMITER = "|"              # delimiter of RAW_DATA_FILE
RAW_DATA_QUOTED = True                # False if RAW_DATA_FILE uses no quoting and descriptions may start with "
SQL_QUERY = "unlabeled_items"          # a file in data/queries/ (name without .sql); may be None only if RAW_DATA_FILE is set

# ---- items to exclude ---------------------------------------------------------
BLACKLIST_FILE = "data/blacklists/bl1.csv"                 # e.g. "data/blacklists/blacklist.txt" (one column)
BLACKLIST_COL = "ID"           # data column the blacklist values are matched against
BLACKLIST_HAS_HEADER = True           # first line of the blacklist file is a header

# ---- columns ------------------------------------------------------------------
TEXT_COL = None                       # None = the description column the model was trained on (e.g. "FullDesc")
KEEP_COLS = ["ID", "Vallen ID"]       # identifiers / extra columns copied into the output ("ID" = ensemble join key)
INCLUDE_TEXT = True                   # also write the raw and cleaned descriptions

# ---- evaluation (only used when the data contains the model's label columns) ----
BUCKET_WIDTH = 0.05                   # confidence bucket width; change and re-run the evaluation cells

# ---- calibration (only used when the model folder has calibration.json) -----------
CALIBRATED_COLUMNS = True             # add L1..L4 Calibrated to the predictions (after the margins)
TOP_K_RECOMMENDATIONS = 0             # > 0: also write the top-k options per level (recommendations_<ts>.csv)
RECOMMENDATION_RANKING = "total"      # "total" (summed over all paths) | "best_path" (best single path)

# COMMAND ----------

# ---- setup (no settings below this line need changing) ----------------------
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from nbclassifier import artifacts, data, evaluation, paths, predict
from nbclassifier.tracking import Tracker

TS = paths.timestamp()
tracker = Tracker(USE_MLFLOW, MLFLOW_EXPERIMENT, run_name=f"04_batch_predict_{TS}",
                  tags={"notebook": "04_batch_predict"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model

# COMMAND ----------

model = artifacts.load_model(MODEL)
clf, cleaner, man = model.clf, model.cleaner, model.manifest
LEVEL_COLUMNS = clf.level_columns
NAME_COLUMNS = [c for c in (model.lookup.columns if model.lookup is not None else []) if c not in LEVEL_COLUMNS]
TEXT_COL = TEXT_COL or model.training_metadata.get("text_col", "FullDesc")
CLEANED_COL = model.training_metadata.get("cleaned_col", "CleanDesc")

print(f"model     : {model.folder.name}  (created {man['created_utc']})")
print(f"trained on: {man['n_training_rows']:,} rows, {man['n_features']:,} features, {man['n_paths']:,} paths")
print(f"cleaning  : {cleaner}")
print(f"calibration: {model.calibrator if model.calibrator is not None else 'none (raw scores only)'}")
print("hyperparameters:")
for k, v in clf.hp.to_dict().items():
    print(f"    {k:<22}{v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

raw, source = data.load_raw(RAW_DATA_FILE, SQL_QUERY, delimiter=RAW_DATA_DELIMITER, quoted=RAW_DATA_QUOTED,
                            spark=globals().get("spark"))
n_raw = len(raw)
n_black = 0
if BLACKLIST_FILE is not None:
    items = data.read_blacklist(BLACKLIST_FILE, has_header=BLACKLIST_HAS_HEADER)
    raw, n_black = data.apply_blacklist(raw, items, BLACKLIST_COL)
n_dup = data.duplicate_count(raw, "ID")
print(f"raw rows: {n_raw:,} | blacklisted: {n_black:,} | to predict: {len(raw):,}")
if n_dup:
    print(f"WARNING: {n_dup:,} rows share an 'ID' with another row (the ensemble joins on ID)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean (with the model's recipe; every row is kept)

# COMMAND ----------

df, counts = data.clean_dataframe(raw, cleaner, TEXT_COL, CLEANED_COL, drop_empty=False)
n_empty = int((df[CLEANED_COL].str.strip() == "").sum())
print(f"rows: {len(df):,} | cleaned to nothing: {n_empty:,} (predicted from priors only; oov_rate = 1)")
stats = data.ngram_stats(df[CLEANED_COL])
data.print_summary({"raw_rows": n_raw, "total_dropped": n_raw - len(df), "cleaned_rows": len(df)}, stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict

# COMMAND ----------

has_labels = all(c in df.columns for c in LEVEL_COLUMNS)
pred = predict.prediction_table(
    clf, df, CLEANED_COL,
    id_cols=KEEP_COLS,
    actual_cols=LEVEL_COLUMNS if has_labels else (),
    text_cols=[TEXT_COL, CLEANED_COL] if INCLUDE_TEXT else (),
    calibrator=model.calibrator if CALIBRATED_COLUMNS else None,
)
pred = predict.attach_names(pred, model.lookup, LEVEL_COLUMNS, NAME_COLUMNS)
print(f"{len(pred):,} predictions; labels present: {has_labels}")
pred.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save predictions

# COMMAND ----------

OUT_DIR = paths.new_folder(paths.OUTPUTS_DIR, "predictions_", TS)
pred_path = OUT_DIR / f"predictions_{TS}.csv"
pred.to_csv(pred_path, index=False)
metadata = {
    "created": TS,
    "model_folder": model.folder.name,
    "model_manifest": man,
    "hyperparameters": clf.hp.to_dict(),
    "cleaning": cleaner.describe(),
    "source": source,
    "blacklist_file": str(paths.resolve(BLACKLIST_FILE)) if BLACKLIST_FILE else None,
    "blacklist_col": BLACKLIST_COL,
    "row_counts": {"raw_rows": n_raw, "blacklisted_dropped": n_black, "predicted_rows": len(pred),
                   "cleaned_to_nothing": n_empty},
    "ngram_stats": stats,
    "columns": list(pred.columns),
    "has_labels": has_labels,
    "calibration": man.get("calibration") if CALIBRATED_COLUMNS else None,
}
data.write_json(OUT_DIR / f"metadata_{TS}.json", metadata)
print("wrote", pred_path)

if TOP_K_RECOMMENDATIONS > 0:
    from nbclassifier.recommend import Recommender
    rec = Recommender.from_loaded(model, k=TOP_K_RECOMMENDATIONS, ranking=RECOMMENDATION_RANKING)
    recs = rec.recommend(texts=df[CLEANED_COL])
    ids = df[[c for c in KEEP_COLS if c in df.columns]].reset_index(drop=True)
    recs = ids.iloc[recs["row"].to_numpy()].reset_index(drop=True).join(recs.drop(columns=["row"]))
    recs.to_csv(OUT_DIR / f"recommendations_{TS}.csv", index=False)
    print(f"wrote {len(recs):,} recommendation rows")

tracker.log_params({"model_folder": model.folder.name, "predictions_folder": OUT_DIR.name,
                    "source": {k: v for k, v in source.items() if k != "query_text"}})
tracker.log_metrics({**metadata["row_counts"], **stats})
tracker.log_artifact(OUT_DIR / f"metadata_{TS}.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate (only if the data contained true labels)
# MAGIC
# MAGIC Headline accuracy by level, then accuracy by confidence bucket. Change `BUCKET_WIDTH` and re-run
# MAGIC from the next cell; nothing is re-predicted.

# COMMAND ----------

if has_labels:
    labelled = pred.dropna(subset=[f"Actual {c}" for c in LEVEL_COLUMNS])
    print(f"{len(labelled):,} labelled rows")
    head = evaluation.headline(labelled, LEVEL_COLUMNS)
    display(head)
    tracker.log_metrics(head.loc["all"].to_dict())
else:
    print("no label columns in the data; skipping evaluation")

# COMMAND ----------

# BUCKET_WIDTH = 0.1   # <- change here and re-run this cell and the next
if has_labels:
    EVAL_DIR = OUT_DIR / "evaluation"
    buckets = evaluation.bucket_tables(labelled, LEVEL_COLUMNS, width=BUCKET_WIDTH)
    EVAL_DIR.mkdir(exist_ok=True)
    buckets.to_csv(EVAL_DIR / f"buckets_width_{BUCKET_WIDTH}.csv", index=False)
    display(buckets[buckets["level"] == f"L{len(LEVEL_COLUMNS)}"])

# COMMAND ----------

if has_labels:
    fig = evaluation.plot_accuracy_by_bucket(labelled, LEVEL_COLUMNS, width=BUCKET_WIDTH, target_accuracy=0.95)
    evaluation.save_figure(fig, EVAL_DIR, f"accuracy_by_bucket_by_level_w{BUCKET_WIDTH}", tracker)
    fig = evaluation.plot_coverage(labelled, LEVEL_COLUMNS)
    evaluation.save_figure(fig, EVAL_DIR, "coverage_by_level", tracker)

# COMMAND ----------

tracker.end()