# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Pull and Clean Data
# MAGIC
# MAGIC Pulls raw data (SQL query or a file), removes blacklisted items, cleans the
# MAGIC description with a named cleaning method and writes everything to `data/<timestamp>/`:
# MAGIC
# MAGIC | file | contents |
# MAGIC |---|---|
# MAGIC | `raw_data_<ts>.txt` | the raw pull after the blacklist filter |
# MAGIC | `cleaned_data_<ts>.txt` | rows kept, with the cleaned description in `CLEANED_COL` |
# MAGIC | `cleaner.pkl` | the compiled cleaning recipe (re-used by notebooks 02/03) |
# MAGIC | `vocabulary.txt` | words kept by step 11 (learned from this data; frozen into `cleaner.pkl`) |
# MAGIC | `<cleaning method>/` | copy of the cleaning-method folder used (code, lists, settings) |
# MAGIC | `metadata_<ts>.json` | source, filters, cleaning configuration, row counts, n-gram counts |
# MAGIC
# MAGIC Notebooks 02 and 03 can use this folder directly (`PREPARED_DATA = "<timestamp>"`) or repeat
# MAGIC the same steps themselves in a single cell.

# COMMAND ----------

# ---- MLflow -------------------------------------------------------------------
USE_MLFLOW = False                    # False = no tracking at all
MLFLOW_EXPERIMENT = None              # None = this notebook's experiment; or e.g. "/Users/patrick.white@vallen.com/nb-classifier"

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
LEVEL_COLUMNS = ["Level 1 ID", "Level 2 ID", "Level 3 ID", "Level 4 ID"]  # rows missing a label are dropped; [] for unlabelled data
ID_COL = "ID"                         # used only to report duplicate identifiers

# ---- output -------------------------------------------------------------------
OUTPUT_DELIMITER = "|"                # delimiter of the .txt files written

# COMMAND ----------

# ---- setup (no settings below this line need changing) ----------------------
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from nbclassifier import cleaning, data, paths
from nbclassifier.tracking import Tracker

TS = paths.timestamp()
OUT_DIR = paths.DATA_DIR / TS
tracker = Tracker(USE_MLFLOW, MLFLOW_EXPERIMENT, run_name=f"01_pull_and_clean_{TS}",
                  tags={"notebook": "01_pull_and_clean_data"})
print("output folder:", OUT_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull raw data

# COMMAND ----------

raw, source = data.load_raw(RAW_DATA_FILE, SQL_QUERY, delimiter=RAW_DATA_DELIMITER, quoted=RAW_DATA_QUOTED,
                            spark=globals().get("spark"))
print(f"source : {source['type']} ({source.get('query_name') or source.get('path')})")
print(f"rows   : {len(raw):,}")
print(f"columns: {list(raw.columns)}")
n_dup = data.duplicate_count(raw, ID_COL)
if n_dup:
    print(f"WARNING: {n_dup:,} rows share an {ID_COL!r} with another row")
raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Blacklist

# COMMAND ----------

n_raw = len(raw)
n_black = 0
if BLACKLIST_FILE is not None:
    items = data.read_blacklist(BLACKLIST_FILE, has_header=BLACKLIST_HAS_HEADER)
    raw, n_black = data.apply_blacklist(raw, items, BLACKLIST_COL)
    print(f"{len(items):,} blacklisted items; {n_black:,} rows removed; {len(raw):,} remain")
else:
    print("no blacklist")

n_nolabel = 0
if LEVEL_COLUMNS:
    raw, n_nolabel = data.drop_missing_labels(raw, LEVEL_COLUMNS)
    print(f"{n_nolabel:,} rows dropped for a missing label in {LEVEL_COLUMNS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save raw data

# COMMAND ----------

OUT_DIR.mkdir(parents=True, exist_ok=False)
raw_path = OUT_DIR / f"raw_data_{TS}.txt"
raw.to_csv(raw_path, sep=OUTPUT_DELIMITER, index=False)
print(f"wrote {raw_path} ({len(raw):,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean
# MAGIC
# MAGIC Runs the 13 steps of `CLEANING_METHOD` (see `cleaning/<method>/method.yaml`), printing how many
# MAGIC descriptions each step changed with examples. Step 11's vocabulary is learned from this data.
# MAGIC Rows whose description is blank (before or after cleaning) are dropped, and marker tokens for
# MAGIC `APPEND_FIELDS` are added.

# COMMAND ----------

print("available cleaning methods:", cleaning.available_methods())
cleaner = cleaning.build_cleaner(CLEANING_METHOD, append_fields=APPEND_FIELDS,
                                 append_nulls_as_none=APPEND_NULLS_AS_NONE)
cleaned, counts = data.clean_dataframe(raw, cleaner, TEXT_COL, CLEANED_COL, drop_empty=True,
                                       fit=True, verbose=False)
print()
for k in ("rows_in", "blank_raw_dropped", "emptied_by_cleaning_dropped", "rows_out"):
    print(f"{k:<30}{counts[k]:>12,}")

# before / after examples
sample = cleaned.sample(min(15, len(cleaned)), random_state=0)[[TEXT_COL, CLEANED_COL]]
with pd.option_context("display.max_colwidth", 120):
    display(sample)

# COMMAND ----------

# most frequent words after cleaning (sanity check of the cleaning method)
cleaning.top_tokens(cleaned[CLEANED_COL], n=30).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save cleaned data

# COMMAND ----------

row_counts = {
    "raw_rows": n_raw,
    "blacklisted_dropped": n_black,
    "missing_label_dropped": n_nolabel,
    "blank_raw_dropped": counts["blank_raw_dropped"],
    "emptied_by_cleaning_dropped": counts["emptied_by_cleaning_dropped"],
    "cleaned_rows": len(cleaned),
}
row_counts["total_dropped"] = row_counts["raw_rows"] - row_counts["cleaned_rows"]
stats = cleaning.ngram_stats(cleaned[CLEANED_COL])

metadata = {
    "created": TS,
    "source": source,
    "blacklist_file": str(paths.resolve(BLACKLIST_FILE)) if BLACKLIST_FILE else None,
    "blacklist_col": BLACKLIST_COL,
    "text_col": TEXT_COL,
    "cleaned_col": CLEANED_COL,
    "level_columns": LEVEL_COLUMNS or None,
    "cleaning": cleaner.describe(),
    "row_counts": row_counts,
    "ngram_stats": stats,
}
files = data.save_dataset(OUT_DIR, TS, raw=None, cleaned=cleaned, cleaner=cleaner,
                          metadata=metadata, delimiter=OUTPUT_DELIMITER)
for k, v in files.items():
    print(f"{k:<10} {v}")
print(f"\nUse in notebooks 02/03 with:  PREPARED_DATA = \"{TS}\"")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vocabulary size

# COMMAND ----------

data.print_summary(row_counts, stats)

tracker.log_params({"source": {k: v for k, v in source.items() if k != "query_text"},
                    "cleaning_method": CLEANING_METHOD,
                    "append_fields": APPEND_FIELDS, "append_nulls_as_none": APPEND_NULLS_AS_NONE,
                    "blacklist_file": BLACKLIST_FILE, "text_col": TEXT_COL, "cleaned_col": CLEANED_COL,
                    "dataset_folder": TS})
tracker.log_metrics({**row_counts, **stats})
tracker.log_artifact(files["metadata"])
tracker.log_artifact(files["vocabulary"])
tracker.end()