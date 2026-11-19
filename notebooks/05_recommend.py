# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Recommend (dropdown interface)
# MAGIC
# MAGIC What the four-dropdown interface calls. For one description it returns the top-k options per level
# MAGIC with calibrated probabilities, and re-scores after a user confirms or selects a category at any level:
# MAGIC
# MAGIC * levels **above** the selection are fixed to its only valid ancestor (probability 1);
# MAGIC * levels **below** list the top-k descendants of the selection, scored given the selection.
# MAGIC
# MAGIC Probabilities are calibrated with the model's `calibration.json` (notebook 02 fits it from out-of-fold
# MAGIC predictions, notebook 03 saves it with the model). Whether a selection *confirmed* the model's own top
# MAGIC pick or *corrected* it is detected automatically, and each case has its own lookup table.
# MAGIC
# MAGIC Two rankings are available (`RANKING`):
# MAGIC
# MAGIC * `"total"` (default): each category's summed probability over every path below it, i.e. the
# MAGIC   categories most likely to be correct;
# MAGIC * `"best_path"`: each category's single most likely path, i.e. the categories on the most convincing
# MAGIC   complete paths (the same as reading distinct categories off the ranked list of paths).
# MAGIC
# MAGIC Level 4 is identical under both. Each ranking has its own calibration tables.
# MAGIC
# MAGIC In an application:
# MAGIC
# MAGIC ```python
# MAGIC from nbclassifier import artifacts
# MAGIC from nbclassifier.recommend import Recommender
# MAGIC
# MAGIC rec = Recommender.from_loaded(artifacts.load_model("model_<ts>"), k=5)   # ranking="best_path" for the other mode
# MAGIC rec.recommend_one(raw_description)                                    # nothing selected
# MAGIC rec.recommend_one(raw_description, selections={"Level 2 ID": "L2-..."})  # after a selection
# MAGIC ```

# COMMAND ----------

# ---- settings ------------------------------------------------------------------
MODEL = None                          # None = newest folder in models/
K = 5                                 # options per dropdown
RANKING = "total"                     # "total" (summed over all paths) | "best_path" (best single path)
DESCRIPTION = "GLOVE NITRILE DISPOSABLE POWDER FREE BLUE LARGE 100/BX"   # a raw description

# COMMAND ----------

import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from nbclassifier import artifacts
from nbclassifier.recommend import Recommender

model = artifacts.load_model(MODEL)
rec = Recommender.from_loaded(model, k=K, ranking=RANKING)
print(f"model: {model.folder.name}")
print(f"calibration: {model.calibrator if model.calibrator is not None else 'none: probabilities are not calibrated'}")


def show(result):
    print("cleaned:", result["cleaned_text"], "| ranking:", result["ranking"])
    for lv in result["levels"]:
        print(f"\n{lv['level']}  [{lv['scenario']}]")
        for o in lv["options"]:
            p = "   n/a " if o["probability"] is None else f"{o['probability']:7.2%}"
            print(f"  {o['rank']}. {p}  {o['label']:<14} {o['name'] or ''}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nothing selected

# COMMAND ----------

first = rec.recommend_one(DESCRIPTION)
show(first)

# COMMAND ----------

# MAGIC %md
# MAGIC ## After a selection
# MAGIC
# MAGIC Pick any option: here the second level-2 option, as if the user corrected the model. Selecting the
# MAGIC first option instead is a *confirmation*. Any level may be selected; the deepest selection decides.

# COMMAND ----------

chosen = first["levels"][1]["options"][1]["label"]
show(rec.recommend_one(DESCRIPTION, selections={"Level 2 ID": chosen}))

# COMMAND ----------

# Several descriptions at once, as a long table (one row per item, level and rank)
rec.recommend(raw_texts=[DESCRIPTION, "SAW BLADE CIRCULAR 7-1/4IN 24T"]).head(40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The two rankings side by side
# MAGIC
# MAGIC Same description, both rankings. They can differ at levels 1-3 (a category whose probability is spread
# MAGIC over many paths ranks higher under `"total"`); level 4 is always the same.

# COMMAND ----------

for r in ["total", "best_path"]:
    print(f"\n========== {r} ==========")
    show(rec.recommend_one(DESCRIPTION, ranking=r))