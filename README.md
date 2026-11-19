# NB Category Classifier

A hierarchical multinomial naive Bayes model that predicts the four-level product category
(`Level 1 ID` … `Level 4 ID`) from item descriptions. It replaces `NB-Category-Classifier` and
`NB-Category-Classifier-Minimized`. It keeps the first repo's workflow but has a single model and
none of the threshold or cost machinery. Calibration is done differently from the old repos: it covers
every option the dropdown interface can show, after any user selection (see [Calibration](#calibration)).

## Workflow

| notebook | purpose | writes |
|---|---|---|
| `notebooks/01_pull_and_clean_data.ipynb` | pull (SQL or file) → blacklist → clean, verbosely | `data/<ts>/` |
| `notebooks/02_evaluation.ipynb` | cross-validation, or a grid search of cross-validations, with checkpoints | `outputs/evaluation_<ts>/` |
| `notebooks/03_train_to_deploy.ipynb` | train on all labelled data and save the model | `models/model_<ts>/` |
| `notebooks/04_batch_predict.ipynb` | predict uncategorized items with a saved model | `outputs/predictions_<ts>/` |
| `notebooks/05_recommend.ipynb` | the dropdown interface: top-k per level, re-scored after a selection | nothing |

Calibration is fitted at the end of notebook 02 (from out-of-fold predictions, written to
`outputs/evaluation_<ts>/calibration/`) and saved with the model by notebook 03 (`CALIBRATION = ...`).

Every notebook starts with a settings cell. All hyperparameters are written out in the notebook, so
none are hidden as defaults. Each notebook has `USE_MLFLOW = True/False` at the top.

Notebooks 02 and 03 can either use a folder written by notebook 01 (`PREPARED_DATA = "<ts>"` or
`"latest"`) or pull and clean the data themselves in one quiet cell. That cell reports only the row
counts and the token/bigram counts.

### Running on Databricks

1. Clone the repo as a Git folder. The notebooks find the package themselves (no `pip install` needed).
2. SQL queries live in `data/queries/*.sql`. `SQL_QUERY = "labeled_items"` runs
   `data/queries/labeled_items.sql` through `spark.sql`.
3. NLTK and its `words` corpus are installed on first use by `nbclassifier/bootstrap.py`.
4. Run the tests once per cluster/runtime: `%pip install pytest` then `!python -m pytest -q ../tests`.

## Folder structure

```
cleaning/
  cleaning_method_20261015/   one folder per named cleaning method:
    method.yaml               settings (list files, step parameters)
    clean_steps.py            step functions      } byte-for-byte copies of cleaning_repo
    term_lists.py             list readers        }
    cleaning_lists/           units, colors, color abbreviations, sizes, good tokens
data/
  queries/               labeled_items.sql, unlabeled_items.sql
  blacklists/            one-column lists of items to exclude
  <ts>/                  output of notebook 01
models/model_<ts>/       trained models (see below), with calibration.json when calibrated
nbclassifier/            the Python package used by the notebooks
notebooks/               01-04
outputs/
  evaluation_<ts>/       notebook 02 checkpoints, results, exported OOF predictions, plots, fold_models/,
                         calibration/candidate_<k>/ (temperatures, lookup tables, diagnostics)
  predictions_<ts>/      notebook 04 predictions + metadata
tests/                   pytest: sklearn equivalence, cleaning parity with cleaning_repo, folds, calibration
```

Package modules: `paths`, `bootstrap`, `config` (hyperparameters, grid, CV settings), `cleaning`, `data`,
`vectorizers`, `model`, `folds`, `predict`, `crossval` (CV and grid search), `evaluation` (buckets, plots),
`calibration` (temperatures and lookup tables fitted from OOF predictions), `recommend` (top-k per level for the
dropdown interface), `artifacts` (model save/load), `tracking` (MLflow). The tracking module isn't named `mlflow.py`
because that name would shadow the real `mlflow` package.

## Cleaning

There is one cleaning method, `cleaning_method_20261015`, selected with
`CLEANING_METHOD = "cleaning_method_20261015"`. It is the 13-step procedure from `cleaning_repo`
(`01_clean_data.ipynb`):

1. uppercase
2. new cleaned field
3. repair words split at the field boundary
4. remove record markers
5. remove the first word unless it is an English word or good token (with safeguards)
6. colors → `<COLOR>`
7. standardize units
8. glue units to numbers
9. sizes → `<SIZE>`, then drop stray digits
10. remove punctuation
11. remove rare words
12. collapse repeated placeholders
13. tidy spaces

The code and term lists are unchanged copies. The parameters from that notebook's settings cell live in
`method.yaml`. `tests/test_cleaning.py` checks the output against the original code on a sample of
`sample_spend_data.txt`.

To create another method, copy the folder under a new name, then edit `method.yaml`, the lists or the
code.

**Step 11 vocabulary.** Words kept by step 11 are learned from the training data when notebooks 01-03
clean it. They are then frozen into the cleaner, so notebook 04 applies the training vocabulary
(cleaning_repo's `BUILD_VOCABULARY = False` path). Steps 3 and 5 count words in whatever data is being
cleaned, exactly as in cleaning_repo.

Optional extras, set in the notebook:

- `APPEND_FIELDS = {"Global Vendno": "VEND", "Prodline": "PLINE"}` appends `VEND_<value>` tokens to the
  cleaned description. A field can also be split into several granularities:
  `{"UNSPSC": {"prefix": "UNSPSC", "levels": [2, 4, 6, 8]}}` turns `46181504` into
  `UNSPSC2_46 UNSPSC4_4618 UNSPSC6_461815 UNSPSC8_46181504`. With `APPEND_NULLS_AS_NONE = True`, nulls
  become `VEND_None`; with `False` the marker is skipped for that row.
- `BLACKLIST_FILE` / `BLACKLIST_COL`: items removed before anything else happens. The file's first column
  holds the identifiers (a header line is expected); `BLACKLIST_COL` (default `ID`) is the data column they
  are matched against.

The cleaner (`cleaner.pkl`) is saved with every dataset and every model. It holds the method's code,
lists and settings, the English word list, the learned vocabulary and the append fields. A readable copy
of the method folder and `vocabulary.txt` are saved alongside it. The step code runs from that frozen
copy, so editing `cleaning/` later never changes how an existing model cleans.

**Tokenizing.** Cleaned text is whitespace-separated words such as `<SIZE>IN` and `<COLOR>`. The
`token_pattern` hyperparameter controls how the vectorizer splits it:

- `"whitespace"` (default) keeps each cleaned word as one token.
- `"sklearn"` uses scikit-learn's default, which splits `<SIZE>IN` into `size` + `in` (and `in` is then
  removed as a stop word).

## Model

`SparseMultinomialNB` reproduces sklearn's `MultinomialNB` exactly, but keeps the class × feature counts
sparse. sklearn stores them densely, which at 3,600 classes × 200,000 features is 5.8 GB. The rewrite is

    log(fc + α) − log_norm = log α − log_norm + log1p(fc/α)

The last term is zero wherever the count is zero, so it stays sparse. Counts don't depend on α or
`fit_prior`, so the grid search fits each vectorizer once per fold and re-uses it for every α /
`fit_prior` / `prediction_mode` combination.

`prediction_mode` controls how the four levels are combined. Only category paths seen in training can be
predicted.

| mode | path score | sums to 1? |
|---|---|---|
| `independent` | product of the four levels' own probabilities | no (raw product) |
| `conditional` | chain rule: P(L1)·P(L2∣L1)·P(L3∣L2)·P(L4∣L3), i.e. one NB per parent node | yes |
| `flattened` | one NB over level-4 categories; levels 1-3 follow from the only valid path | yes |

The `conditional` mode is computed exactly from the per-level counts. A per-parent `MultinomialNB` has
the same feature counts for its children, and its prior differs only by a constant within the sibling
group (`tests/test_model.py` checks this against sklearn models fitted per parent).

Output per level `d`:

- **`Ld Confidence`**: the summed score of all valid paths sharing the predicted path's first `d`
  levels. These are raw scores; nothing is calibrated.
- **`Ld Margin`**: the confidence minus the best competing prefix at that depth. It can be negative.
- **`Ld Calibrated`** (batch predictions of a calibrated model, `CALIBRATED_COLUMNS = True`): the calibrated
  probability that the predicted prefix is correct with nothing selected by a user.

## Calibration

The interface has one dropdown per level, each with the top 5 options. When a user confirms or selects a
category at any level, the levels above it are fixed to its only valid ancestor, and the levels below list
only its descendants. Calibration has to cover every option that can appear, not just each level's top pick.

**One distribution for everything.** In `conditional` mode the path scores already form one distribution
over the valid paths (they sum to 1). Every dropdown number comes from it:

- with nothing selected, a level's options are ranked by **marginal** probability, the summed score of every
  path below the class (so the level-2 list isn't tied to the single best path);
- after a selection, the path scores are restricted to the selected subtree and renormalised, which in
  `conditional` mode is exactly P(child | selection, text) from the per-level models.

**Two rankings.** How a dropdown ranks its level's categories is a choice (`ranking=`):

| ranking | a category's score | answers |
|---|---|---|
| `"total"` (default) | sum of the probabilities of every path below it | which category is most likely correct |
| `"best_path"` | probability of its single most likely path | which categories lie on the most convincing complete paths |

`"best_path"` is the same as walking down the ranked list of paths and collecting distinct categories, so every
dropdown still gets 5 entries. Level 4 is identical under both (each Level 4 category is one path). `"total"`
favours a category whose probability is spread over many paths; `"best_path"` favours one with a single strong
path.

**Step 1: temperatures.** Naive Bayes is badly overconfident (its scores pile up at 0 and 1). One temperature
per level divides that level's log-likelihoods before the sibling softmax. Each T is fitted to minimise the
out-of-fold log-loss of the true child given the true parent. Rankings among siblings don't change, the
scores still sum to 1, and marginals and conditionals stay consistent.

**Step 2: lookup tables per scenario.** A scenario is (target level, how many top levels are fixed, and
whether the selection *confirmed* the model's own top pick or *corrected* it). There are 16 for four levels,
plus 9 for the `"best_path"` ranking (Levels 1-3; Level 4 shares the `"total"` tables). Each ranking needs its
own tables because the same number means different things under the two: on test data, temperatures alone made
best-path scores *less* calibrated, and only their own lookup tables fixed that. Confirmed / corrected is judged
against the ranking in use.
Corrections happen on items the model finds hard, so scores below a correction are less reliable and get
their own tables. For every OOF row the user is simulated selecting the true category at each depth, and
the top `CAL_TOP_K` (default 10) candidates in each scenario are recorded with whether they are correct.
This covers every rank, including options pulled into the top 5 by a filtered scope.

The tables map tempered score to observed accuracy. `CAL_SCHEME = "percent"` gives the 1%-bucket table;
`"logit"` (default) merges fine log-odds bins until each holds `CAL_MIN_BIN_COUNT` candidates, which resolves
the 0-1% and 99-100% regions. Both are made monotone (pool-adjacent-violators), interpolated between bucket
centres and clipped to [0.01%, 99.99%]. A scenario with fewer than `CAL_MIN_SCENARIO_COUNT` candidates uses
the pooled table for its target level and fixed depth (confirmed + corrected), then the nothing-fixed table.

**All from out-of-fold predictions.** `calibration.fit_from_evaluation` uses the evaluation's own folds (the
same ID-hash folds as the ensemble) and scores each fold's held-out rows with a model trained without them.
With `SAVE_FOLD_MODELS = True` (default) the cross-validation in notebook 02 saves each fold's fitted model
under `outputs/evaluation_<ts>/fold_models/` (one per vectorizer setting, ~15-20 MB each), and calibration
reuses them, adapted to the candidate with `clone_with` exactly as the evaluation did, so nothing is refitted.
Folds without a saved model (for example from a run made before this option) are refitted with the candidate's
hyperparameters, rare-path filter on the training part only; both routes give identical results.
`ev.delete_fold_models()` frees the disk space afterwards. Diagnostics are cross-fitted: the "calibrated" rows
of `metrics.csv` use tables fitted without the fold being scored.

`outputs/evaluation_<ts>/calibration/candidate_<k>/` holds `calibration.json` (what the model uses),
`histograms.npz` (so `calibration.rebuild_from_histograms` can change the table settings in seconds),
`temperature_fit.csv`, `lookup_tables.csv` (every table in 1% buckets), `metrics.csv` (ECE, Brier,
log-loss: raw vs tempered vs calibrated), `reliability.csv` + plot, and `rank_accuracy.csv` (observed
accuracy of the 1st, 2nd, ... option in each scenario).

Calibrate the hyperparameters you deploy, `prediction_mode` included. Notebook 03 refuses a calibration
fitted for different hyperparameters. The temperatures and tables are then applied to the model trained on
all rows, which is slightly stronger than the fold models. That small, conservative bias is the usual
trade-off of OOF calibration.

**Using it.** `nbclassifier.recommend.Recommender` is the interface entry point (see notebook 05):

```python
rec = Recommender.from_loaded(artifacts.load_model(), k=5)              # ranking="best_path" for the other mode
rec.recommend_one(raw_description)                                      # four lists of 5
rec.recommend_one(raw_description, selections={"Level 2 ID": "L2-..."}) # after a selection
rec.recommend_one(raw_description, ranking="best_path")                 # per-call override
```

`selections` takes any subset of levels on one valid path; the deepest decides the scope. Confirmed vs
corrected is detected (the selected node was or wasn't the model's top pick at that level), or can be forced
with `selection_kind=`. `rec.recommend(...)` returns the same for many rows as a long table.

## Output format (ensemble input)

Out-of-fold predictions (`outputs/evaluation_<ts>/best/oof_predictions.csv`) and batch predictions
(`outputs/predictions_<ts>/predictions_<ts>.csv`) share one comma-separated layout:

```
ID, Vallen ID, [fold], [Actual Level 1-4 ID],
Predicted Level 1-4 ID, [Predicted Level 1-4 Name],
L1-L4 Confidence, L1-L4 Margin, n_tokens, n_known, oov_rate, FullDesc, CleanDesc
```

The OOF file has a JSON sidecar (`oof_predictions.json`) with `n_folds`, `fold_seed`, `fold_key_col` and
`fold_method`. These are the names the Ensemble repo already reads. Folds use the same ID-hash method as
before (`sha1(f"{seed}|{ID}") % n_folds`, defaults 5 folds, seed 0, key `ID`), so they line up with the
neural network's folds.

## Evaluation, checkpoints and resuming

- `GRID_SEARCH = False`: every hyperparameter must be a scalar or a one-element list, and one
  cross-validation is run.
- `GRID_SEARCH = True`: lists are expanded into every combination. Combinations that would give
  identical models are dropped (e.g. `norm` with a count vectorizer).
- `CV_TYPE = "oof"` (default, k-fold out-of-fold) or `"repeated_holdout"`.
- `SAMPLE_FRAC = 0.2` (optional) evaluates on about 20% of items, chosen by a hash of `FOLD_KEY_COL`, so
  the NB and NN repos sample the same items. Use it to explore a grid quickly, then confirm on all rows.

Folds are assigned on the full dataset, so every candidate is evaluated on the same rows. The
rare-path threshold is applied to each fold's **training** part only, as it is in deployment. Every
held-out row is therefore predicted: the OOF file covers all rows, and paths unseen in training count
as errors.

Each (candidate, fold) prediction file is written to disk as soon as it is done. The dataset itself is
also copied into the evaluation folder. If the session dies, set `RESUME_FROM = "evaluation_<ts>"` and
re-run: finished work is skipped, and nothing is re-pulled. New grid values added to the notebook are
appended to the run.

Bucket tables, the accuracy-and-item-count plots (with a dashed green line at the coverage where
cumulative accuracy reaches `TARGET_ACCURACY`, default 95%) and violin plots (spread across folds) are all
computed from the saved predictions. `BUCKET_WIDTH` can therefore be changed and the cells re-run
without re-fitting.

## Saved model folder

`vectorizer.pkl`, `model_Level_<d>_ID.pkl` (one per level), `path_index.pkl`, `cleaner.pkl`,
`cleaning_method_20261015/` (copy), `vocabulary.txt`, `hierarchy_lookup.csv`, `hyperparameters.json`, `cleaning.json`,
`training_data.json` (source, query, blacklist, row counts), `calibration.json` (optional, see
[Calibration](#calibration); `artifacts.attach_calibration` adds one to an existing folder) and
`manifest.json` (fingerprint, sizes, library versions, calibration summary). Notebook 03 reloads the saved model and checks that it predicts identically.

## MLflow

With `USE_MLFLOW = True` the notebooks log the following:

| notebook | logs |
|---|---|
| 01 | source, cleaning method, row/n-gram counts, metadata, vocabulary |
| 02 | CV settings; one nested run per candidate (parameters, mean accuracy by level, per-fold accuracy as steps); results table; plots |
| 03 | hyperparameters, data summary, the whole model folder |
| 04 | model used, row counts, and accuracy when the data has labels |

`MLFLOW_EXPERIMENT = None` uses each notebook's own experiment. Set a workspace path to collect all
runs in one place.

## What was removed from the old repos

Thresholds and cascades, cost models, trade-off sweeps, scoring simulation, the old top-pick calibration
and confidence references, drift checks, refusal reasons, `per_level` mode, and `abbrevs_to_expand.csv`. The earlier
`aggressive`/`light` rule sets and the find/replace normalization file were replaced by
`cleaning_method_20261015`.
