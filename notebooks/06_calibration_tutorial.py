# Databricks notebook source
# MAGIC %md
# MAGIC # Calibration, step by step
# MAGIC
# MAGIC This notebook walks through how the classifier turns raw naive Bayes scores into the **calibrated probabilities**
# MAGIC shown in the four dropdowns. It follows the 21-step list, one step (or a small group of steps) per section, and
# MAGIC each section has:
# MAGIC
# MAGIC * a short explanation of what the step does and why;
# MAGIC * the formula, where one helps, with **every symbol explained** in a list underneath (naive Bayes and
# MAGIC   out-of-fold cross-validation are assumed familiar and are not explained);
# MAGIC * **commented code** that calls the real functions in `nbclassifier`, with comments that tie each line back to
# MAGIC   the formula;
# MAGIC * an **interactive chart**: use the sliders, menus and buttons, hover for details, and click legend entries to
# MAGIC   hide or show series.
# MAGIC
# MAGIC **Everything runs on simulated data**: a synthetic four-level catalogue generated below, so the notebook runs in
# MAGIC seconds anywhere the repository is available. The *method* is the production code; only the numbers are
# MAGIC synthetic. A few thresholds are scaled down to suit the smaller dataset; each one is noted where it is set.
# MAGIC
# MAGIC | Part | Steps | Topic |
# MAGIC |---|---|---|
# MAGIC | A | 1–3 | A fair test: out-of-fold scores, and how overconfident they are |
# MAGIC | B | 4–5 | Step 1 of calibration: one temperature per level |
# MAGIC | C | 6–10 | Replaying the interface: locks, rankings, recorded options, situations |
# MAGIC | D | 11–16 | Step 2 of calibration: lookup tables |
# MAGIC | E | 17–18 | Checking the result |
# MAGIC | F | 19–21 | Using it |
# MAGIC
# MAGIC **Reading the code.** A few Python conventions recur:
# MAGIC
# MAGIC * `np.something(...)` is a NumPy function. NumPy works on whole arrays at once, so `np.exp(a)` computes
# MAGIC   \\(e^{a}\\) for *every* element of `a` with no loop.
# MAGIC * `a[i]` picks element `i`, counting from **0**, so level 1 is index 0 and level 4 is index 3. `a[:, 3]` means
# MAGIC   "every row, column 3".
# MAGIC * `a[mask]`, with `mask` an array of `True`/`False`, keeps only the `True` positions.
# MAGIC * `def name(...):` defines a reusable function; `for x in xs:` repeats the indented lines for each `x`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The 21 steps in detail
# MAGIC
# MAGIC **A. A fair test** (Part A)
# MAGIC
# MAGIC 1. **Split the labelled data into folds.** The already-categorised products are divided into groups (3 by
# MAGIC    default) by item ID, the same split used by the rest of the evaluation and by the ensemble.
# MAGIC 2. **Train one model per fold.** Each fold gets a model trained on the *other* folds only, so it never sees the
# MAGIC    products it will score. The cross-validation saves these models, so calibration reuses them instead of
# MAGIC    retraining.
# MAGIC 3. **Score every held-out product.** Each product is scored by the model that didn't see it, giving one honest
# MAGIC    prediction per product: the raw material for everything that follows. Comparing these scores with the truth
# MAGIC    shows how overconfident the raw model is.
# MAGIC
# MAGIC **B. Step 1 of calibration: temperature** (Part B)
# MAGIC
# MAGIC 4. **Collect each level's evidence.** For a sample of up to 200,000 held-out products, record the scores each
# MAGIC    level's model gave the *correct* category and its siblings (the categories under the same parent).
# MAGIC 5. **Fit one temperature per level.** Try different amounts of softening (or sharpening) and keep the one that
# MAGIC    minimises the average penalty \\(-\\log(\\text{probability given to the correct category})\\) across all those
# MAGIC    products. Confident mistakes are penalised heavily, so the fit settles where softening stops paying off.
# MAGIC
# MAGIC **C. Replaying the interface** (Part C)
# MAGIC
# MAGIC 6. **Rescore everything with the tempered model.** Every complete path gets a new, softened probability; the
# MAGIC    order of options rarely changes, the confidence does.
# MAGIC 7. **Replay every screen state.** For each held-out product, simulate the user locking the correct category at
# MAGIC    Level 1, then Levels 1–2, then Levels 1–3, as well as nothing locked. Each lock keeps only the paths in the
# MAGIC    locked branch and rescales them to sum to 1.
# MAGIC 8. **Label each lock confirmed or corrected.** *Confirmed* if the locked category was the model's own top pick at
# MAGIC    that level (a checkmark click), *corrected* if the user had to override it. Corrections signal hard items.
# MAGIC 9. **Record the top 10 options in every dropdown, under both rankings.** For each simulated screen state and each
# MAGIC    dropdown below the lock, store the 10 best options' scores and whether each was correct, ranking either by
# MAGIC    *total* (summed over a category's paths) or *best path* (its single best path). Ten rather than five covers
# MAGIC    options that move into view after a lock.
# MAGIC 10. **Sort the records into situations.** A situation is one combination of ranking, dropdown level, number of
# MAGIC     locked levels, and confirmed/corrected: 16 for the total ranking and 9 for best path.
# MAGIC
# MAGIC **D. Step 2 of calibration: lookup tables** (Part D)
# MAGIC
# MAGIC 11. **Sort each situation's scores into fine buckets** on the log-odds scale, which gives extra detail near 0%
# MAGIC     and 100%, where most scores sit.
# MAGIC 12. **Merge neighbouring buckets** until each holds enough options (200 by default), so every bucket's accuracy
# MAGIC     is based on enough data.
# MAGIC 13. **Measure each bucket's real accuracy**: the fraction of its options that were actually correct.
# MAGIC 14. **Enforce "higher score, higher probability".** Where chance makes a higher bucket look less accurate than a
# MAGIC     lower one, neighbouring buckets are averaged until the sequence never decreases (pool-adjacent-violators).
# MAGIC 15. **Clip the extremes and interpolate.** No probability shown is below 0.01% or above 99.99%. A score between
# MAGIC     two buckets is read off by straight-line interpolation on the log-odds scale.
# MAGIC 16. **Set up fallbacks for thin situations.** A situation with too few recorded options (1,000 by default) borrows
# MAGIC     a broader table: confirmed and corrected pooled, then the same level with nothing locked, and finally the
# MAGIC     tempered score itself.
# MAGIC
# MAGIC **E. Checking the result** (Part E)
# MAGIC
# MAGIC 17. **Test on data the tables weren't built from.** For each fold, the tables are rebuilt from the other folds only
# MAGIC     and checked on that fold (cross-fitting). Raw, tempered and calibrated scores are compared.
# MAGIC 18. **Review the diagnostics**: calibration error per situation, reliability charts (stated probability against
# MAGIC     actual accuracy), and how often the 1st, 2nd, 3rd… option in each situation turns out to be right.
# MAGIC
# MAGIC **F. Using it** (Part F)
# MAGIC
# MAGIC 19. **Save the calibration with the model.** Notebook 03 checks it was fitted for the same model settings and
# MAGIC     stores `calibration.json` in the model folder.
# MAGIC 20. **Apply it to each new product.** The tool tempers the model's scores, works out which situation the screen
# MAGIC     is in, ranks the options, and converts each option's score through that situation's lookup table into the
# MAGIC     percentage shown in the dropdown.
# MAGIC 21. **Refit when things change.** Rerun the calibration whenever the model settings, the training data or the
# MAGIC     hierarchy change substantially.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Imports and display settings. Nothing mathematical happens in this cell.
# ----------------------------------------------------------------------------------------------
import os, sys, tempfile, warnings
from pathlib import Path

# Make the repository's `nbclassifier` package importable (this notebook lives in its `notebooks/` folder).
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                          # arrays and element-wise maths
import pandas as pd                         # tables
import plotly.graph_objects as go           # interactive charts
import plotly.io as pio
from plotly.subplots import make_subplots   # several charts in one figure

from nbclassifier import calibration, crossval          # the production calibration code
from nbclassifier.config import CVSettings, HyperParams
from nbclassifier.data import PreparedData
from nbclassifier.model import HierarchicalNB, _log_softmax
from nbclassifier.recommend import Recommender

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:      # Databricks renders Plotly natively
    pio.renderers.default = "plotly_mimetype+notebook_connected"
pio.templates.default = "plotly_white"
pd.set_option("display.width", 160)
warnings.filterwarnings("ignore", category=RuntimeWarning)   # e.g. log(0) warnings in plots

# Colours used consistently: blue = model/tempered, orange = mistakes or observed data,
# green = the ideal (a perfectly calibrated line), grey = raw or background.
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#9aa5b1", "#6250d6"

WORK = Path(tempfile.mkdtemp(prefix="calibration_tutorial_"))   # every output goes here, nothing in the repo
print("working folder:", WORK)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A simulated catalogue
# MAGIC
# MAGIC Four departments, each split into 3 × 3 × 3 sub-categories (108 complete paths). Each category has its own small
# MAGIC vocabulary. A description draws 0–2 genuine words per level, sometimes a *misleading* word from a random other
# MAGIC category, and a few noise words. That gives enough ambiguity for naive Bayes to be overconfident, as it is on real
# MAGIC descriptions.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Generate a synthetic product catalogue with a known 4-level category for every item.
# ----------------------------------------------------------------------------------------------
LEVELS = ["Level 1 ID", "Level 2 ID", "Level 3 ID", "Level 4 ID"]     # columns holding each level's category
NAMES = ["Level 1 Name", "Level 2 Name", "Level 3 Name", "Level 4 Name"]
DEPARTMENTS = ["Safety", "Tools", "Cleaning", "Electrical"]


def make_catalogue(n_items=9000, branching=(4, 3, 3, 3), seed=0):
    rng = np.random.default_rng(seed)        # random-number generator; a fixed seed makes results repeatable

    # All complete paths, as tuples of child positions: (0,), then (0,0), ... up to (3,2,2,2).
    paths = [()]
    for b in branching:
        paths = [p + (i,) for p in paths for i in range(b)]

    # Readable names: (2, 0, 2) -> "Cleaning-1-3".
    label = lambda pre: DEPARTMENTS[pre[0]] + "".join(f"-{i + 1}" for i in pre[1:])
    # Each category (at any level) gets 6 vocabulary words of its own, e.g. "cleaning_1_3_w4".
    vocab = {}
    words = lambda pre: vocab.setdefault(pre, [f"{label(pre).lower().replace('-', '_')}_w{j}" for j in range(6)])

    # Uneven popularity: a Dirichlet draw gives a probability for each path, some much larger than others.
    popularity = rng.dirichlet(np.full(len(paths), 0.7))

    rows = []
    for i, k in enumerate(rng.choice(len(paths), n_items, p=popularity)):   # pick each item's true path
        p, toks = paths[k], []
        for d in range(4):                                                     # for each level d = 0..3
            toks += list(rng.choice(words(p[:d + 1]), rng.integers(0, 3)))    # 0-2 genuine words
            if rng.random() < 0.35:                                            # 35% chance of a misleading word
                toks += list(rng.choice(words(paths[rng.integers(len(paths))][:d + 1]), 1))
        toks += [f"noise{rng.integers(300)}" for _ in range(rng.integers(1, 4))]   # 1-3 noise words
        rng.shuffle(toks)
        ids = [label(p[:d + 1]) for d in range(4)]                             # true category at each level
        rows.append({"ID": f"item{i:05d}", "Desc": " ".join(toks), **dict(zip(LEVELS, ids)), **dict(zip(NAMES, ids))})
    return pd.DataFrame(rows)


catalogue = make_catalogue()
print(f"{len(catalogue):,} items, {catalogue[LEVELS[-1]].nunique()} level-4 categories in use")
catalogue[["ID", "Desc"] + LEVELS].head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part A — A fair test (steps 1–3)
# MAGIC
# MAGIC **Steps 1–3.** The labelled catalogue is split into folds (3 here, as in notebook 02). For each fold, a model is
# MAGIC trained on the other folds and scores the held-out fold. `save_fold_models=True` keeps each fold's fitted model so
# MAGIC the calibration can reuse it instead of refitting.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Out-of-fold cross-validation with the production code (steps 1-3).
# ----------------------------------------------------------------------------------------------
# Model settings, as in production: binary features, no IDF, smoothing alpha = 0.001, and the
# "conditional" mode, where each level's model chooses among the children of the parent.
hp = HyperParams(binary=True, use_idf=False, alpha=0.001, min_df=2, rare_path_threshold=5,
                 prediction_mode="conditional")

# Tell the evaluation which columns hold the text, the four levels and the IDs.
cols = crossval.Columns(text_col="Desc", cleaned_col="Desc", level_columns=LEVELS, name_columns=NAMES, id_cols=["ID"])

# 3 folds: each item is held out exactly once and scored by a model trained on the other two folds.
ev = crossval.Evaluation.create(PreparedData(catalogue, cleaner=None, metadata={}), [hp], CVSettings(n_folds=3),
                                cols, grid_search=False, folder=WORK / "evaluation")
results = ev.run(save_fold_models=True)          # also saves each fold's fitted model for reuse later

print("\nsaved fold models:", sorted(p.name for p in (ev.folder / "fold_models").rglob("*.joblib")))
results.filter(like="Accuracy").round(3)         # top-pick accuracy per level, averaged over folds

# COMMAND ----------

# MAGIC %md
# MAGIC For the illustrations we take **fold 0**: its saved model `clf0` and the rows it never saw, `held`.
# MAGIC
# MAGIC * `jlls[d]` holds the raw naive Bayes score \\(s_d(c;x)\\) of every level-\\(d\\) category for every held-out row.
# MAGIC * `P_raw` holds the probability \\(P(p\\mid x)\\) of every complete path \\(p\\) for every row, combined from the four
# MAGIC   levels.
# MAGIC * `codes` gives each row's **true** category at every level, in the model's own numbering (-1 if the category was
# MAGIC   unseen in training).

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Load fold 0's model and held-out rows, and compute the raw scores used in later sections.
# ----------------------------------------------------------------------------------------------
fold, test_mask = ev._splits(ev.data)[0]             # test_mask is True for rows held out in fold 0
clf0, _ = ev.load_fold_model(hp, fold)               # the model trained WITHOUT those rows
held = ev.data.loc[test_mask].reset_index(drop=True)

X = clf0.transform(held["Desc"])                     # descriptions -> feature vectors x
codes = clf0.encode_paths(held[LEVELS])              # rows x 4 array of true category numbers

# s_d(c; x): one matrix per level, rows = descriptions, columns = that level's categories.
jlls = clf0.level_jll(X)

# P(p | x) for every path p. `joint_log_from_jll` multiplies the four levels' sibling
# probabilities along each path (in log space, as a sum). `_log_softmax` makes each row sum to 1
# (in log space), and `np.exp` turns log-probabilities back into probabilities.
P_raw = np.exp(_log_softmax(clf0.joint_log_from_jll(jlls)))

name_of = [clf0.classes_[d]["label"].to_numpy() for d in range(4)]   # category number -> name, per level
print(f"{len(held):,} held-out rows, {P_raw.shape[1]} paths known to this fold's model")

# COMMAND ----------

# MAGIC %md
# MAGIC **What we are fixing.** A score is *calibrated* when it means what it says:
# MAGIC
# MAGIC $$\Pr(Y = 1 \mid q) \enspace =\enspace  q$$
# MAGIC
# MAGIC * \\(q\\): the score shown next to an option (between 0 and 1).
# MAGIC * \\(Y\\): 1 if that option is the correct category, 0 if not.
# MAGIC * \\(\\Pr(Y=1 \\mid q)\\): among all options shown with score \\(q\\), the fraction that were actually correct.
# MAGIC
# MAGIC The chart shows the held-out **level-4 top picks**. On the left, the scores bunch up near 100%. On the right, rows
# MAGIC are grouped by score and the actual accuracy of each group is plotted: points on the dashed diagonal would
# MAGIC satisfy the equation, and points **below** it are overconfident (the score claims more than is achieved).

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Measure the raw model's calibration for its level-4 top pick (a reliability diagram).
# ----------------------------------------------------------------------------------------------
# Level-4 score of each category = the probability of its path (level 4 has one path per category).
m4 = calibration.level_marginals(clf0, P_raw, 3)     # rows x level-4 categories
top = m4.argmax(1)                                   # the top pick per row (column with the largest score)
score = m4.max(1)                                    # q: the top pick's score
right = top == codes[:, 3]                           # Y: True where the top pick equals the true category

# Group the rows into 10 equal-sized groups by score (deciles of q). Within each group,
# the mean score is the "stated" value and the mean of Y is the "actual" accuracy, i.e. an
# estimate of Pr(Y = 1 | q) for scores around that value.
bins = np.quantile(score, np.linspace(0, 1, 11))                       # 11 edges -> 10 groups
idx = np.clip(np.searchsorted(bins, score, side="right") - 1, 0, 9)    # which group each row falls in
rel = (pd.DataFrame({"q": score, "y": right, "b": idx})
         .groupby("b").agg(q=("q", "mean"), acc=("y", "mean"), n=("y", "size")))

fig = make_subplots(1, 2, subplot_titles=("Raw top-pick scores (level 4)", "Stated vs actual"), horizontal_spacing=0.12)
# Left: how the scores are distributed.
fig.add_histogram(x=score, nbinsx=40, marker_color=GREY, name="rows", row=1, col=1)
# Right: the diagonal is the calibration equation Pr(Y=1|q) = q; the orange points are what was observed.
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREEN, dash="dash"), name="calibrated", row=1, col=2)
fig.add_scatter(x=rel.q, y=rel.acc, mode="lines+markers", marker=dict(size=8, color=ORANGE), name="raw model",
                customdata=rel.n, hovertemplate="stated %{x:.1%}<br>actual %{y:.1%}<br>%{customdata} rows", row=1, col=2)
fig.update_xaxes(tickformat=".0%", row=1, col=1); fig.update_xaxes(tickformat=".0%", range=[0, 1], row=1, col=2)
fig.update_yaxes(tickformat=".0%", range=[0, 1.02], row=1, col=2)
fig.update_layout(height=360, legend=dict(orientation="h", y=-0.2))
fig.show()

# Averaged over all rows: the mean stated score vs the fraction actually right.
print(f"average stated top-pick score {score.mean():.1%}, actually right {right.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part B — Step 1 of calibration: temperature (steps 4–5)
# MAGIC
# MAGIC **Step 4.** For each held-out row and each level, we take the scores the model gave the **true** category and its
# MAGIC **siblings** (the categories under the same parent), because that sibling group is what the level's model compares.
# MAGIC
# MAGIC **Step 5.** A temperature \\(T_d\\) per level divides those scores before they are turned into probabilities:
# MAGIC
# MAGIC $$P^{T}_d(c \mid \text{parent}, x) \enspace =\enspace  \frac{\exp\big(s_d(c;x)/T_d\big)}{\sum_{c' \in \text{siblings}(c)} \exp\big(s_d(c';x)/T_d\big)}$$
# MAGIC
# MAGIC * \\(c\\): a category at level \\(d\\); \\(\\text{siblings}(c)\\): the categories with the same parent, including \\(c\\) itself.
# MAGIC * \\(x\\): the product description.
# MAGIC * \\(s_d(c;x)\\): the raw naive Bayes score of \\(c\\) (its joint log-likelihood). Larger means more likely.
# MAGIC * \\(T_d\\): the temperature of level \\(d\\). \\(T_d = 1\\) is the raw model, \\(T_d > 1\\) softens it and \\(T_d < 1\\) sharpens it.
# MAGIC * \\(\\exp(\\cdot)\\) and the sum in the denominator: the *softmax*, which turns scores into probabilities that sum to 1
# MAGIC   across the siblings. Dividing by \\(T_d\\) shrinks (or stretches) the gaps between scores, but never changes their
# MAGIC   order.
# MAGIC * \\(P^{T}_d(c \\mid \\text{parent}, x)\\): the tempered probability of \\(c\\), given its parent.
# MAGIC
# MAGIC \\(T_d\\) is chosen to minimise the average **penalty** over many held-out rows:
# MAGIC
# MAGIC $$T_d^{\ast} \enspace =\enspace  \arg\min_{T>0}\enspace  \frac{1}{N}\sum_{i=1}^{N} -\log P^{T}_d\big(c^{\ast}_i \mid \text{parent}, x_i\big)$$
# MAGIC
# MAGIC * \\(i\\): one held-out row; \\(N\\): the number of rows used.
# MAGIC * \\(c^{\\ast}_i\\): row \\(i\\)'s **true** category at level \\(d\\); \\(x_i\\): its description.
# MAGIC * \\(-\\log P^{T}_d(\\cdot)\\): the penalty for the probability the true category received. It is near 0 when that
# MAGIC   probability is near 1 and grows without limit as it approaches 0.
# MAGIC * \\(\\arg\\min_{T>0}\\): the value of \\(T\\) that makes the average penalty smallest; \\(T_d^{\\ast}\\) is that value.
# MAGIC
# MAGIC A confident mistake (the true category got about 0%) costs a lot, while a correct pick at 99% costs almost nothing.
# MAGIC Softening trades a small cost on correct picks for a large saving on confident mistakes, and the minimum is where
# MAGIC the two balance.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Step 4: collect each row's true sibling group. Step 5: fit T_d per level (on fold 0 only, to illustrate).
# ----------------------------------------------------------------------------------------------
def sibling_groups(d):
    "Raw scores s_d of each row's true sibling group at level d (the data step 4 collects)."
    ok = codes[:, d] >= 0                                  # skip rows whose true category the model never saw
    starts, sizes, group_of = calibration._softmax_groups(clf0, d)   # where each sibling group sits
    # vals: all the scores, one group after another, as one long array.
    # sz: how many siblings in each row's group. true_pos: position of the true category in its group.
    vals, sz, true_pos = calibration._gather_groups(jlls[d][ok], codes[ok, d], starts, sizes, group_of, 256)
    keep = sz > 1      # a category with no siblings is always predicted with probability 1: no information about T
    return vals[np.repeat(keep, sz)], sz[keep], true_pos[keep]


fold0_temps = {}
for d in range(4):
    # fit_temperature searches over T for the minimum of the average penalty (the second formula above)
    # and returns: T*, the average penalty at T = 1 (raw), and the average penalty at T*.
    T, nll_raw, nll_T = calibration.fit_temperature(*sibling_groups(d))
    fold0_temps[f"L{d + 1}"] = dict(temperature=T, penalty_raw=nll_raw, penalty_tempered=nll_T)
pd.DataFrame(fold0_temps).T.round(3)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interactive: drag the temperature slider (level 4).**
# MAGIC
# MAGIC * *Top left:* one row's sibling group. The model's top pick was wrong here; watch probability flow toward the true
# MAGIC   category (green) as \\(T\\) grows.
# MAGIC * *Top right:* the average penalty against \\(T\\), which is the quantity minimised in the second formula. The dashed
# MAGIC   line marks its lowest point.
# MAGIC * *Bottom:* every held-out row, placed at the probability its *true* category received (\\(x\\)-axis) and the penalty
# MAGIC   \\(-\\log\\) of that probability (\\(y\\)-axis), so every dot lies on the curve \\(y = -\\log x\\). Correct top picks (blue)
# MAGIC   drift slightly left, which raises their small penalties a little. Confident mistakes (orange triangles) climb off
# MAGIC   the steep part of the curve, which lowers their large penalties a lot.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Interactive temperature explorer for level 4.
# ----------------------------------------------------------------------------------------------
vals, sizes, true_pos = sibling_groups(3)              # level 4 is index 3
starts = np.cumsum(sizes) - sizes                      # where each row's group begins inside `vals`


def true_prob(T):
    # Tempered softmax probability of the TRUE category for every row:
    #   P = exp(s_true / T) / sum over siblings of exp(s_c' / T)
    # computed in a numerically safe way ("log-sum-exp"): subtract each group's maximum first, so exp()
    # never overflows. The subtraction cancels between numerator and denominator.
    z = vals / T                                                         # s / T for every score
    m = np.maximum.reduceat(z, starts)                                   # the largest s/T in each group
    lse = m + np.log(np.add.reduceat(np.exp(z - np.repeat(m, sizes)), starts))   # log of the denominator
    return np.exp(z[starts + true_pos] - lse)                           # exp(log numerator - log denominator)


group_max = np.maximum.reduceat(vals, starts)
top_right = vals[starts + true_pos] == group_max       # True where the model's top pick was the true category

# An illustrative row: a group of 3 or more siblings where the true category got under 5% at T = 1.
cand = np.flatnonzero(~top_right & (sizes >= 3))
ex = cand[np.argmax(true_prob(1.0)[cand] < 0.05)]
ex_vals, ex_true = vals[starts[ex]:starts[ex] + sizes[ex]], true_pos[ex]

# The average penalty (1/N) * sum of -log P(true) at many temperatures: the curve in the top-right panel.
Ts_fine = np.linspace(1, 6, 101)
curve = [(-np.log(np.maximum(true_prob(T), 1e-12))).mean() for T in Ts_fine]   # max(.,1e-12) avoids log(0)
best_T = Ts_fine[int(np.argmin(curve))]                                         # the arg-min: T*

Ts = np.round(np.arange(1.0, 6.01, 0.25), 2)          # slider positions
rng = np.random.default_rng(0)
show = rng.choice(len(sizes), min(700, len(sizes)), replace=False)   # plot 700 rows to keep the chart light


def traces(T):
    "Everything that changes with T: the example's bars, the marker on the curve, and the two dot clouds."
    e = np.exp((ex_vals - ex_vals.max()) / T); e /= e.sum()            # tempered softmax of the example group
    q = np.maximum(true_prob(T)[show], 5e-4)                            # probability given to the true category
    r = top_right[show]
    nll = (-np.log(np.maximum(true_prob(T), 1e-12))).mean()             # current average penalty
    return [go.Bar(x=[f"c{j + 1}" for j in range(len(e))], y=e,
                   marker_color=[GREEN if j == ex_true else GREY for j in range(len(e))],
                   marker_line=dict(color=["black" if j == ex_true else GREY for j in range(len(e))], width=2)),
            go.Scatter(x=[T], y=[nll], mode="markers", marker=dict(size=12, color=ORANGE)),
            # dots at (q, -log q): every dot lies on the penalty curve
            go.Scatter(x=q[r], y=-np.log(q[r]), mode="markers", marker=dict(size=5, color=BLUE, opacity=0.5)),
            go.Scatter(x=q[~r], y=-np.log(q[~r]), mode="markers", marker=dict(size=7, color=ORANGE, symbol="triangle-up"))]


fig = make_subplots(2, 2, specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.16, row_heights=[0.45, 0.55],
                    subplot_titles=("One sibling group (true category in green)", "Average penalty vs temperature",
                                    "Every row: probability given to its true category"))
for tr, (r, c) in zip(traces(1.0), [(1, 1), (1, 2), (2, 1), (2, 1)]):
    fig.add_trace(tr, row=r, col=c)
qq = np.linspace(5e-4, 1, 300)
fig.add_scatter(x=qq, y=-np.log(qq), mode="lines", line=dict(color=GREY, width=1), row=2, col=1)   # y = -log x
fig.add_scatter(x=Ts_fine, y=curve, mode="lines", line=dict(color="black", width=2), row=1, col=2)
# One animation frame per slider position; each frame replaces the four T-dependent traces.
fig.frames = [go.Frame(data=traces(T), traces=[0, 1, 2, 3], name=str(T)) for T in Ts]
fig.update_layout(
    height=640, showlegend=False,
    sliders=[dict(active=0, currentvalue=dict(prefix="temperature T = "), pad=dict(t=30),
                  steps=[dict(label=str(T), method="animate",
                              args=[[str(T)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                         for T in Ts])])
fig.update_yaxes(range=[0, 1.05], tickformat=".0%", row=1, col=1)
fig.update_xaxes(title_text="temperature T", row=1, col=2)
fig.update_xaxes(title_text="probability given to the true category", tickformat=".0%", range=[0, 1.02], row=2, col=1)
fig.update_yaxes(title_text="penalty  -log(probability)", range=[0, 8], row=2, col=1)
fig.add_vline(x=best_T, line=dict(color=GREEN, dash="dash"), row=1, col=2)
fig.show()
print(f"best level-4 temperature on fold 0: T* = {best_T:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC In production the same fit runs on up to `CAL_TEMPERATURE_MAX_ROWS` (200,000) out-of-fold rows from *all* folds,
# MAGIC inside `calibration.fit_from_evaluation`, which also performs every remaining step. It runs once here, and the
# MAGIC following sections take its pieces apart.
# MAGIC
# MAGIC The bucket thresholds are scaled down for this small dataset: at least 100 options per bucket (production default
# MAGIC 200), and at least 500 per situation before falling back (production default 1,000).

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# The full calibration, exactly as notebook 02 runs it: temperatures on all folds, then lookup tables.
# ----------------------------------------------------------------------------------------------
settings = calibration.CalibrationSettings(min_bin_count=100,        # n_min, scaled down from 200
                                           min_scenario_count=500)   # N_min, scaled down from 1,000
res = calibration.fit_from_evaluation(ev, candidate=0, settings=settings)
cal = res.calibrator               # holds the temperatures and every lookup table
temps = cal.temperatures           # [T_1, T_2, T_3, T_4]
res.temperature_fit.round(3)       # nll_raw / nll_tempered: average penalty at T = 1 and at T*

# COMMAND ----------

# MAGIC %md
# MAGIC Overconfidence grows with depth, so the deeper levels get larger temperatures. A temperature **below 1** (Level 1
# MAGIC here) is allowed: it means that level was slightly *under*confident, so the fit sharpens it instead of softening
# MAGIC it. Nothing in the method assumes which way the correction goes; the data decides.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part C — Replaying the interface (steps 6–10)
# MAGIC
# MAGIC **Step 6.** Every path is rescored with the tempered model. For one description, here are the top options at each
# MAGIC level before and after tempering. Use the buttons to switch level. The order rarely changes; the confidence does.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Step 6: rescore every path with the temperatures, and compare raw vs tempered for one description.
# ----------------------------------------------------------------------------------------------
def path_probs_for(rows, temperatures=None):
    "P(p | x) for the given rows, optionally tempered (each level's scores divided by its T_d)."
    jl = [None if j is None else j[rows] for j in jlls]
    return np.exp(_log_softmax(clf0.joint_log_from_jll(jl, temperatures)))


P_tmp = path_probs_for(slice(None), temps)                 # tempered P(p | x) for every held-out row
m1 = calibration.level_marginals(clf0, P_tmp, 0)           # level-1 scores (sum over each category's paths)

# Choose an uncertain example: its top level-1 option scores between 45% and 80%.
row = int(np.flatnonzero((m1.max(1) > 0.45) & (m1.max(1) < 0.8) & (codes[:, 0] >= 0))[0])
print("description:", held.loc[row, "Desc"])
print("true path:  ", " > ".join(held.loc[row, LEVELS]))

fig = go.Figure()
for d in range(4):
    raw = calibration.level_marginals(clf0, P_raw[[row]], d)[0]    # raw score of every level-d category
    tmp = calibration.level_marginals(clf0, P_tmp[[row]], d)[0]    # tempered score
    order = np.argsort(-tmp)[:6]                                    # the 6 highest tempered scores
    fig.add_bar(x=name_of[d][order], y=raw[order], name="raw", marker_color=GREY, visible=d == 0)
    fig.add_bar(x=name_of[d][order], y=tmp[order], name="tempered", marker_color=BLUE, visible=d == 0)
# Buttons show one level's pair of bar series at a time.
fig.update_layout(barmode="group", height=360, yaxis_tickformat=".0%", title="Top options, raw vs tempered",
                  updatemenus=[dict(type="buttons", direction="right", x=0, y=1.18, xanchor="left",
                                    buttons=[dict(label=f"Level {d + 1}", method="update",
                                                  args=[{"visible": [k // 2 == d for k in range(8)]}]) for d in range(4)])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 7. Replaying locks.** Locking category \\(v\\) keeps only the paths through \\(v\\) and rescales them so they sum
# MAGIC to 1:
# MAGIC
# MAGIC $$P(p \mid x, v) \enspace =\enspace  \frac{P(p \mid x)\enspace \mathbf{1}[p \text{ passes through } v]}{\sum_{p' \text{ through } v} P(p' \mid x)}$$
# MAGIC
# MAGIC * \\(p\\): a complete path (one category at each of the four levels).
# MAGIC * \\(v\\): the locked category.
# MAGIC * \\(P(p \\mid x)\\): the path's (tempered) probability before the lock.
# MAGIC * \\(\\mathbf{1}[\\cdot]\\): an *indicator*: 1 if the condition is true, 0 otherwise. It removes every path outside
# MAGIC   \\(v\\)'s branch.
# MAGIC * The denominator: the total probability of all paths \\(p'\\) through \\(v\\), i.e. the probability \\(v\\) had before the
# MAGIC   lock. Dividing by it makes the remaining paths sum to 1.
# MAGIC * \\(P(p \\mid x, v)\\): the path's probability once \\(v\\) is locked.
# MAGIC
# MAGIC In the deployed (conditional) mode this equals the product of the lower levels' sibling probabilities, because the
# MAGIC locked part's own probability appears in every path of the branch and cancels. So options below a lock are scored
# MAGIC as if the lock had always been certain.
# MAGIC
# MAGIC **Interactive:** in the sunburst below, the rings are the levels and each slice is sized by its probability.
# MAGIC **Click any slice to lock it**: it becomes the centre, and every percentage is recomputed relative to it. That is
# MAGIC exactly the division in the formula. Click the centre to unlock.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Sunburst of one description's path probabilities.
# A slice's size = the sum of P(p | x) over the paths below it (its "total" score).
# Clicking a slice re-centres the chart on it; "percentRoot" divides each slice by the new centre's
# total, which is the denominator in the lock formula.
# ----------------------------------------------------------------------------------------------
p_row = P_tmp[row]                                                   # tempered P(p | x) for this description
labels_by_level = [clf0._structure()["path_labels"][d] for d in range(4)]   # category of each path, per level

node_val = {}                                  # slice id (e.g. "Cleaning/Cleaning-3") -> summed probability
for d in range(4):
    keys = ["/".join(str(labels_by_level[k][j]) for k in range(d + 1)) for j in range(len(p_row))]
    for key, v in zip(keys, p_row):
        node_val[key] = node_val.get(key, 0.0) + v               # add each path's probability to its ancestors

ids = list(node_val)
fig = go.Figure(go.Sunburst(
    ids=ids, labels=[i.split("/")[-1] for i in ids], parents=["/".join(i.split("/")[:-1]) for i in ids],
    values=[node_val[i] for i in ids], branchvalues="total", sort=False,
    texttemplate="%{label}<br>%{percentRoot:.1%}",               # share of the current centre = P(. | x, v)
    hovertemplate="%{label}<br>%{percentRoot:.2%} of the current centre<extra></extra>",
    insidetextorientation="radial"))
fig.update_layout(height=560, margin=dict(t=10, b=10, l=10, r=10))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The same computation in code: lock the row's **second-ranked** Level 1 category and list the Level 2 options
# MAGIC beneath it. `path_probs` applies the lock formula, in log space so that tiny branches don't underflow to zero.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Apply the lock formula with the production function and compare before/after.
# ----------------------------------------------------------------------------------------------
v = int(np.argsort(-m1[row])[1])               # the second-ranked level-1 category (a "correction")
# path_probs works with log-probabilities: it sets log P = -infinity (P = 0) outside v's branch,
# then re-normalises so the remaining paths sum to 1. That is the indicator and the division.
P_locked = calibration.path_probs(clf0, np.log(np.maximum(P_tmp[[row]], 1e-300)), n_fixed=1, nodes=np.array([v]))
m2_locked = calibration.level_marginals(clf0, P_locked, 1)[0]      # level-2 scores after the lock
inside = np.flatnonzero(clf0._structure()["ancestors"][1][0] == v)  # level-2 categories under v
pd.DataFrame({"level-2 option": name_of[1][inside],
              "before lock": calibration.level_marginals(clf0, P_tmp[[row]], 1)[0][inside],
              "after locking " + name_of[0][v]: m2_locked[inside]}).sort_values("before lock", ascending=False).round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC Each "after" value equals the "before" value divided by their total: the lock rescales the branch without changing
# MAGIC its order.
# MAGIC
# MAGIC **Step 8. Confirmed or corrected?** Every simulated lock at level \\(k\\) is labelled:
# MAGIC
# MAGIC $$\kappa = \begin{cases}\text{confirmed} & \text{if } c^{\ast}_k = \text{the model's top pick at level } k\cr  \text{corrected} & \text{otherwise}\end{cases}$$
# MAGIC
# MAGIC * \\(k\\): the level of the lock (levels \\(1..k\\) are then fixed).
# MAGIC * \\(c^{\\ast}_k\\): the true category at level \\(k\\), which the simulated user locks.
# MAGIC * the model's top pick: the category the model ranked first at level \\(k\\) with nothing locked.
# MAGIC * \\(\\kappa\\): the lock's kind. *Confirmed* corresponds to clicking the checkmark; *corrected* to choosing something
# MAGIC   else.
# MAGIC
# MAGIC A correction is a sign the item is hard for the model, so the suggestions *below* it are less reliable. The chart
# MAGIC compares, after a lock, how often the top option one level down is right.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Accuracy of the top option one level below a lock, split by the lock's kind kappa.
# rank_accuracy holds, for every situation and rank, the fraction of recorded options that were correct.
# ----------------------------------------------------------------------------------------------
ra = res.rank_accuracy
nxt = ra[(ra["ranking"] == "total") & (ra["rank"] == 1)           # the top option...
         & (ra["n_fixed"] > 0)                                     # ...below a lock...
         & (ra["target_level"] == ra["n_fixed"] + 1)]              # ...one level down
fig = go.Figure()
for kind, color in [("confirmed", GREEN), ("corrected", ORANGE)]:
    d = nxt[nxt["kind"] == kind]
    fig.add_bar(x=[f"lock L1..L{k}, show L{k + 1}" for k in d["n_fixed"]], y=d["accuracy"], name=kind, marker_color=color,
                customdata=d["n"], hovertemplate="%{y:.1%} right<br>%{customdata:,} rows<extra>" + kind + "</extra>")
fig.update_layout(barmode="group", height=340, yaxis=dict(tickformat=".0%", range=[0, 1]),
                  title="How often the top option below a lock is right")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 9. Recording the options, under two rankings.** Each dropdown can rank a level's categories in two ways:
# MAGIC
# MAGIC $$m_d(c) = \sum_{p \thinspace \ni\thinspace  c} P(p \mid x) \qquad\qquad b_d(c) = \max_{p \thinspace \ni\thinspace  c} P(p \mid x)$$
# MAGIC
# MAGIC * \\(c\\): a category at level \\(d\\).
# MAGIC * \\(p \\ni c\\): the paths that pass through \\(c\\).
# MAGIC * \\(\\sum\\): add up over those paths; \\(\\max\\): take the largest.
# MAGIC * \\(m_d(c)\\): the **total** score, the probability that \\(c\\) is the correct level-\\(d\\) category.
# MAGIC * \\(b_d(c)\\): the **best-path** score, the probability of the single best complete path through \\(c\\).
# MAGIC * Always \\(b_d(c) \\le m_d(c)\\); at level 4 every category is one path, so the two are equal.
# MAGIC
# MAGIC For every row and situation, the top **10** options in scope are recorded with their score \\(q\\) and whether they
# MAGIC were correct, \\(y\\). Below are the records for the example row with nothing locked, at Level 2.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# What step 9 records for one row and one situation: (rank, option, score q, correct y).
# ----------------------------------------------------------------------------------------------
def records(row, t, ranking, n_fixed=0, node=None, k=10):
    # Path probabilities for this row: tempered, and restricted to the lock if there is one (step 7).
    P = P_tmp[[row]] if n_fixed == 0 else calibration.path_probs(
        clf0, np.log(np.maximum(P_tmp[[row]], 1e-300)), n_fixed, np.array([node]))
    # level_scores computes m_t(c) (ranking="total": a sum) or b_t(c) (ranking="best_path": a max).
    s = calibration.level_scores(clf0, P, t, ranking)
    # The k highest scores among categories inside the lock's branch.
    cand, valid = calibration.top_k_in_scope(clf0, s, t, k, n_fixed, None if node is None else np.array([node]))
    cand = cand[0][valid[0]]
    return pd.DataFrame({"rank": np.arange(1, len(cand) + 1), "option": name_of[t][cand], "score q": s[0, cand],
                         "correct y": (cand == codes[row, t]).astype(int)})       # y = 1 for the true category


pd.concat({"total (m)": records(row, 1, "total"), "best path (b)": records(row, 1, "best_path")}, axis=1).round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interactive: the two rankings across many rows.** Each dot is one category for one held-out row, with
# MAGIC \\(x = m_d(c)\\) and \\(y = b_d(c)\\). Because \\(b \\le m\\), every dot lies on or below the dashed diagonal \\(b = m\\). Dots on
# MAGIC the diagonal have one dominant path; dots far below it have their probability spread over several paths. Use the
# MAGIC buttons to change level.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Scatter of b_d(c) against m_d(c) for 250 rows x every category with m > 1%.
# ----------------------------------------------------------------------------------------------
sample = rng.choice(len(held), 250, replace=False)
fig = go.Figure()
for d in range(3):                                 # levels 1-3 (at level 4, b = m exactly)
    m = calibration.level_scores(clf0, P_tmp[sample], d, "total").ravel()       # m_d(c): sum over paths
    b = calibration.level_scores(clf0, P_tmp[sample], d, "best_path").ravel()   # b_d(c): max over paths
    keep = m > 0.01                                # hide the mass of near-zero categories
    fig.add_scatter(x=m[keep], y=b[keep], mode="markers", marker=dict(size=5, color=BLUE, opacity=0.45),
                    name=f"Level {d + 1}", visible=d == 0)
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREEN, dash="dash"), name="b = m")
fig.update_layout(height=420, xaxis=dict(title="total m", tickformat=".0%"), yaxis=dict(title="best path b", tickformat=".0%"),
                  updatemenus=[dict(type="buttons", direction="right", x=0, y=1.15, xanchor="left",
                                    buttons=[dict(label=f"Level {d + 1}", method="update",
                                                  args=[{"visible": [k == d for k in range(3)] + [True]}]) for d in range(3)])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 10. Sorting the records into situations.** A *situation* is the combination (ranking, which dropdown, how
# MAGIC many levels are locked, confirmed or corrected). Counting them:
# MAGIC
# MAGIC $$\underbrace{\sum_{t=1}^{L}\big(1 + 2(t-1)\big)}_{\text{total ranking}} = L^2 = 16 \qquad\qquad \underbrace{\sum_{t=1}^{L-1}\big(1 + 2(t-1)\big)}_{\text{best-path ranking}} = (L-1)^2 = 9$$
# MAGIC
# MAGIC * \\(L\\): the number of levels (4).
# MAGIC * \\(t\\): the target level, i.e. the dropdown being scored.
# MAGIC * \\(1\\): the situation with nothing locked.
# MAGIC * \\(2(t-1)\\): two kinds (confirmed, corrected) for each of the \\(t-1\\) levels above \\(t\\) that could be the deepest lock.
# MAGIC * The best-path sum stops at \\(L-1\\) because level 4 is identical under both rankings and shares the total tables.
# MAGIC
# MAGIC The heatmap shows how many options were recorded in each situation. The corrected situations are much smaller,
# MAGIC which matters for step 16.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Number of recorded options per situation, read from the saved histograms.
# ----------------------------------------------------------------------------------------------
z = np.load(res.folder / "histograms.npz")                 # everything the lookup tables are built from
scen = calibration._parse_scenarios(z["scenarios"])        # list of (ranking, t, k, kind), t and k from 0
# tempered_n has shape (situation, fold, fine bin); summing over folds and bins counts the options.
counts = {s: z["tempered_n"][i].sum() for i, s in enumerate(scen)}

cols_ = ["nothing locked"] + [f"L1..L{k} {kind}" for k in range(1, 4) for kind in ("confirmed", "corrected")]
fig = go.Figure()
for r_i, ranking in enumerate(["total", "best_path"]):
    zz = np.full((4, len(cols_)), np.nan)                  # rows = target level, columns = lock depth and kind
    for (rk, t, k, kind), n in counts.items():
        if rk == ranking:
            zz[t, 0 if k == 0 else 1 + 2 * (k - 1) + (kind == "corrected")] = n
    fig.add_heatmap(z=zz, x=cols_, y=[f"Level {t + 1} list" for t in range(4)], colorscale="Blues", visible=r_i == 0,
                    text=np.where(np.isnan(zz), "", np.vectorize(lambda v: f"{v:,.0f}" if v == v else "")(zz)),
                    texttemplate="%{text}", hovertemplate="%{y} | %{x}<br>%{z:,.0f} options<extra></extra>", showscale=False)
fig.update_layout(height=330, yaxis_autorange="reversed", title="Recorded options per situation",
                  updatemenus=[dict(type="buttons", direction="right", x=0, y=1.25, xanchor="left",
                                    buttons=[dict(label=r, method="update", args=[{"visible": [k == i for k in range(2)]}])
                                             for i, r in enumerate(["total", "best path"])])])
fig.show()
print(f"{sum(rk == 'total' for rk, *_ in scen)} total situations, {sum(rk == 'best_path' for rk, *_ in scen)} best-path situations")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part D — Step 2 of calibration: lookup tables (steps 11–16)
# MAGIC
# MAGIC **Steps 11–13.** Within a situation, scores are compared on the **log-odds** scale, which spreads out the crowded
# MAGIC regions near 0% and 100%. Records go into fine bins (width 0.05 in log-odds, plus every 1% boundary), and
# MAGIC neighbouring bins are merged until each **bucket** holds at least \\(n_{\\min}\\) options:
# MAGIC
# MAGIC $$z = \log\frac{q}{1-q} \qquad\qquad \hat a_b = \frac{\text{number of correct options in bucket } b}{n_b} \qquad\qquad \bar z_b = \frac{1}{n_b}\sum_{m \in b} z_m$$
# MAGIC
# MAGIC * \\(q\\): a recorded option's (tempered) score; \\(z\\): its log-odds. \\(q = 50\\%\\) gives \\(z = 0\\), \\(q = 99\\%\\) gives
# MAGIC   \\(z \\approx 4.6\\), and \\(q = 1\\%\\) gives \\(z \\approx -4.6\\).
# MAGIC * \\(b\\): one bucket; \\(n_b\\): the number of recorded options in it; \\(m \\in b\\): the options in it.
# MAGIC * \\(\\hat a_b\\): the bucket's **observed accuracy**, the fraction of its options that were correct.
# MAGIC * \\(\\bar z_b\\): the bucket's **position**, the average log-odds of its options.
# MAGIC * \\(n_{\\min}\\): the minimum options per bucket (here 100, production 200).
# MAGIC
# MAGIC **Interactive: move the \\(n_{\\min}\\) slider.** Small buckets follow the data closely but are noisy; large ones are
# MAGIC smooth but coarse. The situation shown is *Level 4, nothing locked*. Grey dots are the fine bins (sized by count),
# MAGIC orange dots are the buckets \\((\\bar z_b, \\hat a_b)\\), and the blue line is the resulting lookup table.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Build one situation's lookup table from its histogram, for several values of n_min.
# ----------------------------------------------------------------------------------------------
def hist(key):
    "Per fine bin, summed over folds: n (count), pos (number correct), sl (sum of log-odds z)."
    i = scen.index(key)
    return z["tempered_n"][i].sum(0), z["tempered_pos"][i].sum(0), z["tempered_sl"][i].sum(0)


def logistic(v):
    # The inverse of log-odds: q = 1 / (1 + e^(-z)) turns z back into a probability.
    return 1 / (1 + np.exp(-v))


key = ("total", 3, 0, "none")            # level 4 (index 3), nothing locked
n, pos, sl = hist(key)
fine = n > 0                             # fine bins that contain at least one option
nmins = [25, 50, 100, 200, 400, 800]
xs = np.linspace(-16, 8, 400)            # log-odds values at which to draw the table


def table_traces(nmin):
    # build_table merges fine bins into buckets with at least n_min options, computes a_b and z_b,
    # makes them non-decreasing (step 14), clips them (step 15), and returns a function g(q).
    s = calibration.CalibrationSettings(min_bin_count=nmin)
    tab = calibration.build_table(n, pos, sl, z["edges"], s)
    return [go.Scatter(x=tab.centers, y=tab.positives / tab.counts,            # (z_b, a_b) per bucket
                       mode="markers", marker=dict(color=ORANGE, size=8), name="bucket accuracy", customdata=tab.counts,
                       hovertemplate="z=%{x:.2f}<br>accuracy %{y:.1%}<br>n=%{customdata}"),
            go.Scatter(x=xs, y=tab(logistic(xs)), mode="lines", line=dict(color=BLUE, width=3),
                       name="lookup table g(q)")]                                # the table at every z


fig = go.Figure()
fig.add_scatter(x=sl[fine] / n[fine], y=pos[fine] / n[fine], mode="markers", name="fine bins",   # mean z, accuracy
                marker=dict(color=GREY, size=np.clip(np.sqrt(n[fine]), 3, 18), opacity=0.35))   # trace 0, underneath
fig.add_traces(table_traces(100))                                                               # traces 1-2
# The identity line: what the table would be if the tempered score were already calibrated (g(q) = q).
fig.add_scatter(x=xs, y=logistic(xs), mode="lines", line=dict(color=GREEN, dash="dash"), name="identity (score taken at face value)")
fig.frames = [go.Frame(data=table_traces(k), traces=[1, 2], name=str(k)) for k in nmins]
fig.update_layout(
    height=460, xaxis_title="tempered score as log-odds z", yaxis=dict(title="probability correct", tickformat=".0%"),
    legend=dict(orientation="h", y=1.12),
    sliders=[dict(active=2, currentvalue=dict(prefix="n_min = "), pad=dict(t=50),
                  steps=[dict(label=str(k), method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                         for k in nmins])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The alternative `CAL_SCHEME = "percent"` uses fixed 1% buckets of \\(q\\) instead. Toggle between the two schemes to
# MAGIC see why the adaptive default exists: with 1% buckets, nearly all options fall into the first and last bucket, and
# MAGIC the middle is sparse and noisy. (This chart uses \\(q\\) itself on the \\(x\\)-axis rather than \\(z\\).)

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# The same situation with adaptive (log-odds) buckets vs fixed 1% buckets. Dot size ~ sqrt(bucket size).
# ----------------------------------------------------------------------------------------------
fig = go.Figure()
for i, scheme in enumerate(["logit", "percent"]):
    tab = calibration.build_table(n, pos, sl, z["edges"], calibration.CalibrationSettings(scheme=scheme, min_bin_count=100))
    fig.add_scatter(x=logistic(tab.centers), y=tab.positives / tab.counts, mode="markers",     # (q of bucket, a_b)
                    marker=dict(color=ORANGE, size=np.clip(np.sqrt(tab.counts) / 2, 4, 22)),
                    customdata=tab.counts, hovertemplate="score %{x:.2%}<br>accuracy %{y:.1%}<br>n=%{customdata:,}",
                    name=f"{scheme} buckets", visible=i == 0)
    fig.add_scatter(x=np.linspace(0, 1, 300), y=tab(np.linspace(0, 1, 300)), mode="lines", line=dict(color=BLUE, width=3),
                    name=f"{scheme} table", visible=i == 0)
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREEN, dash="dash"), name="identity")
fig.update_layout(height=400, xaxis=dict(title="tempered score q", tickformat=".0%"), yaxis=dict(title="probability correct", tickformat=".0%"),
                  updatemenus=[dict(type="buttons", direction="right", x=0, y=1.15, xanchor="left",
                                    buttons=[dict(label=s, method="update", args=[{"visible": [k // 2 == i for k in range(4)] + [True]}])
                                             for i, s in enumerate(["adaptive (logit)", "1% buckets"])])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 14. Higher score, higher probability.** Chance can make a higher bucket look less accurate than a lower one.
# MAGIC The table must never decrease, so it uses the closest non-decreasing sequence to the bucket accuracies:
# MAGIC
# MAGIC $$\hat g = \arg\min_{g_1 \le g_2 \le \dots \le g_B} \enspace \sum_{b=1}^{B} n_b\thinspace \big(\hat a_b - g_b\big)^2$$
# MAGIC
# MAGIC * \\(B\\): the number of buckets, ordered from lowest to highest score.
# MAGIC * \\(g_b\\): the value the table will use for bucket \\(b\\); \\(g_1 \\le g_2 \\le \\dots\\) forces them never to decrease.
# MAGIC * \\(\\hat a_b\\), \\(n_b\\): bucket \\(b\\)'s observed accuracy and size (from steps 11–13).
# MAGIC * \\((\\hat a_b - g_b)^2\\): how far the table departs from what was observed in bucket \\(b\\), weighted by \\(n_b\\) so that
# MAGIC   larger buckets count more.
# MAGIC * \\(\\hat g\\): the sequence with the smallest total weighted departure.
# MAGIC
# MAGIC It is solved exactly by **pool-adjacent-violators**: wherever a bucket is lower than the one before it, the two are
# MAGIC replaced by their size-weighted average, repeating until nothing decreases. This is easiest to see in a small, noisy
# MAGIC situation. The flat stretches of the blue line are pooled buckets.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Before and after pool-adjacent-violators for the largest "corrected" situation, with small buckets.
# ----------------------------------------------------------------------------------------------
small_key = min((k for k in scen if k[0] == "total" and k[3] == "corrected"), key=lambda k: -counts[k])
tab = calibration.build_table(*hist(small_key), z["edges"], calibration.CalibrationSettings(min_bin_count=50))
raw_acc = tab.positives / tab.counts          # a_b: observed accuracy per bucket, before the fit
# tab.values holds g_b, the result of the minimisation above (after clipping, step 15).
fig = go.Figure()
fig.add_scatter(x=tab.centers, y=raw_acc, mode="markers+lines", line=dict(color=ORANGE, width=1, dash="dot"), marker=dict(size=8),
                name="bucket accuracy a_b (before)", customdata=tab.counts, hovertemplate="z=%{x:.2f}<br>%{y:.1%}<br>n=%{customdata}")
fig.add_scatter(x=tab.centers, y=tab.values, mode="markers+lines", line=dict(color=BLUE, width=3), marker=dict(size=6),
                name="g_b after pool-adjacent-violators")
fig.update_layout(height=380, title=calibration.scenario_name(small_key[1], small_key[2], small_key[3]),
                  xaxis_title="log-odds z", yaxis=dict(title="probability correct", tickformat=".0%"), legend=dict(orientation="h", y=-0.25))
fig.show()
# np.diff takes differences between neighbours; a negative difference is a decrease.
print(f"{(np.diff(raw_acc) < 0).sum()} decreases in the raw bucket accuracies; {(np.diff(tab.values) < -1e-12).sum()} after the fit")

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 15. Clipping and reading the table.** The fitted values are kept away from 0 and 1, and a score that falls
# MAGIC between two buckets is read off by straight-line interpolation on the log-odds scale:
# MAGIC
# MAGIC $$g_b \leftarrow \min\big(\max(g_b,\enspace \varepsilon),\enspace 1-\varepsilon\big) \qquad\qquad g(q) = g_b + \frac{z - \bar z_b}{\bar z_{b+1} - \bar z_b}\thinspace \big(g_{b+1} - g_b\big) \quad \text{for } \bar z_b \le z \le \bar z_{b+1}$$
# MAGIC
# MAGIC * \\(\\varepsilon = 10^{-4}\\): the clipping bound, so no calibrated value is below 0.01% or above 99.99%. Even a bucket
# MAGIC   where every option was wrong (or right) can't justify certainty.
# MAGIC * \\(z\\): the log-odds of the score \\(q\\) being looked up.
# MAGIC * \\(\\bar z_b, \\bar z_{b+1}\\): the positions of the two buckets on either side of \\(z\\); \\(g_b, g_{b+1}\\): their values.
# MAGIC * The fraction is how far \\(z\\) lies between the two bucket positions (0 at the left one, 1 at the right one).
# MAGIC * Below the first bucket or above the last, \\(g(q)\\) stays at the end value.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Look up a few extreme and ordinary scores in the level-4, nothing-locked table.
# cal.calibrate(q, target_level=3) does exactly: z = logit(q), interpolate between buckets, clip.
# ----------------------------------------------------------------------------------------------
probe = np.array([1e-9, 1e-4, 0.01, 0.5, 0.99, 1 - 1e-9])
pd.DataFrame({"tempered score q": probe,
              "calibrated g(q)": cal.calibrate(probe, target_level=3)}).style.format("{:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 16. Fallbacks for thin situations.** A situation with fewer than \\(N_{\\min}\\) recorded options (500 here,
# MAGIC 1,000 by default) borrows a broader table instead of using its own:
# MAGIC
# MAGIC own table → confirmed and corrected pooled → same level with nothing locked → the tempered score itself.
# MAGIC
# MAGIC This dataset is large enough that every situation has its own table at 500, and even at the 1,000 default. The
# MAGIC table below rebuilds the calibration from the saved histograms at three thresholds. At 2,000 the smallest
# MAGIC situations (corrected locks) start borrowing a broader table; those cells are highlighted. On a real catalogue,
# MAGIC deep corrected situations in rare branches are where this happens.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Which table each situation actually uses, at three values of N_min.
# ----------------------------------------------------------------------------------------------
H = {k: z[f"tempered_{k}"] for k in ("n", "pos", "sp", "sl")}    # the saved per-bin sums
thresholds = [500, 1000, 2000]
used = {}
for nmin in thresholds:
    # Rebuild every table with this N_min (no rescoring needed: the histograms hold everything).
    c = calibration.build_calibrator(H, z["edges"], scen, temps,
                                     calibration.CalibrationSettings(min_bin_count=100, min_scenario_count=nmin),
                                     LEVELS, hp.prediction_mode, hp.to_dict())
    used[f"N_min = {nmin:,}"] = [c.table_for(t, k, kind, rk)[1] for (rk, t, k, kind) in scen]   # name of table used
own = [calibration.scenario_name(t, k, kind, rk) for (rk, t, k, kind) in scen]
fallback = pd.DataFrame({"situation": own, "recorded options": [int(counts[s_]) for s_ in scen], **used})
for col in used:
    print(f"{col}: {(fallback[col] != fallback['situation']).sum()} of {len(scen)} situations borrow a table")
# Highlight cells where the table used is not the situation's own.
fallback.style.apply(lambda r: ["background-color: #fdecea" if (c_ in used and r[c_] != r["situation"]) else ""
                                for c_ in r.index], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part E — Checking the result (steps 17–18)
# MAGIC
# MAGIC **Step 17. Cross-fitting.** A table checked on the data it was built from always looks good. So for each fold, the
# MAGIC tables are rebuilt from the *other* folds' records only and applied to that fold, and the results are combined.
# MAGIC Three versions of every score are compared: raw, tempered and calibrated.
# MAGIC
# MAGIC **Step 18. The measures.** The main one is the *expected calibration error*:
# MAGIC
# MAGIC $$\text{ECE} = \sum_{j=1}^{100} \frac{N_j}{N}\thinspace \big|\thinspace \bar y_j - \bar q_j\thinspace \big|$$
# MAGIC
# MAGIC * \\(j\\): one of 100 equal-width bins of the score being checked (0–1%, 1–2%, …).
# MAGIC * \\(N_j\\): the number of recorded options in bin \\(j\\); \\(N\\): the total number of options.
# MAGIC * \\(\\bar q_j\\): the average score in bin \\(j\\) (what was *stated*).
# MAGIC * \\(\\bar y_j\\): the fraction of bin \\(j\\)'s options that were correct (what *happened*).
# MAGIC * \\(|\\cdot|\\): the size of the gap, ignoring its direction.
# MAGIC * ECE: the average gap between stated and actual, weighted by how many options sit in each bin. It is 0 for a
# MAGIC   perfectly calibrated score.
# MAGIC
# MAGIC The Brier score and log-loss are also in `metrics.csv`. Lower is better for all three.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# ECE per situation for raw, tempered and cross-fitted calibrated scores.
# ----------------------------------------------------------------------------------------------
ece = calibration.metrics_wide(res.metrics, "ece")       # one row per situation, one column per score version
ece[["ranking", "scenario", "candidates", "raw", "tempered", "calibrated (cross-fitted)"]].round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interactive reliability explorer.** Choose a situation from the menu. Each line plots \\(\\bar y_j\\) (actual) against
# MAGIC \\(\\bar q_j\\) (stated) for one version of the score; a calibrated score lies on the dashed diagonal, and the ECE is the
# MAGIC weighted average vertical distance from it. Grey is raw, blue tempered and orange calibrated (cross-fitted). Note
# MAGIC how *best path* scores sit **above** the diagonal after tempering (underconfident), and how their own lookup tables
# MAGIC bring them back.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Reliability curves for every situation, using 5% bins so the lines are readable.
# ----------------------------------------------------------------------------------------------
rel = res.reliability.copy()                                         # 1% bins: n, mean prediction, observed
rel["bucket"] = np.minimum((rel["bucket_from"] * 20).astype(int), 19)    # merge into 20 bins of 5%
rel["wp"], rel["wo"] = rel["mean_prediction"] * rel["n"], rel["observed"] * rel["n"]   # weighted sums
g = rel.groupby(["ranking", "scenario", "predictor", "bucket"], sort=False)[["n", "wp", "wo"]].sum().reset_index()
g = g[g["n"] >= 30]                                                  # skip bins with too few options to be reliable
g["x"], g["y"] = g["wp"] / g["n"], g["wo"] / g["n"]                 # q_bar_j (stated) and y_bar_j (actual)

situations = list(dict.fromkeys(zip(rel["ranking"], rel["scenario"])))
preds = [("raw", GREY), ("tempered", BLUE), ("calibrated (cross-fitted)", ORANGE)]
fig = go.Figure()
for i, (rk, sc) in enumerate(situations):
    for p, color in preds:
        d = g[(g["ranking"] == rk) & (g["scenario"] == sc) & (g["predictor"] == p)]
        fig.add_scatter(x=d["x"], y=d["y"], mode="lines+markers", name=p, line=dict(color=color), visible=i == 0,
                        customdata=d["n"], hovertemplate="stated %{x:.1%}<br>actual %{y:.1%}<br>n=%{customdata:,}<extra>" + p + "</extra>")
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREEN, dash="dash"), name="calibrated ideal")
n_tr = len(preds)
fig.update_layout(height=460, xaxis=dict(title="stated probability", tickformat=".0%", range=[0, 1]),
                  yaxis=dict(title="actual accuracy", tickformat=".0%", range=[0, 1.02]),
                  updatemenus=[dict(x=0, y=1.16, xanchor="left", buttons=[
                      dict(label=sc, method="update",
                           args=[{"visible": [j // n_tr == i for j in range(len(situations) * n_tr)] + [True]}])
                      for i, (rk, sc) in enumerate(situations)])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Accuracy by rank.** How often the 1st, 2nd, … 10th option in each situation turned out to be correct, i.e. the
# MAGIC fraction of options at that rank with \\(y = 1\\). Most of the probability sits in the first one or two options, and
# MAGIC much more so after a confirmed lock than after a corrected one.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Heatmap: rows = situations, columns = rank, colour = fraction correct.
# ----------------------------------------------------------------------------------------------
ra = res.rank_accuracy
fig = go.Figure()
for i, rk in enumerate(["total", "best_path"]):
    piv = ra[ra["ranking"] == rk].pivot_table(index="scenario", columns="rank", values="accuracy", sort=False)
    fig.add_heatmap(z=piv.values, x=[f"#{c}" for c in piv.columns], y=[s.replace(" | best path", "") for s in piv.index],
                    colorscale="Blues", zmin=0, zmax=1, visible=i == 0, texttemplate="%{z:.0%}", showscale=False,
                    hovertemplate="%{y}<br>rank %{x}: %{z:.1%} correct<extra></extra>")
fig.update_layout(height=620, yaxis_autorange="reversed", margin=dict(l=260),
                  updatemenus=[dict(type="buttons", direction="right", x=0, y=1.08, xanchor="left",
                                    buttons=[dict(label=r, method="update", args=[{"visible": [k == i for k in range(2)]}])
                                             for i, r in enumerate(["total", "best path"])])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part F — Using it (steps 19–21)
# MAGIC
# MAGIC **Step 19. Saving with the model.** Notebook 03 trains the final model on all rows and saves `calibration.json`
# MAGIC beside it, after checking it was fitted for the same hyperparameters. Here the final model is trained in memory
# MAGIC and the calibrator is attached directly.

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Train the final model on every row and attach the calibration fitted from the out-of-fold data.
# ----------------------------------------------------------------------------------------------
final = HierarchicalNB(LEVELS, hp).fit(catalogue["Desc"], catalogue[LEVELS])
cal.check_compatible(final, strict=True)          # same levels, mode and hyperparameters (as notebook 03 checks)
lookup = catalogue[LEVELS + NAMES].drop_duplicates()
rec = Recommender(final, cal, lookup, k=5)        # the object the interface calls; k = options per dropdown
print(cal)

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 20. The interface.** For a new description, the tool tempers the scores, identifies the situation, ranks
# MAGIC the options, and converts each score \\(q\\) through that situation's table \\(g(q)\\).
# MAGIC
# MAGIC **Interactive interface simulator.** The four panels are the four dropdowns, and each bar is a calibrated
# MAGIC probability \\(g(q)\\). Use the menu to replay user actions: confirming the top pick, correcting to the second option,
# MAGIC locking deeper levels, or switching ranking. Grey bars are levels fixed by a lock (probability 1). Hover for the
# MAGIC tempered score \\(q\\) and the table used.
# MAGIC
# MAGIC Two options can show the **same percentage** (Electrical and Cleaning, with nothing locked). Their tempered scores
# MAGIC differ, but both fall in a flat stretch of the lookup table, where step 14 pooled neighbouring buckets. Their order
# MAGIC still follows \\(q\\).

# COMMAND ----------

# ----------------------------------------------------------------------------------------------
# Replay several user actions for one description and draw the four dropdowns for each.
# ----------------------------------------------------------------------------------------------
desc = held.loc[row, "Desc"]
base = rec.recommend(texts=[desc])                              # nothing locked: the default dropdowns
l1 = base[base.depth == 1]["label"].tolist()                    # level-1 options, best first
first2 = base[base.depth == 2]["label"].iloc[0]
# (menu label, selections {level number: category}, ranking)
states = [("Nothing locked", None, "total"),
          (f"Confirm L1 = {l1[0]}", {1: l1[0]}, "total"),        # lock the top pick -> "confirmed" tables
          (f"Correct L1 to {l1[1]}", {1: l1[1]}, "total"),       # lock the 2nd option -> "corrected" tables
          (f"Lock L2 = {first2}", {2: first2}, "total"),          # locking L2 also fixes its L1 parent
          ("Nothing locked (best path)", None, "best_path"),
          (f"Correct L1 to {l1[1]} (best path)", {1: l1[1]}, "best_path")]

fig = make_subplots(4, 1, subplot_titles=[f"Level {d} dropdown" for d in range(1, 5)], vertical_spacing=0.07)
for s_i, (title, sel, ranking) in enumerate(states):
    out = rec.recommend(texts=[desc], selections=sel, ranking=ranking)   # columns include calibrated, score, table
    for d in range(1, 5):
        o = out[out.depth == d].iloc[::-1]                               # reversed so the best is drawn at the top
        fig.add_bar(y=o["label"], x=o["calibrated"], orientation="h", visible=s_i == 0, showlegend=False,
                    marker_color=[GREY if f else (BLUE if ranking == "total" else PURPLE) for f in o["fixed"]],
                    text=[f"{v:.1%}" for v in o["calibrated"]], textposition="outside",
                    customdata=np.stack([o["score"], o["table"].fillna("fixed by lock")], axis=1),
                    hovertemplate="%{y}<br>calibrated g(q) %{x:.2%}<br>tempered score q %{customdata[0]:.3f}<br>table: %{customdata[1]}<extra></extra>",
                    row=d, col=1)
n_per = 4                                                     # four bar traces (one per level) per state
fig.update_xaxes(range=[0, 1.18], tickformat=".0%")
fig.update_layout(height=940, margin=dict(t=150), title=dict(text=f"Description: {desc}", y=0.985),
                  updatemenus=[dict(x=0, y=1.1, xanchor="left", buttons=[
                      dict(label=t, method="update", args=[{"visible": [j // n_per == i for j in range(len(states) * n_per)]}])
                      for i, (t, _, _) in enumerate(states)])])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The same output as data, for the "correct Level 1 to the second option" action. This is what an application calls
# MAGIC (`recommend_one` returns the same information as JSON-ready dictionaries). `score` is \\(q\\), `calibrated` is \\(g(q)\\),
# MAGIC and `table` names the situation whose table was used.

# COMMAND ----------

rec.recommend(texts=[desc], selections={1: l1[1]})[["level", "rank", "label", "calibrated", "score", "fixed", "scenario", "table"]].round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 21. Refitting.** The calibration describes one model setting on one body of data. Rerun notebook 02's
# MAGIC calibration section, then notebook 03, whenever the hyperparameters change, the training data changes materially
# MAGIC (a large intake of newly categorised items, new suppliers or product lines), or the hierarchy changes.
# MAGIC
# MAGIC ---
# MAGIC **Where to go next:** notebook 02 runs all of this on the real catalogue, notebook 05 is the real interface demo,
# MAGIC and `Calibration_Technical_Description_v2.pdf` has the full mathematics. To clean up, delete the working folder
# MAGIC printed in the setup cell.