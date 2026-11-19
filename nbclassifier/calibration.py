"""Calibrated probabilities for every candidate the dropdown interface can show.

Fitted from out-of-fold (OOF) cross-validation: for each split of a
crossval.Evaluation, a model with the candidate's hyperparameters is fitted
on the training part (rare-path filter applied to it, as in deployment) and
the held-out rows are scored. Every held-out score is therefore one the model
produced without seeing that row.

What is calibrated
------------------
Not just the top pick. A *scenario* is (target level t, number of fixed top
levels c, selection kind):

    c = 0            nothing selected; candidates are level-t marginals
                     P(Lt = x | text), the sum over every valid path below x
    c >= 1           levels 1..c are fixed by a user selection; candidates are
                     the level-t descendants of the selected node, scored
                     P(Lt = x | L1..Lc, text) (the path scores restricted to
                     the selected subtree and renormalised)
    kind             c = 0: 'none'. c >= 1: 'confirmed' when the selected
                     level-c node was the model's own top level-c pick (the
                     user clicked the checkmark), 'corrected' when the user
                     chose something else. Corrections happen on items the
                     model finds hard, so the scores below them are less
                     reliable, and get their own tables.

Selecting a node at level c fixes every level above it, so selecting only a
level-3 category is the scenario c = 3.

Rankings
--------
How a dropdown ranks its level's classes (and so which score is shown):

    'total'      the class's marginal probability: the sum over every path below
                 it ("which category is most likely correct"). Default.
    'best_path'  the probability of the class's single most likely path ("which
                 categories lie on the most convincing complete paths"); the same
                 as reading distinct classes off the ranked list of paths.

Both are calibrated, each with its own tables, because the same number means
different things under the two rankings. At level 4 every class is one path,
so both rankings coincide and share the 'total' tables. The scenario is then
(ranking, target level, fixed levels, kind): 16 for 'total' and 9 more for
'best_path' (levels 1-3) with four levels. Confirmed / corrected is judged against the
ranking in use (was the selected node first in that ranking's list?).

For OOF calibration the user is simulated as selecting the *true* node, and
'confirmed' / 'corrected' is decided by whether that node was the model's
top pick. For every row and scenario the top `top_k` candidates within the
scope are recorded with label 1 if the candidate is the true class: this is
calibration over all displayed candidates (any rank), not only rank 1.

Two steps
---------
1. Temperature scaling, one temperature per level. Each level's log-likelihood
   is divided by T_d before its (sibling) softmax; T_d minimises the OOF
   negative log-likelihood of the true child given the true parent. Naive
   Bayes is overconfident, so T_d > 1 is expected. The tempered model is still
   a coherent distribution over paths, so marginals, conditionals and rankings
   all come from one place.
2. Lookup tables, one per scenario, mapping tempered score -> observed
   accuracy. Built from fine histograms of (score, correct) in log-odds
   space. scheme='percent' uses 1% buckets of the score (the classic table);
   scheme='logit' (default) merges fine log-odds bins until each holds
   `min_bin_count` candidates, which resolves the 0-1% and 99-100% regions
   where most naive Bayes scores sit. Either way the bucket accuracies are made
   monotone with pool-adjacent-violators, interpolated between bucket centres
   in log-odds, and clipped to [0.0001, 0.9999]. A scenario with fewer than
   `min_scenario_count` candidates falls back to the pooled table for its
   (t, c) (both kinds), then to the (t, 0) table, then to the clipped
   tempered score.

Fold models: when the Evaluation saved its fold models (Evaluation.run with
save_fold_models=True), they are loaded and adapted with clone_with, exactly
as the evaluation produced its candidates, instead of being refitted.

Diagnostics are cross-fitted: the lookup tables used to score split s are
fitted on the other splits only, so the 'calibrated' rows of metrics.csv
are an honest out-of-sample estimate.

Outputs (outputs/evaluation_<ts>/calibration/candidate_<k>/)
    calibration.json      the Calibrator (temperatures, tables, settings, provenance)
    histograms.npz        the fine histograms (re-build tables without re-fitting)
    temperature_fit.csv   T per level, OOF log-loss before/after
    lookup_tables.csv     every scenario's table in 1% score buckets
    metrics.csv           ECE / Brier / log-loss per scenario: raw, tempered, calibrated
    reliability.csv       reliability curves (1% buckets) per scenario and predictor
    rank_accuracy.csv     observed accuracy by rank within each scenario
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model import HierarchicalNB, _as_csr, _log_softmax

KINDS = ("none", "confirmed", "corrected")
POOLED = "any"
RANKINGS = ("total", "best_path")
CALIBRATION_FILE = "calibration.json"
FORMAT_VERSION = 2              # 2 adds rankings; version 1 files load as 'total' only
_P_EPS = 1e-12


# ---------------------------------------------------------------------------
# Settings and scenarios
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSettings:
    top_k: int = 10                       # candidates recorded per row and scenario (>= dropdown size)
    scheme: str = "logit"                 # 'logit' (adaptive log-odds bins) | 'percent' (1% buckets)
    min_bin_count: int = 200              # 'logit': candidates per merged bin
    min_scenario_count: int = 1000        # fewer candidates -> fall back to a pooled table
    clip_low: float = 0.0001
    clip_high: float = 0.9999
    fit_temperature: bool = True          # False: T = 1 everywhere (lookup tables only)
    temperature_max_rows: int = 200_000   # OOF rows sampled to fit the temperatures
    temperature_max_group: int = 256      # softmax groups larger than this keep their top classes + the truth
    seed: int = 0
    batch_size: int = 1000
    keep_fold_models: bool = True         # without saved fold models: False = refit each fold for pass 2
                                          # (less memory, ~2x fitting time)
    rankings: Tuple[str, ...] = RANKINGS  # which dropdown rankings to calibrate

    def __post_init__(self):
        if self.scheme not in ("logit", "percent"):
            raise ValueError("scheme must be 'logit' or 'percent'")
        if not 0 < self.clip_low < self.clip_high < 1:
            raise ValueError("need 0 < clip_low < clip_high < 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.rankings = tuple(self.rankings)
        bad = [r for r in self.rankings if r not in RANKINGS]
        if bad or not self.rankings:
            raise ValueError(f"rankings must be a non-empty subset of {RANKINGS}, got {self.rankings}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rankings"] = list(self.rankings)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationSettings":
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        d.setdefault("rankings", ("total",))     # version-1 files calibrated 'total' only
        return cls(**d)


def scenarios(n_levels: int, rankings: Sequence[str] = ("total",)) -> List[Tuple[str, int, int, str]]:
    """Every (ranking, target depth t, fixed levels c, kind), depths 0-based, c <= t.

    'best_path' has no deepest-level scenarios: there it equals 'total'.
    """
    out = []
    for r in rankings:
        for t in range(n_levels):
            if r == "best_path" and t == n_levels - 1:
                continue
            for c in range(t + 1):
                out += [(r, t, 0, "none")] if c == 0 else [(r, t, c, "confirmed"), (r, t, c, "corrected")]
    return out


def scenario_name(t: int, c: int, kind: str, ranking: str = "total") -> str:
    """Readable name, e.g. 'L3 | L1-L2 fixed | corrected' (kind 'any' = both kinds pooled);
    best-path scenarios end in ' | best path'."""
    if c == 0:
        name = f"L{t + 1} | nothing fixed"
    else:
        fixed = "L1" if c == 1 else f"L1-L{c}"
        name = f"L{t + 1} | {fixed} fixed | {kind}"
    return name if ranking == "total" else f"{name} | best path"


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _P_EPS, 1 - _P_EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


PERCENT_EDGES = _logit(np.arange(1, 100) / 100.0)   # logits of 1%, 2%, ..., 99%


def default_edges() -> np.ndarray:
    """Fine log-odds bin edges: a 0.05 grid on [-14, 14] plus every 1% boundary.

    Including the 1% boundaries means each fine bin lies inside one 1% bucket,
    so 'percent' tables and 1%-bucket diagnostics are exact merges of fine bins.
    """
    grid = np.round(np.arange(-14.0, 14.0 + 1e-9, 0.05), 6)
    return np.unique(np.concatenate([grid, PERCENT_EDGES]))


def _bin_index(edges: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, logits, side="right")   # 0 .. len(edges)


def _percent_bucket_of_bins(edges: np.ndarray) -> np.ndarray:
    lower = np.concatenate([[-np.inf], edges])            # lower edge of every fine bin
    return np.searchsorted(PERCENT_EDGES, lower, side="right")   # 0 .. 99


def _percent_bucket(p: np.ndarray) -> np.ndarray:
    return np.searchsorted(PERCENT_EDGES, _logit(p), side="right")


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [wsum], [c]
    return np.repeat(vals, cnt)


@dataclass
class LookupTable:
    """Score -> calibrated probability for one scenario.

    centers: mean log-odds of the tempered scores in each bucket (increasing)
    values:  monotone, clipped observed accuracy of each bucket
    """

    centers: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    positives: np.ndarray
    n: int

    def __call__(self, p) -> np.ndarray:
        x = _logit(p)
        if len(self.centers) == 1:
            return np.full(np.shape(x), float(self.values[0]))
        return np.interp(x, self.centers, self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {"n": int(self.n), "centers": [float(v) for v in self.centers],
                "values": [float(v) for v in self.values], "counts": [int(v) for v in self.counts],
                "positives": [int(v) for v in self.positives]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LookupTable":
        return cls(np.asarray(d["centers"], float), np.asarray(d["values"], float),
                   np.asarray(d["counts"], np.int64), np.asarray(d["positives"], np.int64), int(d["n"]))


def build_table(n: np.ndarray, pos: np.ndarray, sum_logit: np.ndarray, edges: np.ndarray,
                settings: CalibrationSettings) -> Optional[LookupTable]:
    """Lookup table from one scenario's fine histogram (arrays over fine bins)."""
    n = np.asarray(n, dtype=np.float64)
    total = n.sum()
    if total <= 0:
        return None
    if settings.scheme == "percent":
        bucket = _percent_bucket_of_bins(edges)
        gn = np.bincount(bucket, weights=n, minlength=100)
        gp = np.bincount(bucket, weights=pos, minlength=100)
        gl = np.bincount(bucket, weights=sum_logit, minlength=100)
    else:
        gn, gp, gl = [], [], []
        an = ap = al = 0.0
        for i in np.flatnonzero(n > 0):
            an += n[i]; ap += pos[i]; al += sum_logit[i]
            if an >= settings.min_bin_count:
                gn.append(an); gp.append(ap); gl.append(al)
                an = ap = al = 0.0
        if an > 0:
            if gn:   # leftover tail joins the last bin
                gn[-1] += an; gp[-1] += ap; gl[-1] += al
            else:
                gn, gp, gl = [an], [ap], [al]
        gn, gp, gl = np.asarray(gn), np.asarray(gp), np.asarray(gl)
    keep = gn > 0
    gn, gp, gl = gn[keep], gp[keep], gl[keep]
    centers = gl / gn
    order = np.argsort(centers, kind="mergesort")
    gn, gp, centers = gn[order], gp[order], centers[order]
    values = np.clip(_pav(gp / gn, gn), settings.clip_low, settings.clip_high)
    return LookupTable(centers, values, gn.astype(np.int64), np.rint(gp).astype(np.int64), int(round(total)))


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

TableKey = Tuple[str, int, int, str]     # (ranking, target depth, fixed levels, kind)


class Calibrator:
    """Per-level temperatures + per-scenario lookup tables (see module docstring)."""

    def __init__(self, level_columns: Sequence[str], prediction_mode: str, hyperparameters: Dict[str, Any],
                 temperatures: Sequence[float], tables: Dict[TableKey, LookupTable],
                 settings: CalibrationSettings, provenance: Optional[Dict[str, Any]] = None,
                 temperature_fit: Optional[List[Dict[str, Any]]] = None):
        self.level_columns = list(level_columns)
        self.prediction_mode = prediction_mode
        self.hyperparameters = dict(hyperparameters)
        self.temperatures = [float(t) for t in temperatures]
        self.tables = dict(tables)
        self.settings = settings
        self.provenance = provenance or {}
        self.temperature_fit = temperature_fit or []

    @property
    def n_levels(self) -> int:
        return len(self.level_columns)

    @property
    def rankings(self) -> Tuple[str, ...]:
        """Rankings this calibrator has tables for."""
        return tuple(r for r in RANKINGS if any(k[0] == r for k in self.tables))

    def has_ranking(self, ranking: str) -> bool:
        return ranking in self.rankings

    # -- lookup ---------------------------------------------------------------
    def table_for(self, target_level: int, n_fixed: int, kind: str,
                  ranking: str = "total") -> Tuple[Optional[LookupTable], str]:
        """The table used for a scenario and which one it is (after fallbacks)."""
        if ranking not in RANKINGS:
            raise ValueError(f"ranking must be one of {RANKINGS}")
        if ranking == "best_path" and target_level == self.n_levels - 1:
            ranking = "total"          # deepest level: one path per class, the two rankings coincide
        if n_fixed == 0:
            kind = "none"
        chain = [(ranking, target_level, n_fixed, kind)]
        if n_fixed > 0:
            chain += [(ranking, target_level, n_fixed, POOLED), (ranking, target_level, 0, "none")]
        for key in chain:
            tab = self.tables.get(key)
            if tab is not None:
                return tab, scenario_name(key[1], key[2], key[3], key[0])
        return None, "tempered score (no table)"

    def calibrate(self, tempered, target_level: int, n_fixed: int = 0, kind: str = "none",
                  ranking: str = "total") -> np.ndarray:
        """Calibrated probability for tempered scores in one scenario.

        target_level and n_fixed are 0-based depth / number of fixed levels;
        ranking is how the scores were computed ('total' or 'best_path').
        """
        tab, _ = self.table_for(target_level, n_fixed, kind, ranking)
        p = np.asarray(tempered, dtype=np.float64)
        if tab is None:
            return np.clip(p, self.settings.clip_low, self.settings.clip_high)
        return tab(p)

    # -- compatibility --------------------------------------------------------
    def check_compatible(self, clf: HierarchicalNB, strict: bool = False) -> None:
        if list(clf.level_columns) != self.level_columns:
            raise ValueError(f"calibration levels {self.level_columns} != model levels {clf.level_columns}")
        if clf.hp.prediction_mode != self.prediction_mode:
            raise ValueError(f"calibration was fitted for prediction_mode={self.prediction_mode!r}, "
                             f"the model uses {clf.hp.prediction_mode!r}")
        from .config import HyperParams

        mine = HyperParams.from_dict(self.hyperparameters).effective()
        theirs = clf.hp.effective()
        diff = sorted(k for k in mine if mine[k] != theirs.get(k))
        if diff:
            msg = ("calibration was fitted with different hyperparameters: "
                   + ", ".join(f"{k}: {mine[k]!r} vs model {theirs.get(k)!r}" for k in diff))
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "level_columns": self.level_columns,
            "prediction_mode": self.prediction_mode,
            "hyperparameters": self.hyperparameters,
            "temperatures": self.temperatures,
            "settings": self.settings.to_dict(),
            "provenance": self.provenance,
            "temperature_fit": self.temperature_fit,
            "tables": [{"ranking": r, "target_level": t + 1, "n_fixed": c, "kind": k,
                        "scenario": scenario_name(t, c, k, r), **tab.to_dict()}
                       for (r, t, c, k), tab in sorted(self.tables.items(), key=lambda kv: _key_order(kv[0]))],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Calibrator":
        if d.get("format_version") not in (1, FORMAT_VERSION):
            raise ValueError(f"unsupported calibration format {d.get('format_version')!r}")
        tables = {(x.get("ranking", "total"), int(x["target_level"]) - 1, int(x["n_fixed"]), x["kind"]):
                  LookupTable.from_dict(x) for x in d["tables"]}
        return cls(d["level_columns"], d["prediction_mode"], d["hyperparameters"], d["temperatures"], tables,
                   CalibrationSettings.from_dict(d["settings"]), d.get("provenance"), d.get("temperature_fit"))

    def save(self, path) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / CALIBRATION_FILE
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path) -> "Calibrator":
        path = Path(path)
        if path.is_dir():
            path = path / CALIBRATION_FILE
        return cls.from_dict(json.loads(path.read_text()))

    # -- views ----------------------------------------------------------------
    def percent_table(self) -> pd.DataFrame:
        """Every table as 1% score buckets: score bucket -> calibrated probability."""
        mid = (np.arange(100) + 0.5) / 100.0
        rows = []
        for (r, t, c, k), tab in sorted(self.tables.items(), key=lambda kv: _key_order(kv[0])):
            cal = tab(mid)
            for i in range(100):
                rows.append({"ranking": r, "target_level": t + 1, "n_fixed": c, "kind": k, "score_from": i / 100,
                             "score_to": (i + 1) / 100, "calibrated_at_midpoint": cal[i]})
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        rows = []
        for (r, t, c, k), tab in sorted(self.tables.items(), key=lambda kv: _key_order(kv[0])):
            rows.append({"ranking": r, "target_level": t + 1, "n_fixed": c, "kind": k, "candidates": tab.n,
                         "buckets": len(tab.centers), "positives": int(tab.positives.sum())})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        temps = ", ".join(f"{t:.3g}" for t in self.temperatures)
        return (f"Calibrator({self.prediction_mode}, temperatures=[{temps}], {len(self.tables)} tables, "
                f"rankings={list(self.rankings)}, scheme={self.settings.scheme!r})")


def _key_order(key: TableKey):
    r, t, c, k = key
    return (RANKINGS.index(r), t, c, (KINDS + (POOLED,)).index(k))


def build_calibrator(hist: Dict[str, np.ndarray], edges: np.ndarray, scen: Sequence[TableKey],
                     temperatures: Sequence[float], settings: CalibrationSettings, level_columns: Sequence[str],
                     prediction_mode: str, hyperparameters: Dict[str, Any], folds: Optional[Sequence[int]] = None,
                     provenance=None, temperature_fit=None) -> Calibrator:
    """Calibrator from tempered-score histograms (shape scenarios x folds x bins).

    folds: use only these folds (cross-fitting); None = all.
    """
    sel = slice(None) if folds is None else list(folds)
    n = hist["n"][:, sel].sum(axis=1)
    pos = hist["pos"][:, sel].sum(axis=1)
    sl = hist["sl"][:, sel].sum(axis=1)
    tables: Dict[TableKey, LookupTable] = {}
    pooled: Dict[Tuple[str, int, int], List[int]] = {}
    for i, (r, t, c, k) in enumerate(scen):
        if c > 0:
            pooled.setdefault((r, t, c), []).append(i)
        if n[i].sum() >= settings.min_scenario_count:
            tab = build_table(n[i], pos[i], sl[i], edges, settings)
            if tab is not None:
                tables[(r, t, c, k)] = tab
    for (r, t, c), idx in pooled.items():
        if n[idx].sum() >= settings.min_scenario_count:
            tab = build_table(n[idx].sum(0), pos[idx].sum(0), sl[idx].sum(0), edges, settings)
            if tab is not None:
                tables[(r, t, c, POOLED)] = tab
    return Calibrator(level_columns, prediction_mode, hyperparameters, temperatures, tables, settings,
                      provenance, temperature_fit)


# ---------------------------------------------------------------------------
# Scope probabilities (shared with nbclassifier.recommend)
# ---------------------------------------------------------------------------

def path_probs(clf: HierarchicalNB, log_joint: np.ndarray, n_fixed: int = 0,
               nodes: Optional[np.ndarray] = None) -> np.ndarray:
    """Path probabilities (rows x paths), restricted to each row's selected subtree.

    log_joint: path log scores (any normalisation). n_fixed = 0: all paths.
    n_fixed = c >= 1: only paths through nodes[i] (a depth c-1 code), renormalised.
    """
    lj = log_joint
    if n_fixed > 0:
        inside = clf.path_codes_[n_fixed - 1][None, :] == np.asarray(nodes)[:, None]
        lj = np.where(inside, lj, -np.inf)
    return np.exp(_log_softmax(lj))


def level_marginals(clf: HierarchicalNB, probs: np.ndarray, depth: int) -> np.ndarray:
    """Summed path probability of every depth-`depth` class (rows x classes)."""
    mask = clf._structure()["masks"][depth]
    return np.ascontiguousarray(np.asarray(mask.T @ probs.T).T)


def level_best_paths(clf: HierarchicalNB, probs: np.ndarray, depth: int) -> np.ndarray:
    """Probability of the single best path through every depth-`depth` class (rows x classes)."""
    if depth == clf.n_levels - 1:
        return probs                          # deepest classes are the paths themselves
    order, starts, has = clf._structure()["path_runs"][depth]
    q = probs if order is None else probs[:, order]
    out = np.zeros((probs.shape[0], len(starts)))
    out[:, has] = np.maximum.reduceat(q, starts[has], axis=1)
    return out


def level_scores(clf: HierarchicalNB, probs: np.ndarray, depth: int, ranking: str = "total") -> np.ndarray:
    """Dropdown score of every depth-`depth` class under a ranking (rows x classes)."""
    if ranking == "total":
        return level_marginals(clf, probs, depth)
    if ranking == "best_path":
        return level_best_paths(clf, probs, depth)
    raise ValueError(f"ranking must be one of {RANKINGS}")


def top_k_in_scope(clf: HierarchicalNB, scores: np.ndarray, depth: int, k: int, n_fixed: int = 0,
                   nodes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(class codes, valid) of each row's top-k depth classes by score, best first.

    valid is False for slots beyond the size of the row's scope.
    """
    C = scores.shape[1]
    k = min(k, C)
    if n_fixed > 0:
        anc = clf._structure()["ancestors"][depth][n_fixed - 1]
        in_scope = anc[None, :] == np.asarray(nodes)[:, None]
        s = np.where(in_scope, scores, -np.inf)
    else:
        s = scores
    if k < C:
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    else:
        part = np.tile(np.arange(C), (s.shape[0], 1))
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1, kind="mergesort")
    cand = np.take_along_axis(part, order, axis=1)
    valid = np.isfinite(np.take_along_axis(s, cand, axis=1))
    return cand, valid


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------

def _gather_groups(jll: np.ndarray, true_codes: np.ndarray, starts: np.ndarray, sizes: np.ndarray,
                   group_of: np.ndarray, max_group: int):
    """Scores of each row's softmax group containing its true class.

    Returns (flat scores, group sizes, position of the truth in its group).
    Groups larger than max_group keep their top (max_group - 1) others + the truth.
    """
    g = group_of[true_codes]
    s0, sz = starts[g], sizes[g]
    out_vals, out_sizes, out_true = [], [], []
    small = sz <= max_group
    if small.any():
        rows = np.flatnonzero(small)
        szs = sz[small]
        rr = np.repeat(rows, szs)
        offs = np.arange(szs.sum()) - np.repeat(np.cumsum(szs) - szs, szs)
        cols = np.repeat(s0[small], szs) + offs
        out_vals.append(jll[rr, cols])
        out_sizes.append(szs)
        out_true.append(true_codes[small] - s0[small])
    for i in np.flatnonzero(~small):
        seg = jll[i, s0[i]:s0[i] + sz[i]]
        tp = true_codes[i] - s0[i]
        top = np.argpartition(-seg, max_group - 1)[:max_group]
        if tp not in top:
            top[np.argmin(seg[top])] = tp
        out_vals.append(seg[top])
        out_sizes.append(np.array([len(top)]))
        out_true.append(np.array([int(np.flatnonzero(top == tp)[0])]))
    if not out_vals:
        return np.empty(0), np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(out_vals), np.concatenate(out_sizes), np.concatenate(out_true)


def _nll(vals: np.ndarray, sizes: np.ndarray, true_pos: np.ndarray, T: float) -> float:
    starts = np.cumsum(sizes) - sizes
    z = vals / T
    m = np.maximum.reduceat(z, starts)
    lse = m + np.log(np.add.reduceat(np.exp(z - np.repeat(m, sizes)), starts))
    return float(np.mean(lse - z[starts + true_pos]))


def fit_temperature(vals: np.ndarray, sizes: np.ndarray, true_pos: np.ndarray) -> Tuple[float, float, float]:
    """(T, mean NLL at T=1, mean NLL at T) minimising the true-class NLL."""
    from scipy.optimize import minimize_scalar

    if len(sizes) == 0:
        return 1.0, float("nan"), float("nan")
    res = minimize_scalar(lambda lt: _nll(vals, sizes, true_pos, float(np.exp(lt))),
                          bounds=(np.log(1e-2), np.log(1e5)), method="bounded", options={"xatol": 1e-4})
    T = float(np.exp(res.x))
    return T, _nll(vals, sizes, true_pos, 1.0), _nll(vals, sizes, true_pos, T)


def _softmax_groups(clf: HierarchicalNB, d: int):
    """(starts, sizes, group of each class) of the softmax used at depth d."""
    C = len(clf.classes_[d])
    if clf.hp.prediction_mode == "conditional":
        starts, sizes = clf._structure()["groups"][d]
    else:
        starts, sizes = np.array([0]), np.array([C])
    group_of = np.repeat(np.arange(len(starts)), sizes)
    return starts, sizes, group_of


def _temperature_levels(clf: HierarchicalNB) -> List[int]:
    return [clf.n_levels - 1] if clf.hp.prediction_mode == "flattened" else list(range(clf.n_levels))


# ---------------------------------------------------------------------------
# Fitting from an evaluation
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    calibrator: Calibrator
    folder: Path
    metrics: pd.DataFrame
    reliability: pd.DataFrame
    rank_accuracy: pd.DataFrame
    temperature_fit: pd.DataFrame
    extra: Dict[str, Any] = field(default_factory=dict)


def _new_hist(S: int, F: int, B: int) -> Dict[str, np.ndarray]:
    return {k: np.zeros((S, F, B)) for k in ("n", "pos", "sp", "sl")}


def _accumulate(hist, s: int, f: int, p: np.ndarray, y: np.ndarray, edges: np.ndarray) -> None:
    if p.size == 0:
        return
    x = _logit(p)
    b = _bin_index(edges, x)
    B = hist["n"].shape[2]
    hist["n"][s, f] += np.bincount(b, minlength=B)
    hist["pos"][s, f] += np.bincount(b, weights=y.astype(np.float64), minlength=B)
    hist["sp"][s, f] += np.bincount(b, weights=p, minlength=B)
    hist["sl"][s, f] += np.bincount(b, weights=x, minlength=B)


def fit_from_evaluation(ev, candidate: Optional[int] = None, settings: Optional[CalibrationSettings] = None,
                        folder: Optional[Path] = None, selection_metric: str = "L4 Accuracy",
                        verbose: bool = True) -> CalibrationResult:
    """Fit temperatures and lookup tables from OOF predictions of an Evaluation.

    ev:        crossval.Evaluation (its saved dataset and splits are used)
    candidate: index of the candidate whose hyperparameters to calibrate
               (None = ev.best(selection_metric)); use the hyperparameters you
               deploy in notebook 03.
    """
    from .data import filter_rare_paths, write_json

    settings = settings or CalibrationSettings()
    if candidate is None:
        candidate, _, _ = ev.best(selection_metric)
    hp = ev.candidates[candidate]
    levels = ev.cols.level_columns
    L = len(levels)
    folder = Path(folder or ev.folder / "calibration" / f"candidate_{candidate:03d}")
    folder.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    if ev.cv.cv_type != "oof":
        say("note: repeated holdout splits overlap, so some rows contribute more than once")

    data = ev.data
    splits = ev._splits(data)
    F = len(splits)
    N = int(sum(int(m.sum()) for _, m in splits))
    rng = np.random.default_rng(settings.seed)
    t_frac = min(1.0, settings.temperature_max_rows / max(N, 1))

    # ---- pass 1: fit fold models, collect temperature data -----------------
    say(f"[calibration] candidate {candidate}: {hp.prediction_mode} mode, {F} {ev.split_col}s, {N:,} OOF rows, "
        f"rankings {list(settings.rankings)}")
    loader = getattr(ev, "load_fold_model", None)

    def fit_fold(split, test):
        """(model, X held-out, true codes, training info, n held-out, source)."""
        held = data.loc[test].reset_index(drop=True)
        saved = loader(hp, split) if loader is not None else None
        if saved is not None:
            clf, info = saved
            source = "saved fold model"
        else:
            train, info = filter_rare_paths(data.loc[~test], levels, hp.rare_path_threshold)
            clf = HierarchicalNB(levels, hp).fit(train[ev.cols.cleaned_col], train[levels])
            del train
            source = "refitted"
        X = _as_csr(clf.transform(held[ev.cols.cleaned_col]))
        return clf, X, clf.encode_paths(held[levels]), info, len(held), source

    fold_cache = []
    sources = []
    t_data = {d: ([], [], []) for d in range(L)}
    for split, test in splits:
        t0 = time.time()
        clf, X, codes, info, n_held, source = fit_fold(split, test)
        sources.append(source)
        # a saved fold model is cheap to reload, so only refitted ones are worth keeping in memory
        fold_cache.append((clf, X, codes) if settings.keep_fold_models and source == "refitted" else None)
        if settings.fit_temperature:
            take = rng.random(n_held) < t_frac
            for s in range(0, n_held, settings.batch_size):
                e = min(s + settings.batch_size, n_held)
                rows = np.flatnonzero(take[s:e])
                if not len(rows):
                    continue
                jlls = clf.level_jll(X[s:e][rows])
                for d in _temperature_levels(clf):
                    tc = codes[s:e][rows, d]
                    ok = tc >= 0
                    if not ok.any():
                        continue
                    starts, sizes, group_of = _softmax_groups(clf, d)
                    v, sz, tp = _gather_groups(jlls[d][ok], tc[ok], starts, sizes, group_of,
                                               settings.temperature_max_group)
                    informative = sz > 1
                    if not informative.all():   # single-child groups say nothing about T
                        keep_vals = np.repeat(informative, sz)
                        v, sz, tp = v[keep_vals], sz[informative], tp[informative]
                    t_data[d][0].append(v); t_data[d][1].append(sz); t_data[d][2].append(tp)
                del jlls
        unseen = [(codes[:, d] < 0).mean() for d in range(L)]
        if not settings.keep_fold_models:
            del clf, X, codes
        say(f"  {ev.split_col} {split}: {source} ({info['rows_after']:,} training rows), scored {n_held:,} held-out "
            f"rows ({time.time() - t0:.0f}s); held-out paths unseen in training: "
            + " ".join(f"L{d + 1}={u:.2%}" for d, u in enumerate(unseen)))

    # ---- temperatures ---------------------------------------------------------
    temps = [1.0] * L
    t_rows = []
    for d in range(L):
        if not t_data[d][0]:
            continue
        v = np.concatenate(t_data[d][0]); sz = np.concatenate(t_data[d][1]); tp = np.concatenate(t_data[d][2])
        T, before, after = fit_temperature(v, sz, tp)
        temps[d] = T
        t_rows.append({"level": levels[d], "depth": d + 1, "temperature": T, "groups_used": int(len(sz)),
                       "nll_raw": before, "nll_tempered": after})
    t_fit = pd.DataFrame(t_rows)
    if settings.fit_temperature:
        say("[calibration] temperatures: " + ", ".join(f"L{r['depth']}={r['temperature']:.3g} "
            f"(log-loss {r['nll_raw']:.4f} -> {r['nll_tempered']:.4f})" for r in t_rows))
    del t_data

    # ---- pass 2: histograms ----------------------------------------------------
    rankings = tuple(settings.rankings)
    scen = scenarios(L, rankings)
    sidx = {sc: i for i, sc in enumerate(scen)}
    S, K = len(scen), settings.top_k
    edges = default_edges()
    B = len(edges) + 1
    H, Hraw = _new_hist(S, F, B), _new_hist(S, F, B)
    R = {k: np.zeros((S, K)) for k in ("n", "pos", "sp")}
    for f, (split, test) in enumerate(splits):
        t0 = time.time()
        clf, X, codes = fold_cache[f] if fold_cache[f] is not None else fit_fold(split, test)[:3]
        fold_cache[f] = None
        for s in range(0, X.shape[0], settings.batch_size):
            e = min(s + settings.batch_size, X.shape[0])
            cb = codes[s:e]
            jlls = clf.level_jll(X[s:e])
            lj_raw = _log_softmax(clf.joint_log_from_jll(jlls, None))
            lj_tmp = _log_softmax(clf.joint_log_from_jll(jlls, temps)) if settings.fit_temperature else lj_raw
            del jlls
            p0 = np.exp(lj_tmp)
            # each ranking's own top pick per level decides confirmed / corrected
            top0 = {r: [level_scores(clf, p0, d, r).argmax(axis=1) for d in range(L)] for r in rankings}
            for c in range(L):
                if c == 0:
                    rows = np.arange(e - s)
                    nodes = None
                    p_t = p0
                    p_r = np.exp(lj_raw)
                else:
                    rows = np.flatnonzero(cb[:, c - 1] >= 0)
                    if not len(rows):
                        continue
                    nodes = cb[rows, c - 1]
                    p_t = path_probs(clf, lj_tmp[rows], c, nodes)
                    p_r = path_probs(clf, lj_raw[rows], c, nodes)
                for r in rankings:
                    if c == 0:
                        kinds = np.zeros(len(rows), dtype=np.int64)                   # 0 = none
                    else:
                        kinds = np.where(top0[r][c - 1][rows] == nodes, 1, 2)        # 1 confirmed, 2 corrected
                    for t in range(c, L):
                        if r == "best_path" and t == L - 1:
                            continue                                                 # identical to 'total'
                        mt = level_scores(clf, p_t, t, r)
                        mr = level_scores(clf, p_r, t, r)
                        cand, valid = top_k_in_scope(clf, mt, t, K, c, nodes)
                        vt = np.take_along_axis(mt, cand, axis=1)
                        vr = np.take_along_axis(mr, cand, axis=1)
                        y = cand == cb[rows, t][:, None]
                        kk = cand.shape[1]
                        for kind_id in np.unique(kinds):
                            si = sidx[(r, t, c, KINDS[kind_id])]
                            rsel = kinds == kind_id
                            vm = valid[rsel]
                            _accumulate(H, si, f, vt[rsel][vm], y[rsel][vm], edges)
                            _accumulate(Hraw, si, f, vr[rsel][vm], y[rsel][vm], edges)
                            R["n"][si, :kk] += vm.sum(axis=0)
                            R["pos"][si, :kk] += (y[rsel] & vm).sum(axis=0)
                            R["sp"][si, :kk] += np.where(vm, vt[rsel], 0.0).sum(axis=0)
            del lj_raw, lj_tmp, p0
        say(f"  {ev.split_col} {split}: candidate histograms done ({time.time() - t0:.0f}s)")
        del clf, X, codes

    provenance = {
        "evaluation": ev.folder.name, "candidate": int(candidate), "cv": ev.cv.to_dict(),
        "n_oof_rows": N, "sample_frac": ev.config.get("sample_frac"),
        "data_source": ev.config.get("data_source"), "cleaning": ev.config.get("cleaning"),
        "created": time.strftime("%Y%m%d_%H%M%S"), "fold_models": sources,
    }
    cal = build_calibrator(H, edges, scen, temps, settings, levels, hp.prediction_mode, hp.to_dict(),
                           provenance=provenance, temperature_fit=t_rows)

    # ---- diagnostics (cross-fitted) ---------------------------------------------
    metrics, reliability = cross_fitted_diagnostics(H, Hraw, edges, scen, temps, settings, levels,
                                                    hp.prediction_mode, hp.to_dict())
    rank_acc = _rank_table(scen, R)

    # ---- save -------------------------------------------------------------------
    cal.save(folder / CALIBRATION_FILE)
    np.savez_compressed(folder / "histograms.npz", edges=edges, temperatures=np.asarray(temps),
                        scenarios=np.array(["|".join(map(str, sc)) for sc in scen]),
                        **{f"tempered_{k}": v for k, v in H.items()}, **{f"raw_{k}": v for k, v in Hraw.items()},
                        **{f"rank_{k}": v for k, v in R.items()})
    t_fit.to_csv(folder / "temperature_fit.csv", index=False)
    cal.percent_table().to_csv(folder / "lookup_tables.csv", index=False)
    metrics.to_csv(folder / "metrics.csv", index=False)
    reliability.to_csv(folder / "reliability.csv", index=False)
    rank_acc.to_csv(folder / "rank_accuracy.csv", index=False)
    write_json(folder / "settings.json", {"settings": settings.to_dict(), "provenance": provenance})
    say(f"[calibration] {len(cal.tables)} lookup tables -> {folder}")
    return CalibrationResult(cal, folder, metrics, reliability, rank_acc, t_fit)


def _rank_table(scen, R) -> pd.DataFrame:
    rows = []
    for i, (rk, t, c, k) in enumerate(scen):
        for r in range(R["n"].shape[1]):
            n = R["n"][i, r]
            if n > 0:
                rows.append({"ranking": rk, "scenario": scenario_name(t, c, k, rk), "target_level": t + 1,
                             "n_fixed": c, "kind": k, "rank": r + 1, "n": int(n), "accuracy": R["pos"][i, r] / n,
                             "mean_tempered_score": R["sp"][i, r] / n})
    return pd.DataFrame(rows)


def _parse_scenarios(names) -> List[TableKey]:
    out = []
    for name in names:
        parts = str(name).split("|")
        if len(parts) == 3:                      # version-1 histograms: 'total' only
            parts = ["total"] + parts
        out.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
    return out


def rebuild_from_histograms(folder, settings: Optional[CalibrationSettings] = None) -> CalibrationResult:
    """Re-build tables and diagnostics from a saved histograms.npz with new
    table settings (scheme, min_bin_count, min_scenario_count, clip). Top-k and
    temperatures are fixed by the original fit."""
    folder = Path(folder)
    old = Calibrator.load(folder / CALIBRATION_FILE)
    settings = settings or old.settings
    settings.top_k, settings.fit_temperature = old.settings.top_k, old.settings.fit_temperature
    z = np.load(folder / "histograms.npz")
    scen = _parse_scenarios(z["scenarios"])
    settings.rankings = tuple(dict.fromkeys(sc[0] for sc in scen))
    H = {k: z[f"tempered_{k}"] for k in ("n", "pos", "sp", "sl")}
    Hraw = {k: z[f"raw_{k}"] for k in ("n", "pos", "sp", "sl")}
    cal = build_calibrator(H, z["edges"], scen, old.temperatures, settings, old.level_columns,
                           old.prediction_mode, old.hyperparameters, provenance=old.provenance,
                           temperature_fit=old.temperature_fit)
    metrics, reliability = cross_fitted_diagnostics(H, Hraw, z["edges"], scen, old.temperatures, settings,
                                                    old.level_columns, old.prediction_mode, old.hyperparameters)
    cal.save(folder / CALIBRATION_FILE)
    cal.percent_table().to_csv(folder / "lookup_tables.csv", index=False)
    metrics.to_csv(folder / "metrics.csv", index=False)
    reliability.to_csv(folder / "reliability.csv", index=False)
    rank = _rank_table(scen, {k: z[f"rank_{k}"] for k in ("n", "pos", "sp")}) if "rank_n" in z else pd.DataFrame()
    tfit = pd.read_csv(folder / "temperature_fit.csv") if (folder / "temperature_fit.csv").is_file() else pd.DataFrame()
    return CalibrationResult(cal, folder, metrics, reliability, rank, tfit)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _binned_metrics(n: np.ndarray, pos: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """ECE (1% buckets of the predictor), Brier and log-loss from binned counts.

    Each fine bin is treated as if every candidate in it had the bin's mean
    score, a close approximation at a 0.05 log-odds bin width.
    """
    keep = n > 0
    n, pos, p = n[keep], pos[keep], np.clip(p[keep], 1e-15, 1 - 1e-15)
    N = n.sum()
    if N == 0:
        return {"candidates": 0}
    b = np.minimum((p * 100).astype(np.int64), 99)
    bn = np.bincount(b, weights=n, minlength=100)
    bpos = np.bincount(b, weights=pos, minlength=100)
    bp = np.bincount(b, weights=n * p, minlength=100)
    ece = np.abs(bpos - bp).sum() / N
    brier = (pos * (1 - p) ** 2 + (n - pos) * p ** 2).sum() / N
    logloss = -(pos * np.log(p) + (n - pos) * np.log1p(-p)).sum() / N
    return {"candidates": int(round(N)), "observed_rate": pos.sum() / N, "mean_prediction": (n * p).sum() / N,
            "ece": ece, "brier": brier, "log_loss": logloss}


def _reliability_rows(name, predictor, n, pos, p):
    keep = n > 0
    n, pos, p = n[keep], pos[keep], p[keep]
    b = np.minimum((np.clip(p, 0, 1) * 100).astype(np.int64), 99)
    bn = np.bincount(b, weights=n, minlength=100)
    bpos = np.bincount(b, weights=pos, minlength=100)
    bp = np.bincount(b, weights=n * p, minlength=100)
    return [{"scenario": name, "predictor": predictor, "bucket_from": i / 100, "bucket_to": (i + 1) / 100,
             "n": int(round(bn[i])), "mean_prediction": bp[i] / bn[i], "observed": bpos[i] / bn[i]}
            for i in np.flatnonzero(bn > 0)]


def cross_fitted_diagnostics(H, Hraw, edges, scen, temps, settings, levels, mode, hyper):
    """Per-scenario metrics and reliability rows for raw, tempered and calibrated
    scores; the calibrated scores for split f come from tables fitted without f."""
    F = H["n"].shape[1]
    cal_by_fold = [build_calibrator(H, edges, scen, temps, settings, levels, mode, hyper,
                                    folds=[g for g in range(F) if g != f]) for f in range(F)] if F > 1 else None
    rows, rel = [], []
    for i, (rk, t, c, k) in enumerate(scen):
        name = scenario_name(t, c, k, rk)
        n_r, pos_r = Hraw["n"][i].sum(0), Hraw["pos"][i].sum(0)
        p_r = np.divide(Hraw["sp"][i].sum(0), n_r, out=np.zeros_like(n_r), where=n_r > 0)
        n_t, pos_t = H["n"][i].sum(0), H["pos"][i].sum(0)
        p_t = np.divide(H["sp"][i].sum(0), n_t, out=np.zeros_like(n_t), where=n_t > 0)
        parts = [("raw", n_r, pos_r, p_r), ("tempered", n_t, pos_t, p_t)]
        if cal_by_fold is not None:
            cn, cpos, cp = [], [], []
            for f in range(F):
                nf = H["n"][i, f]
                keep = nf > 0
                if not keep.any():
                    continue
                centers = H["sl"][i, f][keep] / nf[keep]
                cn.append(nf[keep]); cpos.append(H["pos"][i, f][keep])
                cp.append(cal_by_fold[f].calibrate(_sigmoid(centers), t, c, k, rk))
            if cn:
                parts.append(("calibrated (cross-fitted)", np.concatenate(cn), np.concatenate(cpos),
                              np.concatenate(cp)))
        for predictor, n, pos, p in parts:
            m = _binned_metrics(n, pos, p)
            if m.get("candidates", 0) == 0:
                continue
            rows.append({"ranking": rk, "scenario": name, "target_level": t + 1, "n_fixed": c, "kind": k,
                         "predictor": predictor, **m})
            rel += [{"ranking": rk, **row} for row in _reliability_rows(name, predictor, n, pos, p)]
    return pd.DataFrame(rows), pd.DataFrame(rel)


def metrics_wide(metrics: pd.DataFrame, metric: str = "ece", ranking: Optional[str] = None) -> pd.DataFrame:
    """One row per scenario, one column per predictor (for display); ranking filters."""
    if metrics.empty:
        return metrics
    if "ranking" not in metrics.columns:
        metrics = metrics.assign(ranking="total")
    if ranking is not None:
        metrics = metrics[metrics["ranking"] == ranking]
    idx = ["ranking", "target_level", "n_fixed", "kind", "scenario"]
    w = metrics.pivot_table(index=idx, columns="predictor", values=metric, sort=False)
    n = metrics[metrics["predictor"] == "tempered"].set_index(idx)
    w.insert(0, "candidates", n["candidates"])
    w = w.reset_index()
    w["_r"] = w["ranking"].map(RANKINGS.index)
    return w.sort_values(["_r", "target_level", "n_fixed", "kind"]).drop(columns="_r").reset_index(drop=True)


def plot_reliability(reliability: pd.DataFrame, min_count: int = 50, coarse: int = 5, ranking: str = "total"):
    """Grid of reliability diagrams for one ranking: rows = target level, columns = conditioning.

    Buckets are merged to `coarse`% width for readability; buckets with fewer
    than min_count candidates are not drawn.
    """
    import matplotlib.pyplot as plt

    if "ranking" in reliability.columns:
        reliability = reliability[reliability["ranking"] == ranking]
    if reliability.empty:
        raise ValueError(f"no reliability rows for ranking {ranking!r}")
    rel = reliability.copy()
    suffix = " | best path"
    rel["scenario"] = rel["scenario"].str.replace(suffix, "", regex=False)
    reliability = rel
    rel["coarse"] = np.minimum((rel["bucket_from"] * 100 // coarse).astype(int), 100 // coarse - 1)
    rel["wp"] = rel["mean_prediction"] * rel["n"]
    rel["wo"] = rel["observed"] * rel["n"]
    g = rel.groupby(["scenario", "predictor", "coarse"], sort=False)[["n", "wp", "wo"]].sum().reset_index()
    g["mean_prediction"], g["observed"] = g["wp"] / g["n"], g["wo"] / g["n"]
    names = list(dict.fromkeys(reliability["scenario"]))
    targets = sorted({int(s.split(" |")[0][1:]) for s in names})
    cols = ["nothing fixed"] + [x for x in dict.fromkeys(" | ".join(s.split(" | ")[1:]) for s in names)
                                if x != "nothing fixed"]
    fig, axes = plt.subplots(len(targets), len(cols), figsize=(2.4 * len(cols), 2.4 * len(targets)),
                             squeeze=False, sharex=True, sharey=True)
    colors = {"raw": "#9a9a9a", "tempered": "#2a6fb0", "calibrated (cross-fitted)": "#d4572a"}
    for ax in axes.ravel():
        ax.set_visible(False)
    for s in names:
        t = int(s.split(" |")[0][1:])
        col = " | ".join(s.split(" | ")[1:])
        ax = axes[targets.index(t), cols.index(col)]
        ax.set_visible(True)
        ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8)
        for pred, color in colors.items():
            d = g[(g["scenario"] == s) & (g["predictor"] == pred) & (g["n"] >= min_count)]
            if len(d):
                ax.plot(d["mean_prediction"], d["observed"], marker=".", ms=3, lw=1, color=color, label=pred)
        ax.set_title(s, fontsize=7)
        ax.tick_params(labelsize=6)
    handles, labels = [], []
    for ax in axes.ravel():
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False,
               title=f"ranking: {ranking}", title_fontsize=8)
    fig.supxlabel("predicted probability", fontsize=8)
    fig.supylabel("observed accuracy", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    return fig


def find_calibration(name: Optional[str] = None, candidate: Optional[int] = None) -> Path:
    """calibration.json inside outputs/evaluation_<ts>/calibration/candidate_<k>/.

    name: evaluation folder ('latest', a timestamp, a folder name) or a path to
    a candidate folder / calibration.json. candidate: None = the only (or the
    most recently written) calibrated candidate in that evaluation.
    """
    from . import paths

    p = Path(str(name)) if name is not None else None
    if p is not None and p.is_file():
        return p
    if p is not None and (p / CALIBRATION_FILE).is_file():
        return p / CALIBRATION_FILE
    ev_folder = paths.find_folder(name, paths.OUTPUTS_DIR, prefix="evaluation_")
    base = ev_folder / "calibration"
    if candidate is not None:
        f = base / f"candidate_{candidate:03d}" / CALIBRATION_FILE
        if not f.is_file():
            raise FileNotFoundError(f"no calibration for candidate {candidate} in {ev_folder}")
        return f
    found = sorted(base.glob(f"candidate_*/{CALIBRATION_FILE}"), key=lambda f: f.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no calibration in {ev_folder}; run the calibration cell of notebook 02")
    return found[-1]
