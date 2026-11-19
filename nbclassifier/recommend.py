"""Top-k category recommendations per level for the dropdown interface.

One dropdown per level. With nothing selected, each dropdown lists the k
level classes with the highest *marginal* probability (the summed score of
every valid path below the class), so the level-2 list is not forced onto
the single best path.

Rankings (`ranking=`):
    'total'      (default) each class's summed probability over every path below
                 it: "which category is most likely correct".
    'best_path'  each class's single most likely path: "which categories lie on
                 the most convincing complete categorisations". The same as
                 reading distinct classes off the ranked list of paths. Level 4
                 is identical under both.
Each ranking has its own calibration tables; 'confirmed' / 'corrected' is
judged against the ranking in use.

When a user confirms or selects a category at any level:
    - every level above it is fixed to its only valid ancestor (probability 1);
    - every level below it lists the k best descendants of the selection,
      scored P(class | selection, text): the path scores restricted to the
      selected subtree and renormalised.

Calibrated probabilities come from the model's Calibrator (see
nbclassifier.calibration): scores are tempered per level, then mapped through
the lookup table of the scenario (target level, how many levels are fixed,
and whether the selection confirmed the model's own top pick or corrected
it). Without a calibrator the 'calibrated' column is empty and the raw
scores are still returned.

    rec = Recommender.from_loaded(artifacts.load_model())
    rec.recommend_one("NITRILE GLOVE <SIZE>L", selections={"Level 1 ID": "L1-12783695"})

Descriptions must be cleaned with the model's cleaner first; recommend_one
and recommend(raw_texts=...) do that for you.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import warnings

from .calibration import RANKINGS, Calibrator, level_scores, path_probs, scenario_name, top_k_in_scope
from .model import HierarchicalNB, _as_csr, _log_softmax

Selection = Optional[Dict[Union[str, int], str]]


class Recommender:
    def __init__(self, clf: HierarchicalNB, calibrator: Optional[Calibrator] = None,
                 lookup: Optional[pd.DataFrame] = None, cleaner=None, text_col: Optional[str] = None,
                 cleaned_col: Optional[str] = None, k: int = 5, ranking: str = "total"):
        self.clf = clf
        self.calibrator = calibrator
        if calibrator is not None:
            calibrator.check_compatible(clf)
        self.ranking = self._check_ranking(ranking)
        self.cleaner = cleaner
        self.text_col, self.cleaned_col = text_col, cleaned_col
        self.k = int(k)
        self.levels = list(clf.level_columns)
        self._names: List[Optional[Dict[str, str]]] = [None] * len(self.levels)
        self.name_columns: List[Optional[str]] = [None] * len(self.levels)
        if lookup is not None:
            others = [c for c in lookup.columns if c not in self.levels]
            for d, lvl in enumerate(self.levels):
                # 'Level 2 ID' -> 'Level 2 Name' when present, else the d-th non-level column
                guess = lvl.replace(" ID", " Name")
                name = guess if guess in lookup.columns else (others[d] if d < len(others) else None)
                if name is not None and lvl in lookup.columns:
                    self.name_columns[d] = name
                    m = lookup[[lvl, name]].dropna().drop_duplicates(lvl)
                    self._names[d] = dict(zip(m[lvl].astype(str), m[name].astype(str)))

    @classmethod
    def from_loaded(cls, model, k: int = 5, ranking: str = "total") -> "Recommender":
        """From artifacts.load_model(...) (uses its calibration.json when present)."""
        tm = model.training_metadata or {}
        return cls(model.clf, getattr(model, "calibrator", None), model.lookup, model.cleaner,
                   tm.get("text_col", "FullDesc"), tm.get("cleaned_col", "CleanDesc"), k=k, ranking=ranking)

    def _check_ranking(self, ranking: str) -> str:
        if ranking not in RANKINGS:
            raise ValueError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
        cal = self.calibrator
        if cal is not None and ranking != "total" and not cal.has_ranking(ranking):
            warnings.warn(f"the calibration has no {ranking!r} tables (fitted before rankings existed, or "
                          "without that ranking): levels above the deepest show the clipped tempered score "
                          "instead of a calibrated probability. Re-run notebook 02's calibration to fix this.")
        return ranking

    # -- selections -------------------------------------------------------------
    def _level_index(self, key) -> int:
        if isinstance(key, (int, np.integer)):
            if not 1 <= key <= len(self.levels):
                raise KeyError(f"level number must be 1..{len(self.levels)}")
            return int(key) - 1
        if key in self.levels:
            return self.levels.index(key)
        raise KeyError(f"unknown level {key!r}; use one of {self.levels} or 1..{len(self.levels)}")

    def resolve(self, selections: Selection):
        """(n_fixed, node code at depth n_fixed-1, fixed labels) for a selection.

        selections: {level column or 1-based level number: label}. Any subset of
        levels may be given; they must lie on one valid path. The deepest
        selected level decides the scope; levels above it are implied.
        """
        if not selections:
            return 0, None, []
        st = self.clf._structure()
        sel = {self._level_index(k): str(v) for k, v in selections.items() if v is not None and str(v) != ""}
        if not sel:
            return 0, None, []
        mask = np.ones(self.clf.n_paths, dtype=bool)
        for d, v in sel.items():
            mask &= st["path_labels"][d].astype(str) == v
        if not mask.any():
            raise ValueError(f"no valid category path matches the selection {selections}")
        deepest = max(sel)
        nodes = np.unique(self.clf.path_codes_[deepest][mask])
        if len(nodes) > 1:
            raise ValueError(f"selection {selections} is ambiguous: {sel[deepest]!r} occurs under several "
                             "parents; also select a higher level")
        node = int(nodes[0])
        path = int(np.flatnonzero(mask)[0])
        fixed = [str(st["path_labels"][d][path]) for d in range(deepest + 1)]
        return deepest + 1, node, fixed

    # -- scoring ------------------------------------------------------------------
    def _name(self, d: int, label: str) -> Optional[str]:
        m = self._names[d]
        return None if m is None else m.get(str(label))

    def clean(self, raw_texts) -> pd.Series:
        """Clean raw descriptions with the model's own cleaner."""
        if self.cleaner is None:
            raise ValueError("no cleaner attached; pass cleaned texts")
        from .data import clean_dataframe

        col = self.text_col or "FullDesc"
        out = self.cleaned_col or "CleanDesc"
        df = pd.DataFrame({col: pd.Series(raw_texts, dtype=object).fillna("").astype(str).to_numpy()})
        cleaned, _ = clean_dataframe(df, self.cleaner, col, out, drop_empty=False)
        return cleaned[out].reset_index(drop=True)

    def recommend(self, texts=None, X=None, raw_texts=None, selections: Union[Selection, Sequence[Selection]] = None,
                  k: Optional[int] = None, selection_kind: Optional[str] = None,
                  batch_size: int = 1000, ranking: Optional[str] = None) -> pd.DataFrame:
        """Top-k per level for every row, as a long table.

        texts: cleaned descriptions (or X: vectorized; or raw_texts: cleaned here).
        selections: None, one dict applied to every row, or one dict (or None) per row.
        selection_kind: None = decide per row ('confirmed' when the selected
            node was the model's top pick at that level, else 'corrected');
            or force 'confirmed' / 'corrected' if the interface knows.
        ranking: 'total' or 'best_path' (None = the Recommender's default).

        Columns: row, level, depth, rank, label, [name], calibrated, score,
        raw_score, fixed, scenario, table, ranking. 'score' is the tempered model score
        used for ranking; 'calibrated' its calibrated probability; fixed levels
        have rank 1 and probability 1.
        """
        if selection_kind not in (None, "confirmed", "corrected"):
            raise ValueError("selection_kind must be None, 'confirmed' or 'corrected'")
        k = int(k or self.k)
        ranking = self.ranking if ranking is None else self._check_ranking(ranking)
        if raw_texts is not None:
            texts = self.clean(raw_texts)
        if X is None:
            X = self.clf.transform(texts)
        X = _as_csr(X)
        n = X.shape[0]
        if selections is None or isinstance(selections, dict):
            groups = {0: (selections, np.arange(n))}
        else:
            if len(selections) != n:
                raise ValueError(f"{len(selections)} selections for {n} rows")
            keyed: Dict[Any, List[int]] = {}
            first: Dict[Any, Selection] = {}
            for i, s in enumerate(selections):
                key = tuple(sorted((str(a), str(b)) for a, b in (s or {}).items()))
                keyed.setdefault(key, []).append(i)
                first.setdefault(key, s)
            groups = {key: (first[key], np.asarray(rows)) for key, rows in keyed.items()}
        parts = []
        for sel, rows in groups.values():
            n_fixed, node, fixed = self.resolve(sel)
            for s in range(0, len(rows), batch_size):
                r = rows[s:s + batch_size]
                parts.append(self._recommend_block(X[r], r, n_fixed, node, fixed, k, selection_kind, ranking))
        out = pd.concat(parts, ignore_index=True)
        out["ranking"] = ranking
        return out.sort_values(["row", "depth", "rank"], kind="mergesort").reset_index(drop=True)

    def _recommend_block(self, X, rows, n_fixed, node, fixed, k, selection_kind, ranking) -> pd.DataFrame:
        clf, cal, L = self.clf, self.calibrator, len(self.levels)
        temps = cal.temperatures if cal is not None else None
        st = clf._structure()
        jlls = clf.level_jll(X)
        lj_raw = _log_softmax(clf.joint_log_from_jll(jlls, None))
        lj_tmp = _log_softmax(clf.joint_log_from_jll(jlls, temps)) if temps is not None else lj_raw
        del jlls
        m = len(rows)
        nodes = np.full(m, node, dtype=np.int64) if n_fixed else None
        if n_fixed:
            if selection_kind is None:
                top = level_scores(clf, np.exp(lj_tmp), n_fixed - 1, ranking).argmax(axis=1)
                kinds = np.where(top == node, "confirmed", "corrected")
            else:
                kinds = np.full(m, selection_kind)
        else:
            kinds = np.full(m, "none")
        p_t = path_probs(clf, lj_tmp, n_fixed, nodes)
        p_r = path_probs(clf, lj_raw, n_fixed, nodes) if temps is not None else p_t
        frames = []
        for t, lvl in enumerate(self.levels):
            if t < n_fixed:
                frames.append(pd.DataFrame({
                    "row": rows, "level": lvl, "depth": t + 1, "rank": 1, "label": fixed[t],
                    "name": self._name(t, fixed[t]), "calibrated": 1.0, "score": 1.0, "raw_score": 1.0,
                    "fixed": True, "scenario": "selected" if t == n_fixed - 1 else "implied by selection",
                    "table": None}))
                continue
            mt = level_scores(clf, p_t, t, ranking)
            mr = level_scores(clf, p_r, t, ranking) if p_r is not p_t else mt
            cand, valid = top_k_in_scope(clf, mt, t, k, n_fixed, nodes)
            vt = np.take_along_axis(mt, cand, axis=1)
            vr = np.take_along_axis(mr, cand, axis=1)
            calv = np.full(vt.shape, np.nan)
            table = np.full(vt.shape, None, dtype=object)
            if cal is not None:
                for kind in np.unique(kinds):
                    rs = kinds == kind
                    calv[rs] = cal.calibrate(vt[rs], t, n_fixed, kind, ranking)
                    table[rs] = cal.table_for(t, n_fixed, kind, ranking)[1]
            labels = clf.classes_[t]["label"].to_numpy()[cand]
            kk = cand.shape[1]
            df = pd.DataFrame({
                "row": np.repeat(rows, kk), "level": lvl, "depth": t + 1,
                "rank": np.tile(np.arange(1, kk + 1), m), "label": labels.ravel().astype(str),
                "calibrated": calv.ravel(), "score": vt.ravel(), "raw_score": vr.ravel(), "fixed": False,
                "scenario": np.repeat([scenario_name(t, n_fixed, kd, ranking) for kd in kinds], kk),
                "table": table.ravel()})
            df.insert(5, "name", [self._name(t, x) for x in df["label"]])
            frames.append(df[valid.ravel()])
        return pd.concat(frames, ignore_index=True)

    def recommend_one(self, text: Optional[str] = None, selections: Selection = None, k: Optional[int] = None,
                      cleaned_text: Optional[str] = None, selection_kind: Optional[str] = None,
                      ranking: Optional[str] = None) -> Dict[str, Any]:
        """One description -> a JSON-ready dict for the four dropdowns.

        text is a raw description (cleaned with the model's cleaner); pass
        cleaned_text instead if it is already cleaned.
        """
        if cleaned_text is None:
            if text is None:
                raise ValueError("give text or cleaned_text")
            cleaned_text = self.clean([text]).iloc[0] if self.cleaner is not None else text
        df = self.recommend(texts=[cleaned_text], selections=selections, k=k, selection_kind=selection_kind,
                            ranking=ranking)
        out: Dict[str, Any] = {"cleaned_text": cleaned_text, "calibrated": self.calibrator is not None,
                               "ranking": df["ranking"].iloc[0], "selections": selections or {}, "levels": []}
        for t, lvl in enumerate(self.levels):
            d = df[df["depth"] == t + 1]
            out["levels"].append({
                "level": lvl, "fixed": bool(d["fixed"].any()),
                "scenario": d["scenario"].iloc[0] if len(d) else None,
                "options": [{"rank": int(r["rank"]), "label": r["label"], "name": r["name"],
                             "probability": None if pd.isna(r["calibrated"]) else round(float(r["calibrated"]), 6),
                             "score": round(float(r["score"]), 6)} for _, r in d.iterrows()],
            })
        return out
