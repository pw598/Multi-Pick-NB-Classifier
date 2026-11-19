"""Calibration, temperatures and the Recommender on a small synthetic hierarchy."""

import json

import numpy as np
import pandas as pd
import pytest

from nbclassifier import artifacts, calibration, crossval
from nbclassifier.calibration import (CalibrationSettings, Calibrator, build_table, default_edges, fit_temperature,
                                      level_marginals, path_probs, scenarios)
from nbclassifier.config import CVSettings, HyperParams
from nbclassifier.data import PreparedData
from nbclassifier.model import HierarchicalNB, _log_softmax
from nbclassifier.recommend import Recommender

LEVELS = ["Level 1 ID", "Level 2 ID", "Level 3 ID", "Level 4 ID"]
NAMES = ["Level 1 Name", "Level 2 Name", "Level 3 Name", "Level 4 Name"]


def synthetic(n=3000, seed=0, branching=(3, 3, 2, 3)):
    """Descriptions made of per-node keywords plus confusing words and noise."""
    rng = np.random.default_rng(seed)
    paths = [()]
    for b in branching:
        paths = [p + (i,) for p in paths for i in range(b)]
    words = {}

    def vocab(prefix):
        return words.setdefault(prefix, [f"W{'_'.join(map(str, prefix))}X{j}" for j in range(5)])

    pop = rng.dirichlet(np.ones(len(paths)))
    idx = rng.choice(len(paths), n, p=pop)
    rows = []
    for i, k in enumerate(idx):
        p = paths[k]
        toks = []
        for d in range(4):
            toks += list(rng.choice(vocab(p[: d + 1]), rng.integers(0, 3)))
            if rng.random() < 0.35:
                toks += list(rng.choice(vocab(paths[rng.integers(len(paths))][: d + 1]), 1))
        toks += [f"NOISE{rng.integers(200)}" for _ in range(rng.integers(1, 4))]
        ids = [f"L{d + 1}-" + "_".join(map(str, p[: d + 1])) for d in range(4)]
        text = " ".join(toks)
        rows.append({"ID": f"id{i}", "Vallen ID": f"v{i}", "FullDesc": text, "CleanDesc": text,
                     **dict(zip(LEVELS, ids)), **{nm: f"name {v}" for nm, v in zip(NAMES, ids)}})
    return pd.DataFrame(rows)


HP = dict(binary=True, use_idf=False, alpha=0.001, min_df=1, rare_path_threshold=3)


@pytest.fixture(scope="module")
def df():
    return synthetic()


@pytest.fixture(scope="module")
def clf(df):
    hp = HyperParams(prediction_mode="conditional", **HP)
    return HierarchicalNB(LEVELS, hp).fit(df["CleanDesc"], df[LEVELS])


@pytest.fixture(scope="module")
def fitted(df, tmp_path_factory):
    folder = tmp_path_factory.mktemp("evaluation")
    prep = PreparedData(df, cleaner=None, metadata={"source": {"type": "synthetic"}})
    cols = crossval.Columns("FullDesc", "CleanDesc", LEVELS, NAMES, ["ID", "Vallen ID"])
    ev = crossval.Evaluation.create(prep, [HyperParams(prediction_mode="conditional", **HP)],
                                    CVSettings(n_folds=3), cols, grid_search=False, folder=folder)
    settings = CalibrationSettings(min_bin_count=50, min_scenario_count=200, batch_size=400)
    res = calibration.fit_from_evaluation(ev, 0, settings, verbose=False)
    return ev, res


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

def test_encode_paths(clf, df):
    codes = clf.encode_paths(df[LEVELS].head(50))
    labels = clf.valid_paths()
    for i in range(50):
        for d in range(4):
            assert clf.classes_[d]["label"].iloc[codes[i, d]] == df[LEVELS[d]].iloc[i]
    unseen = pd.DataFrame([[df[LEVELS[0]].iloc[0], "nope", "x", "y"]], columns=LEVELS)
    assert clf.encode_paths(unseen).tolist() == [[clf.encode_paths(df[LEVELS].head(1))[0, 0], -1, -1, -1]]
    assert len(labels) == clf.n_paths


def test_temperature_one_is_identity_and_sums_to_one(clf, df):
    X = clf.transform(df["CleanDesc"].head(100))
    raw = clf.joint_log(X)
    np.testing.assert_allclose(clf.joint_log(X, [1, 1, 1, 1]), raw)
    hot = clf.joint_log(X, [1.5, 2, 3, 4])
    np.testing.assert_allclose(np.exp(hot).sum(axis=1), 1.0, rtol=1e-10)
    # the most likely child of every parent is unchanged by tempering
    jll = clf.level_jll(X)[1]
    starts, sizes = clf._structure()["groups"][1]
    for g, (s, z) in enumerate(zip(starts, sizes)):
        assert (np.argmax(jll[:, s:s + z], 1) == np.argmax(jll[:, s:s + z] / 3, 1)).all()
    with pytest.raises(ValueError):
        clf.joint_log(X, [1, 2])


def test_conditioning_matches_chain_rule(clf, df):
    """Restricting to a subtree and renormalising = the product of the conditionals below it."""
    X = clf.transform(df["CleanDesc"].head(30))
    temps = [1.2, 1.7, 2.5, 3.0]
    lj = clf.joint_log(X, temps)
    jlls = clf.level_jll(X)
    st = clf._structure()
    node = int(clf.path_codes_[1][0])   # a level-2 node
    p = path_probs(clf, lj, 2, np.full(30, node))
    lp = []
    for d in range(4):
        s, z = st["groups"][d]
        from nbclassifier.model import _segmented_log_softmax
        lp.append(_segmented_log_softmax(jlls[d] / temps[d], s, z))
    inside = clf.path_codes_[1] == node
    expect = np.zeros_like(p)
    expect[:, inside] = np.exp(lp[2][:, clf.path_codes_[2][inside]] + lp[3][:, clf.path_codes_[3][inside]])
    np.testing.assert_allclose(p, expect, atol=1e-12)
    m = level_marginals(clf, p, 1)
    np.testing.assert_allclose(m[:, node], 1.0)


def test_fit_temperature_recovers_known_value():
    rng = np.random.default_rng(1)
    n, C, T = 4000, 6, 3.0
    logits = rng.normal(0, 4, (n, C))
    probs = np.exp(_log_softmax(logits / T))
    y = np.array([rng.choice(C, p=pp) for pp in probs])
    fitted, before, after = fit_temperature(logits.ravel(), np.full(n, C), y)
    assert abs(fitted - T) / T < 0.1
    assert after < before


def test_build_table_monotone_and_clipped():
    edges = default_edges()
    B = len(edges) + 1
    rng = np.random.default_rng(0)
    n = rng.integers(0, 50, B).astype(float)
    pos = np.minimum(n, rng.integers(0, 50, B)).astype(float)   # non-monotone rates
    centers = np.concatenate([[-15], (edges[:-1] + edges[1:]) / 2, [15]])
    sl = n * centers
    for scheme in ("logit", "percent"):
        s = CalibrationSettings(scheme=scheme, min_bin_count=100)
        tab = build_table(n, pos, sl, edges, s)
        assert np.all(np.diff(tab.values) >= -1e-12)
        assert tab.values.min() >= s.clip_low and tab.values.max() <= s.clip_high
        out = tab(np.linspace(0, 1, 101))
        assert np.all(np.diff(out) >= -1e-12)
        if scheme == "percent":
            assert len(tab.centers) <= 100
    assert len(scenarios(4)) == 16


# ---------------------------------------------------------------------------
# fitting from OOF cross-validation
# ---------------------------------------------------------------------------

def test_fit_from_evaluation(fitted):
    ev, res = fitted
    cal = res.calibrator
    assert len(cal.temperatures) == 4 and all(t > 0 for t in cal.temperatures)
    tf = res.temperature_fit
    assert (tf["nll_tempered"] <= tf["nll_raw"] + 1e-12).all()
    assert ("total", 0, 0, "none") in cal.tables and ("total", 3, 0, "none") in cal.tables
    assert ("best_path", 0, 0, "none") in cal.tables
    assert not any(k[0] == "best_path" and k[1] == 3 for k in cal.tables)    # level 4 shares 'total'
    assert cal.rankings == ("total", "best_path")
    for name in ("calibration.json", "histograms.npz", "metrics.csv", "reliability.csv",
                 "lookup_tables.csv", "rank_accuracy.csv", "temperature_fit.csv"):
        assert (res.folder / name).is_file()
    assert set(res.metrics["predictor"]) == {"raw", "tempered", "calibrated (cross-fitted)"}
    # every OOF row contributes its top-1 candidate to every nothing-fixed scenario
    ra = res.rank_accuracy[res.rank_accuracy["ranking"] == "total"]
    r1 = ra[(ra["n_fixed"] == 0) & (ra["rank"] == 1)]
    assert (r1["n"] == len(ev.data)).all()
    # deeper levels are harder: top-1 accuracy decreases with depth
    assert r1.sort_values("target_level")["accuracy"].is_monotonic_decreasing


def test_calibrator_roundtrip_and_rebuild(fitted, tmp_path):
    _, res = fitted
    cal = res.calibrator
    path = cal.save(tmp_path / "c.json")
    back = Calibrator.load(path)
    p = np.linspace(0, 1, 57)
    for r, t, c, k in cal.tables:
        kind = k if k != "any" else "confirmed"
        np.testing.assert_allclose(back.calibrate(p, t, c, kind, r), cal.calibrate(p, t, c, kind, r))
    assert back.temperatures == cal.temperatures
    # a missing table falls back to the pooled / unconditioned table
    tab, name = cal.table_for(3, 3, "corrected")
    assert tab is not None
    import shutil
    copy = tmp_path / "cand"
    shutil.copytree(res.folder, copy)
    again = calibration.rebuild_from_histograms(copy, CalibrationSettings(scheme="percent", min_scenario_count=200))
    assert again.calibrator.settings.scheme == "percent"
    assert all(len(t.centers) <= 100 for t in again.calibrator.tables.values())


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

def test_recommender_scopes(clf, fitted, df):
    _, res = fitted
    lookup = df[LEVELS + NAMES].drop_duplicates()
    rec = Recommender(clf, res.calibrator, lookup, k=5)
    texts = df["CleanDesc"].head(20)
    out = rec.recommend(texts=texts)
    assert out.groupby(["row", "depth"]).size().max() <= 5
    assert out["calibrated"].between(0, 1).all()
    assert out["name"].notna().all()
    # within a row and level, calibrated follows the ranking (monotone tables)
    for _, g in out.groupby(["row", "depth"]):
        assert g["score"].is_monotonic_decreasing
        assert g["calibrated"].is_monotonic_decreasing

    # select the second-ranked level-2 class of row 0: L1 fixed to its parent, L3/L4 in scope
    l2 = out[(out["row"] == 0) & (out["depth"] == 2)]["label"].iloc[1]
    sel = rec.recommend(texts=texts.head(1), selections={"Level 2 ID": l2})
    fixed = sel[sel["fixed"]]
    assert list(fixed["depth"]) == [1, 2] and (fixed["calibrated"] == 1).all()
    assert fixed["label"].iloc[0] == l2.split("_")[0].replace("L2", "L1")
    below = sel[~sel["fixed"]]
    assert below["label"].map(lambda s: s.startswith(l2.replace("L2-", "L3-")) or
                              s.startswith(l2.replace("L2-", "L4-"))).all()
    assert (below["scenario"].str.endswith("corrected")).all()
    # the model's own top pick counts as 'confirmed'
    top = out[(out["row"] == 0) & (out["depth"] == 2)]["label"].iloc[0]
    conf = rec.recommend(texts=texts.head(1), selections={2: top})
    assert conf[~conf["fixed"]]["scenario"].str.endswith("confirmed").all()
    # per-row selections
    many = rec.recommend(texts=texts.head(3), selections=[None, {"Level 2 ID": l2}, None])
    assert set(many[many["fixed"]]["row"]) == {1}


def test_recommender_conditional_scores_sum_to_one(clf, df):
    rec = Recommender(clf, None, k=50)
    node_label = clf.classes_[1]["label"].iloc[0]
    out = rec.recommend(texts=df["CleanDesc"].head(5), selections={"Level 2 ID": node_label})
    s = out[(~out["fixed"])].groupby(["row", "depth"])["score"].sum()
    np.testing.assert_allclose(s.to_numpy(), 1.0, rtol=1e-9)
    assert out.loc[~out["fixed"], "calibrated"].isna().all()


def test_recommender_rejects_bad_selections(clf):
    rec = Recommender(clf)
    with pytest.raises(ValueError):
        rec.resolve({"Level 1 ID": "L1-0", "Level 2 ID": "L2-1_0"})   # not on one path
    with pytest.raises(KeyError):
        rec.resolve({"Level 9 ID": "x"})


def test_ambiguous_label_needs_parent():
    df = pd.DataFrame({"t": ["a x", "b x", "a y", "b y"] * 5,
                       "L1": ["A", "B", "A", "B"] * 5, "L2": ["same", "same", "s2", "s2"] * 5})
    m = HierarchicalNB(["L1", "L2"], HyperParams(min_df=1, rare_path_threshold=1)).fit(df["t"], df[["L1", "L2"]])
    rec = Recommender(m)
    with pytest.raises(ValueError, match="ambiguous"):
        rec.resolve({"L2": "same"})
    assert rec.resolve({"L1": "B", "L2": "same"})[0] == 2


# ---------------------------------------------------------------------------
# saving with a model
# ---------------------------------------------------------------------------

class FakeCleaner:
    method_name = "none"

    def export(self, folder):
        return {}

    def describe(self):
        return {}


def test_save_and_load_with_model(fitted, df, tmp_path):
    ev, res = fitted
    hp = ev.candidates[0]
    model = HierarchicalNB(LEVELS, hp).fit(df["CleanDesc"], df[LEVELS])
    folder = tmp_path / "model_x"
    files = artifacts.save_model(folder, model, FakeCleaner(), df[LEVELS + NAMES], {"n_training_rows": len(df)},
                                 calibrator=res.calibrator)
    assert "calibration" in files
    loaded = artifacts.load_model(str(folder))
    assert loaded.calibrator is not None
    assert json.loads((folder / "manifest.json").read_text())["calibration"]["n_tables"] == len(res.calibrator.tables)
    pf = loaded.clf.predict_frame(df["CleanDesc"].head(10), calibrator=loaded.calibrator)
    assert [f"L{d} Calibrated" for d in range(1, 5)] == [c for c in pf.columns if c.endswith("Calibrated")]
    # a calibrator for other hyperparameters is refused
    other = HierarchicalNB(LEVELS, HyperParams(**{**hp.to_dict(), "alpha": 0.5})).fit(df["CleanDesc"], df[LEVELS])
    with pytest.raises(ValueError):
        artifacts.save_model(tmp_path / "model_y", other, FakeCleaner(), None, {}, calibrator=res.calibrator)
    # attach to an existing folder
    plain = tmp_path / "model_z"
    artifacts.save_model(plain, model, FakeCleaner(), None, {})
    artifacts.attach_calibration(plain, res.calibrator)
    assert artifacts.load_model(str(plain)).calibrator is not None


@pytest.mark.parametrize("mode", ["independent", "flattened"])
def test_other_prediction_modes(mode, tmp_path):
    data = synthetic(n=1200, seed=3)
    prep = PreparedData(data, cleaner=None, metadata={})
    cols = crossval.Columns("FullDesc", "CleanDesc", LEVELS, NAMES, ["ID"])
    ev = crossval.Evaluation.create(prep, [HyperParams(prediction_mode=mode, **HP)], CVSettings(n_folds=2), cols,
                                    grid_search=False, folder=tmp_path / "ev")
    s = CalibrationSettings(min_bin_count=30, min_scenario_count=100, keep_fold_models=False, batch_size=300)
    res = calibration.fit_from_evaluation(ev, 0, s, verbose=False)
    cal = res.calibrator
    if mode == "flattened":
        assert cal.temperatures[:3] == [1.0, 1.0, 1.0]
    assert cal.temperatures[-1] != 1.0
    model = HierarchicalNB(LEVELS, ev.candidates[0]).fit(data["CleanDesc"], data[LEVELS])
    out = Recommender(model, cal, k=5).recommend(texts=data["CleanDesc"].head(5))
    assert out["calibrated"].between(0, 1).all()
    pf = model.predict_frame(data["CleanDesc"].head(5), calibrator=cal)
    assert pf["L4 Calibrated"].between(0, 1).all()



# ---------------------------------------------------------------------------
# rankings
# ---------------------------------------------------------------------------

def test_best_path_scores_match_brute_force(clf, df):
    from nbclassifier.calibration import level_best_paths

    X = clf.transform(df["CleanDesc"].head(15))
    p = np.exp(_log_softmax(clf.joint_log(X)))
    for d in range(4):
        got = level_best_paths(clf, p, d)
        for j in range(len(clf.classes_[d])):
            np.testing.assert_allclose(got[:, j], p[:, clf.path_codes_[d] == j].max(axis=1))
    # ranking classes by best path = reading distinct classes off the ranked paths
    row = p[0]
    order = np.argsort(-row, kind="mergesort")
    seen = list(dict.fromkeys(clf.path_codes_[1][order]))[:5]
    best = level_best_paths(clf, p[:1], 1)[0]
    assert list(np.argsort(-best, kind="mergesort")[:5]) == seen


def test_rankings_in_recommender(clf, fitted, df):
    _, res = fitted
    texts = df["CleanDesc"].head(10)
    tot = Recommender(clf, res.calibrator, k=5).recommend(texts=texts)
    best = Recommender(clf, res.calibrator, k=5, ranking="best_path").recommend(texts=texts)
    assert (best["ranking"] == "best_path").all() and (tot["ranking"] == "total").all()
    # level 4 identical under both rankings
    cols = ["row", "rank", "label", "score", "calibrated"]
    pd.testing.assert_frame_equal(tot[tot.depth == 4][cols].reset_index(drop=True),
                                  best[best.depth == 4][cols].reset_index(drop=True))
    # a class's best path never exceeds its total, and calibration stays monotone in the ranking
    m = tot.merge(best, on=["row", "depth", "label"], suffixes=("_t", "_b"))
    assert (m["score_b"] <= m["score_t"] + 1e-12).all()
    for _, g in best.groupby(["row", "depth"]):
        assert g["score"].is_monotonic_decreasing and g["calibrated"].is_monotonic_decreasing
    assert best[best.depth < 4]["scenario"].str.endswith("best path").all()
    # per-call override, and after a lock
    one = Recommender(clf, res.calibrator).recommend_one(cleaned_text=texts.iloc[0], ranking="best_path",
                                                         selections={1: tot["label"].iloc[0]})
    assert one["ranking"] == "best_path"
    with pytest.raises(ValueError):
        Recommender(clf, res.calibrator, ranking="nope")


def test_version1_calibration_still_loads(fitted, clf, df, tmp_path):
    _, res = fitted
    d = res.calibrator.to_dict()
    d["format_version"] = 1
    d["tables"] = [{k: v for k, v in t.items() if k != "ranking"} for t in d["tables"] if t["ranking"] == "total"]
    d["settings"].pop("rankings")
    old = Calibrator.from_dict(d)
    assert old.rankings == ("total",)
    with pytest.warns(UserWarning, match="no 'best_path' tables"):
        Recommender(clf, old, ranking="best_path")
    np.testing.assert_allclose(old.calibrate([0.3, 0.7], 1, 1, "corrected"),
                               res.calibrator.calibrate([0.3, 0.7], 1, 1, "corrected"))


# ---------------------------------------------------------------------------
# fold models saved by the evaluation
# ---------------------------------------------------------------------------

def test_calibration_reuses_saved_fold_models(df, tmp_path):
    data = df.head(1500)
    prep = PreparedData(data, cleaner=None, metadata={})
    cols = crossval.Columns("FullDesc", "CleanDesc", LEVELS, NAMES, ["ID"])
    grid = [HyperParams(prediction_mode="conditional", **HP),
            HyperParams(prediction_mode="conditional", **{**HP, "alpha": 0.05})]   # same vectorizer group
    ev = crossval.Evaluation.create(prep, grid, CVSettings(n_folds=3), cols, grid_search=True,
                                    folder=tmp_path / "ev")
    ev.run()
    assert ev.has_fold_models(grid[0]) and ev.has_fold_models(grid[1])
    assert len(list((ev.folder / "fold_models").glob("*/split_*.joblib"))) == 3     # one group, 3 folds
    s = CalibrationSettings(min_bin_count=30, min_scenario_count=100, batch_size=300)
    reused = calibration.fit_from_evaluation(ev, 1, s, folder=tmp_path / "a", verbose=False)
    assert reused.calibrator.provenance["fold_models"] == ["saved fold model"] * 3
    # the reused model reproduces the evaluation's own out-of-fold predictions
    clf, _ = ev.load_fold_model(grid[1], 0)
    test = ev._splits(ev.data)[0][1]
    pred = clf.predict_frame(ev.data.loc[test, "CleanDesc"])
    saved = pd.read_parquet(ev._pred_path(1, 0))
    assert (pred[f"Predicted {LEVELS[-1]}"].to_numpy() == saved[f"Predicted {LEVELS[-1]}"].to_numpy()).all()
    # and gives the same calibration as refitting from scratch
    assert ev.delete_fold_models() > 0 and not ev.has_fold_models(grid[1])
    refit = calibration.fit_from_evaluation(ev, 1, s, folder=tmp_path / "b", verbose=False)
    assert refit.calibrator.provenance["fold_models"] == ["refitted"] * 3
    np.testing.assert_allclose(reused.calibrator.temperatures, refit.calibrator.temperatures, rtol=1e-9)
    assert reused.calibrator.tables.keys() == refit.calibrator.tables.keys()
    for key, tab in reused.calibrator.tables.items():
        np.testing.assert_allclose(tab.values, refit.calibrator.tables[key].values)
