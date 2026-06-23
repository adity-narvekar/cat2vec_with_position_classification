#!/usr/bin/env python3
"""
experiment_baseline_and_significance_brfss.py
=============================================

Adaptation of experiment_baseline_and_significance.py for the CDC Diabetes
Health Indicators dataset (BRFSS 2015),
file: diabetes_binary_health_indicators_BRFSS2015.csv

Dataset notes:
- 253,680 rows, 21 features, no missing values.
- Target: Diabetes_binary (0=no diabetes, 1=prediabetes/diabetes); ~86%/14% split.
- 4 ordinal features (GenHlth, Age, Education, Income) — all already integer-coded.
- 14 binary 0/1 features treated as binary-as-nominal.
- Continuous features (BMI, MentHlth, PhysHlth) are EXCLUDED — only ordinal and nominal features used.
- All values stored as float64 in the CSV.
- Ordinal rank maps use explicit float lists to avoid incorrect string-sort ordering
  (e.g. "10.0" < "2.0" lexicographically — explicit maps prevent this for Age).

SVM WARNING
-----------
~190k training rows. SVM is OFF by default (RUN_SVM = False). Pass --svm to enable.
"""

from __future__ import annotations
import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar  # pip install statsmodels

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================
DATA_PATH  = "../data/diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET_COL = "Diabetes_binary"   # already 0.0/1.0 float; cast to int in evaluate()
ID_COL     = "id"                # not present in this dataset; silently ignored

# 4 ordinal features — all integer-coded as floats in the CSV.
# Explicit float rank maps are required: string-sorting "10.0" < "2.0" (wrong for Age).
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]
ORDINAL_RANK_MAPS = {
    "GenHlth":   [1.0, 2.0, 3.0, 4.0, 5.0],           # 1=excellent … 5=poor
    "Age":       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                  8.0, 9.0, 10.0, 11.0, 12.0, 13.0],   # 1=18-24 … 13=80+
    "Education": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],       # 1=never attended … 6=college grad
    "Income":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # 1=<$10k … 8=>$75k
}

# 14 binary 0/1 features treated as binary-as-nominal
NOMINAL_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]


N_SEEDS = 5
TEST_SIZE = 0.25
RUN_SVM = False          # see SVM WARNING above
TOPK = 100               # TopKReducer K for one-hot pipelines
POS_ENCODING_DIM = 12
CAT2VEC_EPOCHS = 25
CAT2VEC_BATCH_SIZE = 1024
CAT2VEC_LR = 1e-3
CAT2VEC_MIN_COUNT = 3
CAT2VEC_MAX_EMB_DIM = 24
EMBED_SAMPLE = None      # set to int to subsample embedding fit on large data


# =====================================================================
# Embedding transformer classes (ported from Classification_dataset 1.ipynb)
# =====================================================================
def positional_encoding_1d(length: int, d_model: int, n: float = 10000.0) -> np.ndarray:
    positions = np.arange(length)[:, None]
    dims = np.arange(d_model)[None, :]
    div_term = np.power(n, (2 * (dims // 2)) / d_model)
    sin_vals = np.sin(positions / div_term)
    cos_vals = np.cos(positions / div_term)
    pe = np.zeros((length, d_model), dtype=np.float32)
    pe[:, 0::2] = sin_vals[:, 0::2]
    pe[:, 1::2] = cos_vals[:, 1::2]
    return pe.astype(np.float32)


class TopKReducer:
    """Cap per-column cardinality before one-hot encoding."""
    def __init__(self, top_k=100, other_token="__RARE__"):
        self.top_k = top_k
        self.other_token = other_token

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.kept_ = {}
        for c in self.columns_:
            top = X[c].astype(str).value_counts(dropna=False).head(self.top_k).index.tolist()
            self.kept_[c] = set(top)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            X[c] = X[c].astype(str)   # must convert before assignment (float cols crash)
            mask = ~X[c].isin(self.kept_[c])
            X.loc[mask, c] = self.other_token
        return X

    def get_params(self, deep=True):
        return {"top_k": self.top_k, "other_token": self.other_token}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class Cat2VecTransformer:
    """Learned embeddings for nominal/high-cardinality categoricals."""
    def __init__(self, epochs=25, batch_size=1024, lr=1e-3,
                 min_count=3, max_emb_dim=24, sample_n=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_count = min_count
        self.max_emb_dim = max_emb_dim
        self.sample_n = sample_n

    def _fit_single(self, x_col: pd.Series, y: np.ndarray):
        vc = x_col.value_counts()
        kept = vc[vc >= self.min_count].index.tolist()
        cat2idx = {cat: i + 1 for i, cat in enumerate(sorted(map(str, kept)))}
        K = len(cat2idx) + 1
        emb_dim = min(self.max_emb_dim, int(max(2, math.sqrt(K))) + 1)

        idxs = x_col.astype(str).map(cat2idx).fillna(0).astype(int).values
        yy = y.copy()
        if self.sample_n is not None and len(idxs) > self.sample_n:
            sel = np.random.RandomState(42).choice(len(idxs), self.sample_n, replace=False)
            idxs, yy = idxs[sel], yy[sel]

        num_classes = int(np.max(yy)) + 1
        inp = layers.Input(shape=(1,), dtype="int32")
        emb = layers.Embedding(K, emb_dim, name="emb")(inp)
        x = layers.Flatten()(emb)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        m = models.Model(inp, out)
        m.compile(optimizers.Adam(self.lr), losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
        es = callbacks.EarlyStopping(monitor="val_accuracy", patience=2,
                                     restore_best_weights=True, verbose=0)
        m.fit(idxs, yy, validation_split=0.1, epochs=self.epochs,
              batch_size=self.batch_size, verbose=0, callbacks=[es])
        return cat2idx, m.get_layer("emb").get_weights()[0]

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.cols_ = list(X.columns)
        if getattr(y, "dtype", None) is not None and y.dtype.kind not in "iu":
            _, y = np.unique(y, return_inverse=True)
        self.maps_, self.embs_ = {}, {}
        for col in self.cols_:
            m, e = self._fit_single(X[col], y)
            self.maps_[col], self.embs_[col] = m, e
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        outs = []
        for col in self.cols_:
            m, e = self.maps_[col], self.embs_[col]
            idxs = X[col].astype(str).map(m).fillna(0).astype(int).values
            outs.append(e[idxs])
        return (np.concatenate(outs, 1).astype(np.float32)
                if outs else np.zeros((len(X), 0), np.float32))

    def get_params(self, deep=True):
        return dict(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                    min_count=self.min_count, max_emb_dim=self.max_emb_dim,
                    sample_n=self.sample_n)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class Cat2VecWPosTransformer:
    """Learned embeddings + sinusoidal positional encoding for ordinal categoricals."""
    def __init__(self, ordinal_order_map=None, pos_dim=12, epochs=25,
                 batch_size=1024, lr=1e-3, min_count=3, max_emb_dim=24, sample_n=None):
        self.ordinal_order_map = ordinal_order_map or {}
        self.pos_dim = pos_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_count = min_count
        self.max_emb_dim = max_emb_dim
        self.sample_n = sample_n

    def _rank_map(self, x_col: pd.Series, col: str):
        if self.ordinal_order_map.get(col):
            ordered = list(map(str, self.ordinal_order_map[col]))
        else:
            ordered = sorted(map(str, x_col.dropna().unique()))
        return {cat: i for i, cat in enumerate(ordered)}

    def _fit_single(self, x_col: pd.Series, y: np.ndarray):
        vc = x_col.value_counts()
        kept = vc[vc >= self.min_count].index.tolist()
        cat2idx = {cat: i + 1 for i, cat in enumerate(sorted(map(str, kept)))}
        K = len(cat2idx) + 1
        emb_dim = min(self.max_emb_dim, int(max(2, math.sqrt(K))) + 1)

        idxs = x_col.astype(str).map(cat2idx).fillna(0).astype(int).values
        yy = y.copy()
        if self.sample_n is not None and len(idxs) > self.sample_n:
            sel = np.random.RandomState(42).choice(len(idxs), self.sample_n, replace=False)
            idxs, yy = idxs[sel], yy[sel]

        num_classes = int(np.max(yy)) + 1
        inp = layers.Input(shape=(1,), dtype="int32")
        emb = layers.Embedding(K, emb_dim, name="emb")(inp)
        x = layers.Flatten()(emb)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        m = models.Model(inp, out)
        m.compile(optimizers.Adam(self.lr), losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
        es = callbacks.EarlyStopping(monitor="val_accuracy", patience=2,
                                     restore_best_weights=True, verbose=0)
        m.fit(idxs, yy, validation_split=0.1, epochs=self.epochs,
              batch_size=self.batch_size, verbose=0, callbacks=[es])
        return cat2idx, m.get_layer("emb").get_weights()[0]

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.cols_ = list(X.columns)
        if getattr(y, "dtype", None) is not None and y.dtype.kind not in "iu":
            _, y = np.unique(y, return_inverse=True)
        self.maps_, self.ranks_, self.embs_ = {}, {}, {}
        for col in self.cols_:
            rmap = self._rank_map(X[col], col)
            m, e = self._fit_single(X[col], y)
            self.maps_[col], self.ranks_[col], self.embs_[col] = m, rmap, e
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        outs = []
        for col in self.cols_:
            m, rmap, e = self.maps_[col], self.ranks_[col], self.embs_[col]
            idxs = X[col].astype(str).map(m).fillna(0).astype(int).values
            base = e[idxs]
            ranks = X[col].astype(str).map(rmap).fillna(0).astype(int).values
            K = (max(rmap.values()) + 1) if rmap else 1
            pe_mat = positional_encoding_1d(K, self.pos_dim)
            pe = pe_mat[np.clip(ranks, 0, K - 1)]
            outs.append(np.concatenate([base, pe], 1))
        return (np.concatenate(outs, 1).astype(np.float32)
                if outs else np.zeros((len(X), 0), np.float32))

    def get_params(self, deep=True):
        return dict(ordinal_order_map=self.ordinal_order_map, pos_dim=self.pos_dim,
                    epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                    min_count=self.min_count, max_emb_dim=self.max_emb_dim,
                    sample_n=self.sample_n)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# =====================================================================
# Rank-as-numeric transformer (the new baseline)
# =====================================================================
def _build_rank_lookup():
    """Return dict: col -> {category: integer_rank}."""
    lookup = {}
    for col in ORDINAL_COLS:
        order = ORDINAL_RANK_MAPS.get(col)
        if order is not None:
            lookup[col] = {cat: i for i, cat in enumerate(order)}
        else:
            lookup[col] = None  # natural numeric order, resolved at fit time
    return lookup


class RankAsNumeric:
    """Map each ordinal value to its integer rank -> ONE numeric column.

    This is the trivial order-preserving baseline. It is intentionally
    minimal: no embeddings, no positional encoding -- just the rank.
    """

    def __init__(self):
        self.lookup_ = _build_rank_lookup()

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=ORDINAL_COLS)
        for col in ORDINAL_COLS:
            if self.lookup_[col] is None:
                # natural order over observed values (numeric if castable)
                vals = pd.unique(X[col].dropna())
                try:
                    order = sorted(vals, key=lambda v: float(v))
                except (TypeError, ValueError):
                    order = sorted(map(str, vals))
                self.lookup_[col] = {cat: i for i, cat in enumerate(order)}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=ORDINAL_COLS).copy()
        out = np.zeros((len(X), len(ORDINAL_COLS)), dtype=float)
        for j, col in enumerate(ORDINAL_COLS):
            m = self.lookup_[col]
            default = (max(m.values()) + 1) / 2.0 if m else 0.0
            out[:, j] = X[col].map(lambda v: m.get(v, default)).astype(float)
        return out

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


# =====================================================================
# Pipeline builders
# =====================================================================
def _ordinal_known_categories():
    cats = []
    lookup = _build_rank_lookup()
    for col in ORDINAL_COLS:
        order = ORDINAL_RANK_MAPS.get(col)
        cats.append(order if order is not None else "auto")
    return cats


def build_preprocessor(name: str) -> ColumnTransformer:
    """Return the ColumnTransformer for a given pipeline name."""
    scaler = ("scaler", StandardScaler())

    # --- ordinal branch options ---
    def ordinal_rank_numeric():
        # RankAsNumeric is a proper transformer; do NOT use a lambda wrapper
        # (a lambda would refit on each call, causing data leakage on test sets)
        return Pipeline([("rank", RankAsNumeric()), scaler])

    def ordinal_encoder():
        return Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
            scaler,
        ])

    def ordinal_wpos():
        return Pipeline([
            ("wpos", Cat2VecWPosTransformer(
                ordinal_order_map=ORDINAL_RANK_MAPS,
                pos_dim=POS_ENCODING_DIM,
                epochs=CAT2VEC_EPOCHS,
                batch_size=CAT2VEC_BATCH_SIZE,
                lr=CAT2VEC_LR,
                min_count=CAT2VEC_MIN_COUNT,
                max_emb_dim=CAT2VEC_MAX_EMB_DIM,
                sample_n=EMBED_SAMPLE,
            )),
            scaler,
        ])

    # --- nominal branch options ---
    def nominal_onehot():
        return Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("topk", TopKReducer(top_k=TOPK)),
            ("oh", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False, max_categories=TOPK)),
        ])

    def nominal_cat2vec():
        return Pipeline([
            ("c2v", Cat2VecTransformer(
                epochs=CAT2VEC_EPOCHS,
                batch_size=CAT2VEC_BATCH_SIZE,
                lr=CAT2VEC_LR,
                min_count=CAT2VEC_MIN_COUNT,
                max_emb_dim=CAT2VEC_MAX_EMB_DIM,
                sample_n=EMBED_SAMPLE,
            )),
            scaler,
        ])

    # name -> (ordinal branch, nominal branch)
    table = {
        "P0": (ordinal_rank_numeric, nominal_onehot),   # NEW baseline
        "P1": (ordinal_encoder,      nominal_onehot),
        "P2": (ordinal_encoder,      nominal_cat2vec),
        "P3": (ordinal_wpos,         nominal_onehot),
        "P4": (ordinal_wpos,         nominal_cat2vec),
    }
    if name not in table:
        raise ValueError(f"unknown pipeline {name}")
    ord_branch, nom_branch = table[name]
    return ColumnTransformer([
        ("ord", ord_branch(), ORDINAL_COLS),
        ("nom", nom_branch(), NOMINAL_COLS),
    ], remainder="drop")


def available_pipelines():
    return ["P0", "P1", "P2", "P3", "P4"]


def build_models():
    models = {
        "LogReg": lambda: LogisticRegression(max_iter=1000, n_jobs=-1),
        "RF":     lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1),
        "KNN":    lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    }
    if HAVE_XGB:
        models["XGB"] = lambda: XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, eval_metric="logloss", n_jobs=-1, tree_method="hist")
    if RUN_SVM:
        models["SVM"] = lambda: SVC(probability=True)
    return models


# =====================================================================
# Metrics + tuned-threshold helper
# =====================================================================
def tuned_threshold(y_true, scores):
    """Pick the threshold (over a grid) that maximizes macro-F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (scores >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


class CellResult:
    def __init__(self):
        self.acc = []
        self.f1 = []
        self.roc = []
        self.pr = []
        self.last_preds = None
        self.last_true = None


def evaluate(df):
    X = df.drop(columns=[c for c in (TARGET_COL, ID_COL) if c in df.columns])
    y = df[TARGET_COL].astype(int).values

    pipes = available_pipelines()
    models = build_models()
    results = {(p, m): CellResult() for p in pipes for m in models}

    for seed in range(N_SEEDS):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        # carve a validation slice from train for threshold tuning
        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)

        for p in pipes:
            pre = build_preprocessor(p)
            t0 = time.time()
            pre.fit(Xtr2, ytr2)
            Ztr, Zval, Zte = pre.transform(Xtr2), pre.transform(Xval), pre.transform(Xte)
            prep_s = time.time() - t0

            for mname, mfac in models.items():
                clf = mfac()
                clf.fit(Ztr, ytr2)
                val_scores = clf.predict_proba(Zval)[:, 1]
                thr = tuned_threshold(yval, val_scores)
                te_scores = clf.predict_proba(Zte)[:, 1]
                yhat = (te_scores >= thr).astype(int)

                r = results[(p, mname)]
                r.acc.append(accuracy_score(yte, yhat))
                r.f1.append(f1_score(yte, yhat, average="macro"))
                r.roc.append(roc_auc_score(yte, te_scores))
                r.pr.append(average_precision_score(yte, te_scores))
                if seed == N_SEEDS - 1:
                    r.last_preds = yhat
                    r.last_true = yte
            print(f"  seed {seed} {p} prep={prep_s:.1f}s done")
    return results, pipes, list(models.keys())


# =====================================================================
# Reporting + significance
# =====================================================================
def ms(x):
    a = np.array(x)
    return f"{a.mean():.4f}+/-{a.std(ddof=1):.4f}"


def report(results, pipes, model_names):
    print("\n" + "=" * 78)
    print(f"MEAN +/- STD over {N_SEEDS} seeds")
    print("=" * 78)
    for m in model_names:
        print(f"\n[{m}]")
        print(f"  {'pipe':<5}{'accuracy':<18}{'macroF1':<18}{'ROC-AUC':<18}PR-AUC")
        for p in pipes:
            r = results[(p, m)]
            print(f"  {p:<5}{ms(r.acc):<18}{ms(r.f1):<18}{ms(r.roc):<18}{ms(r.pr)}")

    print("\n" + "=" * 78)
    print("PAIRED SIGNIFICANCE: P4 vs each other pipeline")
    print("  (paired t-test on macro-F1 across seeds; McNemar on final-seed preds)")
    print("=" * 78)
    if "P4" not in pipes:
        print("  P4 not run (USE_EMBEDDINGS=False); skipping.")
        return
    for m in model_names:
        print(f"\n[{m}]")
        r4 = results[("P4", m)]
        for p in pipes:
            if p == "P4":
                continue
            rp = results[(p, m)]
            # paired t-test on macro-F1 across the seeds
            tstat, pval = stats.ttest_rel(r4.f1, rp.f1)
            # McNemar on the final-seed test predictions
            try:
                a = r4.last_preds == r4.last_true
                b = rp.last_preds == rp.last_true
                n01 = int(np.sum(~a & b))   # P4 wrong, p right
                n10 = int(np.sum(a & ~b))   # P4 right, p wrong
                mc = mcnemar([[0, n01], [n10, 0]], exact=False, correction=True)
                mc_p = mc.pvalue
            except Exception:
                mc_p = float("nan")
            dF1 = np.mean(r4.f1) - np.mean(rp.f1)
            flag = "  *" if (pval < 0.05) else ""
            print(f"  P4 vs {p}: dMacroF1={dF1:+.4f}  t-test p={pval:.4f}"
                  f"  McNemar p={mc_p:.4f}{flag}")
    print("\n  * = paired t-test significant at alpha=0.05")
    print("  Report these exact numbers in the paper. Do not round selectively.")


def main():
    global N_SEEDS, RUN_SVM
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--svm", action="store_true", help="include SVM (slow!)")
    args = ap.parse_args()

    N_SEEDS = args.seeds
    RUN_SVM = RUN_SVM or args.svm

    print(f"Loading {args.data} ...")
    try:
        df = pd.read_csv(args.data)
    except (UnicodeDecodeError, Exception):
        # file may be an Excel file saved with a .csv extension
        df = pd.read_excel(args.data)
    before = len(df)
    needed = [c for c in (ORDINAL_COLS + NOMINAL_COLS + [TARGET_COL]) if c in df.columns]
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"  {before} -> {len(df)} rows after dropping NaN in required cols")
    print(f"  pipelines: {available_pipelines()}   models: {list(build_models())}")

    results, pipes, model_names = evaluate(df)
    report(results, pipes, model_names)


if __name__ == "__main__":
    main()
