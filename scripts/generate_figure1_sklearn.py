#!/usr/bin/env python3
"""
Build the five Cat2Vec preprocessing pipelines (P0–P4) as real sklearn objects
and render a side-by-side HTML/PDF diagram for the PeerJ paper (Figure 1).

Transformers are defined here as lightweight sklearn-compatible stubs —
identical class names and parameters to the training code, no TF dependency.

Output:
  Figure1_pipelines.html   — open in Chrome/Safari → Print → Save as PDF
  Figure 1.pdf             — via weasyprint if installed
                             (pip install weasyprint)
"""

import os

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import estimator_html_repr

set_config(display="diagram")

# ── Feature columns (mirrors the experiment script exactly) ───────────────
ORDINAL_COLS = ["ord_0", "ord_1", "ord_2", "day", "month"]
NOMINAL_COLS = (
    [f"nom_{i}" for i in range(10)]
    + [f"bin_{i}" for i in range(5)]
    + ["ord_3", "ord_4", "ord_5"]
)

# ── Sklearn-compatible stubs for the custom transformers ──────────────────
# Inheriting BaseEstimator gives free get_params / set_params and clean reprs.

class RankAsNumeric(BaseEstimator, TransformerMixin):
    """Map each ordinal value to its integer rank (trivial order baseline)."""
    pass


class TopKReducer(BaseEstimator, TransformerMixin):
    """Retain top-K categories per column; map the rest to a RARE token."""
    def __init__(self, top_k=100, other_token="__RARE__"):
        self.top_k = top_k
        self.other_token = other_token


class Cat2VecTransformer(BaseEstimator, TransformerMixin):
    """Learned dense embeddings for nominal / high-cardinality categoricals."""
    def __init__(self, epochs=25, batch_size=1024, lr=1e-3,
                 min_count=3, max_emb_dim=24, sample_n=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_count = min_count
        self.max_emb_dim = max_emb_dim
        self.sample_n = sample_n


class Cat2VecWPosTransformer(BaseEstimator, TransformerMixin):
    """Learned embeddings + sinusoidal positional encoding for ordinals."""
    def __init__(self, pos_dim=12, epochs=25, batch_size=1024,
                 lr=1e-3, min_count=3, max_emb_dim=24, sample_n=None):
        self.pos_dim = pos_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_count = min_count
        self.max_emb_dim = max_emb_dim
        self.sample_n = sample_n


# ── Pipeline builders ─────────────────────────────────────────────────────

def _ord_rank():
    return Pipeline([("rank", RankAsNumeric()), ("scaler", StandardScaler())])

def _ord_encoder():
    return Pipeline([
        ("imp",    SimpleImputer(strategy="most_frequent")),
        ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler()),
    ])

def _ord_wpos():
    return Pipeline([
        ("wpos",   Cat2VecWPosTransformer(pos_dim=12, epochs=25, batch_size=1024,
                                          lr=1e-3, min_count=3, max_emb_dim=24)),
        ("scaler", StandardScaler()),
    ])

def _nom_onehot():
    return Pipeline([
        ("imp",    SimpleImputer(strategy="most_frequent")),
        ("topk",   TopKReducer(top_k=100)),
        ("oh",     OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 max_categories=100)),
    ])

def _nom_cat2vec():
    return Pipeline([
        ("c2v",    Cat2VecTransformer(epochs=25, batch_size=1024, lr=1e-3,
                                      min_count=3, max_emb_dim=24)),
        ("scaler", StandardScaler()),
    ])


PIPELINE_SPECS = {
    "P0": (_ord_rank,    _nom_onehot,   "Trivial rank baseline"),
    "P1": (_ord_encoder, _nom_onehot,   "Conventional baseline"),
    "P2": (_ord_encoder, _nom_cat2vec,  "Nominal embeddings only"),
    "P3": (_ord_wpos,    _nom_onehot,   "Ordinal pos. encoding only"),
    "P4": (_ord_wpos,    _nom_cat2vec,  "Fully embedded  ★  (proposed)"),
}

pipelines = {}
for name, (ord_fn, nom_fn, _) in PIPELINE_SPECS.items():
    pipelines[name] = ColumnTransformer(
        [("ordinal", ord_fn(), ORDINAL_COLS),
         ("nominal", nom_fn(), NOMINAL_COLS)],
        remainder="drop",
    )

# ── HTML generation ────────────────────────────────────────────────────────

html_reprs = {name: estimator_html_repr(ct) for name, ct in pipelines.items()}

CSS = """
@page { size: A4 portrait; margin: 1.5cm; }
* { box-sizing: border-box; }
body {
  font-family: "DejaVu Sans", Arial, sans-serif;
  font-size: 8pt;
  margin: 0;
  background: white;
  color: #111;
  width: 100%;
}
.grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  align-items: start;
  width: 100%;
}
.col {
  border: 1px solid #CCCCCC;
  border-radius: 5px;
  padding: 5px 3px;
  background: #FAFAFA;
  min-width: 0;
  overflow: hidden;
}
.col-header {
  text-align: center;
  margin-bottom: 3px;
}
.col-header .pipe-id {
  font-size: 13pt;
  font-weight: bold;
  color: #222;
}
.col-header .pipe-role {
  font-size: 6.5pt;
  color: #666;
  font-style: italic;
  margin-top: 1px;
}
.col.proposed {
  border: 2px solid #B22222;
  background: #FFF8F8;
}
.col.proposed .pipe-id { color: #B22222; }

/* ── Shrink the sklearn diagram ────────────────────────────── */
.sk-estimator-doc-link { display: none !important; }
.sk-text-repr-fallback  { display: none !important; }
/* Target every text node inside the diagram */
[class^="sk-"] p,
[class^="sk-"] span,
[class^="sk-"] div,
[class^="sk-"] label {
  font-size: 7pt !important;
}
/* Pull in horizontal padding on estimator boxes */
.sk-estimator { padding: 2px 4px !important; }
.sk-label-container { padding: 2px 4px !important; }
.sk-serial { padding: 0 2px !important; }
.sk-parallel-item { padding: 2px !important; }
.sk-top-container { padding: 2px !important; }

/* Abbreviations footer */
.abbr {
  font-size: 6pt;
  color: #555;
  margin-top: 10px;
  border-top: 1px solid #DDD;
  padding-top: 5px;
  line-height: 1.5;
}
"""

ABBR = (
    "<b>Abbreviations:</b> "
    "ord = OrdinalEncoder &nbsp;·&nbsp; "
    "oh = OneHotEncoder &nbsp;·&nbsp; "
    "imp = SimpleImputer(most_frequent) &nbsp;·&nbsp; "
    "TopKReducer: retain top-100 categories, remainder → <code>__RARE__</code> &nbsp;·&nbsp; "
    "Cat2VecTransformer: learned dense embeddings (nominal) &nbsp;·&nbsp; "
    "Cat2VecWPosTransformer: learned embeddings + sinusoidal pos. enc. (ordinal) &nbsp;·&nbsp; "
    "★ = proposed method"
)

col_blocks = ""
for name, (_, _, role) in PIPELINE_SPECS.items():
    is_p4 = name == "P4"
    cls = "col proposed" if is_p4 else "col"
    col_blocks += f"""
<div class="{cls}">
  <div class="col-header">
    <div class="pipe-id">{name}</div>
    <div class="pipe-role">{role}</div>
  </div>
  {html_reprs[name]}
</div>
"""

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
<div class="grid">
{col_blocks}
</div>
<div class="abbr">{ABBR}</div>
</body>
</html>
"""

here = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(here, "../figures/Figure1_pipelines.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(page)
print(f"Saved HTML: {html_path}")
print("→ Open in Chrome/Safari, File → Print → Save as PDF to export.")

# ── Optional: PDF via weasyprint ──────────────────────────────────────────
try:
    from weasyprint import HTML as WP_HTML
    pdf_path = os.path.join(here, "../figures/Figure1_pipelines.pdf")
    WP_HTML(string=page, base_url=here).write_pdf(pdf_path)
    print(f"Saved PDF:  {pdf_path}")
except ImportError:
    print("(weasyprint not installed — PDF skipped; install with: pip install weasyprint)")
except Exception as e:
    print(f"(weasyprint PDF failed: {e})")
