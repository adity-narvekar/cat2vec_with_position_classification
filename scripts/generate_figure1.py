#!/usr/bin/env python3
"""
Generate Figure 1 for the Cat2Vec PeerJ submission.
Five-pipeline comparison diagram (P0–P4).

Output: "Figure 1.pdf"  — US Letter, 2.5 cm margins, vector, no caption baked in.
Supply the title separately per PeerJ instructions:
  "The five categorical-encoding pipelines (P0–P4) compared in this study."
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

# ── Palette (ColorBrewer Paired + neutrals — colorblind-safe) ────────────
# Visual distinction: color AND dashed/solid border AND corner tag — never color alone.
C_SHARED  = "#DDDDDD"   # neutral gray   — shared input / concat / eval blocks
C_CLASSIC = "#B2D8E8"   # muted blue     — classical encoders
C_EMB     = "#FDBF6F"   # orange         — Cat2Vec (learned embedding)
C_POSENC  = "#CAB2D6"   # lavender       — Cat2Vec_wpos (embedding + sinusoidal pos. enc.)
C_EVAL    = "#B2E2B2"   # light green    — downstream evaluation blocks
EC_STD    = "#444444"
EC_EMB    = "#C85A00"   # darker orange  — Cat2Vec box edges
EC_POS    = "#5E2A8A"   # darker purple  — Cat2Vec_wpos box edges
EC_P4     = "#B22222"   # crimson        — P4 column (proposed method)
FONT      = "DejaVu Sans"

# ── Page geometry ─────────────────────────────────────────────────────────
PW, PH    = 8.5, 11.0
MARGIN_CM = 2.5
MX = MARGIN_CM / (PW * 2.54)   # ≈ 0.1158
MY = MARGIN_CM / (PH * 2.54)   # ≈ 0.0895
X0, X1 = MX, 1.0 - MX
Y0, Y1 = MY, 1.0 - MY
UW = X1 - X0                   # ≈ 0.769

fig = plt.figure(figsize=(PW, PH))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Drawing helpers ────────────────────────────────────────────────────────

def rbox(cx, cy, w, h, fc, text="", fs=7.0, ec=EC_STD, lw=0.9,
         ls="solid", bold=False, tag=None):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.007",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        transform=ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontfamily=FONT,
                fontweight="bold" if bold else "normal",
                color="#111111", transform=ax.transAxes,
                zorder=4, linespacing=1.3)
    if tag:
        ax.text(cx + w / 2 - 0.007, cy + h / 2 - 0.005, tag,
                ha="right", va="top", fontsize=5.0, fontfamily=FONT,
                color=ec, style="italic", transform=ax.transAxes, zorder=5)


def arr(x1, y1, x2, y2, lw=0.8, color="#666666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, shrinkA=0, shrinkB=0),
                zorder=2)


def hbar(y, xa, xb, color="#AAAAAA", lw=0.75, ls="-"):
    ax.plot([xa, xb], [y, y], color=color, lw=lw, ls=ls,
            transform=ax.transAxes, zorder=1)


def vbar(x, ya, yb, color="#AAAAAA", lw=0.6):
    ax.plot([x, x], [ya, yb], color=color, lw=lw,
            transform=ax.transAxes, zorder=1)


# ── Column geometry ────────────────────────────────────────────────────────
NCOLS = 5
COLW  = UW / NCOLS    # ≈ 0.154 per column
BW    = COLW * 0.82   # treatment box width
BH    = 0.048         # treatment box height
HH    = 0.036         # pipeline header height

cc = [X0 + COLW * (i + 0.5) for i in range(NCOLS)]   # column centers P0–P4

# ── Y positions (figure fractions, measured down from Y1) ────────────────
# Input box is 2 lines tall (0.048); everything below shifted ~0.013 vs v1.
Y_INP  = Y1 - 0.025   # input dataset block center        (H = 0.048)
Y_BUS0 = Y1 - 0.063   # fan bar: input → feature lanes
Y_LAN  = Y1 - 0.100   # feature lane box centers          (LH = 0.050)
Y_BUS1 = Y1 - 0.140   # bus: lanes → pipeline headers
Y_PHD  = Y1 - 0.171   # pipeline header centers           (HH = 0.036)
Y_ROL  = Y1 - 0.210   # role text (italic, no box)
Y_SEP1 = Y1 - 0.231   # separator  ─── header | pipeline band
Y_ORD  = Y1 - 0.267   # ordinal transformer row           (BH = 0.048)
Y_NOM  = Y1 - 0.339   # nominal transformer row           (BH = 0.048)
Y_SEP2 = Y1 - 0.388   # separator  ─── pipeline band | downstream
Y_BUS2 = Y1 - 0.418   # bus: nominal boxes → concat
Y_CON  = Y1 - 0.448   # concatenate block                 (H = 0.034)
Y_FAN  = Y1 - 0.483   # fan bar: concat → classifiers
Y_CLF  = Y1 - 0.519   # classifier bank                   (H = 0.052)
Y_BUS3 = Y1 - 0.568   # bus: classifiers → tuning
Y_TUN  = Y1 - 0.601   # threshold-tuning block            (H = 0.034)
Y_EVA  = Y1 - 0.651   # evaluation block                  (H = 0.034)
Y_ISEP = Y1 - 0.691   # dashed separator above inset
Y_ILBL = Y1 - 0.706   # inset section label
Y_INS  = Y1 - 0.743   # inset box center                  (H = 0.052)
Y_ABBR = Y1 - 0.815   # abbreviations line (center ≈ 0.096, just above Y0)

# ── 1. Input block ─────────────────────────────────────────────────────────
rbox(0.5, Y_INP, UW * 0.55, 0.048, C_SHARED,
     "All-categorical tabular dataset\n"
     "(Cat-in-the-Dat II, ≈ 300 k rows, binary classification)",
     fs=6.5)

# Input → fan bar → 3 lane tops
arr(0.5, Y_INP - 0.024, 0.5, Y_BUS0, lw=0.9)

# ── 2. Feature-group lanes ─────────────────────────────────────────────────
LH  = 0.050
LCX = [X0 + UW * f for f in (0.14, 0.50, 0.86)]
LW  = [UW * 0.23, UW * 0.28, UW * 0.21]
LANE_DATA = [
    ("Ordinal features",                "ord_0, ord_1, ord_2, day, month"),
    ("Nominal features",                "nom_0–nom_9, ord_3, ord_4, ord_5"),
    ("Binary features\n(nom. treatment)", "bin_0–bin_4"),
]

hbar(Y_BUS0, LCX[0], LCX[-1], lw=0.85)
for lx in LCX:
    arr(lx, Y_BUS0, lx, Y_LAN + LH / 2, lw=0.7, color="#777777")

for (title, feats), lx, lw_ in zip(LANE_DATA, LCX, LW):
    rbox(lx, Y_LAN, lw_, LH, C_SHARED, f"{title}\n{feats}", fs=6.0)

# Lane bottoms → BUS1 → pipeline headers
for lx in LCX:
    arr(lx, Y_LAN - LH / 2, lx, Y_BUS1, lw=0.7, color="#888888")
hbar(Y_BUS1, X0 + UW * 0.02, X1 - UW * 0.02, lw=0.85)
for cx in cc:
    arr(cx, Y_BUS1, cx, Y_PHD + HH / 2, lw=0.7, color="#888888")

# ── 3. Pipeline headers ────────────────────────────────────────────────────
HEADERS = ["P0", "P1", "P2", "P3", "P4"]
ROLES = [
    "Trivial rank\nbaseline",
    "Conventional\nbaseline",
    "Nominal emb.\nonly",
    "Ordinal pos. enc.\nonly",
    "Fully embedded\n(proposed  ★)",
]

for i, (h, r) in enumerate(zip(HEADERS, ROLES)):
    is_p4 = i == 4
    rbox(cc[i], Y_PHD, BW, HH,
         "#FFF0F0" if is_p4 else "#F5F5F5",
         h, fs=9.0, bold=True,
         ec=EC_P4 if is_p4 else EC_STD,
         lw=2.2 if is_p4 else 1.0)
    ax.text(cc[i], Y_ROL, r, ha="center", va="center",
            fontsize=5.5, fontfamily=FONT, color="#555555",
            transform=ax.transAxes, zorder=4, linespacing=1.25,
            style="italic")

# Light column separators through the pipeline band
for i in range(1, NCOLS):
    xv = X0 + COLW * i
    vbar(xv, Y_SEP1, Y_SEP2, color="#DDDDDD", lw=0.5)

hbar(Y_SEP1, X0, X1, color="#AAAAAA", lw=0.85)

# Row labels in left margin
for y_row, label in [
    (Y_ORD, "Ordinal\ntreatment"),
    (Y_NOM, "Nominal\ntreatment\n(binary follows)"),
]:
    ax.text(X0 - 0.013, y_row, label, ha="right", va="center",
            fontsize=5.8, fontfamily=FONT, fontweight="bold",
            color="#333333", transform=ax.transAxes, zorder=4, linespacing=1.2)

# ── 4. Ordinal transformer row ─────────────────────────────────────────────
ORD_CFG = [
    ("Rank-as-numeric\n+ scaler",  C_CLASSIC, EC_STD, "solid",  0.9, None),
    ("Ordinal enc.\n+ scaler",     C_CLASSIC, EC_STD, "solid",  0.9, None),
    ("Ordinal enc.\n+ scaler",     C_CLASSIC, EC_STD, "solid",  0.9, None),
    ("Cat2Vec_wpos\n+ scaler",     C_POSENC,  EC_POS, "dashed", 1.6, "[emb+pos]"),
    ("Cat2Vec_wpos\n+ scaler",     C_POSENC,  EC_POS, "dashed", 1.6, "[emb+pos]"),
]
for i, (txt, fc, ec, ls, lw_, tag) in enumerate(ORD_CFG):
    rbox(cc[i], Y_ORD, BW, BH, fc, txt, fs=6.5, ec=ec, lw=lw_, ls=ls, tag=tag)

# ── 5. Nominal transformer row ─────────────────────────────────────────────
NOM_CFG = [
    ("TopKReducer\n+ one-hot",  C_CLASSIC, EC_STD, "solid",  0.9, None),
    ("TopKReducer\n+ one-hot",  C_CLASSIC, EC_STD, "solid",  0.9, None),
    ("Cat2Vec\n+ scaler",       C_EMB,     EC_EMB, "dashed", 1.6, "[emb]"),
    ("TopKReducer\n+ one-hot",  C_CLASSIC, EC_STD, "solid",  0.9, None),
    ("Cat2Vec\n+ scaler",       C_EMB,     EC_EMB, "dashed", 1.6, "[emb]"),
]
for i, (txt, fc, ec, ls, lw_, tag) in enumerate(NOM_CFG):
    rbox(cc[i], Y_NOM, BW, BH, fc, txt, fs=6.5, ec=ec, lw=lw_, ls=ls, tag=tag)
    arr(cc[i], Y_ORD - BH / 2, cc[i], Y_NOM + BH / 2, lw=0.75)

hbar(Y_SEP2, X0, X1, color="#AAAAAA", lw=0.85)

# ── 6. Concatenate block ───────────────────────────────────────────────────
for cx in cc:
    arr(cx, Y_NOM - BH / 2, cx, Y_BUS2, lw=0.75, color="#666666")
hbar(Y_BUS2, X0 + UW * 0.02, X1 - UW * 0.02, lw=0.9, color="#666666")
arr(0.5, Y_BUS2, 0.5, Y_CON + 0.017, lw=0.9)

rbox(0.5, Y_CON, UW * 0.68, 0.034, C_SHARED,
     "Concatenate  (ColumnTransformer output → combined feature matrix)",
     fs=6.5)

# ── 7. Classifier bank ─────────────────────────────────────────────────────
arr(0.5, Y_CON - 0.017, 0.5, Y_FAN, lw=0.9)

CLF_NAMES = ["Logistic\nRegression", "Random\nForest", "XGBoost", "k-NN"]
CW = UW * 0.175
CH = 0.052
clf_x = [0.5 + (j - 1.5) * UW * 0.228 for j in range(4)]

hbar(Y_FAN, clf_x[0], clf_x[-1], lw=0.85, color="#666666")
for cx in clf_x:
    arr(cx, Y_FAN, cx, Y_CLF + CH / 2, lw=0.7)
for cx, cn in zip(clf_x, CLF_NAMES):
    rbox(cx, Y_CLF, CW, CH, C_EVAL, cn, fs=6.5)

# ── 8. Threshold tuning ────────────────────────────────────────────────────
hbar(Y_BUS3, clf_x[0], clf_x[-1], lw=0.85, color="#666666")
for cx in clf_x:
    arr(cx, Y_CLF - CH / 2, cx, Y_BUS3, lw=0.7)
arr(0.5, Y_BUS3, 0.5, Y_TUN + 0.017, lw=0.9)

rbox(0.5, Y_TUN, UW * 0.68, 0.034, C_EVAL,
     "Decision-threshold tuning on validation split  (maximize macro-F1)",
     fs=6.5)

arr(0.5, Y_TUN - 0.017, 0.5, Y_EVA + 0.017, lw=0.9)

# ── 9. Evaluation block ────────────────────────────────────────────────────
rbox(0.5, Y_EVA, UW * 0.74, 0.034, C_EVAL,
     "Evaluation over 5 random seeds  ·  paired significance tests (Wilcoxon signed-rank)",
     fs=6.5)

# ── 10. Inset: embedding-transformer architecture details ──────────────────
hbar(Y_ISEP, X0, X1, color="#CCCCCC", lw=0.6, ls="--")
ax.text(0.5, Y_ILBL,
        "▼  Embedding-transformer architecture details  (used in P2, P3, P4)  ▼",
        ha="center", va="center", fontsize=6.0, fontfamily=FONT,
        style="italic", color="#666666", transform=ax.transAxes, zorder=4)

INS_H = 0.052
INS_W = UW * 0.455

rbox(X0 + UW * 0.27, Y_INS, INS_W, INS_H, C_EMB,
     "Cat2Vec  [emb]  —  nominal → dense embedding\n"
     "Rare-category filter (min-count = 3); index 0 = unseen / rare\n"
     "Embed dim = ⌊√cardinality⌋, max 24\n"
     "Net: Emb → Dense(32, ReLU) → Dense(16, ReLU) → softmax\n"
     "≤ 25 epochs · early stopping · batch 1024 · Adam · weights reused at inference",
     fs=5.5, ec=EC_EMB, lw=1.3)

rbox(X0 + UW * 0.73, Y_INS, INS_W, INS_H, C_POSENC,
     "Cat2Vec_wpos  [emb+pos]  —  ordinal → embedding ⊕ pos. enc.\n"
     "Same architecture as Cat2Vec; output dim = ⌊√K⌋ + 1 (bounded)\n"
     "Sinusoidal pos. enc.: P(m, 2i) = sin(m / n^(2i/d)),  P(m, 2i+1) = cos(m / n^(2i/d))\n"
     "d = 12,  n = 10 000,  m = ordinal rank,  i = column index\n"
     "Final representation = concat(learned embedding, positional vector)",
     fs=5.5, ec=EC_POS, lw=1.3)

# ── 11. Abbreviations ─────────────────────────────────────────────────────
ax.text(0.5, Y_ABBR,
        'enc. = encoder  ·  scaler = StandardScaler  ·  '
        'TopKReducer: retain top-100 categories, remainder → "RARE"  ·  '
        '[emb] = learned embedding  ·  [emb+pos] = embedding + sinusoidal pos. enc.  ·  '
        '★ = proposed method',
        ha="center", va="center", fontsize=5.0, fontfamily=FONT,
        color="#555555", transform=ax.transAxes, zorder=4, linespacing=1.4)

# ── 12. Legend — anchored in the gap between input block and feature lanes ─
# Top-right corner; sits in the vertical gap between Y_INP bottom and Y_LAN top.
legend_handles = [
    Patch(facecolor=C_CLASSIC, edgecolor=EC_STD, linewidth=0.9,
          label="Classical encoder"),
    Patch(facecolor=C_EMB,    edgecolor=EC_EMB, linewidth=1.6,
          linestyle="dashed", label="Cat2Vec  [emb]"),
    Patch(facecolor=C_POSENC, edgecolor=EC_POS, linewidth=1.6,
          linestyle="dashed", label="Cat2Vec_wpos  [emb+pos]"),
    Patch(facecolor="#FFF0F0", edgecolor=EC_P4,  linewidth=2.2,
          label="P4 — proposed method  ★"),
]
# Place the legend upper-right corner at (X1, just below input box bottom)
leg_anchor_y = Y_INP - 0.024 - 0.006   # a few points below input box bottom
leg = ax.legend(handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(X1, leg_anchor_y),
                fontsize=5.5, framealpha=0.97,
                edgecolor="#BBBBBB", ncol=1,
                handlelength=1.8, handletextpad=0.5,
                borderpad=0.5)
leg.get_frame().set_linewidth(0.6)

# ── Save ──────────────────────────────────────────────────────────────────
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/Figure 1.pdf")
fig.savefig(out, format="pdf", bbox_inches=None,
            metadata={"Creator": "generate_figure1.py"})
print(f"Saved: {out}")
