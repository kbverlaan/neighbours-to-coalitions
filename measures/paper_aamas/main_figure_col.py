"""Figuren AAMAS in één kolom (7 okt 2026): Cliff's delta per stap (paper) en rang-eta2 (supplement). Leest main_figure_rows.json (geschreven door main_figure.py); berekent niets.
Alleen de veertien vooraf vastgelegde maten.
  uv run --python 3.12 --with numpy --with matplotlib --with seaborn python paper_aamas/main_figure_col.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "plots"))
from _style import FAINT, INK, PT_ASIDE, style  # noqa: E402
from _paths import PAPER
OUT = PAPER / "figures"
import _paperfont as _pf
CAP, PRICE, UP, DOWN = _pf.GREEN, _pf.PEACH, _pf.GREEN, _pf.PEACH
STEPS = ("L1-L2", "L2-L3", "L3-L4")
rows = [r for r in json.loads((HIER / "main_figure_rows.json").read_text()) if r["family"] in ("F1", "F2")]
n = len(rows); y = np.arange(n)[::-1]
def lab(r): return r["label"] + ("" if len(r["rungs"]) == 4 else f"  ({r['rungs'][0]}–{r['rungs'][-1]})")
def seps(axs):
    for i in range(1, n):
        if rows[i]["family"] != rows[i - 1]["family"]:
            for ax in axs: ax.axhline(y[i] + 0.5, color=FAINT, linewidth=0.5)
style()
import _paperfont; _paperfont.apply()

# --- paper: eta2, one column
W, H = 3.33, 3.0; L, T, B = 1.45, 0.08, 0.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([L / W, B / H, (W - L - 0.1) / W, 1 - (T + B) / H]); h = 0.36
for yi, r in zip(y, rows):
    for off, val, ci, col in ((h / 2, r["cap"], r["cap_ci"], CAP), (-h / 2, r["price"], r["price_ci"], PRICE)):
        ax.barh(yi + off, val, h, color=col)
        ax.errorbar(val, yi + off, xerr=[[val - ci[0]], [ci[1] - val]], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1)
ax.set_yticks(y); ax.set_yticklabels([lab(r) for r in rows], fontsize=PT_ASIDE - 1.5)
ax.set_xlim(0, 1); ax.set_xlabel(r"partial $\eta^2$ on ranks", fontsize=PT_ASIDE - 1, labelpad=2)
ax.tick_params(axis="x", labelsize=PT_ASIDE - 2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", visible=False); seps([ax])
fig.legend(handles=[Patch(color=CAP, label="action set"), Patch(color=PRICE, label="price")],
           loc="lower center", ncol=2, frameon=False, fontsize=PT_ASIDE - 1, bbox_to_anchor=(0.5, 0.0))
fig.savefig(OUT / "main_levers_col.pdf")

# --- supplement: Cliff's delta per step
W2, H2 = 3.33, 3.0; L2, T2, B2 = 1.45, 0.08, 0.55
fig2 = plt.figure(figsize=(W2, H2))
bx = fig2.add_axes([L2 / W2, B2 / H2, (W2 - L2 - 0.05) / W2, 1 - (T2 + B2) / H2])
for i, s in enumerate(STEPS):
    for yi, r in zip(y, rows):
        st = r["steps"].get(s)
        if st is None:
            bx.text(i, yi, "–", ha="center", va="center", fontsize=PT_ASIDE - 2, color=FAINT); continue
        d, ci = st["delta"], st["ci"]; sig = not (ci[0] <= 0 <= ci[1])
        bx.scatter(i, yi, s=10 + 90 * abs(d), color=UP if d > 0 else DOWN, alpha=1 if sig else 0.25, linewidths=0)
        bx.text(i + 0.22, yi, f"{d:+.2f}", va="center", fontsize=PT_ASIDE - 3.5, color=INK if sig else FAINT)
bx.set_yticks(y); bx.set_yticklabels([lab(r) for r in rows], fontsize=PT_ASIDE - 1.5)
bx.set_xlim(-0.4, 2.75); bx.set_xticks(range(3)); bx.set_xticklabels(["L1$\\to$L2", "L2$\\to$L3", "L3$\\to$L4"], fontsize=PT_ASIDE - 1)
bx.set_xlabel("Cliff's $\\delta$ per step", fontsize=PT_ASIDE - 1, labelpad=2)
for s in ("top", "right"): bx.spines[s].set_visible(False)
bx.grid(axis="y", visible=False); seps([bx])
fig2.legend(handles=[Line2D([], [], marker="o", ls="", color=UP, label="rises at the step"),
                     Line2D([], [], marker="o", ls="", color=DOWN, label="falls at the step")],
            loc="lower center", ncol=2, frameon=False, fontsize=PT_ASIDE - 1, bbox_to_anchor=(0.5, 0.0))
fig2.savefig(OUT / "steps_delta.pdf")
# --- paper: capacity eta2 + Cliff's delta per step, full text width (figure*), shared labels
W3, H3 = 7.0, 2.45; L3, T3, B3 = 1.45, 0.2, 0.42
fig3 = plt.figure(figsize=(W3, H3))
wa = 1.7; gap = 0.25
ca = fig3.add_axes([L3 / W3, B3 / H3, wa / W3, 1 - (T3 + B3) / H3])
cb = fig3.add_axes([(L3 + wa + gap) / W3, B3 / H3, (W3 - L3 - wa - gap - 1.25) / W3, 1 - (T3 + B3) / H3], sharey=ca)
for yi, r in zip(y, rows):
    ca.barh(yi, r["cap"], 0.55, color=_pf.GREEN_TINT)
    ca.errorbar(r["cap"], yi, xerr=[[r["cap"] - r["cap_ci"][0]], [r["cap_ci"][1] - r["cap"]]], fmt="none", ecolor=_pf.INKGREEN, elinewidth=0.8, capsize=1)
ca.set_xlim(0, 1); ca.set_xticks([0, 0.25, 0.5, 0.75, 1]); ca.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=PT_ASIDE - 2)
ca.set_yticks(y); ca.set_yticklabels([lab(r) for r in rows], fontsize=PT_ASIDE - 1.5)
ca.set_title(r"partial $\eta^2$ on ranks, action set", fontsize=PT_ASIDE - 1, pad=3)
for i, s_ in enumerate(STEPS):
    for yi, r in zip(y, rows):
        st = r["steps"].get(s_)
        if st is None:
            cb.text(i, yi, "–", ha="center", va="center", fontsize=PT_ASIDE - 2.5, color=FAINT); continue
        d, ci = st["delta"], st["ci"]; sig = not (ci[0] <= 0 <= ci[1])
        cb.scatter(i, yi, s=6 + 60 * abs(d), color=UP if d > 0 else DOWN, alpha=1 if sig else 0.25, linewidths=0)
        cb.text(i + 0.12, yi, f"{d:+.2f}", va="center", fontsize=PT_ASIDE - 2.5, color=INK if sig else FAINT)
cb.set_xlim(-0.3, 2.55); cb.set_xticks(range(3)); cb.set_xticklabels(["L1$\\to$L2", "L2$\\to$L3", "L3$\\to$L4"], fontsize=PT_ASIDE - 1.5)
cb.tick_params(axis="y", left=False, labelleft=False)
cb.set_title(r"Cliff's $\delta$ per step", fontsize=PT_ASIDE - 1, pad=3)
for ax_ in (ca, cb):
    for sp in ("top", "right"): ax_.spines[sp].set_visible(False)
    ax_.grid(axis="y", visible=False)
cb.spines["left"].set_visible(False)
seps([ca, cb])
fig3.legend(handles=[Line2D([], [], marker="o", ls="", color=UP, label="rises at the step"),
                     Line2D([], [], marker="o", ls="", color=DOWN, label="falls at the step")],
            loc="center right", ncol=1, frameon=False, fontsize=PT_ASIDE - 1.5, bbox_to_anchor=(0.995, 0.5))
fig3.savefig(OUT / "capacity_steps.pdf")
print("ok", n)
