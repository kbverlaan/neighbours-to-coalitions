"""Hoofdfiguur AAMAS: alle geregistreerde dv's + de na registratie toegevoegde maten.
Links: rang-eta2 capaciteit vs prijs (95%-bootstrap). Rechts: Cliff's delta per
trede-overgang, gepoold over de prijs; vaag waar het interval nul bevat.
Leest prereg_fingerprint.json (geregistreerd) en per_run_8.json (toegevoegd).
    /opt/homebrew/bin/python3.10 paper_aamas/main_figure.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "plots"))
from _style import FAINT, INK, PT_ASIDE, style  # noqa: E402
from _paths import PAPER
RUNGS = ("L1", "L2", "L3", "L4"); PAYS = ("scar", "knife", "abund"); STEPS = ("L1-L2", "L2-L3", "L3-L4")
CAP, PRICE, UP, DOWN = "#0072B2", "#D55E00", "#1b7837", "#762a83"
SEED, RES = 20260817, 2000

reg = json.loads((HIER / "prereg_fingerprint.json").read_text())
lo8 = json.loads((HIER / "leave_one_rung_out.json").read_text())["alle"]
per8 = {tuple(k.split("|")): v for k, v in json.loads((HIER / "per_run_8.json").read_text()).items()}
rng = np.random.default_rng(SEED)
def cliff(a, b):
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[None, :]
    return float(((b > a).sum() - (b < a).sum()) / (a.size * b.size))
def boot(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = [cliff(lo[rng.integers(0, len(lo), len(lo))], hi[rng.integers(0, len(hi), len(hi))]) for _ in range(RES)]
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
ADDED = [("names_any", "named agreement present"), ("pairs_any", "reciprocal pair present"),
         ("dyads", "number of pairs"), ("econ", "summed final economy"),
         ("floor_ratio", "poorest / mean holding"), ("gini", "final Gini")]
added = []
for k, lab in ADDED:
    st = {}
    for a, b in zip(RUNGS, RUNGS[1:]):
        xa = [v[k] for p in PAYS for v in per8[(a, p)]]; xb = [v[k] for p in PAYS for v in per8[(b, p)]]
        st[f"{a}-{b}"] = {"delta": round(cliff(xa, xb), 2), "ci": boot(xa, xb)}
    v = lo8[k]
    added.append({"key": k, "label": lab, "family": "added", "rungs": list(RUNGS), "cap": v["cap"], "price": v["price"],
                  "cap_ci": v["cap_ci"], "price_ci": v["price_ci"], "steps": st})
rows = [r for r in reg if r["family"] == "F1"] + [r for r in reg if r["family"] == "F2"]
REGISTERED_ONLY = True  # toegevoegde maten staan in de tekst, niet in de figuur
(HIER / "main_figure_rows.json").write_text(json.dumps(rows + added, indent=1))
GROUPS = {"F1": "structure", "F2": "norms"}

style()
W, H = 7.0, 3.5
fig = plt.figure(figsize=(W, H))
L, T, B = 1.75, 0.12, 0.62
axA = fig.add_axes([L / W, B / H, 2.35 / W, 1 - (T + B) / H])
axB = fig.add_axes([(L + 2.35 + 0.45) / W, B / H, 1.6 / W, 1 - (T + B) / H], sharey=axA)
n = len(rows); y = np.arange(n)[::-1]; h = 0.36
def lab(r):
    return r["label"] + ("" if len(r["rungs"]) == 4 else f"  ({r['rungs'][0]}--{r['rungs'][-1]})")
for yi, r in zip(y, rows):
    for off, val, ci, col in ((h/2, r["cap"], r["cap_ci"], CAP), (-h/2, r["price"], r["price_ci"], PRICE)):
        axA.barh(yi + off, val, h, color=col)
        axA.errorbar(val, yi + off, xerr=[[val - ci[0]], [ci[1] - val]], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1)
axA.set_yticks(y); axA.set_yticklabels([lab(r) for r in rows], fontsize=PT_ASIDE - 1)
axA.set_xlim(0, 1); axA.set_xlabel(r"partial $\eta^2$ on ranks", fontsize=PT_ASIDE - 1, labelpad=3)
axA.tick_params(axis="x", labelsize=PT_ASIDE - 2)
for i, s in enumerate(STEPS):
    for yi, r in zip(y, rows):
        st = r["steps"].get(s)
        if st is None:
            axB.text(i, yi, "--", ha="center", va="center", fontsize=PT_ASIDE - 2, color=FAINT); continue
        d, ci = st["delta"], st["ci"]; sig = not (ci[0] <= 0 <= ci[1])
        axB.scatter(i, yi, s=10 + 90 * abs(d), color=UP if d > 0 else DOWN, alpha=1 if sig else 0.25, linewidths=0)
        axB.text(i + 0.28, yi, f"{d:+.2f}", va="center", fontsize=PT_ASIDE - 3, color=INK if sig else FAINT)
axB.set_xlim(-0.4, 2.75); axB.set_xticks(range(3)); axB.set_xticklabels(["L1$\\to$L2", "L2$\\to$L3", "L3$\\to$L4"], fontsize=PT_ASIDE - 1)
axB.tick_params(axis="y", left=False, labelleft=False); axB.set_xlabel("Cliff's $\\delta$ per step", fontsize=PT_ASIDE - 1, labelpad=3)
for ax in (axA, axB):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", visible=False)
axB.spines["left"].set_visible(False)
for i in range(1, n):
    if rows[i]["family"] != rows[i-1]["family"]:
        for ax in (axA, axB): ax.axhline(y[i] + 0.5, color=FAINT, linewidth=0.5)
for fam, name in GROUPS.items():
    idx = [i for i, r in enumerate(rows) if r["family"] == fam]
    axB.text(2.95, (y[idx[0]] + y[idx[-1]]) / 2, name, fontsize=PT_ASIDE - 2, color=FAINT, style="italic", va="center")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hd = [Patch(color=CAP, label="capacity level"), Patch(color=PRICE, label="price"),
      Line2D([], [], marker="o", ls="", color=UP, label="rises at the step"), Line2D([], [], marker="o", ls="", color=DOWN, label="falls at the step")]
fig.legend(handles=hd, loc="lower center", ncol=4, frameon=False, fontsize=PT_ASIDE - 1, bbox_to_anchor=(0.5, 0.0))
fig.savefig(PAPER / "figures/main_levers.pdf"); print("ok", n, "rijen")
