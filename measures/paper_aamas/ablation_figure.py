"""Figuur: Cliff's delta voor kanaal- en modelablatie (leest ablations.json)."""
import json, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
HIER = Path(__file__).resolve().parent; sys.path.insert(0, str(HIER.parent / "plots"))
from _style import FAINT, INK, PT_ASIDE, style  # noqa: E402
from _paths import PAPER
import _paperfont as _pf
UP, DOWN = _pf.GREEN, _pf.PEACH
d = json.loads((HIER / "ablations.json").read_text())
style()
_pf.apply()
panels = [("channel", ("L1","L2","L3","L4"), "channel removed"), ("model", ("L2","L3","L4"), "Qwen instead of Gemma")]
nrows = [len(d[p]) for p, _, _ in panels]
W, H = 3.33, 0.23 * (sum(nrows)) + 1.1
fig = plt.figure(figsize=(W, H))
top = 1 - 0.28 / H; L = 1.35 / W; wA = 1 - L - 0.12 / W
y0 = top
for (name, rungs, title), n in zip(panels, nrows):
    h = n * 0.23 / H
    ax = fig.add_axes([L, y0 - h, wA, h]); y0 -= h + 0.42 / H
    items = list(d[name].values()); ys = np.arange(n)[::-1]
    for i, r in enumerate(rungs):
        for yi, it in zip(ys, items):
            st = it.get(r)
            if st is None:
                ax.text(i, yi, "–", ha="center", va="center", fontsize=PT_ASIDE - 3, color=FAINT); continue
            dl, ci = st["delta"], st["ci"]; sig = not (ci[0] <= 0 <= ci[1])
            ax.scatter(i, yi, s=6 + 55 * abs(dl), color=UP if dl > 0 else DOWN, alpha=1 if sig else 0.25, linewidths=0)
            ax.text(i + 0.27, yi, f"{dl:+.2f}", va="center", fontsize=PT_ASIDE - 4, color=INK if sig else FAINT)
    ax.set_yticks(ys); ax.set_yticklabels([it["label"] for it in items], fontsize=PT_ASIDE - 2)
    ax.set_xticks(range(len(rungs))); ax.set_xticklabels(rungs, fontsize=PT_ASIDE - 2)
    ax.set_xlim(-0.4, len(rungs) - 0.3); ax.set_ylim(-0.6, n - 0.4)
    ax.set_title(title, fontsize=PT_ASIDE - 1, loc="left", pad=3)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", left=False); ax.grid(False)
fig.savefig(PAPER / "figures/ablations.pdf"); print("ok")
