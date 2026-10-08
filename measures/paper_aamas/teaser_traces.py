"""Openingsfiguur AAMAS (7 okt 2026): de redeneertraces per trede, uit plots/fig9_trajectories.py.

Hergebruikt projectie (gecachte, geseede UMAP), trajecten en scheidingsmaten van de scriptiefiguur;
alleen labels (rewiring, shared stock), breedte (ACM textwidth) en de paneelletter wijken af.
  uv run --python 3.12 --with numpy --with matplotlib --with umap-learn python paper_aamas/teaser_traces.py
"""
import sys
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent
sys.path.insert(0, str(M / "plots"))
import fig9_trajectories as f9                     # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402

from _paths import PAPER
OUT = PAPER / "figures/teaser_traces.pdf"
LABEL = {"L1": "giving", "L2": "+ predation", "L3": "+ rewiring", "L4": "+ shared stock"}

f9.style()
import _paperfont; _paperfont.apply()
import _style; _style.COLOUR.update(_paperfont.PRICE_CELLS)
E, rows, meta = f9.cache.load()
xy = f9.projection(E)
traj = f9.trajectories(xy, rows)
sep = f9.separation(xy, rows)
f9.WIDTH = 7.0                                      # ACM sigconf \textwidth
fig = f9.plate_trajectory(xy, rows, traj, meta)
for ax, rung in zip(fig.axes[:4], f9.RUNGS):
    ax.set_title(f"{rung} · {LABEL[rung]}", fontsize=f9.PT_TITLE, loc="left", pad=3, fontweight="bold")
fig.set_size_inches(7.0, 2.1)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=300); fig.savefig(OUT.with_suffix(".png"), dpi=160)
print(OUT); print(meta); print(f9.geography(sep))
print({f"{a}-{b}": round(v, 2) for (a, b), v in sep["levels"].items()})
