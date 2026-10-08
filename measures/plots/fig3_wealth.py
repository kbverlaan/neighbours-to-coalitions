"""Figure 3 — what each round does to what an agent holds.

    python3 plots/fig3_wealth.py

Thirty agents down the side, sixty rounds across, and a cell coloured by the
change in that agent's holdings over that round: blue where it gained, vermilion
where it lost, pale where nothing much happened. One run per capacity level.

Why the change and not the level. A map of holdings shows that a row is brighter
than its neighbours and says nothing about how it got that way; and since almost
everything drains slowly, a level map is four panels of gradual fading. The
change separates the two things that are actually happening. The slow bleed is
the fee, and it is the pale wash over everything: 81 per cent of agent-rounds at
L3 end lower than they began, at a median of one resource. The hunt is the
handful of cells that are not pale --- one deep vermilion mark where an agent is
brought down, with a row of blue in the same column where the coalition divides
what it took.

What the reader should be able to see without being told. At L1 pairs light up
together and stay lit. At L2 the marks are scattered and small. At L3 the
brightest rows are struck one after another, and each strike pays a crowd. At L4
almost nothing happens to anybody: the commons is harvested on a schedule and
the grid is nearly empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HERE.parent / _m))

import combat          # noqa: E402
import logs            # noqa: E402
import runstat         # noqa: E402
import runset          # noqa: E402
from _style import (PAYOFF, PT_ASIDE, PT_LABEL, PT_TITLE, RUNG, WIDTH,  # noqa: E402
                    figure_tex, save, style)

FIGURE = "fig3_wealth"
ROUNDS = 60
PANELS = ["prod_L1_abund", "prod_L2_knife", "prod_L3_knife", "prod_L4_knife"]

LOSS = "#D55E00"        # vermilion, as in Figure 2: the agent being taken from
GAIN = "#0072B2"


def deltas(path: Path):
    """Change in holdings per agent per round, and what the run did overall.

    The fights are read too, so a loss can be told apart from a bleed. Without
    them the panel says an agent ended the round lower and leaves the reader to
    guess whether that was the fee or a coalition, which is the whole
    distinction the capacity level turns on.
    """
    entries = {e.get("round"): e for e in logs.rounds(path)}
    names = sorted({n for e in entries.values() for n in (e.get("agents") or {})})
    index = {n: i for i, n in enumerate(names)}

    held = np.full((len(names), ROUNDS + 1), np.nan)
    for r in range(1, ROUNDS + 1):
        for n, a in ((entries.get(r) or {}).get("agents") or {}).items():
            held[index[n], r] = a.get("resources") or 0
    grid = np.diff(held, axis=1)          # column r is round r+1 less round r

    finite = grid[np.isfinite(grid)]

    # The deepest fall from a peak, per agent, and the worst of them. A single
    # round is the wrong unit for what the chapter calls the hunt: the agent
    # that loses most does not lose it in one blow but over several rounds, and
    # naming only the heaviest of those rounds understates it by a factor of
    # two or three. What is marked instead is the whole descent --- the peak,
    # the floor it reaches, and how many rounds lie between.
    fall = None
    for i, name in enumerate(names):
        row = held[i]
        peak = peak_at = None
        for r in range(1, ROUNDS + 1):
            v = row[r]
            if not np.isfinite(v):
                continue
            if peak is None or v > peak:
                peak, peak_at = v, r
            drop = peak - v
            if peak > 0 and (fall is None or drop > fall["drop"]):
                fall = {"who": name, "row": i, "from": float(peak),
                        "to": float(v), "drop": float(drop),
                        "peak_at": peak_at, "at": r}
    struck, joined = [], []
    for r, fight in combat.fights(path):
        if not 1 <= (r or 0) <= ROUNDS:
            continue
        d = fight.get("defender")
        if d in index:
            struck.append((r, index[d]))
        for a in (fight.get("attackers") or []):
            if a in index and a != d:
                joined.append((r, index[a]))

    return {"grid": grid, "names": names, "path": path, "fall": fall,
            "struck": struck, "joined": joined,
            "share_down": 100 * float((finite < 0).mean()) if finite.size else 0,
            "worst": float(finite.min()) if finite.size else 0,
            "best": float(finite.max()) if finite.size else 0}


def pick(cell: str):
    """The run whose final inequality is the cell's median.

    Not the most extreme run and not the calmest: the panel should be one a
    reader could have drawn at random from the cell and recognised.
    """
    scored = sorted((runstat.final_gini(p), p.name, p) for p in runset.cel(cell))
    return scored[len(scored) // 2][2]


def panel(ax, d, cell, norm, cmap):
    n = len(d["names"])
    ax.pcolormesh(np.arange(0.5, ROUNDS + 1.5), np.arange(-0.5, n + 0.5),
                  d["grid"], cmap=cmap, norm=norm, rasterized=True)
    ax.set_xlim(0.5, ROUNDS + 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([1, 15, 30, 45, 60])
    ax.tick_params(axis="x", labelsize=PT_ASIDE, length=2, pad=1.5)
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(visible=False)

    # The heaviest single round, named. A wash of colour says an economy moved;
    # a name and a number say what moving meant for somebody in it.
    # A cross on the agent that was attacked, an open dot on each agent that
    # joined the attack. The colour already says an agent lost; these say the
    # loss was taken from it.
    if d["joined"]:
        jr, jc = zip(*[(r, c) for r, c in d["joined"]])
        ax.scatter(jr, jc, s=1.6, marker="o", facecolor="none",
                   edgecolor="#1a1a1a", linewidth=0.3, alpha=0.65, zorder=3)
    if d["struck"]:
        sr, sc = zip(*[(r, c) for r, c in d["struck"]])
        ax.scatter(sr, sc, s=5.0, marker="x", color="#1a1a1a", linewidth=0.5,
                   alpha=0.85, zorder=4)

    # The steepest fall is computed and reported in the text, not drawn. A name
    # and an arrow in the panel turns four regimes into one agent's story, and
    # this is the one figure in the chapter whose subject is the regime.

    rung, payoff = cell.split("_")[1], cell.split("_")[2]
    ax.set_title(f"{rung} · {RUNG[rung]}, {PAYOFF[payoff]}", fontsize=PT_TITLE,
                 loc="left", pad=3, fontweight="bold")
    # Clear of the round numbers. Beside the title, as in Figures 1 and 4, it
    # collides with it instead: these panels are half the width of those.
    ax.annotate(f"{d['share_down']:.0f}% of agent-rounds end lower  ·  "
                f"run {d['path'].name.split('__')[1].split('_')[0]}",
                (0.0, 0.0), xycoords="axes fraction", xytext=(0, -19),
                textcoords="offset points", fontsize=PT_ASIDE,
                color="#8a8a8a", ha="left", va="top")


def build():
    style()
    data = [(cell, deltas(pick(cell))) for cell in PANELS]

    # Symmetric and symmetric-log. The median round is a loss of one and the
    # worst is a loss of two hundred, so a linear scale renders the bleed as
    # nothing and a log scale cannot cross zero. `linthresh` is set at 2, which
    # is where a round stops being the fee and starts being an event.
    span = max(max(abs(d["worst"]), abs(d["best"])) for _, d in data)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "delta", [LOSS, "#f6dccb", "#f7f7f7", "#cfe2ee", GAIN])
    cmap.set_bad("#ffffff")
    norm = mcolors.SymLogNorm(linthresh=2, vmin=-span, vmax=span, base=10)

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.85))
    fig.subplots_adjust(left=0.035, right=0.87, top=0.91, bottom=0.15,
                        wspace=0.10, hspace=0.60)
    for ax, (cell, d) in zip(axes.flat, data):
        panel(ax, d, cell, norm, cmap)

    fig.text(0.018, 0.5, "thirty agents, alphabetical", rotation=90,
             va="center", ha="center", fontsize=PT_ASIDE, color="#666666")
    # No "round" label: the axis reads 1 to 60 under every panel and saying so
    # a fifth time only crowds the line the legend needs.
    fig.text(0.45, 0.018,
             "\u00d7 attacked by a coalition that round   \u00b7   "
             "\u25cb took part in the attack   \u00b7   "
             "pale wash: the 3% fee, paid every round by everyone",
             ha="center", fontsize=PT_ASIDE, color="#666666")

    bar = fig.add_axes([0.90, 0.22, 0.014, 0.56])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar)
    cb.set_label("change in holdings over the round", fontsize=PT_ASIDE,
                 labelpad=4)
    cb.set_ticks([-100, -10, 0, 10, 100])
    cb.ax.tick_params(labelsize=PT_ASIDE, length=2, pad=1.5)
    cb.outline.set_linewidth(0)
    return fig, data


def caption(data) -> str:
    bits = []
    for cell, d in data:
        seed = d["path"].name.split("__")[1].split("_")[0]
        bits.append(f"\\texttt{{{seed}}} ({cell.split('_')[1]} "
                    f"{PAYOFF[cell.split('_')[2]]}, {d['share_down']:.0f}\\% of "
                    f"agent-rounds ending lower)")
    return ("Thirty agents down the side in alphabetical order, sixty rounds "
            "across, and a cell coloured by the change over that round, blue "
            "for a gain and vermilion for a loss and pale where little "
            "happened. A cross marks the agent a coalition attacked that round "
            "and an open dot each agent that joined it, which is what separates "
            "a loss to a blow from the slow bleed of the fee. The scale is "
            "symmetric and logarithmic either side of zero, since the median "
            "round is a loss of one resource and the worst a loss of a hundred. "
            "One run per capacity level, each the run whose final inequality is "
            "its cell's median: " + "; ".join(bits) + ".")

if __name__ == "__main__":
    fig, data = build()
    save(fig, FIGURE, "deltas")
    path = HERE / "figures" / FIGURE / f"{FIGURE}.tex"
    path.write_text(figure_tex(FIGURE, ["deltas"],
                               "Round-by-round change in holdings",
                               caption(data), "fig:wealth"))
    for cell, d in data:
        print(f"   {cell:16} {d['share_down']:4.0f}% down  worst {d['worst']:8.1f}  "
              f"best {d['best']:7.1f}  run {d['path'].name.split('__')[1].split('_')[0]}")
    print(f"-> {path.parent}/")
