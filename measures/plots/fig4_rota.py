"""Figure 4 — the rota nobody agreed, and the run where it never formed.

    python3 plots/fig4_rota.py

Thirty agents down the side in alphabetical order, sixty rounds across, a mark
where that agent harvested in that round. A round whose harvesters form a
contiguous alphabetical block is drawn in colour; every other harvest round is
grey.

Why two panels, and what the difference between them is not. Of the ten scarce
runs four score no alphabetical block at all and six score them in half to nine
tenths of their rounds, so the cell figure of 42.7 per cent describes no run in
the cell. It is tempting to read the two halves as rota and no rota. Both panels
are rotas. The upper one harvests five agents a round on a six-round cycle, the
lower one ten a round on a three-round cycle, and in both the great majority of
an agent's intervals sit at that period. What differs is the order the turns are
taken in: down the alphabet on the left, and by some other assignment on the
right that the block test cannot see.

Across the scarce cell 84 per cent of intervals sit at the run's own period
against 22 per cent for the same harvests scattered at random, and no run
harvests every round \meth{m:harvest-rhythm}. Turn-taking at the stock is
general. The alphabet is one way of filling it in.
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HERE.parent / _m))

import logs            # noqa: E402
import runset          # noqa: E402
from _style import (COLOUR, PAYOFF, PT_ASIDE, PT_LABEL, PT_TITLE, RUNG,  # noqa: E402
                    WIDTH, figure_tex, margins, save, style, tint)

FIGURE = "fig4_rota"
CELL = "prod_L4_scar"
ROUNDS = 60
ROW_H = 0.088                 # inches per agent
HEAD_FOOT = 0.52              # inches for the title and the axis
NAME_MARGIN = 0.86            # inches reserved for the agent names


def harvests(path: Path):
    """(agents, matrix, block-rounds, share). Rows are agents, columns rounds.

    The block test is the measure's, verbatim: the harvesters of a round are a
    contiguous run in the alphabetical list of that round's agents. Rounds with
    a single harvester are not scored --- one name is a block by definition ---
    and they are drawn as harvests all the same, since the panel shows what
    happened and not only what was scored.
    """
    entries = {e.get("round"): e for e in logs.rounds(path)}
    names = sorted({n for e in entries.values() for n in (e.get("agents") or {})})
    index = {n: i for i, n in enumerate(names)}

    grid = np.zeros((len(names), ROUNDS), dtype=int)   # 0 none, 1 harvest, 2 in slot
    blocks, scored, chance = set(), 0, 0.0
    gaps, last, per_round, picked_in = [], {}, [], {}
    for r in range(1, ROUNDS + 1):
        e = entries.get(r) or {}
        agents = e.get("agents") or {}
        picked = sorted(n for n, a in agents.items() if a.get("action") == "harvest")
        for n in picked:
            grid[index[n], r - 1] = 1
            if n in last:
                gaps.append(r - last[n])
            last[n] = r
        if picked:
            per_round.append(len(picked))
        picked_in[r] = picked
        if len(picked) < 2:
            continue
        alive = sorted(agents)
        idx = sorted(alive.index(x) for x in picked)
        scored += 1
        if idx == list(range(idx[0], idx[0] + len(idx))):
            blocks.add(r)
        n_, k = len(alive), len(picked)
        chance += (n_ - k + 1) / comb(n_, k) if k <= n_ else 0.0
    share = 100 * len(blocks) / scored if scored else 0.0
    # The run's own rhythm, the same quantity as \meth{m:harvest-rhythm}: the
    # commonest interval between one agent's harvests. Stated on the panel
    # because the two runs differ in it, and a reader who sees only the block
    # score would take the lower panel for a run without a rota.
    period = max(set(gaps), key=gaps.count) if gaps else 0
    harvesters = float(np.median(per_round)) if per_round else 0.0

    # Colour marks a harvest taken in the agent's own slot of the rota, not one
    # that happens to fall in an alphabetical block. Both panels are rotas and
    # colouring on the alphabet left the lower one uniformly grey, which reads
    # as a run with no schedule --- the opposite of what it is. Each agent's
    # slot is the phase it harvests in most often from round 20 on, once the
    # schedule has settled; what then differs between the panels is the shape
    # the slots make, not whether they exist.
    settled_at = 20
    phases = {}
    if period > 1:
        for r, picked in picked_in.items():
            if r <= settled_at:
                continue
            for n in picked:
                phases.setdefault(n, []).append(r % period)
        slot = {n: max(set(v), key=v.count) for n, v in phases.items()}
        for r, picked in picked_in.items():
            for n in picked:
                if n in slot and r % period == slot[n]:
                    grid[index[n], r - 1] = 2
    in_slot = 100 * float((grid == 2).sum()) / max(float((grid > 0).sum()), 1)
    return {"names": names, "grid": grid, "rounds": scored, "share": share,
            "chance": (100 * chance / scored if scored else 0.0),
            "period": period, "harvesters": harvesters, "in_slot": in_slot,
            "path": path}


def pick():
    """The scarce run with the most alphabetical blocks, and one with none.

    Both from the same cell, so the comparison is between runs and not between
    prices. Ties are broken by the number of harvest rounds, so neither panel is
    a run that barely harvested.
    """
    scored = [harvests(path) for path in runset.cel(CELL)]
    scored.sort(key=lambda d: (d["share"], d["rounds"]))
    return scored[-1], scored[0]


def panel(ax, item, title_note):
    names, grid = item["names"], item["grid"]
    share, n_rounds = item["share"], item["rounds"]
    period, harvesters, path = item["period"], item["harvesters"], item["path"]
    in_slot = item["in_slot"]
    payoff = CELL.split("_")[2]
    colour = COLOUR[payoff]

    for row in range(len(names)):
        for r in range(ROUNDS):
            v = grid[row, r]
            if not v:
                continue
            ax.add_patch(plt.Rectangle(
                (r + 0.6, row - 0.36), 0.8, 0.72,
                facecolor=colour if v == 2 else "#c8c8c8",
                edgecolor="none", zorder=2))

    ax.set_xlim(0.5, ROUNDS + 0.5)
    ax.set_ylim(len(names) - 0.5, -0.5)      # first name at the top
    ax.set_xticks([1, 15, 30, 45, 60])
    ax.set_xlabel("round", fontsize=PT_LABEL)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=PT_LABEL, length=2.5, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(visible=False)

    rung = "L4"
    ax.set_title(f"{rung} · {RUNG[rung]}, {PAYOFF[payoff]} — {title_note}",
                 fontsize=PT_TITLE, loc="left", pad=4, fontweight="bold")
    ax.annotate(f"{harvesters:.0f} a round every {period}  ·  {in_slot:.0f}% in own "
                f"slot  ·  {share:.0f}% alphabetical  ·  "
                f"run {path.name.split('__')[1].split('_')[0]}",
                (1.0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", ha="right", va="baseline",
                fontsize=PT_ASIDE, color="#8a8a8a")


def build():
    style()
    top, bottom = pick()
    figs = []
    for item, note, name in ((top, "an alphabetical rota", "alphabetical"),
                             (bottom, "a rota on another order", "other_order")):
        height = ROW_H * len(item["names"]) + HEAD_FOOT
        fig, ax = plt.subplots(figsize=(WIDTH, height))
        panel(ax, item, note)
        margins(fig, NAME_MARGIN, height)
        figs.append((fig, name, item))
    return figs


def caption(items) -> str:
    bits = []
    for item, name in items:
        seed = item["path"].name.split("__")[1].split("_")[0]
        bits.append(f"run \\texttt{{{seed}}}, {item['share']:.0f}\\% of "
                    f"{item['rounds']} harvest rounds, "
                    f"{item['harvesters']:.0f} agents a round every "
                    f"{item['period']} rounds")
    return ("Thirty agents in alphabetical order against sixty rounds, with a "
            "mark where that agent harvested in that round, drawn in colour "
            "where the harvest fell in the agent's own slot, so the grey "
            "opening of each panel is the rounds before the schedule settles "
            "and the colour is the schedule running. Both panels are the L4 "
            "scarce cell and both are rotas: " + "; ".join(bits) + ". In each "
            "the population divides exactly into cohorts, harvesters a round "
            "times the period equalling the thirty agents, and once an agent "
            "has a slot it keeps it. What differs is the order the turns are "
            "taken in, since the upper panel's cohorts are blocks of the agent "
            "list while the lower panel's are drawn from across it "
            "\\meth{m:harvest-rhythm} \\meth{m:commons-capacity-level}.")

if __name__ == "__main__":
    figs = build()
    files, items = [], []
    for fig, name, item in figs:
        save(fig, FIGURE, name)
        plt.close(fig)
        files.append(name)
        items.append((item, name))
        print(f"   {name:14} {item['share']:5.1f}% blocks of {item['rounds']:2} "
              f"rounds  ·  {item['harvesters']:.0f} a round every "
              f"{item['period']}  ·  run "
              f"{item['path'].name.split('__')[1].split('_')[0]}")
    path = HERE / "figures" / FIGURE / f"{FIGURE}.tex"
    path.write_text(figure_tex(
        FIGURE, files, "The alphabetical harvest rota", caption(items), "fig:rota"))
    print(f"-> {path.parent}/")
