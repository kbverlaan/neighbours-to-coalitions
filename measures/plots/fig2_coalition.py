"""Figure 2 — the largest attack each capacity level produced.

    python3 plots/fig2_coalition.py

Two scenes. The target at the centre, everyone it was connected to in that round
around it: filled where they joined the attack, hollow where they could have and
did not. A line is drawn for each attacker.

Why this figure exists. The chapter reads the L2 ceiling of seven attackers as
behavioural rather than structural, and supports it by reporting how many
neighbours the struck agents had --- nine, ten and thirteen. That asks a reader
to hold three numbers and picture the gap. Here the gap is the six empty circles
in the left-hand ring.

Which scene. The largest coalition each capacity level produced anywhere. At L2 that is
seven attackers, reached in all three payoff cells, and the scene drawn is the
one where the target had the most neighbours, so the room left unused is at its
clearest. At L3 it is twenty-nine attackers on an agent with twenty-nine
neighbours: everyone still alive who could reach the target did.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HERE.parent / _m))

import combat          # noqa: E402
import logs            # noqa: E402
import runset          # noqa: E402
from _style import (INK, PAYOFF, PT_ASIDE, PT_LABEL, PT_TITLE,  # noqa: E402
                    RUNG, WIDTH, figure_tex, save, style)

FIGURE = "fig2_coalition"
TARGET = "#D55E00"            # the agent attacked
JOINED = "#0072B2"            # a neighbour who struck
CELLS = {"L2": ("prod_L2_scar", "prod_L2_knife", "prod_L2_abund"),
         "L3": ("prod_L3_scar", "prod_L3_knife", "prod_L3_abund")}


def neighbours(entry, name) -> set:
    """Who the agent was connected to in that round, from the round's graph."""
    out = set()
    for a, b in (entry.get("network") or {}).get("edges") or []:
        if a == name:
            out.add(b)
        elif b == name:
            out.add(a)
    return out


def scene(rung: str):
    """The largest coalition the capacity level produced, and the graph around its target.

    Ties on coalition size are broken by the target's neighbourhood, so the
    scene shown is the one with the most room left unused. That is a choice
    about which fight illustrates the claim and the caption says so; the claim
    itself rests on every fight in the cell, not this one.
    """
    best = None
    for cell in CELLS[rung]:
        for path in runset.cel(cell):
            per_round = {e.get("round"): e for e in logs.rounds(path)}
            for r, fight in combat.fights(path):
                attackers = list(fight.get("attackers") or [])
                target = fight.get("defender")
                ring = neighbours(per_round.get(r) or {}, target)
                key = (len(attackers), len(ring))
                if best is None or key > best[0]:
                    best = (key, cell, path.name, r, target, attackers, ring)
    (_, ring_n), cell, run, r, target, attackers, ring = best
    return {"cell": cell, "run": run, "round": r, "target": target,
            "attackers": attackers, "ring": ring, "rung": rung}


def draw(ax, s: dict):
    """The target at the centre, its neighbourhood on a circle around it.

    Attackers are placed first and abstainers after them, so the coalition sits
    together rather than interleaved. That is a readability choice and it costs
    something: the ring is not the graph's own layout, and adjacency between two
    neighbours is not drawn at all. What the figure claims is only who could
    have joined and who did.
    """
    joined = [a for a in s["attackers"] if a != s["target"]]
    # An attacker need not be a neighbour in the logged graph --- at L3 an agent
    # can rewire in the same round it strikes --- so the ring is the union.
    held_back = sorted(s["ring"] - set(joined))
    ring = joined + held_back
    n = len(ring)

    # Start at the top and run clockwise, so the first attacker is at twelve
    # o'clock and the block of abstainers, if any, closes the circle.
    angles = np.pi / 2 - np.arange(n) * 2 * np.pi / n
    R = 1.0
    xs, ys = R * np.cos(angles), R * np.sin(angles)

    # The ring itself, faintly. Without it the empty circles read as scattered
    # points; with it they read as gaps in something, which is the claim.
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(circle), R * np.sin(circle), color="#dddddd",
            linewidth=0.5, zorder=0)

    for x, y, name in zip(xs, ys, ring):
        struck = name in joined
        if struck:
            ax.plot([0, x], [0, y], color=JOINED, linewidth=0.55, alpha=0.55,
                    zorder=1, solid_capstyle="round")
        ax.scatter([x], [y], s=26,
                   facecolor=JOINED if struck else "white",
                   edgecolor=JOINED, linewidth=0.7, zorder=3)
        # Names read outward, upright on the right of the circle and flipped on
        # the left, so none of them is upside down.
        angle = np.degrees(np.arctan2(y, x))
        flip = 90 < angle % 360 < 270
        ax.annotate(name, (x, y), xytext=(x * 1.13, y * 1.13),
                    textcoords="data", ha="right" if flip else "left",
                    va="center", fontsize=5.4, color="#444444",
                    rotation=angle + 180 if flip else angle,
                    rotation_mode="anchor", annotation_clip=False)

    ax.scatter([0], [0], s=95, facecolor=TARGET, edgecolor="white",
               linewidth=0.8, zorder=4)
    # The target's name in ink rather than in the target colour, and outlined:
    # vermilion on white at 6.4pt is thin, and here it sits on top of however
    # many lines converge on the centre --- twenty-nine of them on the right.
    label = ax.annotate(s["target"], (0, 0), xytext=(0, -0.245),
                        textcoords="data", ha="center", va="top",
                        fontsize=PT_LABEL, color=INK, fontweight="bold",
                        zorder=5)
    label.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")

    payoff = s["cell"].split("_")[2]
    rung = s["rung"]
    ax.set_title(f"{rung} · {RUNG[rung]}, {PAYOFF[payoff]} payoff",
                 fontsize=PT_TITLE, loc="left", pad=2, fontweight="bold")
    ax.annotate(f"{len(joined)} of {len(ring)} neighbours struck "
                f"{s['target']} in round {s['round']}",
                (0.0, -0.035), xycoords="axes fraction", fontsize=PT_ASIDE,
                color="#666666", ha="left", va="top")
    # Which run, as in Figure 1: a panel should be checkable against the data
    # without going to the caption for the seed.
    ax.annotate(f"run {s['run'].split('__')[1].split('_')[0]}",
                (1.0, 1.0), xycoords="axes fraction", xytext=(0, 2),
                textcoords="offset points", ha="right", va="baseline",
                fontsize=PT_ASIDE, color="#8a8a8a")


def build():
    style()
    scenes = [scene("L2"), scene("L3")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.15))
    for ax, s in zip(axes, scenes):
        draw(ax, s)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.10,
                        wspace=0.06)

    # One legend for both panels, said once and in words rather than as a key.
    fig.text(0.5, 0.012,
             "filled: struck the target that round   ·   hollow: was connected "
             "to it and did not   ·   line: one blow",
             fontsize=PT_ASIDE, color="#555555", ha="center")
    return fig, scenes


def caption(scenes) -> str:
    bits = []
    for s in scenes:
        joined = len([a for a in s["attackers"] if a != s["target"]])
        ring = len(set(s["attackers"]) - {s["target"]} | s["ring"])
        seed = s["run"].split("__")[1].split("_")[0]
        bits.append(f"{joined} of {ring} at {s['rung']} "
                    f"{PAYOFF[s['cell'].split('_')[2]]} in round {s['round']} "
                    f"of run \\texttt{{{seed}}}")
    return ("The largest coalition each capacity level produced anywhere in "
            "the run set, drawn as the graph around the agent it struck. The "
            "target is at the centre and every agent connected to it that round "
            "is on the ring, filled where it joined the blow and hollow where it "
            "did not, with a line for each attacker. Attackers are placed first "
            "and abstainers after them, so the ring is not the graph's own "
            "layout and ties between two neighbours are not drawn. The two "
            "scenes are " + "; ".join(bits) + " \\meth{m:many-attack-once}.")

if __name__ == "__main__":
    fig, scenes = build()
    save(fig, FIGURE, "coalitions")
    path = HERE / "figures" / FIGURE / f"{FIGURE}.tex"
    path.write_text(figure_tex(
        FIGURE, ["coalitions"], "The largest attack at each capacity level",
        caption(scenes), "fig:coalition"))
    for s in scenes:
        joined = len([a for a in s["attackers"] if a != s["target"]])
        print(f"   {s['rung']}  {joined:2} attackers on {s['target']:8} "
              f"({len(s['ring'])} neighbours)  {s['cell']:16} round {s['round']}")
    print(f"-> {path.parent}/")
