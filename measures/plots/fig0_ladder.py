"""fig0_ladder --- the capacity ladder as a design schematic (Theory, §1.4).

Not a data figure: it draws the design itself. Four steps left to right, each
carrying the action set of the level below plus exactly one addition, with the
addition in ink and the inheritance in faint; the top step also carries the
stock its new action draws on, which is the package the text prices. Under the
steps one bar names what never changes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _style import FAINT, GRID, INK, WIDTH, figure_tex, save, style, FIGURES

RUNGS = [
    ("L1", "give or hold", ["give", "hold"], []),
    ("L2", "predation", ["take", "strengthen"], ["give", "hold"]),
    ("L3", "association", ["drop", "invite"], ["take", "strengthen", "give", "hold"]),
    ("L4", "commons", ["harvest"], ["drop", "invite", "take", "strengthen", "give", "hold"]),
]

def main() -> None:
    style()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    xw, gap, x0 = 0.225, 0.0275, 0.005
    base = 0.235
    tops = [0.56, 0.70, 0.84, 0.98]
    for i, (rung, naam, nieuw, oud) in enumerate(RUNGS):
        x = x0 + i * (xw + gap)
        ax.add_patch(FancyBboxPatch((x, base), xw, tops[i] - base,
                     boxstyle="round,pad=0.004,rounding_size=0.012",
                     linewidth=0.8, edgecolor=FAINT, facecolor="white"))
        y = tops[i] - 0.075
        ax.text(x + xw / 2, y, f"{rung} · {naam}", ha="center",
                va="center", fontsize=8.2, color=INK, fontweight="bold")
        y -= 0.115
        ax.text(x + xw / 2, y, "+ " + ", ".join(nieuw), ha="center",
                va="center", fontsize=8.0, color=INK)
        if rung == "L4":
            y -= 0.105
            ax.text(x + xw / 2, y, "+ a shared stock", ha="center",
                    va="center", fontsize=8.0, color=INK, fontstyle="italic")
        if oud:
            y -= 0.105
            erf = ", ".join(oud) if len(oud) <= 4 else ", ".join(oud[:3]) + ",\n" + ", ".join(oud[3:])
            ax.text(x + xw / 2, y, erf, ha="center",
                    va="center", fontsize=7.4, color=FAINT)

    ax.add_patch(FancyBboxPatch((x0, 0.03), 4 * xw + 3 * gap, 0.155,
                 boxstyle="round,pad=0.004,rounding_size=0.012",
                 linewidth=0.8, edgecolor=FAINT, facecolor=GRID))
    ax.text(x0 + (4 * xw + 3 * gap) / 2, 0.107,
            "identical at every level — the channel · the memory · "
            "the network · the prices, set once",
            ha="center", va="center", fontsize=8.0, color=INK)

    fig.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005)
    save(fig, "fig0_ladder", "ladder")

    caption = (
        "Each level carries the action set of the level below plus exactly one "
        "addition (in ink; the inheritance in grey), and the top level also "
        "carries the stock its new action draws on: the package "
        "Section~\\ref{sec:varied} prices, while the bar beneath the four names "
        "what never changes across them.")
    tex = figure_tex("fig0_ladder", ["ladder"], "The capacity ladder",
                     caption, "fig:ladder", span=True)
    (FIGURES / "fig0_ladder" / "fig0_ladder.tex").write_text(tex)
    print("fig0_ladder geschreven")

if __name__ == "__main__":
    main()
