"""Figure 1 — the names a collective coins, and how long they last.

    python3 plots/fig1_names.py
    python3 plots/fig1_names.py --median      # the median run instead

Every use of a coined name is one observation, placed at the round it was said.
A name said by twenty agents in round four contributes twenty observations at
four. A violin is therefore wide where the name was in many mouths at once and
long where it stayed in the language, and the bar inside it is the median round
and the middle half of its uses.

Four runs, one per cell, written as four files and stacked in the chapter, so a
row is the same height from panel to panel whatever a panel's name count. Names
run down the left in order of first appearance, which is what makes the
succession legible: the opening pact at the top, the last plan at the bottom.

Which run. The run with the most coined names in its cell. A panel drawn on a
typical run would in several cells be a single violin --- six of the fifteen L2
knife-edge runs coin exactly one name. That is a choice about which run
illustrates, not about which is typical, and the caption carries the spread.

Two choices the caption no longer explains, since they belong with the measure
and not with the picture. Every violin is drawn to one width because a name in
three hundred utterances would otherwise flatten its neighbours, and the shape
is wanted for its timing rather than for its volume. Names are ordered by first
appearance because that is what makes the diagonal legible: under scarcity each
name arrives later than the last, while at knife-edge the early ones are still
being said at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HERE.parent / _m))

import runset          # noqa: E402
import text            # noqa: E402
from _style import (COLOUR, PAYOFF, PT_ASIDE, PT_LABEL, PT_TITLE, RUNG,  # noqa: E402
                    WIDTH, figure_tex, margins, save, style, tint)

FIGURE = "fig1_names"
CELLS = ["prod_L2_scar", "prod_L2_knife", "prod_L3_scar", "prod_L3_knife"]

# A4 with 0.75in margins leaves 737pt of column height, and the four panels plus
# a caption of a dozen lines has to fit inside it or the float is pushed off the
# page. 37 names at 0.142in plus four headings and axes comes to 536pt.
ROW_H = 0.142                 # inches per name, the same in every panel
HEAD_FOOT = 0.55              # inches for the title and the axis, per panel
NAME_MARGIN = 1.62            # inches reserved for the names, in every panel


def utterances(cell: str):
    """The richest run of the cell, one row per utterance, plus the cell's spread."""
    counted = sorted((len(text.named_agreements(p, with_rounds=True)), p.name, p)
                     for p in runset.cel(cell))
    path = (counted[len(counted) // 2] if "--median" in sys.argv else counted[-1])[2]
    rows = []
    for name, v in text.named_agreements(path, with_rounds=True).items():
        for round_, speakers in v["speakers_by_round"].items():
            rows += [{"name": name, "round": round_}] * speakers
    df = pd.DataFrame(rows)
    order = df.groupby("name", observed=True)["round"].min().sort_values().index
    df["name"] = pd.Categorical(df["name"], categories=order, ordered=True)
    return df.sort_values("name"), path.name, [n for n, _, _ in counted]


def panel(cell: str):
    df, run, spread = utterances(cell)
    rung, payoff = cell.split("_")[1], cell.split("_")[2]
    n = df["name"].cat.categories.size
    spans = df.groupby("name", observed=True)["round"].agg(["min", "max"])
    median_life = float((spans["max"] - spans["min"] + 1).median())

    height = ROW_H * n + HEAD_FOOT
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    # inner="box" puts the median and the middle half inside each violin: the
    # shape says when a name was in the air, the box says where the bulk of its
    # use sat, which separates a name used throughout from one used hard for
    # five rounds and then trailing.
    sns.violinplot(data=df, y="name", x="round", color=tint(COLOUR[payoff], 0.78),
                   cut=0, density_norm="width", width=0.9, linewidth=0.55,
                   saturation=1.0, inner="box",
                   inner_kws={"box_width": 1.8, "whis_width": 0.45,
                              "color": COLOUR[payoff]},
                   ax=ax, orient="h")
    for patch in ax.collections:
        patch.set_edgecolor(COLOUR[payoff])
    sns.stripplot(data=df, y="name", x="round", color=COLOUR[payoff], size=0.8,
                  jitter=0.26, alpha=0.5, ax=ax, orient="h")

    ax.set_xlim(0.5, 60.5)
    ax.set_xticks([1, 15, 30, 45, 60])
    ax.set_xlabel("round", fontsize=PT_LABEL)
    ax.set_ylabel("")
    ax.tick_params(labelsize=PT_LABEL)
    ax.set_title(f"{rung} · {RUNG[rung]}, {PAYOFF[payoff]} payoff",
                 fontsize=PT_TITLE, loc="left", pad=4, fontweight="bold")
    # Which run this is and what it does, beside the title rather than in the
    # caption only: a reader checking a panel against the data should not have
    # to find the caption to know what to open.
    ax.annotate(f"{n} names, median life {median_life:.0f} rounds  ·  "
                f"run {run.split('__')[1].split('_')[0]}",
                (1.0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", ha="right", va="baseline",
                fontsize=PT_ASIDE, color="#8a8a8a")
    sns.despine(ax=ax, left=True, bottom=True)
    margins(fig, NAME_MARGIN, height)
    return fig, run, spread, n


def caption(meta) -> str:
    lines = []
    for cell, run, spread, n in meta:
        rung, payoff = cell.split("_")[1], PAYOFF[cell.split("_")[2]]
        seed = run.split("__")[1].split("_")[0]
        # Seed and count only. How many names the cell coins at the median,
        # and over what spread, is the measure's business and is cited below.
        lines.append(f"\\texttt{{{seed}}} ({rung} {payoff}, {n})")
    return ("One violin per coined name, one run per cell. Every use of a "
            "name is an observation placed at the round it was said, so a "
            "violin is wide where the name was in many mouths at once and long "
            "where it stayed in the language, while the bar inside it marks the "
            "median round and the middle half of the uses. All violins are "
            "drawn to one width, so the shape says \\emph{when} a name was said "
            "and not how often, and names run down the panel in order of first "
            "appearance. Each panel is the run in its cell that coined the most "
            "names, since a panel drawn on a typical run would in several cells "
            "be a single violin: " + "; ".join(lines) + ". The cell figures "
            "these four illustrate are "
            "\\meth{m:institutions-patronage-welfare} and "
            "\\meth{m:how-long-a-name-lives}.")

if __name__ == "__main__":
    style()
    meta, files = [], []
    for cell in CELLS:
        fig, run, spread, n = panel(cell)
        save(fig, FIGURE, cell)
        plt.close(fig)
        files.append(cell)
        meta.append((cell, run, spread, n))
        print(f"   {cell:16} {n:3} names  run {run.split('__')[1].split('_')[0]}")
    path = HERE / "figures" / FIGURE / f"{FIGURE}.tex"
    path.write_text(figure_tex(
        FIGURE, files, "Coined names and their lifetime",
        caption(meta), "fig:names"))
    print(f"-> {path.parent}/")
