"""Figure 9 --- the reasoning traces projected, and where each cell drifts.

    python3 plots/fig9_trajectories.py

The 87,546 reasoning traces of the 150 production runs, projected to two
dimensions, one panel per capacity level, with the mean position of each payoff
cell traced from the first sampled round to the last.

Read it for the geography and not for the distances. A UMAP preserves
neighbourhoods and is free to stretch everything else, so two clouds twice as
far apart on the page are not twice as different, and the separation numbers in
the caption are quoted in pooled deviations for that reason. The companion plate
in Figure~\\ref{fig:reasoning} carries the same question with a unit attached.

The trajectories are short. Mean positions move 6 to 18 per cent of the cloud's
diameter over the sixty rounds, which is drift and not a route between regimes,
and a mean of projected coordinates summarises a cloud rather than tracing a
path anything took. The caption says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _m in ("core", "_shared", "tools"):
    sys.path.insert(0, str(HERE.parent / _m))

import _reasoning_cache as cache            # noqa: E402
from _style import (COLOUR, FAINT, INK, PAYOFF, PT_ASIDE, PT_LABEL,  # noqa: E402
                    PT_TITLE, RUNG, WIDTH, figure_tex, save, style, tint)

FIGURE = "fig9_trajectories"
RUNGS = ("L1", "L2", "L3", "L4")
PAYOFFS = ("scar", "knife", "abund")

UMAP_NEIGHBOURS = 30
UMAP_MIN_DIST = 0.15
UMAP_SEED = 20250825
BACKDROP = 14000          # traces drawn as the grey cloud; drawing all 87,546
                          # makes a 30 MB PDF and no darker a cloud

def projection(E: np.ndarray) -> np.ndarray:
    """The 2-D UMAP, computed once and kept beside the embeddings.

    Seeded, which costs UMAP its parallelism and buys a figure that is the same
    figure when it is redrawn. An unseeded projection rotates and reflects
    between runs, and a reader comparing this plate to the one in an earlier
    draft would be comparing two different pictures of one dataset.
    """
    path = cache.CACHE / "umap.npy"
    if path.exists():
        xy = np.load(path)
        if len(xy) == len(E):
            return xy
    import umap                                          # noqa: PLC0415
    print(f"projecting {len(E)} traces ...", flush=True)
    xy = umap.UMAP(n_neighbors=UMAP_NEIGHBOURS, min_dist=UMAP_MIN_DIST,
                   n_components=2, metric="cosine",
                   random_state=UMAP_SEED).fit_transform(E)
    xy = np.asarray(xy, dtype=np.float32)
    np.save(path, xy)
    return xy


def trajectories(xy: np.ndarray, rows: list[dict]) -> dict:
    """Mean position per cell per sampled round.

    A mean of UMAP coordinates is not a UMAP of the mean --- the projection is
    not linear and the centre of a cloud need not be in the cloud. It is used
    here as what it is: a summary of where a cell's traces sat at a round,
    drawn over the cloud it summarises so the reader can see how much of it the
    line stands for.
    """
    bucket: dict[str, dict[int, list[int]]] = {}
    for i, r in enumerate(rows):
        bucket.setdefault(r["cell"], {}).setdefault(r["round"], []).append(i)
    out = {}
    for cell, per_round in bucket.items():
        rs = sorted(per_round)
        out[cell] = (np.array(rs),
                     np.array([xy[per_round[r]].mean(axis=0) for r in rs]))
    return out


def separation(xy: np.ndarray, rows: list[dict]) -> dict:
    """How far apart two groups sit in the projection, in pooled deviations.

    The distance between two groups' centroids over the pooled spread of the
    two, which is the two-dimensional form of what a reader estimates by eye
    when asked whether two clouds are one cloud. It is a description of the
    picture and not a test of anything: no null is fitted and no threshold
    separates a real gap from an apparent one. It is in the script rather than
    in the caption's prose because the sentence the caption writes about the
    projection has to change when the projection does.
    """
    where: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        where.setdefault(r["rung"], []).append(i)
        where.setdefault(r["cell"], []).append(i)
    pts = {k: xy[np.asarray(v)] for k, v in where.items()}

    def pair(a: str, b: str) -> float:
        pa, pb = pts[a], pts[b]
        gap = float(np.hypot(*(pa.mean(0) - pb.mean(0))))
        pooled = float(np.sqrt((pa.var(0).sum() + pb.var(0).sum()) / 2))
        return gap / pooled

    return {
        "levels": {(a, b): pair(a, b)
                   for i, a in enumerate(RUNGS) for b in RUNGS[i + 1:]},
        "cells": {(rung, a, b): pair(f"prod_{rung}_{a}", f"prod_{rung}_{b}")
                  for rung in RUNGS
                  for i, a in enumerate(PAYOFFS) for b in PAYOFFS[i + 1:]},
    }


def geography(sep: dict) -> str:
    """What the projection separates and what it does not, from the numbers.

    Written from the separations rather than from what the picture looked like
    on the day, so a projection rebuilt on a different sample cannot leave the
    caption describing a geography the panels no longer have.
    """
    lv, cl = sep["levels"], sep["cells"]
    close = [(a, b) for (a, b), v in lv.items() if v < 1.0]
    worst = min(lv.items(), key=lambda kv: kv[1])
    inside = np.array(list(cl.values()))
    (rung, a, b), best = max(cl.items(), key=lambda kv: kv[1])

    if not close:
        where = ("Every pair of capacity levels is at least one pooled "
                 f"deviation apart, the closest being {worst[0][0]} and "
                 f"{worst[0][1]} at {worst[1]:.2f}. ")
    elif len(close) == 1:
        where = ("Every pair of capacity levels is at least one pooled "
                 f"deviation apart except {close[0][0]} and {close[0][1]}, "
                 "which lie almost on top of each other at "
                 f"{worst[1]:.2f} of a deviation. ")
    else:
        pairs = ", ".join(f"{x} and {y}" for x, y in close)
        where = ("Some capacity levels sit well apart and others do not: "
                 f"{pairs} are under one pooled deviation apart, the closest "
                 f"at {worst[1]:.2f}. ")

    return (where +
            "Inside a level the three payoff cells largely do not separate, at "
            f"a median of {np.median(inside):.2f} of a deviation over the "
            f"twelve pairs; the exception is {rung}, where {PAYOFF[a]} sits "
            f"{best:.2f} deviations from {PAYOFF[b]} --- the same cell that is "
            "narrowest in panel (a). ")


def plate_trajectory(xy, rows, traj, meta):
    height = 2.42
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, height), sharex=True, sharey=True)

    rng = np.random.default_rng(11)
    keep = rng.choice(len(xy), size=min(BACKDROP, len(xy)), replace=False)
    cloud = xy[keep]
    payoff_of = np.array([r["payoff"] for r in rows])[keep]
    rung_of = np.array([r["rung"] for r in rows])[keep]

    pad = 0.04 * (xy[:, 0].max() - xy[:, 0].min())
    xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    for ax, rung in zip(axes, RUNGS):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.8, color="#e8e8e8",
                   edgecolor="none", zorder=1, rasterized=True)
        # One scatter over the panel's own traces, in shuffled order. Drawing
        # the three payoff cells one after another puts whichever is drawn last
        # on top of the other two everywhere they overlap, and since they
        # overlap almost everywhere the panel then reports one payoff cell while
        # appearing to report three --- the artefact `fig7_targets.py` avoids by
        # drawing transparently rather than in sequence. Here the points are too
        # small for transparency to read, so the order is randomised instead and
        # no cell is systematically above another.
        here = np.flatnonzero(rung_of == rung)
        here = here[rng.permutation(len(here))]
        colours = [tint(COLOUR[p], 0.45) for p in payoff_of[here]]
        ax.scatter(cloud[here, 0], cloud[here, 1], s=0.9, c=colours,
                   edgecolor="none", zorder=2, rasterized=True)
        for payoff in PAYOFFS:
            rs, pts = traj[f"prod_{rung}_{payoff}"]
            # A white stroke under the line. Without it a trajectory crossing
            # its own cell's points is the same hue as what it crosses, and the
            # eye loses the line exactly where the panel is densest.
            ax.plot(pts[:, 0], pts[:, 1], color=COLOUR[payoff], lw=1.5,
                    solid_capstyle="round", zorder=4,
                    path_effects=[pe.Stroke(linewidth=2.9, foreground="white"),
                                  pe.Normal()])
            ax.plot(pts[0, 0], pts[0, 1], marker="o", markersize=3.6,
                    markerfacecolor="white", markeredgecolor=COLOUR[payoff],
                    markeredgewidth=1.1, zorder=5)
            ax.plot(pts[-1, 0], pts[-1, 1], marker="o", markersize=3.6,
                    markerfacecolor=COLOUR[payoff], markeredgecolor="white",
                    markeredgewidth=0.7, zorder=5)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        # The letter goes on the first panel only. The caption refers to the
        # whole plate as (c) and four separately lettered panels would suggest
        # four separate claims where there is one drawn four times.
        mark = "c  " if rung == RUNGS[0] else ""
        ax.set_title(f"{mark}{rung} · {RUNG[rung]}", fontsize=PT_TITLE,
                     loc="left", pad=3, fontweight="bold")

    axes[0].annotate("open circle round 1, filled round 58",
                     (0.0, 0.0), xycoords="axes fraction", xytext=(1, 1),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=PT_ASIDE, color=FAINT)
    axes[3].annotate(f"UMAP of {meta['traces']:,} traces from "
                     f"{meta['runs']} runs",
                     (1.0, 0.0), xycoords="axes fraction", xytext=(-1, 1),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=PT_ASIDE, color=FAINT)

    # One legend for the figure, at the foot of the lower plate. The two plates
    # are stacked in LaTeX, so this is the foot of the whole figure, and the
    # colours it names are the same three in both.
    handles = [plt.Line2D([], [], color=COLOUR[p], lw=1.5, label=PAYOFF[p])
               for p in PAYOFFS]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=3, frameon=False, fontsize=PT_LABEL, handletextpad=0.5,
               columnspacing=1.8, bbox_to_anchor=(0.5, 0.004))
    fig.subplots_adjust(left=0.004, right=0.996, top=1 - 0.20 / height,
                        bottom=0.24 / height, wspace=0.03)
    return fig


def caption(meta, sep, traj) -> str:
    """What the panel shows, in the length a caption is meant to be.

    The separations this used to quote are in pooled deviations, which is a
    unit the projection does not carry on the page, and the companion plate
    reports the same comparison in cosine distance where it does mean
    something. The appendix text makes the reading; what stays here is the
    grammar of the panel.
    """
    return (
        f"The {meta['traces']:,} reasoning traces of the {meta['runs']} "
        "production runs, projected to two dimensions, one panel per capacity "
        "level. Grey is every trace in the study and the tinted points are that "
        "panel's own; the line is the mean position of a payoff cell at each "
        "sampled round, from an open circle at the first to a filled one at the "
        "last. Distances on the page carry no unit, since the projection "
        "preserves neighbourhoods while it is free to stretch everything else. "
        "Nothing here is a test.")

def main() -> int:
    style()
    E, rows, meta = cache.load()
    xy = projection(E)
    traj = trajectories(xy, rows)
    sep = separation(xy, rows)
    fig = plate_trajectory(xy, rows, traj, meta)
    save(fig, FIGURE, "trajectories")
    tex = figure_tex(FIGURE, ["trajectories"],
                     "Reasoning traces projected, by capacity level",
                     caption(meta, sep, traj), "fig:trajectories")
    (HERE / "figures" / FIGURE / f"{FIGURE}.tex").write_text(tex)
    print(f"wrote figures/{FIGURE}/trajectories.pdf and {FIGURE}.tex")
    print("\n" + geography(sep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
