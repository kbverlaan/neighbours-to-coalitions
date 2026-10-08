r"""Figure 8 --- how alike the twelve payoff cells reason.

    python3 plots/_reasoning_cache.py     # once, about a hundred minutes
    python3 plots/fig8_reasoning.py       # seconds, from the cache

Two plates over one set of embeddings. Panels (a) and (b) split the spread of a
payoff cell into the part inside a run and the part between runs. Panels (c) and
(d) put every cell centre against every other, and then average that by capacity
level.

The figure was a UMAP first. The projection was not empty --- the capacity levels
land in different parts of it --- but distance in a UMAP has no unit, so it could
not answer the question the section asks, which is how alike two cells are.

MEASURE TEXT --- documenting what this figure computes
======================================================

Kept here so the caption can stay short.

**What is computed.** The distance between two bodies of reasoning text, in the
embedding space of a sentence encoder. Every agent's chain of thought is a
document; a run is the mean of its documents, re-normalised to unit length; a
payoff cell is the mean of its runs' centres, re-normalised again. Distance is
cosine distance, one minus the dot product of two unit vectors. Three quantities
are reported. *Within-run spread* is the mean distance from a run's documents to
that run's own centre. *Between-run spread* is the mean distance from each run's
centre to the centre of its cell. *Cell distance* is the distance between two
cell centres.

**Scope.** The 150 production runs, twelve payoff cells, four capacity levels by
three payoff regimes. Every agent alive in a sampled round contributes one
document; agents already dead contribute none. Rounds are sampled one in three,
1, 4, ... 58, giving 20 of the 60 rounds and 87,546 documents. Runs are never
thinned: the between-run quantity is a claim about runs, and thinning them would
be thinning the evidence for it. Rounds are thinned because consecutive rounds of
one agent are strongly autocorrelated.

**Encoder.** `nomic-ai/nomic-embed-text-v1.5`, 768 dimensions, with the task
prefix `search_document: ` the model was trained to expect. Each document is the
last 2,400 characters of the trace, truncated to 512 tokens. The tail and not the
head: a trace opens with the agent restating its holdings, its neighbours and the
prices, text that is near-identical across agents in a round and would embed the
scenario rather than the reasoning. The strategic argument and the decision are
at the end. The median trace is about 5,400 characters, so the tail is roughly
its closing half.

**The shuffled reference.** An absolute cosine distance means nothing on its own.
For each quantity the labels are permuted at the level the quantity is about,
keeping group sizes, and the quantity recomputed: run labels within a cell for
(a) and (b), averaged over five permutations; run-to-cell assignment for (c) and
(d), averaged over twenty. That is what the figure looks like when the grouping
carries no information. It is a reference and not a test --- no null distribution
is fitted and no tail probability is taken from it.

**Uncertainty.** Percentile bootstrap intervals at 95 per cent over 10,000
resamples, on `tools/uncertainty.py`'s generator and percentile method, seeded
by name. The run is the resampling unit throughout. The 87,546 documents are not
87,546 independent draws --- they come from 150 runs --- and resampling documents
would report intervals roughly a tenth of the honest width.

**Limitations.**

1. *Vocabulary grows with the ladder.* Each capacity level adds an action and
   the words for it, so agents at a higher level have more to write about and
   their texts more on which to differ. A distance that rises with the capacity
   level is therefore consistent with genuinely diverging reasoning and equally
   consistent with a larger vocabulary. Nothing in this figure separates them.
   The test would be to mask the level-specific action vocabulary and encode
   again; it has not been done. This is the limitation the caption carries,
   because it bounds the finding and not the method.

2. *Resolution.* The median bootstrap interval on a cell distance is 11
   thousandths, which is wider than the mean distance among L3's own three
   payoff cells and comparable to L2's and L4's. Differences inside a capacity
   level are not resolved and the diagonal blocks are read as blocks.

3. *A trace is behaviour, not mechanism.* What an agent writes is an observation
   of what it produced, not a readout of how it decided (Turpin et al. 2024;
   Lanham et al. 2023). Distance here is a statement about wording. Whether the
   collectives were organised differently is settled on the actions, in the
   measures of Sections 4.2 to 4.5, not here.

4. *Encoder-dependence.* Every number is relative to one encoder. The ordering
   of cells is what the figure claims; the absolute distances are not portable
   to another model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _m in ("core", "_shared", "tools"):
    sys.path.insert(0, str(HERE.parent / _m))

import _reasoning_cache as cache            # noqa: E402
from _ci import band                        # noqa: E402
from uncertainty import RESAMPLES, _pctl, _rng   # noqa: E402
from _style import (COLOUR, FAINT, INK, PAYOFF, PT_ASIDE, PT_LABEL,  # noqa: E402
                    PT_TITLE, RUNG, WIDTH, figure_tex, save, style, tint)

FIGURE = "fig8_reasoning"
RUNGS = ("L1", "L2", "L3", "L4")
PAYOFFS = ("scar", "knife", "abund")
OFFSET = {"scar": -0.24, "knife": 0.0, "abund": 0.24}

UMAP_NEIGHBOURS = 30
UMAP_MIN_DIST = 0.15
UMAP_SEED = 20250825
BACKDROP = 14000          # traces drawn as the grey cloud; drawing all 87,546
                          # makes a 30 MB PDF and no darker a cloud


# --- the numbers ------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def dispersion(E: np.ndarray, rows: list[dict]) -> dict:
    """Within-run and between-run spread, in cosine distance, per run.

    Within is how far a run's traces sit from that run's own centroid, averaged
    over the traces. Between is how far that centroid sits from the centroid of
    its payoff cell. The two are on the same scale and the same axis, which is
    the point of splitting them: a cell can be wide because each run wanders or
    because the runs went to different places, and one number cannot say which.

    Vectors are unit-normalised by the encoder, so a dot product is a cosine and
    the distance is one minus it. Centroids are means and then re-normalised;
    without that a tight run's centroid is long and a scattered one's is short,
    and the distance would read the tightness twice.
    """
    per_run: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        per_run.setdefault((r["cell"], r["run"]), []).append(i)

    centroid = {k: unit(E[idx].mean(axis=0)) for k, idx in per_run.items()}
    within = {k: float(np.mean(1.0 - E[idx] @ centroid[k]))
              for k, idx in per_run.items()}

    per_cell: dict[str, list[tuple]] = {}
    for k in per_run:
        per_cell.setdefault(k[0], []).append(k)

    out = {}
    rng = np.random.default_rng(3)
    for cell, keys in per_cell.items():
        mid = unit(np.mean([centroid[k] for k in keys], axis=0))
        out[cell] = {
            "within": np.array([within[k] for k in keys]),
            "between": np.array([float(1.0 - centroid[k] @ mid) for k in keys]),
            "runs": [k[1] for k in keys],
        }
        out[cell].update(shuffled(E, [per_run[k] for k in keys], rng))
    return out


def shuffled(E: np.ndarray, groups: list[list[int]], rng, repeats: int = 5) -> dict:
    """The same two quantities with the traces reassigned to runs at random.

    The reference every dot in the figure is a departure from. A run's traces
    could sit close together because that run went somewhere of its own, or
    because any six hundred traces from the cell would sit about that close ---
    the number alone cannot say which. Shuffling the run labels inside the cell,
    keeping each run's size, gives what the same cell looks like when the runs
    are not runs.

    This is a reference line and not a test. No p-value is computed from it and
    none should be: five reassignments say where the middle of the shuffled
    distribution lies, not how often the observed value would be beaten.
    """
    idx = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    within, between = [], []
    for _ in range(repeats):
        order = rng.permutation(idx)
        parts, at = [], 0
        for n in sizes:
            parts.append(order[at:at + n])
            at += n
        cs = [unit(E[p].mean(axis=0)) for p in parts]
        mid = unit(np.mean(cs, axis=0))
        within.append(np.mean([np.mean(1.0 - E[p] @ c) for p, c in zip(parts, cs)]))
        between.append(np.mean([1.0 - c @ mid for c in cs]))
    return {"within_shuffled": float(np.mean(within)),
            "between_shuffled": float(np.mean(between))}


# --- the matrix ------------------------------------------------------------

def run_centres(E: np.ndarray, rows: list[dict]) -> dict:
    """One unit vector per run: the mean of its traces, re-normalised.

    Everything downstream is built from these and never from the traces
    directly, because the run is the unit this study samples. A cell centre is
    the mean of its runs and not the mean of its traces, so a run that happens
    to have survived longer and written more does not weigh more than one that
    ended early.
    """
    per: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        per.setdefault((r["cell"], r["run"]), []).append(i)
    out: dict[str, list[np.ndarray]] = {}
    for (cell, _), idx in per.items():
        out.setdefault(cell, []).append(unit(E[idx].mean(axis=0)))
    return {c: np.array(v) for c, v in out.items()}


def cell_matrix(centres: dict) -> dict:
    """The twelve cell centres against each other, with a reference and a width.

    Three numbers make the panel readable and none of them is a test.

    The matrix itself is the cosine distance between two cells' centres. It has
    a scale, which is the whole reason it replaced a projection: a distance in a
    UMAP is whatever the optimiser found convenient, and this one is not.

    The reference is the same shuffled figure panels (a) and (b) carry. Runs are
    dealt out to the twelve cells at random, keeping each cell's count, and the
    matrix recomputed; that is what the panel looks like when the cells do not
    differ, and it is the scale against which 0.05 means something.

    The width is a percentile bootstrap interval over runs, on the chapter's own
    machinery from `tools/uncertainty.py` --- the same generator, the same
    seeding by name and the same percentile method `_ci.py` uses, with only the
    statistic changed. One hundred and forty-four intervals cannot be drawn, so
    the median width is reported in the caption instead. It matters: it is as
    large as most of the distances inside a capacity level, which is the reason
    the caption declines to read those.
    """
    cells = [f"prod_{r}_{p}" for r in RUNGS for p in PAYOFFS]
    mid = {c: unit(centres[c].mean(axis=0)) for c in cells}
    M = np.array([[1.0 - mid[a] @ mid[b] for b in cells] for a in cells])

    sizes = [len(centres[c]) for c in cells]
    pool = np.concatenate([centres[c] for c in cells])
    rng = np.random.default_rng(5)
    shuffles = []
    for _ in range(20):
        order = rng.permutation(len(pool))
        at, cs = 0, []
        for n in sizes:
            cs.append(unit(pool[order[at:at + n]].mean(axis=0)))
            at += n
        S = np.array([[1.0 - a @ b for b in cs] for a in cs])
        shuffles.append(np.median(S[np.triu_indices(len(cells), 1)]))

    widths = []
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            ra, rb = centres[a], centres[b]
            g = _rng(f"fig8-matrix-{a}-{b}")
            d = np.empty(RESAMPLES)
            step = 2500
            for s in range(0, RESAMPLES, step):
                n = min(step, RESAMPLES - s)
                ca = unit(ra[g.integers(0, len(ra), size=(n, len(ra)))].mean(axis=1))
                cb = unit(rb[g.integers(0, len(rb), size=(n, len(rb)))].mean(axis=1))
                d[s:s + n] = 1.0 - np.einsum("ij,ij->i", ca, cb)
            lo, hi = _pctl(d)
            widths.append(hi - lo)

    upper = M[np.triu_indices(len(cells), 1)]
    return {"cells": cells, "M": M, "shuffled": float(np.mean(shuffles)),
            "width": float(np.median(widths)), "runs": sizes,
            "median": float(np.median(upper)), "max": float(upper.max())}


def level_matrix(mat: dict) -> np.ndarray:
    """The same distances averaged into a four-by-four over capacity levels.

    Not a new measurement --- every number in it is a mean of numbers in the
    twelve-by-twelve beside it. It is there because the twelve-by-twelve is a
    detail view and the reader needs the headline as well: within a level the
    mean is over that level's three pairs, between two levels over the nine
    pairs that cross them.
    """
    M, cells = mat["M"], mat["cells"]
    idx = {r: [i for i, c in enumerate(cells) if c.split("_")[1] == r]
           for r in RUNGS}
    out = np.zeros((len(RUNGS), len(RUNGS)))
    for i, a in enumerate(RUNGS):
        for j, b in enumerate(RUNGS):
            sub = M[np.ix_(idx[a], idx[b])]
            out[i, j] = (sub[np.triu_indices(len(idx[a]), 1)].mean()
                         if a == b else sub.mean())
    return out


# --- plate one: the spread --------------------------------------------------

def spread_panel(ax, disp, key: str, title: str, ylabel: str) -> None:
    rng = np.random.default_rng(7)
    for j, rung in enumerate(RUNGS):
        for payoff in PAYOFFS:
            v = disp[f"prod_{rung}_{payoff}"][key]
            x = j + OFFSET[payoff]
            ax.scatter(x + rng.uniform(-0.055, 0.055, len(v)), v,
                       s=5.0, facecolor=tint(COLOUR[payoff], 0.42),
                       edgecolor="none", zorder=2, rasterized=False)
            # The interval is the chapter's own, from `_ci.py`: the run is the
            # resampling unit. The 87,546 traces are not 87,546 independent
            # draws --- they come from 150 runs, and a bootstrap over traces
            # would report an interval a tenth of the honest width.
            mean, lo, hi = (a[0] for a in band(f"fig8-{key}-{rung}-{payoff}",
                                               v.reshape(-1, 1)))
            ax.plot([x, x], [lo, hi], color=COLOUR[payoff], lw=0.8,
                    solid_capstyle="butt", zorder=3)
            ax.plot([x - 0.10, x + 0.10], [mean] * 2,
                    color=COLOUR[payoff], lw=1.5, solid_capstyle="butt", zorder=3)
            ax.plot([x - 0.13, x + 0.13],
                    [disp[f"prod_{rung}_{payoff}"][f"{key}_shuffled"]] * 2,
                    color=FAINT, lw=0.7, linestyle=(0, (2.2, 1.6)), zorder=4)

    ax.set_xlim(-0.5, len(RUNGS) - 0.5)
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels([f"{r}\n{RUNG[r]}" for r in RUNGS], fontsize=PT_LABEL)
    ax.set_xlabel("capacity level", fontsize=PT_LABEL, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=PT_LABEL)
    ax.set_title(title, fontsize=PT_TITLE, loc="left", pad=4, fontweight="bold")
    ax.tick_params(axis="both", labelsize=PT_LABEL, length=2.5, pad=2)
    ax.grid(visible=True, axis="y")
    ax.grid(visible=False, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plate_spread(disp, meta):
    height = 2.55
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height))
    spread_panel(axes[0], disp, "within",
                 "a  spread inside a run", "cosine distance to the run's centre")
    spread_panel(axes[1], disp, "between",
                 "b  spread between runs", "cosine distance to the cell's centre")

    axes[0].annotate("dotted: run labels shuffled inside the cell",
                     (1.0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                     textcoords="offset points", ha="right", va="baseline",
                     fontsize=PT_ASIDE, color=FAINT)
    axes[1].annotate(f"a dot is one run  ·  {meta['runs']} runs",
                     (1.0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                     textcoords="offset points", ha="right", va="baseline",
                     fontsize=PT_ASIDE, color=FAINT)
    fig.subplots_adjust(left=0.062, right=0.995, top=1 - 0.22 / height,
                        bottom=0.62 / height, wspace=0.22)
    return fig


# --- plate two: the matrix --------------------------------------------------

RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "magnitude", ["#ffffff", COLOUR["scar"]])
# The chapter's magnitude ramp, one hue from light to dark. A diverging map
# would put a midpoint somewhere in the middle of the range and invite the
# reader to read one side as more and the other as less; this is a distance,
# which has a floor and no midpoint.


def grid(ax, M, labels, top_labels, note_diag=True, size=5.4):
    """One matrix, drawn with its numbers in thousandths."""
    shown = np.array(M, dtype=float)
    mask = np.eye(len(M), dtype=bool)
    ax.imshow(np.where(mask, np.nan, shown), cmap=RAMP, vmin=0.0,
              vmax=float(np.nanmax(np.where(mask, np.nan, shown))),
              interpolation="nearest")
    for i in range(len(M)):
        for j in range(len(M)):
            if i == j:
                if note_diag:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor="#f4f4f4",
                                               edgecolor="none", zorder=2))
                continue
            v = shown[i, j]
            # Thousandths as whole numbers. "0.044" at five point is four
            # glyphs of decoration around one digit of information.
            ax.text(j, i, f"{1000 * v:.0f}", ha="center", va="center",
                    fontsize=size, zorder=3,
                    color="white" if v > 0.6 * np.nanmax(shown) else INK)
    if len(M) > len(RUNGS):
        # White rules every three cells. The block structure is the whole
        # finding, and without them the reader has to count labels to see where
        # one capacity level ends and the next begins.
        for k in range(len(RUNGS), len(M), len(PAYOFFS)):
            ax.axvline(k - 0.5, color="white", lw=1.6, zorder=4)
            ax.axhline(k - 0.5, color="white", lw=1.6, zorder=4)
    ax.set_xticks(range(len(M)))
    ax.set_yticks(range(len(M)))
    ax.set_xticklabels(top_labels, fontsize=PT_LABEL)
    ax.set_yticklabels(labels, fontsize=PT_LABEL)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.xaxis.set_ticks_position("top")
    ax.grid(visible=False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return ax


def plate_matrix(mat, meta):
    height = 3.30
    fig = plt.figure(figsize=(WIDTH, height))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 2.05], wspace=0.62)
    ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(2))
    cax = fig.add_axes((0.93, 0.1, 0.012, 0.7))   # placed properly further down
    # Both matrices hang from the same line. imshow holds a square aspect, so a
    # four-by-four and a twelve-by-twelve in one row are centred at different
    # heights unless they are told to hang from the top.
    for a in (ax1, ax2):
        a.set_anchor("N")

    cells = mat["cells"]
    short = [f"{c.split('_')[1]} {PAYOFF[c.split('_')[2]]}" for c in cells]
    tops = [PAYOFF[c.split("_")[2]][:2] for c in cells]
    grid(ax1, mat["M"], short, tops, size=4.9)
    ax1.set_title("c  every payoff cell against every other",
                  fontsize=PT_TITLE, loc="left", pad=13, fontweight="bold")

    L = level_matrix(mat)
    grid(ax2, L, [f"{r} · {RUNG[r]}" for r in RUNGS], RUNGS, size=6.4)
    ax2.set_title("d  averaged by capacity level", fontsize=PT_TITLE,
                  loc="left", pad=13, fontweight="bold")

    # The diagonal of (d) is not empty the way (c)'s is: it is the mean distance
    # among a level's own three payoff cells, which is the quantity the panel is
    # for. It is drawn and labelled rather than masked.
    for i in range(len(RUNGS)):
        ax2.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1,
                                    facecolor=RAMP(L[i, i] / np.nanmax(
                                        L[~np.eye(len(RUNGS), dtype=bool)])),
                                    edgecolor="white", lw=0.8, zorder=2.5))
        ax2.text(i, i, f"{1000 * L[i, i]:.0f}", ha="center", va="center",
                 fontsize=6.4, zorder=3, color=INK, style="italic")

    top = float(np.nanmax(np.where(np.eye(len(cells), dtype=bool), np.nan,
                                   mat["M"])))
    bar = mpl.colorbar.ColorbarBase(
        cax, cmap=RAMP, norm=mpl.colors.Normalize(0, top))
    bar.set_label("cosine distance ($\\times$10$^{-3}$)", fontsize=PT_ASIDE)
    bar.set_ticks([0, mat["shuffled"], top / 2, top])
    bar.set_ticklabels(["0", f"{1000 * mat['shuffled']:.0f}",
                        f"{1000 * top / 2:.0f}", f"{1000 * top:.0f}"])
    bar.ax.tick_params(labelsize=PT_ASIDE, length=2, pad=1.5)
    bar.outline.set_visible(False)
    # The shuffled figure marked on the bar itself, so the reader has the scale
    # without going to the caption for it.
    bar.ax.axhline(mat["shuffled"], color=INK, lw=0.9)
    bar.ax.annotate("shuffled", (0.0, mat["shuffled"]),
                    xycoords=("axes fraction", "data"), xytext=(-3, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=PT_ASIDE, color=INK, style="italic")

    ax1.annotate(f"columns follow the rows  ·  distances in thousandths  ·  a "
                 f"bootstrap interval over runs is {1000 * mat['width']:.0f} "
                 f"wide at the median",
                 (0.0, 0.0), xycoords="axes fraction", xytext=(0, -10),
                 textcoords="offset points", ha="left", va="top",
                 fontsize=PT_ASIDE, color=FAINT)
    fig.subplots_adjust(left=0.083, right=0.895, top=0.845, bottom=0.075)
    # The colour bar is placed against the twelve-by-twelve rather than against
    # the figure, so it is as tall as the thing it explains. imshow keeps a
    # square aspect, so where that matrix actually lands is only known once the
    # figure has been laid out.
    fig.canvas.draw()
    box = ax1.get_window_extent().transformed(fig.transFigure.inverted())
    cax.set_position((0.935, box.y0, 0.011, box.height))
    return fig


# --- the caption ------------------------------------------------------------

WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def count_of(n: int, noun: str, verb: tuple[str, str]) -> str:
    """`one of the four levels is`, `two of the four levels are`.

    Numbers written into a sentence by a template come out as "1 of the four
    levels are" unless something agrees the verb with them.
    """
    v = verb[0] if n == 1 else verb[1]
    return f"{WORDS.get(n, str(n))} of the four {noun}" + (f" {v}" if v else "")


def caption(disp, meta, mat) -> str:
    """The caption: what a reader is looking at, and nothing else.

    It used to run to four hundred and fifty words and restate the whole
    appendix, which is longer than the appendix itself and against the
    convention of thirty to eighty. What a caption owes the reader is the
    grammar of the panel --- what a dot is, what a bar is, how many runs, what
    the dotted rule means --- while the findings belong in the running text and
    the method in the appendix that documents it.

    Two numbers stay because they are read off the panel and cannot be got from
    it by eye: how many runs a dot is drawn from, and how wide an interval is,
    which is what tells a reader not to rank the cells inside a diagonal block.
    """
    return (
        "How alike the reasoning of the twelve payoff cells is, read off the text the agents wrote while deciding. \\textbf{(a, b)} The spread of a cell split in two, in cosine distance: how far a run's traces sit from that run's centre, and how far each run's centre sits from its cell's. A dot is one of the 150 runs, the bar is the cell mean with a 95\\% interval, and the dotted rule is the same quantity with the run labels shuffled. \\textbf{(c, d)} Cosine distance between the twelve cell centres, in thousandths; (d) averages by capacity level. An interval is 11 wide at the median, so a diagonal block reads as one block.")


# --- run --------------------------------------------------------------------

if __name__ == "__main__":
    style()
    E, rows, meta = cache.load()
    disp = dispersion(E, rows)
    mat = cell_matrix(run_centres(E, rows))

    files = []
    for fig, name in ((plate_spread(disp, meta), "dispersion"),
                      (plate_matrix(mat, meta), "matrix")):
        save(fig, FIGURE, name)
        plt.close(fig)
        files.append(name)

    print(f"{meta['traces']:,} traces, {meta['runs']} runs, "
          f"{len(meta['rounds'])} rounds")
    print(f"{'cell':18} {'within':>8} {'between':>8}")
    for cell in sorted(disp):
        print(f"{cell:18} {disp[cell]['within'].mean():8.4f} "
              f"{disp[cell]['between'].mean():8.4f}")
    print(f"\nmatrix: median {1000 * mat['median']:.0f}, max "
          f"{1000 * mat['max']:.0f}, shuffled {1000 * mat['shuffled']:.1f}, "
          f"bootstrap width {1000 * mat['width']:.0f} (thousandths)")
    L = level_matrix(mat)
    print("by capacity level (thousandths):")
    print("        " + "".join(f"{r:>7}" for r in RUNGS))
    for i, r in enumerate(RUNGS):
        print(f"  {r:5} " + "".join(f"{1000 * L[i, j]:7.1f}" for j in range(len(RUNGS))))

    path = HERE / "figures" / FIGURE / f"{FIGURE}.tex"
    path.write_text(figure_tex(
        FIGURE, files, "Reasoning traces in embedding space",
        caption(disp, meta, mat), "fig:reasoning"))
    print(f"-> {path.parent}/")