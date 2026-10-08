"""Per-run quantities, and how a cell summarises them.

The other counting axis. `turns` pools every agent-turn in a cell, which weights
a run by how busy it was; this module computes one number per run and then
summarises across runs, which weights every run equally. Both are legitimate and
they answer different questions, so a figure must say which it used --- and
where the two disagree, that disagreement is itself the finding. The scarce
transfer classification is the standing example: pooled, one run supplied a
quarter of the cell's transfers and moved the headline by ten points.

`summarise` returns mean, sd, median and range together rather than making the
caller pick, so a figure that reports a mean over a skewed cell is visibly doing
that rather than quietly doing it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from logs import rounds        # noqa: E402
from result import Result      # noqa: E402
import runset                  # noqa: E402

# An agent above this floor counts as alive. Dead agents stay in the round record
# at zero rather than disappearing, so `len(agents)` reads thirty everywhere,
# including in cells where half the population has died.
ALIVE_FLOOR = 0.0

# An agent that never acts decays to this. Any floor at or near it means nobody
# was stripped by another agent; the decay did all the work.
DO_NOTHING = 100 * 0.97 ** 60


def final(path: Path) -> dict[str, float]:
    """Round-60 holdings per agent, for one run."""
    rs = rounds(path)
    if not rs:
        raise runset.RunsetError(f"{path.name} has no rounds")
    return {nm: float(a.get("resources") or 0.0)
            for nm, a in (rs[-1].get("agents") or {}).items()}


def gini(values) -> float:
    """Gini over a series; 0 is perfect equality."""
    xs = sorted(max(0.0, float(x)) for x in values if x is not None)
    n, s = len(xs), sum(max(0.0, float(x)) for x in values if x is not None)
    if n == 0 or s == 0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(xs, 1))
    g = (2 * cum) / (n * s) - (n + 1) / n
    # A perfectly equal series lands at -2.2e-16 and prints as "-0.000", which
    # reads as negative inequality. Clamp only what is indistinguishable from
    # zero in double precision; anything genuinely below that is a bug worth seeing.
    return 0.0 if -1e-12 < g < 1e-12 else g


# --- the scalar library: one number per run -------------------------------

def final_gini(p: Path) -> float:
    return gini(final(p).values())


def floor(p: Path) -> float:
    """The poorest agent's round-60 holding."""
    return min(final(p).values())


def alive(p: Path) -> int:
    return sum(1 for v in final(p).values() if v > ALIVE_FLOOR)


def total_wealth(p: Path) -> float:
    return sum(final(p).values())


def mean_holding(p: Path) -> float:
    v = final(p)
    return sum(v.values()) / len(v) if v else 0.0


# --- summarising across runs ----------------------------------------------

def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    h = len(xs) // 2
    return xs[h] if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2


def per_run(paths, scalar: Callable[[Path], float]) -> list[float]:
    return [scalar(p) for p in paths]


def summarise(paths, scalar: Callable[[Path], float], unit="runs",
              digits: int = 3) -> Result:
    """Mean, sd, median and range of a per-run quantity over one cell."""
    xs = per_run(paths, scalar)
    n = len(xs)
    if not n:
        return Result(value=None, n=0, unit=unit, note="empty cell")
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return Result(
        value=round(m, digits), n=n, denominator=n, unit=unit,
        sensitivity={"sd": round(sd, digits),
                     "median": round(_median(xs), digits),
                     "min": round(min(xs), digits),
                     "max": round(max(xs), digits)},
        note="mean over runs; per-run values in sensitivity")


# --- round-level shapes, summarised into one number per run ----------------
#
# These two are not properties of the final state but of the trajectory, and
# both were reimplemented per script before. Ported here unchanged so the
# figures that use them cannot drift from the ones that did.

def coalition_size(p: Path) -> float:
    """Mean over attacking rounds of the largest group aimed at one target.

    Read from the declared `take` actions rather than from the resolved combat
    entries, because a coalition that formed and did not resolve into a fight is
    still a coalition. The combat-entry reading is available as
    `coalition_size_resolved` and the two agree closely.
    """
    per = []
    for e in rounds(p):
        doelen: dict[str, set] = {}
        for nm, a in (e.get("agents") or {}).items():
            if a.get("action") == "take" and a.get("target"):
                doelen.setdefault(a["target"], set()).add(nm)
        if doelen:
            per.append(max(len(v) for v in doelen.values()))
    return sum(per) / len(per) if per else 0.0


def coalition_size_resolved(p: Path) -> float:
    """The same quantity from the resolved combat entries, as a cross-check."""
    per = []
    for e in rounds(p):
        cs = e.get("combat") or []
        if cs:
            per.append(max(len(c.get("attackers") or []) for c in cs
                           if isinstance(c, dict)))
    return sum(per) / len(per) if per else 0.0


def _effective(a: dict):
    """A hold that carries a rewire intent counts as the rewire.

    An agent that drops a tie must play some economic action alongside it, and
    that action is almost always `hold`. Reading the raw field therefore counts
    a round of rewiring as a round of doing nothing.
    """
    act = a.get("action")
    ri = a.get("rewire_intent") or {}
    # The engine writes "no_action", not "do_nothing"; the old spelling
    # never matched, so a dead agent's empty turn counted as the modal action
    # in 24 of 600 rounds at L3 knife-edge.
    if act in (None, "hold", "do_nothing", "no_action") and isinstance(ri, dict):
        for k in ("drop", "invite"):
            if ri.get(k):
                return k
    return act


def consensus_spread(p: Path) -> float:
    """Standard deviation, over rounds, of the share playing the modal action.

    High when the collective swings between agreement and disagreement about
    what to do; low when it either always agrees or never does. It is a measure
    of instability in coordination, not of its level.
    """
    reeks = []
    for e in rounds(p):
        tel: dict[str, int] = {}
        for a in (e.get("agents") or {}).values():
            if (a.get("resources") or 0) > 0 and a.get("action"):
                k = str(_effective(a) or "hold").lower()
                tel[k] = tel.get(k, 0) + 1
        if tel:
            reeks.append(max(tel.values()) / sum(tel.values()))
    if not reeks:
        return 0.0
    m = sum(reeks) / len(reeks)
    return (sum((x - m) ** 2 for x in reeks) / len(reeks)) ** 0.5


def economy_growth(p: Path) -> float:
    """Percentage change in the collective's total holdings, round 1 to round 60.

    Growth rather than level, because the level is the mean holding times thirty
    and says the same thing twice.
    """
    reeks = [sum(a.get("resources") or 0.0 for a in (e.get("agents") or {}).values())
             for e in rounds(p)]
    return (reeks[-1] / reeks[0] - 1) * 100 if reeks and reeks[0] > 0 else 0.0


# --- shared summary shape --------------------------------------------------

def median(xs):
    """Median of a series, ignoring absent values; None on an empty one."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    h = len(xs) // 2
    return xs[h] if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2


def summary(xs, digits: int = 2) -> dict:
    """Mean beside median and range, never a mean alone.

    With ten or fifteen runs a cell mean is a claim about a distribution, and
    most of the reporting errors this package was built to remove came from
    giving one without the other.

    `defined_in` against `runs` is the second half of it: a quantity that is
    undefined for some runs --- the round of a first attack in a run with no
    fights --- must say how many runs it actually rests on, rather than quietly
    averaging over the ones where it happened to exist.

    Lives here rather than in a figures module because two modules had begun to
    keep their own copy, which is the drift this layer exists to prevent.
    """
    v = [x for x in xs if x is not None]
    if not v:
        return {"mean": None, "defined_in": 0, "runs": len(xs)}
    return {"mean": round(sum(v) / len(v), digits), "median": round(median(v), digits),
            "min": round(min(v), digits), "max": round(max(v), digits),
            "defined_in": len(v), "runs": len(xs)}


def destitute_and_rescues(path: Path, floor: float = 1.0) -> tuple[int, int]:
    """(agent-rounds spent below `floor`, transfers that reached one).

    The one thing that has to be right here is *when* the holdings are read. The
    round record is end-of-round state, and a successful transfer lifts its
    recipient above the floor, so an agent rescued in round r is no longer below
    the floor in round r's own record. Counting from that record erases exactly
    the events being counted --- it returned zero rescues at L3 knife-edge where
    there are eleven, and zero for the whole Gemma arm where there are twelve.

    So the destitute set is read from the board going *into* the round, and the
    transfers from the round itself.

    Lives in core because two figures asked this question and answered it
    differently: one had the correction and one did not, and the chapter printed
    both answers two paragraphs apart.
    """
    rs = rounds(path)
    per_ronde = {e.get("round"): e for e in rs}
    kansen = gered = 0
    for e in rs:
        voor = (per_ronde.get((e.get("round") or 0) - 1) or {}).get("agents") or {}
        arm = {nm for nm, a in voor.items() if 0 < (a.get("resources") or 0) < floor}
        kansen += len(arm)
        for a in (e.get("agents") or {}).values():
            if a.get("action") == "transfer" and a.get("target") in arm:
                gered += 1
    return kansen, gered


# --- an on-disk memo for per-run scalars ------------------------------------

_MEMO_PAD = Path(__file__).resolve().parents[1] / "out" / "scalars.json"
_MEMO: dict | None = None


def _memo() -> dict:
    global _MEMO
    if _MEMO is None:
        import json
        try:
            _MEMO = json.loads(_MEMO_PAD.read_text())
        except (OSError, ValueError):
            _MEMO = {}
    return _MEMO


def memo_flush() -> None:
    import json
    _MEMO_PAD.parent.mkdir(exist_ok=True)
    _MEMO_PAD.write_text(json.dumps(_memo(), sort_keys=True))


def cached_scalar(naam: str, path: Path, fn):
    """One number per (run, quantity), remembered on disk between processes.

    The run set is 24 GB and a figure that walks every cell reads a large part
    of it. The path-dependence screen does that four times over --- once per
    predictor --- and on this machine it did not survive doing so: the process
    was killed part-way through, three times, with no output and no traceback.

    The values memoised here are scalars derived from a run that never changes,
    so the cache is safe in the only way that matters: a run's log is immutable
    once written, and a changed definition gets a new `naam`. It is not a
    performance nicety --- it is what makes the figure computable here at all,
    and it makes a killed run resume instead of restart.

    Delete `out/scalars.json` to force a full recomputation.
    """
    sleutel = f"{naam}|{Path(path).name}"
    m = _memo()
    if sleutel not in m:
        m[sleutel] = fn(path)
    return m[sleutel]
