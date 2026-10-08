"""Counting agent-turns: the share of a cell's turns on which something is true.

Most action figures in the chapter are one question with a different predicate:
what fraction of the turns played in this cell were transfers, were attacks,
carried a drop intent, targeted a named agent. One loop, a predicate argument.

The denominator is every turn on which an agent played an action. Dead agents
leave the record, so they fall out on their own, which is what should happen: an
agent that has died is not holding anything. `share()` returns that denominator
rather than hiding it, because the same percentage means different things over
27,000 turns and over 300.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from logs import rounds          # noqa: E402
from result import Result        # noqa: E402  (same package)
import runset                    # noqa: E402

# The economic actions, listed rather than derived: an action that never occurs
# in a cell must read as 0.0 and not go missing, or a reader cannot tell "never
# done" from "never measured".
ACTIONS = ("hold", "transfer", "take", "strengthen", "harvest")


def turns(paths):
    """Yield (run, round, agent, record) for every turn an agent actually played.

    Filtered on being alive, not on the action field being set. A dead agent
    stays in the round record with `action: "no_action"`, which is truthy, so
    filtering on the action counted it as a turn: at L3 knife-edge that is 2,506
    of 18,000 turns, and every action share in those cells was diluted by a
    seventh. The appendix has always said these turns are excluded; the code did
    not do it.

    Living agents that produced no valid action are kept --- there are 112 of
    them at L3 knife-edge, and they are the schema fallbacks. Dropping those
    would hide the contamination the run set records rather than measure it.
    """
    for p in paths:
        for e in rounds(p):
            for nm, a in (e.get("agents") or {}).items():
                if (a.get("resources") or 0) > 0 and a.get("action"):
                    yield p, e.get("round"), nm, a


def share(paths, predicate: Callable[[dict], bool], unit="agent-turns") -> Result:
    """Share of turns satisfying `predicate`, with the denominator attached."""
    hits = total = 0
    for _, _, _, a in turns(paths):
        total += 1
        if predicate(a):
            hits += 1
    return Result(value=round(100 * hits / total, 2) if total else 0.0,
                  n=len(list(paths)), denominator=total, unit=unit,
                  note=f"{hits} of {total}")


def count(paths, predicate: Callable[[dict], bool], unit="agent-turns") -> Result:
    """Absolute number of turns satisfying `predicate`, with the denominator."""
    hits = total = 0
    for _, _, _, a in turns(paths):
        total += 1
        if predicate(a):
            hits += 1
    return Result(value=hits, n=len(list(paths)), denominator=total, unit=unit)


def action_profile(paths) -> dict[str, Result]:
    """Share per action over one cell. The five economic actions, always all five."""
    tally: Counter = Counter()
    total = 0
    for _, _, _, a in turns(paths):
        total += 1
        tally[str(a["action"]).lower()] += 1
    n = len(list(paths))
    return {act: Result(value=round(100 * tally.get(act, 0) / total, 2) if total else 0.0,
                        n=n, denominator=total, unit="agent-turns",
                        note=f"{tally.get(act, 0)} of {total}")
            for act in ACTIONS}


def rewire_profile(paths) -> dict[str, Result]:
    """Share of turns carrying a drop or an invite intent.

    Kept apart from `action_profile` because it is a different axis: an agent can
    give and invite on the same turn, so these do not sum to a hundred with the
    actions. Counting them as actions returned zero everywhere, including on L3
    where rewiring is the whole capacity.
    """
    tally: Counter = Counter()
    total = 0
    for _, _, _, a in turns(paths):
        total += 1
        ri = a.get("rewire_intent") or {}
        for k in ("drop", "invite"):
            if ri.get(k):
                tally[k] += 1
    n = len(list(paths))
    return {k: Result(value=round(100 * tally.get(k, 0) / total, 2) if total else 0.0,
                      n=n, denominator=total, unit="agent-turns",
                      note=f"{tally.get(k, 0)} of {total}")
            for k in ("drop", "invite")}


# Predicates worth naming, because they recur and because a lambda in a figure
# file is a definition nobody can find again.
def is_action(name: str) -> Callable[[dict], bool]:
    return lambda a: str(a.get("action", "")).lower() == name


def has_target(a: dict) -> bool:
    return bool(a.get("target"))
