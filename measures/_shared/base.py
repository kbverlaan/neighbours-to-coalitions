"""Shared primitives: written once, instead of forty-seven times over.

The old ad-hoc scripts each reimplemented their own round loop, Gini and action
count. That is exactly where silent divergence creeps in — one counts dead agents
in the denominator, another does not, and the difference only surfaces when two
numbers in the text stop adding up.

Everything here reads the compact `_log.jsonl` rather than the `reasoning_live`:
same actions and resources, without the reasoning text, and ten times smaller.
Measures that do need the traces or the messages load those separately.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from runset import log_path

# The economic actions from the `action` field, in the order the rungs add them.
# Listed explicitly rather than derived from the data: an action that never
# occurs in a cell should appear as 0.0% and not as a missing column — otherwise
# a reader cannot tell "never done" from "never measured".
#
# `drop` and `invite` are deliberately absent. Rewiring is not an action but an
# intent alongside one: it lives in `rewire_intent`, and an agent does it in the
# same turn in which it also holds or gives. Including them here returned zero
# everywhere, including on T3 where rewiring is the whole rung.
ACTIONS = ("hold", "transfer", "take", "strengthen", "harvest")

# Above this floor an agent counts as alive. Dead agents stay in the round record
# with resources 0 instead of disappearing, so `len(agents)` returned 30
# everywhere — including cells where half the population had died.
ALIVE_FLOOR = 0.0


def rounds(path: Path) -> list[dict]:
    """Every round of one run. Delegates to `core/logs.py::rounds`.

    It had its own parser, which meant it also had its own blind spot: one
    compact log in the set stops at round 36 while its full trace carries all
    sixty, and this copy returned the short version without complaint. The
    generated tables read this path, so they reported that run's final state
    from round 36 while the chapter --- reading the guarded primitive --- had
    round 60.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core"))
    import logs as _logs
    return _logs.rounds(path)


def action_shares(paths: list[Path]) -> tuple[dict[str, float], int]:
    """Share per action across a cell's played turns, plus that denominator.

    Delegates to `core/turns.py::action_profile`. It used to have its own loop,
    which is how the generated tables came to disagree with the chapter: this
    copy filtered on the action field being set, and a dead agent stays in the
    record with `action: "no_action"`, which is truthy. At L3 knife-edge that
    put 2,506 of 18,000 turns in the denominator and diluted every action share
    in the table by a seventh, while the text --- reading the corrected
    primitive --- had it right.

    Two paths to one number is the failure this package exists to remove. There
    is now one implementation and this is a thin wrapper on it.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core"))
    from turns import action_profile
    per = action_profile(list(paths))
    noemer = next(iter(per.values())).denominator if per else 0
    return {k: v.value for k, v in per.items()}, (noemer or 0)


def rewire_shares(paths: list[Path]) -> tuple[dict[str, float], int]:
    """Share of turns carrying a drop or invite intent, on the same denominator.

    Kept apart from `action_shares` because it is a different axis: an agent can
    give and invite in the same turn. These percentages therefore do not sum to a
    hundred together with the actions.
    """
    tally: Counter = Counter()
    n = 0
    for p in paths:
        for e in rounds(log_path(p)):
            for a in (e.get("agents") or {}).values():
                if not a.get("action"):
                    continue
                n += 1
                ri = a.get("rewire_intent") or {}
                if ri.get("drop"):
                    tally["drop"] += 1
                if ri.get("invite"):
                    tally["invite"] += 1
    shares = {k: (100 * tally.get(k, 0) / n if n else 0.0) for k in ("drop", "invite")}
    shares["rewire"] = shares["drop"] + shares["invite"]
    return shares, n


def gini(values) -> float:
    """Gini over a series; 0 is perfect equality. Negative values count as zero."""
    xs = sorted(max(0.0, float(x)) for x in values if x is not None)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(xs, 1))
    g = (2 * cum) / (n * s) - (n + 1) / n
    # A perfectly equal series lands just below zero — the L1 scarce runs all come
    # out at -2.2e-16 — and prints as "-0.000", which reads as a negative
    # inequality. Clamp only what is indistinguishable from zero in double
    # precision; anything genuinely below that is a bug and should stay visible.
    return 0.0 if -1e-12 < g < 1e-12 else g


def final_gini(path: Path) -> float:
    """Gini over the resources of the living agents in the last round."""
    rs = rounds(log_path(path))
    if not rs:
        return 0.0
    return gini(a.get("resources") for a in (rs[-1].get("agents") or {}).values())


def alive(path: Path) -> int:
    """Agents holding resources above `ALIVE_FLOOR` in the last round.

    Not `len(agents)`: dead agents remain in the record at zero.
    """
    rs = rounds(log_path(path))
    if not rs:
        return 0
    return sum(1 for a in (rs[-1].get("agents") or {}).values()
               if (a.get("resources") or 0) > ALIVE_FLOOR)


def mean_sd(xs) -> tuple[float, float]:
    """Mean and sample standard deviation; (0, 0) on an empty series."""
    xs = [float(x) for x in xs]
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5
