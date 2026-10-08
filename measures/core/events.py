"""Classifying an event by what surrounds it in the log.

Some figures ask what an action was *for*: a transfer that repays an earlier
arming, an attack that follows an accusation, a harvest that respects a rota.
The answer is never in the action record itself; it is in what the other agent
did before or after. This module builds the index once and then applies a list
of rules to each event.

Rules are ordered and exclusive. That ordering is a real choice and not a
neutral one: a transfer that both repays an arming and belongs to a mutual pair
lands in whichever rule comes first, which makes the earlier categories upper
bounds and the last one a residual. `classify` reports the rule order it used so
the figure can state it, and `overlaps` reports how many events matched more
than one rule --- because if that number is large, the exclusive split is
hiding most of what is going on.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from logs import rounds        # noqa: E402
from result import Result      # noqa: E402
import runset                  # noqa: E402


class Index(NamedTuple):
    """Everything a rule might need to look up, built once per run."""
    acted: dict            # (agent, action) -> [rounds]
    directed: dict         # (agent, action, target) -> [rounds]
    pairs: set             # (agent, action, target) seen at any point
    events: list           # [(round, agent, target, action)]


def index(path: Path) -> Index:
    acted, directed = defaultdict(list), defaultdict(list)
    pairs, events = set(), []
    for e in rounds(path):
        r = e.get("round")
        for nm, a in (e.get("agents") or {}).items():
            act, tgt = a.get("action"), a.get("target")
            if not act:
                continue
            acted[(nm, act)].append(r)
            events.append((r, nm, tgt, act))
            if tgt:
                directed[(nm, act, tgt)].append(r)
                pairs.add((nm, act, tgt))
    return Index(dict(acted), dict(directed), pairs, events)


# --- rule builders: the vocabulary a figure writes its rules in -------------

def did_before(action: str, to_giver: bool = True) -> Callable:
    """The other party played `action` on this one, in an earlier round."""
    def rule(ix: Index, r, actor, other):
        key = (other, action, actor) if to_giver else (other, action, None)
        return any(x < r for x in ix.directed.get(key, ()))
    return rule


def did_within(action: str, window: int, targeted: bool = False) -> Callable:
    """The other party played `action` in the `window` rounds before this one."""
    def rule(ix: Index, r, actor, other):
        rs = (ix.directed.get((other, action, actor), ()) if targeted
              else ix.acted.get((other, action), ()))
        return any(r - window <= x < r for x in rs)
    return rule


def did_ever(action: str) -> Callable:
    """The other party played `action` on this one at any point in the run."""
    def rule(ix: Index, r, actor, other):
        return (other, action, actor) in ix.pairs
    return rule


def classify(paths, action: str, rules: list[tuple[str, Callable]],
             residual: str = "unmatched") -> tuple[Result, dict]:
    """Assign every `action` event to the first matching rule.

    Returns the pooled split and the per-run splits, because pooling weights a
    run by how many events it produced and that weighting has already changed a
    headline in this chapter once.
    """
    pooled: Counter = Counter()
    per_run: dict[str, Counter] = {}
    meervoudig = 0
    for p in paths:
        ix = index(p)
        t: Counter = Counter()
        for r, actor, other, act in ix.events:
            if act != action or not other:
                continue
            t["total"] += 1
            raak = [naam for naam, rule in rules if rule(ix, r, actor, other)]
            if len(raak) > 1:
                meervoudig += 1
            t[raak[0] if raak else residual] += 1
        pooled.update(t)
        per_run[p.name] = t
    tot = pooled["total"]
    namen = [n for n, _ in rules] + [residual]
    return (Result(
        value={n: round(100 * pooled[n] / tot, 2) if tot else 0.0 for n in namen},
        n=len(paths), denominator=tot, unit=f"{action} events",
        sensitivity={"counts": {n: pooled[n] for n in namen},
                     "rule_order": [n for n, _ in rules],
                     "matched_more_than_one_rule": meervoudig,
                     "per_run_median": _run_median(per_run, namen),
                     "events_per_run": sorted(t["total"] for t in per_run.values())},
        note="rules are ordered and exclusive; earlier rules are upper bounds "
             "and the residual is a lower bound"),
        per_run)


def _run_median(per_run, namen):
    uit = {}
    for n in namen:
        xs = sorted(100 * t[n] / t["total"] for t in per_run.values() if t["total"])
        if xs:
            h = len(xs) // 2
            uit[n] = round(xs[h] if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2, 2)
    return uit
