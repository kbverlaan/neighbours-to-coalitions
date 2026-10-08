"""Building a graph out of who did what to whom, and reading structure off it.

Pairs, camps, coalitions and harvest groups are all the same construction with
different arguments: take some interactions, decide when they constitute a tie,
and take the connected components. Doing that once means a "camp" and a "pair"
cannot quietly use different rules for what counts as a link.

Two arguments carry the whole definition and both are free parameters, so a
figure must state them and the standards test requires the alternative to be
computed:

  `mutual`    whether a tie needs both directions, or one is enough
  `min_count` how many interactions make a tie

The third choice is the window. A structure read off the whole run and then
scored against actions from that same run is a tautology; every camp figure
here fixes the structure on an early window and scores a later one.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from logs import rounds        # noqa: E402
from result import Result      # noqa: E402
import runset                  # noqa: E402


def components(edges, members) -> list[list[str]]:
    """Connected components of an undirected edge set, singletons dropped."""
    ouder = {m: m for m in members}

    def vind(x):
        while ouder[x] != x:
            ouder[x] = ouder[ouder[x]]
            x = ouder[x]
        return x

    for a, b in edges:
        if a in ouder and b in ouder:
            ra, rb = vind(a), vind(b)
            if ra != rb:
                ouder[ra] = rb
    groepen = defaultdict(list)
    for m in members:
        groepen[vind(m)].append(m)
    return [sorted(g) for g in groepen.values() if len(g) > 1]


def interactions(path: Path, actions=(), flows=False, rounds_in=None):
    """(round, a, b) for every interaction of the requested kind.

    `actions` reads declared actions with a target; `flows` reads the settled
    per-round transfer amounts. They are not the same thing: an action that was
    declared but yielded nothing leaves no flow. A figure about who ended up
    connected should use flows; one about who tried should use actions.
    """
    for e in rounds(path):
        r = e.get("round")
        if rounds_in and not (rounds_in[0] <= r <= rounds_in[1]):
            continue
        if actions:
            for nm, a in (e.get("agents") or {}).items():
                if a.get("action") in actions and a.get("target"):
                    yield r, nm, a["target"]
        if flows:
            for k, v in (e.get("bilateral_flows") or {}).items():
                if v and "→" in k:
                    a, b = k.split("→", 1)
                    yield r, a.strip(), b.strip()


def build(path: Path, actions=(), flows=False, rounds_in=None,
          mutual: bool = False, min_count: int = 1):
    """(groups, members) for one run under one definition of a tie."""
    gericht: Counter = Counter()
    leden = set()
    for e in rounds(path):
        leden.update((e.get("agents") or {}).keys())
    for _, a, b in interactions(path, actions, flows, rounds_in):
        gericht[(a, b)] += 1
        leden.update((a, b))
    if mutual:
        edges = {tuple(sorted((a, b))) for (a, b), n in gericht.items()
                 if n >= min_count and gericht.get((b, a), 0) >= min_count}
    else:
        ongericht: Counter = Counter()
        for (a, b), n in gericht.items():
            ongericht[tuple(sorted((a, b)))] += n
        edges = {p for p, n in ongericht.items() if n >= min_count}
    return components(edges, leden), leden


def group_profile(paths, digits: int = 1, **kw) -> Result:
    """Group counts and sizes over a cell, plus each group's median endowment."""
    from runstat import final, _median
    per_grootte = defaultdict(list)
    totaal = 0
    for p in paths:
        groepen, _ = build(p, **kw)
        eind = final(p)
        for g in groepen:
            per_grootte[len(g)].append(_median([eind[m] for m in g if m in eind]))
            totaal += 1
    maten = {}
    for k in sorted(per_grootte):
        v = sorted(x for x in per_grootte[k] if x is not None)
        if v:
            maten[k] = {"groups": len(v), "median": round(_median(v), digits),
                        "min": round(v[0], digits), "max": round(v[-1], digits),
                        "values": [round(x, digits) for x in v] if len(v) <= 10 else None}
    return Result(value=totaal, n=len(paths), denominator=totaal, unit="groups",
                  sensitivity={"by_size": maten, "definition": kw},
                  note="connected components; per size the median over group medians")


def mutual_dyads(path: Path, min_count: int = 1, **kw) -> int:
    """Pairs of agents who gave to each other, counted as dyads not components.

    A dyad and not a connected component: counting components of size exactly
    two asks how many pairs are isolated from everyone else, which at L1 is a
    much smaller number because most pairs share a member with another pair.

    `min_count` is the free parameter --- one transfer each way makes an event,
    two makes something that recurred --- and callers report both. The chapter
    previously used one threshold at L1 and another at L4 while calling both
    "mutual pairs"; that is the drift this package exists to remove.

    Reads declared `transfer` actions by default rather than settled flows.
    Where combat exists the two diverge sharply, because a resolved fight also
    moves resources between two named agents and so registers as a flow: at L3
    the flow reading gives 83 pairs per run against 2 on actions.
    """
    kw.setdefault("actions", ("transfer",))
    tel: Counter = Counter()
    for _, a, b in interactions(path, **kw):
        tel[(a, b)] += 1
    return sum(1 for (a, b), n in tel.items()
               if n >= min_count and tel.get((b, a), 0) >= min_count) // 2
