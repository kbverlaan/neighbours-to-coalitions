"""Section 4.3, Take the language away --- the four no-channel controls.

Every figure here is a comparison of one control cell against its speaking
counterpart, knife-edge only, five runs against fifteen or ten. The asymmetry in
n is not incidental: the controls are small, so a figure is reported with both
sides' n and no difference is tested.

What makes this section readable is that the two sides share everything except
the channel --- same board, same private notes, same seeds where available --- so
a quantity that moves is attributable to the messaging and not to the payoff.
"""
from __future__ import annotations

import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HIER.parent / _m))

import re                                                    # noqa: E402
import combat, graph, logs, model, runstat, text, turns      # noqa: E402
from result import Result                                    # noqa: E402
import runset                                                # noqa: E402

PAIRS = {f"L{i}": (f"prod_L{i}_knife", f"prod_L{i}_knife_nocomm") for i in (1, 2, 3, 4)}

# An agent writing down a plan aimed at whoever is currently ahead. Narrow on
# purpose: it must name a move and a reason to make it, not merely mention the
# leader. UNVALIDATED — reported as an upper bound.
PLAN_AGAINST_LEADER = re.compile(
    r"\b(?:target|attack|strike|take from|coordinate against|move against|"
    r"bring down|reduce)\b[^.!?]{0,60}\b(?:leader|strongest|richest|wealthiest|"
    r"top|dominant|ahead)\b"
    r"|\b(?:leader|strongest|richest|wealthiest|dominant)\b[^.!?]{0,60}"
    r"\b(?:should be|must be|needs? to be)\b[^.!?]{0,30}"
    r"\b(?:attacked|targeted|stopped|reduced)\b", re.I)


# --- per-run quantities that only this section needs -----------------------

def _first_pair_round(p) -> float | None:
    """The round in which the first mutual pair of givers completes.

    Read from declared `transfer` actions, not from the settled flows. The flow
    field records what a resolved fight moves as well, under the attacker's and
    defender's names, so two fights in opposite directions completed a "mutual
    pair": at L2 knife-edge four of the first six runs got their first pair from
    combat, one of them in round 7 where the first real pair of givers is round
    34, and at L3 runs scored a pair in rounds 5 or 6 while containing no
    reciprocal transfer at all.
    """
    gezien, eerst = set(), None
    for e in logs.rounds(p):
        r = e.get("round")
        for nm, a in (e.get("agents") or {}).items():
            if a.get("action") != "transfer" or not a.get("target"):
                continue
            paar = (nm, a["target"])
            if (paar[1], paar[0]) in gezien and eerst is None:
                eerst = r
            gezien.add(paar)
    return eerst


def _pairs_per_run(p) -> int:
    return graph.mutual_dyads(p, min_count=1)


def _pairs_per_run_strict(p) -> int:
    return graph.mutual_dyads(p, min_count=2)


def _combats(p) -> list[dict]:
    return [c for e in logs.rounds(p) for c in (e.get("combat") or [])
            if isinstance(c, dict)]


def _first_attack_round(p) -> float | None:
    for e in logs.rounds(p):
        if e.get("combat"):
            return e.get("round")
    return None


def _first_attack_coalition(p) -> float | None:
    for e in logs.rounds(p):
        for c in (e.get("combat") or []):
            if isinstance(c, dict):
                return len(c.get("attackers") or [])
    return None


def _strength_ratio(p) -> float | None:
    """Mean coalition power over defender power, across a run's fights."""
    v = [c["coalition_power"] / c["defender_power"] for c in _combats(p)
         if c.get("defender_power")]
    return sum(v) / len(v) if v else None


def _win_rate(p) -> float | None:
    cs = _combats(p)
    return 100 * sum(1 for c in cs if c.get("winner") == "coalition") / len(cs) if cs else None


def _hit_the_richest(p) -> float | None:
    """Share of attacks whose target was among the three richest going in.

    Read from the board *before* the round, not from the round's own record.
    The record is end-of-round state and a blow takes its share out of the
    target, so scoring against it asks whether the agent was rich after being
    robbed. It read 37 per cent for L3 knife-edge where the corrected reading
    gives 50, and the chapter printed both --- this one in Section 4.3 and
    `whom_they_hit`'s in Section 4.2.
    """
    rs = logs.rounds(p)
    per = {e.get("round"): e for e in rs}
    raak = tot = 0
    for e in rs:
        ag = (per.get((e.get("round") or 0) - 1) or {}).get("agents") \
             or e.get("agents") or {}
        rijk = {nm for nm, _ in sorted(ag.items(),
                key=lambda kv: -(kv[1].get("resources") or 0))[:3]}
        for c in (e.get("combat") or []):
            if isinstance(c, dict) and c.get("defender"):
                tot += 1
                raak += c["defender"] in rijk
    return 100 * raak / tot if tot else None


def _target_relative_wealth(p) -> float | None:
    """Target's holdings as a multiple of the mean holding of the living.

    Same correction as `_hit_the_richest`: the standing is read from the board
    going into the round, so a target is not scored on what the blow left it.
    """
    v = []
    rs = logs.rounds(p)
    _per = {e.get("round"): e for e in rs}
    for e in rs:
        ag = (_per.get((e.get("round") or 0) - 1) or {}).get("agents") \
             or e.get("agents") or {}
        levend = [a.get("resources") or 0 for a in ag.values() if (a.get("resources") or 0) > 0]
        if not levend:
            continue
        m = sum(levend) / len(levend)
        for c in (e.get("combat") or []):
            d = isinstance(c, dict) and c.get("defender")
            if d and d in ag and m:
                v.append((ag[d].get("resources") or 0) / m)
    return sum(v) / len(v) if v else None


def _known_names(p) -> float:
    """How many distinct agent names a typical agent has written down.

    Read from the private trace, so it survives the channel being removed. It
    measures what an agent has learnt about the population, which is what a rota
    needs and what silence takes away.

    Each agent's trace is joined once and scanned with a single alternation of
    all thirty names. Searching name by name, round by round, is the same answer
    at 54,000 regex passes per run instead of one, and it made this section take
    longer to compute than every other figure in the chapter combined.
    """
    return _names_in(p, ("thinking", "memory"))


def _known_names_memory(p) -> float:
    """The same count over the memory field alone --- what an agent wrote down
    to keep, rather than everything it thought about in passing."""
    return _names_in(p, ("memory",))


def _names_in(p, velden) -> float:
    rs = logs.rounds(p)
    alle = set()
    for e in rs:
        alle |= set((e.get("agents") or {}).keys())
    if not alle:
        return 0.0
    naald = re.compile(r"\b(" + "|".join(sorted(alle)) + r")\b")
    per = []
    for nm in alle:
        stukken = []
        for e in rs:
            a = (e.get("agents") or {}).get(nm) or {}
            for v in velden:
                stukken.append(str(a.get(v) or ""))
        per.append(len(set(naald.findall(" ".join(stukken))) - {nm}))
    return sum(per) / len(per)


def _final_stock(p) -> float | None:
    for e in reversed(logs.rounds(p)):
        c = e.get("commons")
        if isinstance(c, dict) and c.get("stock_before") is not None:
            return float(c["stock_before"])
    return None


def _severings(p) -> int:
    return sum(1 for e in logs.rounds(p)
               for a in (e.get("agents") or {}).values()
               if (a.get("rewire_intent") or {}).get("drop"))


# --- the figures -----------------------------------------------------------

QUANTITIES = {
    "transfer_pct": lambda ps: turns.share(ps, turns.is_action("transfer")).value,
    "hold_pct": lambda ps: turns.share(ps, turns.is_action("hold")).value,
    "take_pct": lambda ps: turns.share(ps, turns.is_action("take")).value,
    "strengthen_pct": lambda ps: turns.share(ps, turns.is_action("strengthen")).value,
}

PER_RUN = {
    "first_pair_round": _first_pair_round,
    "pairs_per_run": _pairs_per_run,
    "pairs_per_run_min2": _pairs_per_run_strict,
    "mean_final_holding": runstat.mean_holding,
    "final_gini": runstat.final_gini,
    "combats": lambda p: len(_combats(p)),
    "first_attack_round": _first_attack_round,
    "first_attack_coalition": _first_attack_coalition,
    "strength_ratio": _strength_ratio,
    "win_rate_pct": _win_rate,
    "hit_the_richest_pct": _hit_the_richest,
    "target_relative_wealth": _target_relative_wealth,
    "known_names": _known_names,
    "known_names_memory_only": _known_names_memory,
    "severings": _severings,
}


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    h = len(xs) // 2
    return round(xs[h] if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2, 2)


def _side(paths) -> dict:
    uit = {k: round(fn(paths), 2) for k, fn in QUANTITIES.items()}
    for k, scalar in PER_RUN.items():
        xs = [scalar(p) for p in paths]
        geldig = [x for x in xs if x is not None]
        uit[k] = {"mean": round(sum(geldig) / len(geldig), 2) if geldig else None,
                  "median": _median(xs),
                  "runs_with_a_value": len(geldig), "runs": len(xs)}
    return uit


def removing_channel() -> dict:
    """Every level, speaking against silent, on the same quantities.

    A quantity that is undefined for a run --- a first attack in a run with no
    fights, a strength ratio with no combat --- is reported as absent rather
    than as zero, with the number of runs that had a value. Those two are not
    the same claim, and treating them alike is what makes "the fighting stops"
    indistinguishable from "the fighting was never measured".
    """
    uit = {}
    for niveau, (sprekend, stil) in PAIRS.items():
        uit[niveau] = Result(
            value={"speaking": _side(runset.cel(sprekend)),
                   "silent": _side(runset.cel(stil))},
            n=len(runset.cel(sprekend)) + len(runset.cel(stil)),
            denominator=len(runset.cel(stil)), unit="runs",
            note=f"{sprekend} ({len(runset.cel(sprekend))} runs) against "
                 f"{stil} ({len(runset.cel(stil))} runs); no difference is "
                 "tested, the control arm is five runs").as_dict()
    return uit


def private_plans() -> dict:
    """Plans against whoever is ahead, written privately, speaking against silent.

    Runs on the reasoning trace only, so it is the one measure in this section
    the channel cannot mechanically suppress: whatever an agent still works out
    for itself stays visible. The detector is unvalidated and broad enough to
    catch a plan being rejected as well as adopted, so it is an upper bound.
    """
    uit = {}
    stream = text.private("thinking")
    for niveau, (sprekend, stil) in PAIRS.items():
        uit[niveau] = {
            "speaking": text.share(runset.cel(sprekend), stream, PLAN_AGAINST_LEADER).as_dict(),
            "silent": text.share(runset.cel(stil), stream, PLAN_AGAINST_LEADER).as_dict()}
    return uit


def _collapse_round(p) -> int | None:
    """The first round whose record opens with an empty stock, or None."""
    for e in logs.rounds(p):
        c = e.get("commons")
        if isinstance(c, dict) and c.get("stock_before") is not None:
            if float(c["stock_before"]) <= 0.1:
                return e.get("round")
    return None


def commons_collapse() -> dict:
    """Whether the shared stock survives, across every cell that has one.

    The stock is a state variable and not an inference, so this is a count and
    not an estimate: a run either ends at carrying capacity or it does not.

    The Qwen cell was added later. The chapter explained the L4 peace by
    the arithmetic --- holdings stay level, so an attack at parity is worth -2
    per cent and no target worth hitting appears --- and never reported what the
    second arm does with the same rung. It empties the stock and then fights,
    which makes level holdings an outcome of that arm rather than a property of
    the rung. Leaving the cell out of this figure was what let the explanation
    stand unchallenged.

    `fights_before_the_collapse` is the load-bearing column. §4.3 reports of the
    five silent controls that every fight fell in or after the round its own
    stock reached zero; if the Qwen arm has the same shape with a channel open,
    the reading generalises from the wire to the breadth of coordination.
    """
    uit = {}
    for c in ([f"prod_L4_{p}" for p in ("scar", "knife", "abund")]
              + ["prod_L4_knife_nocomm", "robust_qwen_L4_knife"]):
        paths = runset.cel(c)
        eind = [_final_stock(p) for p in paths]
        vol = sum(1 for x in eind if x is not None and x >= 119.9)
        leeg = sum(1 for x in eind if x is not None and x <= 0.1)
        nul = [_collapse_round(p) for p in paths]
        voor = na = 0
        for p, nr in zip(paths, nul):
            for r, _ in combat.fights(p):
                if nr is None:
                    voor += 1          # never collapsed: every fight precedes nothing
                elif r < nr:
                    voor += 1
                else:
                    na += 1
        uit[c] = Result(value={"at_capacity": vol, "collapsed": leeg, "runs": len(paths),
                               "fights": voor + na,
                               "fights_before_the_collapse": voor,
                               "fights_in_or_after_it": na},
                        n=len(paths), denominator=len(paths), unit="runs",
                        sensitivity={"final_stock": [round(x, 1) if x is not None else None
                                                     for x in eind],
                                     "collapse_round": nul},
                        note="stock at the last round; capacity is 120.0. A run that "
                             "never empties has all its fights counted as before, "
                             "which is the conservative direction for the claim that "
                             "violence follows collapse").as_dict()
    return uit


FIGURES = {
    "m:removing-channel": removing_channel,
    "m:stock-survives-silence": commons_collapse,
    "m:private-notes-keep": private_plans,
}
