"""Section 4.2, What each capacity brings --- the four rungs, one at a time.

The largest section in the chapter and the one with the most figures. Split by
what a figure reads: the ones here take the action and combat log only, so they
carry no detector and no validation caveat. The language figures for this
section live alongside them and say what they rest on.
"""
from __future__ import annotations

import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HIER.parent / _m))

import re
from collections import Counter                                          # noqa: E402
import combat, events, graph, lexicons as LEX, logs, model, runstat, text, turns  # noqa: E402
from result import Result                                                # noqa: E402
import runset                                                            # noqa: E402

PAYOFFS = ("scar", "knife", "abund")
RUNGS = ("L1", "L2", "L3", "L4")


def _cells(*rungs):
    return [f"prod_{r}_{p}" for r in rungs for p in PAYOFFS]


# `_median` and `_summary` moved to core/runstat.py when a second figures
# module needed them; two copies of a summary shape is exactly the drift this
# package exists to remove. Kept as names here so the figures below read the same.
_median = runstat.median
_summary = runstat.summary


# --- m:form-profile --------------------------------------------------------

def _reciprocal_pairs(p) -> int:
    return graph.mutual_dyads(p, min_count=1)


def _cycles(p) -> int:
    """Closed giving cycles of three or more, over the whole run.

    A cycle is a directed loop in which every agent gives to the next: the
    smallest arrangement that cannot be maintained by two agents alone, and so
    the first form that needs a third party.
    """
    rand = set()
    for _, a, b in graph.interactions(p, actions=("transfer",)):
        rand.add((a, b))
    tel = 0
    for a, b in rand:
        for c in {y for x, y in rand if x == b} - {a}:
            if (c, a) in rand:
                tel += 1
    return tel // 3


def form_profile() -> dict:
    """Which arrangements each cell produces: pairs, cycles, coined names.

    The three are on the same denominator --- one run --- so they can be read
    against each other. A pair needs two agents, a cycle three, and a coined name
    three distinct users, which is why the three rise and fall together and why
    the third is not independent evidence of the first two.
    """
    uit = {}
    for c in _cells(*RUNGS):
        paths = runset.cel(c)
        uit[c] = Result(
            value={"reciprocal_pairs": _summary([_reciprocal_pairs(p) for p in paths]),
                   "giving_cycles": _summary([_cycles(p) for p in paths]),
                   "coined_names": _summary([len(text.named_agreements(p)) for p in paths])},
            n=len(paths), denominator=len(paths), unit="runs").as_dict()
    return uit


# --- m:many-attack-once ----------------------------------------------------

def _target_degree(e, naam) -> int:
    """How many neighbours the struck agent had in that round."""
    from collections import Counter as _C
    tel: _C = _C()
    for a, b in (e.get("network") or {}).get("edges") or []:
        tel[a] += 1
        tel[b] += 1
    return tel.get(naam, 0)


def coalition_sizes() -> dict:
    """How many agents strike together, across every cell with fighting.

    Reported as the distribution and not the mean: the chapter's claim is that
    the blow is either solo or overwhelming with little between, which is a
    statement about the shape.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        tel: Counter = Counter()
        # The largest coalition a run produced, against how many neighbours the
        # agent it struck actually had. The chapter read the L2 ceiling of seven
        # as the target's degree --- as structure rather than as behaviour ---
        # and that is testable.
        grootste = []
        for p in paths:
            rs = logs.rounds(p)
            per = {e.get("round"): e for e in rs}
            top = None
            for r, x in combat.fights(p):
                k = len(x.get("attackers") or [])
                tel[k] += 1
                if top is None or k > top[0]:
                    graad = _target_degree(per.get(r) or {}, x.get("defender"))
                    top = (k, graad)
            if top:
                grootste.append(top)
        n = sum(tel.values())
        if not n:
            continue
        omsingeld = sum(1 for k, g in grootste if g and k >= g)
        uit[c] = Result(
            value={k: round(100 * tel[k] / n, 1) for k in sorted(tel)},
            n=len(paths), denominator=n, unit="fights",
            sensitivity={"counts": dict(sorted(tel.items())),
                         "runs_with_a_fight": sum(1 for p in paths if combat.count(p)),
                         "largest_coalition_per_run": _summary([k for k, _ in grootste], 1),
                         "degree_of_that_target": _summary([g for _, g in grootste], 1),
                         "runs_where_every_neighbour_joined": omsingeld,
                         "runs_compared": len(grootste)},
            note="share of fights by number of attackers; the sensitivity block "
                 "asks whether the largest coalition was bounded by the target's "
                 "neighbourhood or fell short of it").as_dict()
    return uit


# --- m:whom-hit-over -------------------------------------------------------

def whom_they_hit() -> dict:
    """Where the target stands when it is struck.

    Holdings relative to the mean of the living, and rank within the cell's
    population at that round. Both, because "the richest" and "above average"
    are different claims and the chapter makes both in different places.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        rel, rang, top3 = [], [], 0
        n = 0
        # The chapter also claims the target moves up the ordering as the run
        # proceeds, which the whole-run median cannot show. Same quantities,
        # split into the first and last twenty rounds.
        vroeg, laat = {"rang": [], "top3": 0, "n": 0}, {"rang": [], "top3": 0, "n": 0}
        for p in paths:
            rs = logs.rounds(p)
            vorige = {e.get("round"): e for e in rs}
            for e in rs:
                # Where the target stood going *into* the round, not after the
                # blow had already taken its share out.
                bord = (vorige.get((e.get("round") or 0) - 1) or {}).get("agents") or {}
                ag = bord or (e.get("agents") or {})
                levend = [(nm, a.get("resources") or 0) for nm, a in ag.items()
                          if (a.get("resources") or 0) > 0]
                if not levend:
                    continue
                m = sum(v for _, v in levend) / len(levend)
                gesorteerd = [nm for nm, _ in sorted(levend, key=lambda kv: -kv[1])]
                for x in (e.get("combat") or []):
                    d = isinstance(x, dict) and x.get("defender")
                    if not d or d not in ag:
                        continue
                    n += 1
                    if m:
                        rel.append((ag[d].get("resources") or 0) / m)
                    if d in gesorteerd:
                        i = gesorteerd.index(d)
                        rang.append(i + 1)
                        top3 += i < 3
                        r = e.get("round") or 0
                        emmer = vroeg if r <= 20 else (laat if r > 40 else None)
                        if emmer is not None:
                            emmer["rang"].append(i + 1)
                            emmer["top3"] += i < 3
                            emmer["n"] += 1
        if not n:
            continue
        uit[c] = Result(
            value={"holdings_relative_to_living_mean": _summary(rel),
                   "rank_among_the_living": _summary(rang, 1),
                   "share_in_the_top_three_pct": round(100 * top3 / n, 1),
                   "rounds_1_to_20": {
                       "rank": _summary(vroeg["rang"], 1),
                       "top_three_pct": round(100 * vroeg["top3"] / vroeg["n"], 1)
                           if vroeg["n"] else None,
                       "fights": vroeg["n"]},
                   "rounds_41_to_60": {
                       "rank": _summary(laat["rang"], 1),
                       "top_three_pct": round(100 * laat["top3"] / laat["n"], 1)
                           if laat["n"] else None,
                       "fights": laat["n"]}},
            n=len(paths), denominator=n, unit="fights",
            baseline=round(100 * 3 / 30, 1),
            note="the baseline is the share expected if targets were chosen "
                 "at random from thirty agents").as_dict()
    return uit


# --- what losing costs -----------------------------------------------------
#
# Not registered. It is correct and it still runs, but no claim in the chapter
# rests on it any more: the support-after-defeat sentence it used to carry moved
# to `support_after_defeat`, which measures the support rather than the loss.
# Kept rather than deleted because retiring a measure should not destroy it.

def what_happens_to_the_loser() -> dict:
    """Whether losing a fight is recoverable, and who loses.

    `winner` is a role label --- 'coalition' or 'defender' --- and never an agent
    name. Comparing it against the defender's name is true of every fight and
    made an earlier version of this measure count all fights as defender losses.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        verlies, dood, n = [], 0, 0
        verdedigers_winnen = 0
        for p in paths:
            rs = logs.rounds(p)
            per_ronde = {e.get("round"): e for e in rs}
            for r, x in combat.fights(p):
                n += 1
                if x.get("winner") == "defender":
                    verdedigers_winnen += 1
                    continue
                d = x.get("defender")
                # `voor` must be the round before the fight: the record for the
                # round itself already has the loss taken out, so reading it
                # measured decay and aftermath instead of the blow.
                voor = (per_ronde.get(r - 1, {}).get("agents") or {}).get(d, {})
                na = (per_ronde.get(r, {}).get("agents") or {}).get(d)
                v0 = voor.get("resources") or 0
                if na is not None and v0:
                    verlies.append(100 * (1 - (na.get("resources") or 0) / v0))
                laatste = (rs[-1].get("agents") or {}).get(d, {})
                dood += (laatste.get("resources") or 0) <= 0
        if not n:
            continue
        uit[c] = Result(
            value={"defender_wins_pct": round(100 * verdedigers_winnen / n, 1),
                   "loss_as_pct_of_holdings": _summary(verlies, 1),
                   "defeats_followed_by_elimination": dood},
            n=len(paths), denominator=n, unit="fights").as_dict()
    return uit


# --- m:where-l3-gini / m:drives-l3-levelling --------------------------------

def where_the_inequality_comes_from() -> dict:
    """Whether L3's inequality is made by combat or by giving.

    Per run, the total moved by resolved fights against the total moved by
    transfers, both read from the engine's own accounting rather than inferred
    from holdings.
    """
    uit = {}
    for c in _cells("L2", "L3"):
        paths = runset.cel(c)
        rijen = []
        for p in paths:
            # Both channels come from the engine's own per-agent accounting.
            # The bilateral flow field cannot serve for the giving channel: it
            # records what a resolved fight moves as well, under the attacker's
            # and defender's names, so a round with no transfer action at all
            # still produces a flow. Reading giving from there counts every
            # fight as a gift and understates the combat share.
            door_gevecht = door_gift = 0.0
            for e in logs.rounds(p):
                for a in (e.get("agents") or {}).values():
                    b = a.get("breakdown") or {}
                    door_gift += max(0.0, b.get("invest_received") or 0.0)
                    door_gevecht += max(0.0, b.get("combat_transfer") or 0.0)
            rijen.append({"combat": door_gevecht, "gift": door_gift,
                          "gini": runstat.final_gini(p)})
        combat_tot = sum(r["combat"] for r in rijen)
        gift_tot = sum(r["gift"] for r in rijen)
        uit[c] = Result(
            value={"moved_by_combat": round(combat_tot, 1),
                   "moved_by_transfer": round(gift_tot, 1),
                   "combat_share_pct": round(100 * combat_tot / (combat_tot + gift_tot), 1)
                                       if combat_tot + gift_tot else None},
            n=len(paths), denominator=len(paths), unit="runs",
            sensitivity={
                "combat_vs_final_gini": model.correlate(
                    [r["combat"] for r in rijen], [r["gini"] for r in rijen]).value,
                "transfer_vs_final_gini": model.correlate(
                    [r["gift"] for r in rijen], [r["gini"] for r in rijen]).value},
            note="both channels come from the engine's per-agent breakdown; "
                 "the bilateral flow field records combat movement as well and "
                 "cannot separate the two").as_dict()
    return uit


# --- m:opening-non-aggression ----------------------------------------------

def opening_pact() -> dict:
    """Non-aggression language in the opening rounds, and how far it reaches.

    The lexicon is unvalidated beyond its round-one behaviour, so the share is
    an upper bound. What carries the claim is the reach: how many of the thirty
    agents use the language, and by which round, which a broad lexicon inflates
    far less than it inflates a rate.
    """
    uit = {}
    for c in (_cells(*RUNGS) + [f"prod_{r}_knife_nocomm" for r in RUNGS]
              + [f"robust_qwen_{r}_knife" for r in ("L2", "L3", "L4")]
              + [f"robust_deepseek_{r}_knife" for r in ("L2", "L3", "L4")]):
        paths = runset.cel(c)
        eerste, sprekers, ronde90, ronde1 = [], [], [], []
        for p in paths:
            wie, eerst = set(), None
            per_ronde: dict[int, set] = {}
            for _, r, spreker, s in text.public([p]):
                if LEX.NAP.search(s):
                    wie.add(spreker)
                    per_ronde.setdefault(r, set()).add(spreker)
                    if eerst is None:
                        eerst = r
            eerste.append(eerst)
            sprekers.append(len(wie))
            ronde1.append(len(per_ronde.get(1, ())))
            lopend, bereikt = set(), None
            for r in sorted(per_ronde):
                lopend |= per_ronde[r]
                if bereikt is None and len(lopend) >= 27:
                    bereikt = r
            ronde90.append(bereikt)
        uit[c] = Result(
            value={"first_round_with_the_language": _summary(eerste, 1),
                   "distinct_agents_using_it": _summary(sprekers, 1),
                   "agents_using_it_in_round_one": _summary(ronde1, 1),
                   "round_reaching_ninety_per_cent": _summary(ronde90, 1)},
            n=len(paths), denominator=len(paths), unit="runs",
            note="lexicon unvalidated beyond its round-one behaviour; the reach "
                 "figures are far less sensitive to that than a rate would be").as_dict()
    return uit


FIGURES = {
    "m:form-profile": form_profile,
    "m:many-attack-once": coalition_sizes,
    "m:whom-hit-over": whom_they_hit,
    "m:where-l3-gini": where_the_inequality_comes_from,
    "m:opening-non-aggression": opening_pact,
}


# --- m:ceiling-rule-l3 / m:ceiling-anything --------------------------------

def ceiling_rule() -> dict:
    """Whether a stated wealth ceiling appears, when, and whether it binds.

    The CEILING pattern is registered as unvalidated and broad: any number near
    a negation matches it, so "I hold 45 and will not attack" reads as a stated
    threshold. Every figure here is therefore an upper bound on how often a
    ceiling is proposed, and the load-bearing questions are the ones a false
    positive cannot manufacture --- whether the highest holding in a run ever
    comes down to the number, and whether the targeting it describes was already
    happening before it was said.

    A placebo arm is reported alongside: the same pattern run over rounds before
    any fighting has begun, where a ceiling has nothing to regulate. A detector
    that fires as often there as it does later is describing vocabulary rather
    than an arrangement.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        met, eerste, plafonds, hoogste, placebo, placebo_n = 0, [], [], [], 0, 0
        for p in paths:
            eerst, getallen, vroeg = None, [], False
            # The placebo needs a fight to be earlier than. In a run with no
            # fighting every round is 'before the first blow', which would score
            # the whole cell as placebo and say nothing. Those runs are excluded
            # from the placebo arm rather than counted in it.
            grens = combat.first_round(p)
            for _, r, _, s in text.public([p]):
                m = LEX.CEILING.search(s)
                if not m:
                    continue
                if grens is not None and r < grens:
                    vroeg = True
                if eerst is None:
                    eerst = r
                for g in m.groups():
                    if g:
                        try:
                            getallen.append(float(g))
                        except ValueError:
                            pass
            if eerst is not None:
                met += 1
                eerste.append(eerst)
                if getallen:
                    plafonds.append(_median(getallen))
            if grens is not None:
                placebo_n += 1
                placebo += vroeg
            eind = runstat.final(p)
            hoogste.append(max(eind.values()) if eind else 0.0)
        uit[c] = Result(
            value={"runs_stating_a_ceiling": met, "runs": len(paths),
                   "first_round_stated": _summary(eerste, 1),
                   "median_number_named": _summary(plafonds, 1),
                   "highest_holding_at_round_60": _summary(hoogste, 1)},
            n=len(paths), denominator=len(paths), unit="runs",
            baseline={"runs_firing_before_any_fight": placebo,
                      "runs_with_a_fight": placebo_n},
            note="the baseline is the placebo arm: runs in which the same "
                 "pattern fires before any fighting has begun, where a ceiling "
                 "has nothing to regulate. UNVALIDATED lexicon; every count "
                 "here is an upper bound").as_dict()
    return uit


FIGURES["m:opening-pact-survive"] = opening_pact
FIGURES["m:ceiling-rule-l3"] = ceiling_rule


# --- m:accusation-attack / m:accusations-hold / collective sanction ---------

ACCUSATION_WINDOW = 5
BACKING_WINDOW = 3


def _viel_aan(rs, ronde, wie, doelwit) -> bool:
    """Whether `wie` attacked `doelwit` specifically, in that round."""
    for e in rs:
        if e.get("round") != ronde:
            continue
        a = (e.get("agents") or {}).get(wie) or {}
        return a.get("action") == "take" and a.get("target") == doelwit
    return False


def accusation_and_attack() -> dict:
    """Whether naming someone as an offender is followed by an attack on them.

    The question the chapter asks of this is whether predation has been
    repurposed as enforcement, so the test has to distinguish "the accused was
    attacked" from "somebody was attacked". The baseline does that: the same
    window, the same run, the same accusations, with the accused name replaced
    by another living agent drawn at random. That preserves how much fighting
    there is and destroys only the link being claimed.

    Both a solo and a collective follow-up are counted, separately. A collective
    attack --- two or more agents on one target --- is the one that carries the
    claim, since a single agent striking whoever it just complained about needs
    no enforcement to explain it.

    The detector undercounts accusations that name several offenders at once,
    and those are exactly the ones most likely to precede a coalition. The
    collective figures here are therefore lower bounds by more than the solo
    ones.

    A second baseline was added because the first does not settle what the
    chapter wants it to. Drawing the stand-in at random holds the level of
    violence fixed, but an accused agent is not a random agent: it is likelier
    to be poor, to have lost a fight, to be the one everybody is already
    circling. Any of those would draw both the accusation and the attack, and
    the multiple would then measure the vulnerability rather than the naming.

    `matched_on_wealth` draws the stand-in from the five living agents whose
    holdings are closest to the accused's in that round. If the multiple
    survives that, the naming contributes something of its own; if it collapses,
    the first baseline was measuring who was already exposed.
    """
    import random
    uit = {}
    # L4 as well, though its cells produce almost no accusations. A cell where
    # the detector finds nothing is a result --- it is the claim the L4 section
    # makes --- and leaving it out of the figure turns that result into a blank.
    for c in _cells("L2", "L3", "L4") + [f"robust_qwen_{r}_knife" for r in ("L2", "L3", "L4")]:
        paths = runset.cel(c)
        n = solo = coll = 0
        b_solo = b_coll = 0
        m_solo = m_coll = m_n = 0
        gedekt = gedekt_op_aanklager = 0
        rng = random.Random(20260815)
        for p in paths:
            rs = logs.rounds(p)
            levend_per_ronde = {e.get("round"): [nm for nm, a in (e.get("agents") or {}).items()
                                                 if (a.get("resources") or 0) > 0] for e in rs}
            bezit_per_ronde = {e.get("round"): {nm: (a.get("resources") or 0)
                                                for nm, a in (e.get("agents") or {}).items()
                                                if (a.get("resources") or 0) > 0} for e in rs}
            aanvallen = {}
            for r, x in combat.fights(p):
                d = x.get("defender")
                if d:
                    aanvallen.setdefault(r, {}).setdefault(d, 0)
                    aanvallen[r][d] = max(aanvallen[r][d], len(x.get("attackers") or []))

            def geraakt(naam, vanaf):
                # The window opens the round *after* the accusation. Including
                # the accusation's own round counted a fight that resolved
                # simultaneously with it as a consequence of it, and made the
                # window six rounds where the appendix says five.
                s = k = False
                for rr in range(vanaf + 1, vanaf + ACCUSATION_WINDOW + 1):
                    m = (aanvallen.get(rr) or {}).get(naam)
                    if m:
                        s = True
                        k = k or m >= 2
                return s, k

            # Whether the charge is true: did the accused in fact attack, in the
            # round it was named or the three before. Answered against the
            # engine's record and not against anyone's account of it.
            aanvaller = {}
            for e in rs:
                for nm, a in (e.get("agents") or {}).items():
                    if a.get("action") == "take" and a.get("target"):
                        aanvaller.setdefault(nm, []).append(e.get("round"))

            for _, r, aanklager, beschuldigd in text.accusations([p]):
                n += 1
                rondes = aanvaller.get(beschuldigd, ())
                if any(r - 3 <= x <= r for x in rondes):
                    gedekt += 1
                    if any((x, beschuldigd) and True for x in rondes
                           if r - 3 <= x <= r and _viel_aan(rs, x, beschuldigd, aanklager)):
                        gedekt_op_aanklager += 1
                s, k = geraakt(beschuldigd, r)
                solo += s
                coll += k
                pool = [x for x in levend_per_ronde.get(r, []) if x != beschuldigd]
                if pool:
                    nep = rng.choice(pool)
                    bs, bk = geraakt(nep, r)
                    b_solo += bs
                    b_coll += bk
                    # second null model: equally rich, so equally attractive as a target
                    bezit = bezit_per_ronde.get(r) or {}
                    mijn = bezit.get(beschuldigd)
                    if mijn is not None:
                        dichtbij = sorted((x for x in pool if x in bezit),
                                          key=lambda x: abs(bezit[x] - mijn))[:5]
                        if dichtbij:
                            ms, mk = geraakt(rng.choice(dichtbij), r)
                            m_solo += ms
                            m_coll += mk
                            m_n += 1
        # A cell where the detector finds nothing stays in the figure. Dropping
        # it made "no accusations at L4" indistinguishable from "L4 was never
        # scanned", and the table printed the same dash for both.
        if not n:
            uit[c] = Result(
                value={"accusations": 0}, n=len(paths), denominator=0,
                unit="accusations",
                note="the detector finds no accusation in this cell; the "
                     "shares below it are undefined rather than zero").as_dict()
            continue
        uit[c] = Result(
            value={"accusations": n,
                   "charge_backed_by_an_attack_pct": round(100 * gedekt / n, 1),
                   "backed_by_an_attack_on_the_accuser_pct": round(100 * gedekt_op_aanklager / n, 1),
                   "followed_by_any_attack_pct": round(100 * solo / n, 1),
                   "followed_by_a_collective_attack_pct": round(100 * coll / n, 1)},
            n=len(paths), denominator=n, unit="accusations",
            baseline={"any_attack_pct": round(100 * b_solo / n, 1),
                      "collective_attack_pct": round(100 * b_coll / n, 1),
                      "how": "the same accusations against a randomly drawn "
                             "living agent, same window, same run"},
            sensitivity={"window_rounds": ACCUSATION_WINDOW,
                         "matched_on_wealth": (
                             {"any_attack_pct": round(100 * m_solo / m_n, 1),
                              "collective_attack_pct": round(100 * m_coll / m_n, 1),
                              "accusations_with_a_match": m_n,
                              "how": "stand-in drawn from the five living agents whose "
                                     "holdings are closest to the accused's in that round, "
                                     "so the comparison holds exposure fixed as well as "
                                     "the level of violence"}
                             if m_n else None)},
            note="lower bound: the detector returns one accused per sentence, "
                 "so accusations naming several offenders --- the ones most "
                 "likely to precede a coalition --- are undercounted").as_dict()
    return uit


FIGURES["m:accusation-attack"] = accusation_and_attack
FIGURES["m:accusations-hold"] = accusation_and_attack
FIGURES["m:accusation-collective-sanction"] = accusation_and_attack


# --- m:arming-buys / m:new-capacity ----------------------------------------

def what_a_gift_buys() -> dict:
    """What follows an act of arming, and what follows an invitation.

    Two questions with the same shape: an agent does something for another, and
    the question is what the other does next. Both are counted against a
    name-shuffle baseline, because in a cell where a third of agents attack in
    any window "the armed agent attacked" is not evidence that the arming
    bought anything.

    Repayment is read as a transfer from the armed agent back to its armourer at
    any later point --- the loosest reading, so the figure is an upper bound on
    how often arming is repaid.
    """
    import random
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        tel = Counter()
        rng = random.Random(20260815)
        for p in paths:
            ix = events.index(p)
            rs = logs.rounds(p)
            levend = {e.get("round"): [nm for nm, a in (e.get("agents") or {}).items()
                                       if (a.get("resources") or 0) > 0] for e in rs}

            def deed(actor, actie, doel, vanaf, venster=5):
                for rr in range(vanaf + 1, vanaf + venster + 1):
                    if rr in ix.directed.get((actor, actie, doel), ()):
                        return True
                return False

            def deed_na(actor, actie, doel, vanaf):
                """Whether the act happened after the round in question.

                `ix.pairs` knows no time, so an earlier version counted a
                transfer made *before* the arming as repayment for it: at L2
                knife-edge that read 52.6 per cent where the strictly-after
                figure is 42.4.
                """
                return any(x > vanaf for x in ix.directed.get((actor, actie, doel), ()))

            for r, gever, ontvanger, act in ix.events:
                if not ontvanger:
                    continue
                if act == "strengthen":
                    tel["armings"] += 1
                    tel["armed_attacks_anyone"] += any(
                        rr in ix.acted.get((ontvanger, "take"), ())
                        for rr in range(r + 1, r + 6))
                    tel["armed_attacks_the_armourer"] += deed(ontvanger, "take", gever, r)
                    tel["repaid"] += deed_na(ontvanger, "transfer", gever, r)
                    pool = [x for x in levend.get(r, []) if x not in (gever, ontvanger)]
                    if pool:
                        nep = rng.choice(pool)
                        tel["baseline_attacks_anyone"] += any(
                            rr in ix.acted.get((nep, "take"), ())
                            for rr in range(r + 1, r + 6))
                        tel["baseline_attacks_the_armourer"] += deed(nep, "take", gever, r)
        n = tel["armings"]
        if not n:
            continue
        uit[c] = Result(
            value={"armings": n,
                   "armed_agent_attacks_anyone_pct": round(100 * tel["armed_attacks_anyone"] / n, 1),
                   "armed_agent_attacks_its_armourer_pct": round(100 * tel["armed_attacks_the_armourer"] / n, 1),
                   "armed_agent_ever_repays_pct": round(100 * tel["repaid"] / n, 1)},
            n=len(paths), denominator=n, unit="arming actions",
            baseline={"attacks_anyone_pct": round(100 * tel["baseline_attacks_anyone"] / n, 1),
                      "attacks_the_armourer_pct": round(100 * tel["baseline_attacks_the_armourer"] / n, 1),
                      "how": "a randomly drawn living agent, same round, same window"},
            note="repayment counts a transfer back at any later point, which is "
                 "the loosest reading and so an upper bound").as_dict()
    return uit


FIGURES["m:arming-buys"] = what_a_gift_buys


# --- m:new-capacity --------------------------------------------------------

def what_an_invitation_buys() -> dict:
    """Whether reaching for someone commits the inviter to anything.

    An invitation is a rewire intent and not an action with a target, so it is
    read from `rewire_intent.invite` rather than from the action field. Only
    invitations the engine accepted are counted --- `invite_outcome` of `added`
    --- since an invitation that failed created no tie and cannot be followed by
    an attack on a new neighbour.

    The comparison that matters is against the same inviter attacking anyone
    else in the same window. If an inviter attacks its invitee no more often
    than it attacks the rest of the board, the invitation predicts nothing.
    """
    import random
    uit = {}
    for c in _cells("L3", "L4"):
        paths = runset.cel(c)
        tel = Counter()
        rng = random.Random(20260815)
        for p in paths:
            ix = events.index(p)
            rs = logs.rounds(p)
            levend = {e.get("round"): [nm for nm, a in (e.get("agents") or {}).items()
                                       if (a.get("resources") or 0) > 0] for e in rs}
            for e in rs:
                r = e.get("round")
                buren = {}
                for a1, b1 in ((e.get("network") or {}).get("edges") or []):
                    buren.setdefault(a1, set()).add(b1)
                    buren.setdefault(b1, set()).add(a1)
                for nm, a in (e.get("agents") or {}).items():
                    ri = a.get("rewire_intent") or {}
                    genodigde = ri.get("invite")
                    if not genodigde or ri.get("invite_outcome") != "added":
                        continue
                    tel["invitations"] += 1
                    venster = range(r + 1, r + 6)
                    tel["inviter_attacks_the_invitee"] += any(
                        x in ix.directed.get((nm, "take", genodigde), ()) for x in venster)
                    # Only the inviter's other neighbours. At this capacity
                    # level an agent can attack nobody else, so comparing
                    # against the whole board would compare "could attack"
                    # against "could not" and manufacture the multiple.
                    pool = [q for q in buren.get(nm, ()) if q not in (nm, genodigde)
                            and q in levend.get(r, [])]
                    if pool:
                        nep = rng.choice(pool)
                        tel["inviter_attacks_someone_else"] += any(
                            x in ix.directed.get((nm, "take", nep), ()) for x in venster)
        n = tel["invitations"]
        if not n:
            continue
        uit[c] = Result(
            value={"invitations_accepted": n,
                   "inviter_attacks_the_invitee_pct": round(100 * tel["inviter_attacks_the_invitee"] / n, 2)},
            n=len(paths), denominator=n, unit="accepted invitations",
            baseline={"inviter_attacks_another_agent_pct":
                      round(100 * tel["inviter_attacks_someone_else"] / n, 2),
                      "how": "the same inviter, same window, against a randomly "
                             "drawn living agent it was already adjacent to and "
                             "did not invite --- an agent it could in fact attack"},
            note="only invitations the engine accepted are counted").as_dict()
    return uit


FIGURES["m:new-capacity"] = what_an_invitation_buys


# --- m:welfare-rules-as ----------------------------------------------------

def welfare_rules() -> dict:
    """Runs in which a collective welfare rule is stated, and whether it acts.

    The detector is hand-validated at 82.5 per cent precision --- of forty
    flagged messages, thirty-three state a real rule, five are borderline and
    two are wrong --- so it is one of the two lexicons in the register that
    carries a number rather than an upper bound. Recall is unknown, so the run
    counts are lower bounds.

    Stating a rule and acting on one are separate and both are reported. The
    behavioural half asks whether any transfer in that run reached an agent
    below one resource, which needs no detector at all, and it is the half that
    decides whether a stated floor is an arrangement or a sentence.
    """
    uit = {}
    for c in _cells("L2", "L3") + [f"robust_qwen_{r}_knife" for r in ("L2", "L3", "L4")]:
        paths = runset.cel(c)
        met, gered, kansen = 0, 0, 0
        per_run = []
        for p in paths:
            treffers = sum(1 for _, _, _, s in text.public([p]) if LEX.WELFARE.search(s))
            met += treffers > 0
            # Both halves of this come from one primitive, so the model-arm
            # figure in Section 4.4 and this one cannot answer the same
            # question differently. They did: see core/runstat.py.
            k, r_gered = runstat.destitute_and_rescues(p)
            kansen += k
            gered += r_gered
            per_run.append({"welfare_sentences": treffers, "rescues": r_gered})
        uit[c] = Result(
            value={"runs_stating_a_rule": met, "runs": len(paths),
                   "transfers_reaching_an_agent_below_one": gered},
            n=len(paths), denominator=kansen,
            unit="agent-rounds below one resource",
            sensitivity={"per_run": per_run},
            note="detector precision 82.5 per cent, recall unknown, so the run "
                 "count is a lower bound; the rescue count needs no detector").as_dict()
    return uit


FIGURES["m:welfare-rules-as"] = welfare_rules


# --- m:target-was-named ----------------------------------------------------

def target_named_beforehand() -> dict:
    """Whether the first target is spoken of before it is struck.

    The test is whether the defender's name appears in any public message in the
    rounds before the first fight. Taken alone that figure means little: in a
    cell where agents address each other constantly, most names appear
    beforehand.

    The baseline is therefore a name shuffle --- how often a randomly drawn agent
    who was *not* the first target had its name spoken in the same window, in the
    same run. That holds how talkative the cell is fixed and destroys only the
    link being claimed.
    """
    import random
    uit = {}
    for c in _cells("L2", "L3"):
        paths = runset.cel(c)
        genoemd = nep_genoemd = n = 0
        for p in paths:
            grens = combat.first_round(p)
            doelwit = combat.defender_of_first(p)
            if grens is None or not doelwit:
                continue
            n += 1
            rng = random.Random(20260815 + n)
            gezegd, allen = set(), set()
            for e in logs.rounds(p):
                allen |= set((e.get("agents") or {}).keys())
                if (e.get("round") or 0) >= grens:
                    continue
                for m in (e.get("messages") or []):
                    tekst = m.get("text") or ""
                    for nm in allen:
                        if re.search(rf"\b{re.escape(nm)}\b", tekst):
                            gezegd.add(nm)
            genoemd += doelwit in gezegd
            pool = [x for x in allen if x != doelwit]
            if pool:
                nep_genoemd += rng.choice(pool) in gezegd
        if not n:
            continue
        uit[c] = Result(
            value=round(100 * genoemd / n, 1), n=len(paths), denominator=n,
            unit="runs with a fight",
            baseline=round(100 * nep_genoemd / n, 1),
            note="the baseline is a randomly drawn agent that was not the first "
                 "target, named in the same window of the same run").as_dict()
    return uit


FIGURES["m:target-was-named"] = target_named_beforehand


# --- m:drives-l3-levelling -------------------------------------------------

def what_levels_them() -> dict:
    """Whether decay or the combat pot flattens the distribution at L3.

    Decay is proportional, so it cannot change a Gini at all: multiplying every
    holding by the same factor leaves the ratios untouched. That is arithmetic
    rather than a finding, and stating it is the point --- it removes the only
    rival explanation for the levelling without needing a measure.

    What can move the distribution is the combat pot, so the figure follows the
    wealthiest six from the round inequality peaks to round 60: what they hold
    as a share of the total, and how much passes out of them by combat.
    """
    uit = {}
    for c in _cells("L2", "L3"):
        paths = runset.cel(c)
        piek_aandeel, eind_aandeel, netto = [], [], []
        for p in paths:
            rs = logs.rounds(p)
            per = [(e, runstat.gini([(a.get("resources") or 0)
                                     for a in (e.get("agents") or {}).values()]))
                   for e in rs]
            piek_e = max(per, key=lambda x: x[1])[0]
            piek_r = piek_e.get("round")

            def top6_aandeel(e):
                v = sorted(((a.get("resources") or 0)
                            for a in (e.get("agents") or {}).values()), reverse=True)
                return 100 * sum(v[:6]) / sum(v) if sum(v) else None

            rijk = [nm for nm, _ in sorted(((nm, a.get("resources") or 0)
                    for nm, a in (piek_e.get("agents") or {}).items()),
                    key=lambda kv: -kv[1])[:6]]
            a0, a1 = top6_aandeel(piek_e), top6_aandeel(rs[-1])
            if a0 is not None and a1 is not None:
                piek_aandeel.append(a0)
                eind_aandeel.append(a1)
            # Net combat flow out of that six, from the peak round onward.
            uit_pot = 0.0
            for e in rs:
                if (e.get("round") or 0) < piek_r:
                    continue
                for nm, a in (e.get("agents") or {}).items():
                    if nm in rijk:
                        uit_pot -= (a.get("breakdown") or {}).get("combat_transfer") or 0.0
            netto.append(uit_pot)
        uit[c] = Result(
            value={"top_six_share_at_the_peak": _summary(piek_aandeel, 1),
                   "top_six_share_at_round_60": _summary(eind_aandeel, 1),
                   "net_combat_flow_out_of_the_top_six": _summary(netto, 1)},
            n=len(paths), denominator=len(paths), unit="runs",
            note="decay is proportional and cannot move a Gini, so it is "
                 "excluded by arithmetic rather than tested").as_dict()
    return uit


FIGURES["m:drives-l3-levelling"] = what_levels_them


# --- m:attacker-thinking ---------------------------------------------------

WIN_ESTIMATE = re.compile(
    r"\b(?:win|victory|succeed|success|odds|chance|probability|expected value|"
    r"ev|payoff)\b[^.!?]{0,60}\b(?:\d+(?:\.\d+)?\s*%|\d\.\d+|high|good|likely|"
    r"favou?rable|certain)\b"
    r"|\b(?:combined|together|our)\s+(?:strength|power)\b[^.!?]{0,40}"
    r"\b(?:exceed\w*|beat\w*|outweigh\w*|greater|higher|more than)\b", re.I)


def what_the_first_attacker_had() -> dict:
    """What is on record for the agent that strikes first, against a control.

    Three things are checked in the rounds before the blow: whether it armed or
    was armed by anyone, whether it struck with company, and whether its own
    private trace carries an estimate of winning. The first two come from the
    action log; the third is a pattern and is UNVALIDATED, so it is an upper
    bound and is reported as one.

    The control is the agent that would have been struck --- the defender ---
    over the same rounds. Comparing the attacker against nobody would show only
    that agents in these cells arm and talk about odds, which they do.
    """
    uit = {}
    for c in _cells("L2", "L3"):
        paths = runset.cel(c)
        n = 0
        tel = Counter()
        for p in paths:
            grens = combat.first_round(p)
            c0 = combat.first(p)
            if grens is None or not c0:
                continue
            aanvaller = (c0.get("attackers") or [None])[0]
            verdediger = c0.get("defender")
            if not aanvaller:
                continue
            n += 1
            ix = events.index(p)
            tel["struck_with_company"] += len(c0.get("attackers") or []) > 1
            for rol, wie in (("attacker", aanvaller), ("defender", verdediger)):
                if not wie:
                    continue
                bewapend = any(r < grens for r in ix.acted.get((wie, "strengthen"), ()))
                ontvangen = any(r < grens for (a, act, d), rs2 in ix.directed.items()
                                if act == "strengthen" and d == wie for r in rs2)
                tel[f"{rol}_armed_someone"] += bewapend
                tel[f"{rol}_was_armed"] += ontvangen
                schatting = False
                for e in logs.rounds(p):
                    if (e.get("round") or 0) >= grens:
                        break
                    a = (e.get("agents") or {}).get(wie) or {}
                    blob = f"{a.get('thinking') or ''} {a.get('memory') or ''}"
                    if WIN_ESTIMATE.search(blob):
                        schatting = True
                        break
                tel[f"{rol}_has_a_win_estimate"] += schatting
        if not n:
            continue
        pct = lambda k: round(100 * tel[k] / n, 1)
        uit[c] = Result(
            value={"first_fights": n,
                   "struck_with_company_pct": pct("struck_with_company"),
                   "attacker_armed_someone_pct": pct("attacker_armed_someone"),
                   "attacker_was_armed_pct": pct("attacker_was_armed"),
                   "attacker_has_a_win_estimate_pct": pct("attacker_has_a_win_estimate")},
            n=len(paths), denominator=n, unit="first fights",
            baseline={"defender_armed_someone_pct": pct("defender_armed_someone"),
                      "defender_was_armed_pct": pct("defender_was_armed"),
                      "defender_has_a_win_estimate_pct": pct("defender_has_a_win_estimate"),
                      "how": "the agent that was struck, over the same rounds"},
            note="the win-estimate pattern is UNVALIDATED; both columns are "
                 "upper bounds and the comparison between them is what is used").as_dict()
    return uit


FIGURES["m:attacker-thinking"] = what_the_first_attacker_had


# --- m:commons-round-two ---------------------------------------------------

COLLAPSE_ARITHMETIC = re.compile(
    r"\b(?:30|thirty|all|everyone|each)\b[^.!?]{0,60}"
    r"\b(?:harvest\w*|take\w*|extract\w*)\b[^.!?]{0,60}"
    r"\b(?:collapse\w*|deplet\w*|exhaust\w*|empty|wipe|destroy\w*|zero|nothing left|"
    r"unsustainab\w*|crash\w*)\b"
    r"|\b(?:stock|commons|pool)\b[^.!?]{0,50}\b(?:collapse\w*|deplet\w*|exhaust\w*|"
    r"crash\w*|wiped)\b[^.!?]{0,50}\b(?:if|when|unless)\b", re.I)


def collapse_arithmetic() -> dict:
    """Agents who state, in round 2, that everyone harvesting empties the stock.

    Read from the private traces, so it measures what an agent worked out rather
    than what it was told. Round 2 is the first round in which the stock has
    visibly fallen, so it is the earliest the arithmetic can be done from
    evidence rather than from the prompt.

    A share near a hundred per cent means little on its own --- three measures in
    this chapter turned out to be true of everybody --- so two controls run
    beside it. The placebo is the same pattern at L2 and L3, where there is no
    commons and the arithmetic has nothing to describe; a detector that fires
    there is matching vocabulary. The prompt control is the same pattern in
    round 1, before any stock movement is visible.
    """
    uit = {}
    doel = [f"prod_L4_{p}" for p in PAYOFFS] + ["prod_L4_knife_nocomm"] \
        + [f"robust_qwen_L4_knife"] + ["prod_L2_knife", "prod_L3_knife"]
    for c in doel:
        paths = runset.cel(c)
        r1 = r2 = totaal = 0
        for p in paths:
            for e in logs.rounds(p):
                r = e.get("round") or 0
                if r > 2:
                    break
                for nm, a in (e.get("agents") or {}).items():
                    blob = f"{a.get('thinking') or ''} {a.get('memory') or ''}"
                    raak = bool(COLLAPSE_ARITHMETIC.search(blob))
                    if r == 1:
                        r1 += raak
                    elif r == 2:
                        r2 += raak
                        totaal += 1
        if not totaal:
            continue
        uit[c] = Result(
            value=round(100 * r2 / totaal, 1), n=len(paths), denominator=totaal,
            unit="agents in round 2",
            baseline={"round_one_pct": round(100 * r1 / totaal, 1),
                      "how": "the same pattern before any stock movement is visible"},
            note="UNVALIDATED pattern; the L2 and L3 cells are the placebo, "
                 "where there is no commons for the arithmetic to describe").as_dict()
    return uit


FIGURES["m:commons-round-two"] = collapse_arithmetic


# --- m:overlapping-pact-memberships ----------------------------------------

def overlapping_pacts() -> dict:
    """Whether an attack proposal names someone inside a listener's pact.

    Two agents count as pact partners once one has addressed non-aggression
    language to the other, in a message with the other among its recipients.
    That is a directed act made specific by the address, which is what keeps
    this from collapsing into "everyone talked about peace" --- and everyone did,
    so a membership graph built from mere mention would connect the whole board.

    An attack proposal is a public sentence carrying attack vocabulary and an
    agent's name. The lexicon is UNVALIDATED and matches regardless of polarity,
    so "let us not attack Ash" counts.

    That polarity blindness was once defended with the argument that a false
    positive is as likely to fall in one category as another. It is not: a
    negated-attack sentence *is* pact language, and pact language is addressed
    to pact partners,
    so the false positives pile into the speaker's-own-partner bin --- the very
    category the chapter's claim rests on. The argument was wrong in a direction
    that flattered the finding.

    So the split is computed twice. `polarity_blind` is the original reading.
    `negations_removed` drops any sentence that also matches the non-aggression
    lexicon, which is a blunt filter --- it will discard a genuine proposal
    phrased as "they agreed not to attack, so let us strike Ash" --- and it errs
    against the claim, which is the direction a sensitivity check should err in.
    The chapter should quote whichever it quotes with the reading named.

    Each proposal is placed once: the named target is a pact partner of the
    speaker, or an outsider to the speaker but a partner of somebody the message
    was addressed to, or outside everyone's pact.
    """
    uit = {}
    for c in _cells("L2", "L3"):
        paths = runset.cel(c)
        tel, tel_pos = Counter(), Counter()
        for p in paths:
            partners: dict[str, set] = {}
            voorstellen = []
            namen = set()
            for e in logs.rounds(p):
                namen |= set((e.get("agents") or {}).keys())
            for e in logs.rounds(p):
                for m in (e.get("messages") or []):
                    spreker, tekst = m.get("from"), m.get("text") or ""
                    horen = list(m.get("to") or [])
                    if LEX.NAP.search(tekst):
                        for h in horen:
                            partners.setdefault(spreker, set()).add(h)
                            partners.setdefault(h, set()).add(spreker)
                    for s in text.sentences(tekst):
                        if not LEX.ATTACK_TALK.search(s):
                            continue
                        ontkennend = bool(LEX.NAP.search(s))
                        for nm in namen:
                            if nm != spreker and re.search(rf"\b{re.escape(nm)}\b", s):
                                voorstellen.append((spreker, horen, nm, ontkennend))
            for spreker, horen, doel, ontkennend in voorstellen:
                if doel in partners.get(spreker, ()):
                    sleutel = "target_is_the_speakers_own_partner"
                elif any(doel in partners.get(h, ()) for h in horen):
                    sleutel = "outsider_to_speaker_partner_to_a_listener"
                else:
                    sleutel = "outside_everyones_pact"
                tel["proposals"] += 1
                tel[sleutel] += 1
                if not ontkennend:
                    tel_pos["proposals"] += 1
                    tel_pos[sleutel] += 1
        n, n_pos = tel["proposals"], tel_pos["proposals"]
        if not n:
            continue
        SLEUTELS = ("target_is_the_speakers_own_partner",
                    "outsider_to_speaker_partner_to_a_listener",
                    "outside_everyones_pact")
        uit[c] = Result(
            value={k: round(100 * tel[k] / n, 1) for k in SLEUTELS},
            n=len(paths), denominator=n, unit="attack proposals",
            sensitivity={"counts": dict(tel),
                         "negations_removed": (
                             {k: round(100 * tel_pos[k] / n_pos, 1) for k in SLEUTELS}
                             | {"proposals": n_pos,
                                "share_of_proposals_that_were_negated_pct":
                                    round(100 * (n - n_pos) / n, 1)}
                             if n_pos else None)},
            note="attack lexicon UNVALIDATED and polarity-blind, so the headline "
                 "split is an upper bound on the own-partner category in "
                 "particular: a negated-attack sentence is pact language and pact "
                 "language is addressed to partners. `negations_removed` in the "
                 "sensitivity block is the reading with those sentences dropped").as_dict()
    return uit


FIGURES["m:overlapping-pact-memberships"] = overlapping_pacts


# --- m:rewiring-network ----------------------------------------------------

def rewiring_network() -> dict:
    """What rewiring does to the graph at L3, beyond how often a tie is cut.

    The chapter reads five quantities off this and the figure had been reduced
    to a drop count, so four of them rested on nothing. They are: how many
    invitations are issued and what becomes of them, how the mean degree moves,
    how wide the degree spread grows, who gets invited, and whether holding the
    hub position pays.

    The last is the one worth having a number for. A network measure that only
    reported growth would say the graph densifies, which is visible; the useful
    question is whether the agent at the centre of it ends richer, and that has
    to be a correlation across runs rather than a description of one.
    """
    uit = {}
    for c in _cells("L3"):
        paths = runset.cel(c)
        tel = Counter()
        graad_start, graad_eind, spreiding, max_graad = [], [], [], []
        rijk_relatief = []
        hub_vs_bezit = ([], [])
        for p in paths:
            rs = logs.rounds(p)
            def graden(e):
                b = Counter()
                for a1, b1 in ((e.get("network") or {}).get("edges") or []):
                    b[a1] += 1
                    b[b1] += 1
                for nm in (e.get("agents") or {}):
                    b.setdefault(nm, 0)
                return b
            g0, g1 = graden(rs[0]), graden(rs[-1])
            graad_start.append(sum(g0.values()) / len(g0))
            graad_eind.append(sum(g1.values()) / len(g1))
            m = sum(g1.values()) / len(g1)
            spreiding.append((sum((x - m) ** 2 for x in g1.values()) / len(g1)) ** 0.5)
            max_graad.append(max(g1.values()))
            for e in rs:
                ag = e.get("agents") or {}
                levend = [(a.get("resources") or 0) for a in ag.values()
                          if (a.get("resources") or 0) > 0]
                gem = sum(levend) / len(levend) if levend else 0
                gr = graden(e)
                gem_gr = sum(gr.values()) / len(gr) if gr else 0
                for nm, a in ag.items():
                    ri = a.get("rewire_intent") or {}
                    doel = ri.get("invite")
                    tel[ri.get("invite_outcome") or "none"] += doel is not None
                    if doel and doel in ag and gem:
                        rijk_relatief.append((ag[doel].get("resources") or 0) / gem)
            halverwege = graden(rs[len(rs) // 2])
            eind = runstat.final(p)
            for nm in eind:
                hub_vs_bezit[0].append(halverwege.get(nm, 0))
                hub_vs_bezit[1].append(eind[nm])
        uit[c] = Result(
            value={"invitations": sum(tel.values()),
                   "outcomes": dict(tel),
                   "mean_degree_start": _summary(graad_start, 1),
                   "mean_degree_end": _summary(graad_eind, 1),
                   "degree_spread_end": _summary(spreiding, 1),
                   "highest_degree": _summary(max_graad, 1),
                   "invitee_wealth_over_living_mean": _summary(rijk_relatief)},
            n=len(paths), denominator=sum(tel.values()), unit="invitations",
            sensitivity={"degree_at_midpoint_vs_final_holdings":
                         model.correlate(*hub_vs_bezit).value},
            note="degree counted from the network edge list each round").as_dict()
    return uit


FIGURES["m:rewiring-network"] = rewiring_network


# --- when the violence happens, and where the ceiling sits in that sequence ---

def _pct(paar):
    """(hits on the top three, hits) as a percentage, or None if never tested."""
    t, n = paar
    return {"pct": round(100 * t / n, 1), "hits": t, "of": n} if n else None


def violence_timing() -> dict:
    """The shape of a run's fighting over time, and the ceiling's place in it.

    The chapter reads L3 as a sequence --- first blow, peak, stated ceiling, last
    blow --- and three of those four had no measure behind them. This supplies
    them from the combat log, per run, summarised over the cell.

    The peak is taken over a five-round moving sum rather than the single busiest
    round. A run with two fights in round 11 and two in round 12 has no peak at
    all on the raw count, and the question the chapter asks --- when does the
    violence crest --- is about a stretch of rounds, not a spike.

    `rounds_after_the_peak` needs both a peak and a stated ceiling, so it is
    defined on fewer runs than the columns beside it; `defined_in` says how many.
    A negative value means the ceiling was stated before the fighting crested.
    """
    VENSTER = 5
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        eerste, piek, laatste, lag, plafondronde = [], [], [], [], []
        voor, na = [0, 0], [0, 0]
        betrokken, alleen_doelwit = [], []
        for p in paths:
            per_ronde = Counter(r for r, _ in combat.fights(p))
            if not per_ronde:
                continue
            eerste.append(min(per_ronde))
            laatste.append(max(per_ronde))
            # Who was in the violence at all, on either side. "Everyone takes
            # part" is a claim about the population, not about the fights, and
            # it needs the two roles counted separately to mean anything.
            slaat, geraakt = set(), set()
            for _, f in combat.fights(p):
                slaat |= set(f.get("attackers") or [])
                if f.get("defender"):
                    geraakt.add(f["defender"])
            betrokken.append(len(slaat | geraakt))
            alleen_doelwit.append(len(geraakt - slaat))
            hoogste = max(per_ronde)
            glad = {r: sum(per_ronde.get(r + d, 0) for d in range(VENSTER))
                    for r in range(1, hoogste + 1)}
            pr = max(glad, key=lambda r: (glad[r], -r)) + VENSTER // 2
            piek.append(pr)

            # the same first-statement round the ceiling figure reports
            eerst_gezegd = None
            for _, r, _, s in text.public([p]):
                if LEX.CEILING.search(s):
                    eerst_gezegd = r
                    break
            if eerst_gezegd is not None:
                plafondronde.append(eerst_gezegd)
                lag.append(eerst_gezegd - pr)
                # Was the targeting already aimed at the top before anyone said
                # so? Ranked on the board going into the round, as in
                # `whom_they_hit`, so the blow itself does not set the rank.
                rs = logs.rounds(p)
                bord = {e.get("round"): e for e in rs}
                for e in rs:
                    r = e.get("round") or 0
                    ag = (bord.get(r - 1) or {}).get("agents") or e.get("agents") or {}
                    levend = sorted(((nm, a.get("resources") or 0) for nm, a in ag.items()
                                     if (a.get("resources") or 0) > 0),
                                    key=lambda kv: -kv[1])
                    top = {nm for nm, _ in levend[:3]}
                    if not top:
                        continue
                    emmer = voor if r < eerst_gezegd else na
                    for x in (e.get("combat") or []):
                        d = isinstance(x, dict) and x.get("defender")
                        if not d:
                            continue
                        emmer[1] += 1
                        emmer[0] += d in top

        uit[c] = Result(
            value={"first_fight_round": _summary(eerste, 1),
                   "peak_round": _summary(piek, 1),
                   "last_fight_round": _summary(laatste, 1),
                   "ceiling_stated_round": _summary(plafondronde, 1),
                   "rounds_after_the_peak": _summary(lag, 1),
                   "top_three_share_before_the_ceiling_pct": _pct(voor),
                   "top_three_share_after_the_ceiling_pct": _pct(na),
                   "agents_in_a_fight_on_either_side": _summary(betrokken, 1),
                   "agents_only_ever_a_target": _summary(alleen_doelwit, 1)},
            n=len(eerste), denominator=len(paths), unit="runs with a fight",
            sensitivity={"peak_window_rounds": VENSTER},
            note="runs with no fighting are absent from every column, not "
                 "counted as zero; the ceiling columns carry the unvalidated "
                 "CEILING lexicon and are upper bounds").as_dict()
    return uit


FIGURES["m:violence-timing"] = violence_timing


# --- m:mutual-aid-after-defeat ---------------------------------------------

def support_after_defeat() -> dict:
    """Whether losing a fight draws help, against whether anyone draws help.

    The chapter's claim is that the mutual aid of L2 disappears once rewiring
    exists. It was carried by a rate --- support actions per round --- which
    cannot say whether a defeated agent is helped more than an undefeated one,
    and in a cell with a great deal of giving that is the whole question.

    A defeat is a resolved fight the defender lost. Support is a `transfer` or a
    `strengthen` naming that agent, in the three rounds after. The baseline
    re-runs each defeat against a living agent drawn at random from the same
    round of the same run, which holds the amount of giving fixed and destroys
    only the link to the defeat. Ten draws per defeat.
    """
    import random
    VENSTER = 3
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        geholpen = nederlagen = 0
        bl_geholpen = bl_n = 0
        for p in paths:
            rs = logs.rounds(p)
            per_ronde = {e.get("round"): e for e in rs}
            # who received support, per round
            steun: dict[int, set] = {}
            levend: dict[int, list] = {}
            for e in rs:
                r = e.get("round")
                ag = e.get("agents") or {}
                levend[r] = [nm for nm, a in ag.items() if (a.get("resources") or 0) > 0]
                for a in ag.values():
                    if a.get("action") in ("transfer", "strengthen") and a.get("target"):
                        steun.setdefault(r, set()).add(a["target"])

            def gesteund(wie, vanaf):
                return any(wie in steun.get(vanaf + d, set()) for d in range(VENSTER))

            rng = model.seeded(p.name)
            for r, f in combat.fights(p):
                d = f.get("defender")
                if not d or f.get("winner") != "coalition":
                    continue
                nederlagen += 1
                geholpen += gesteund(d, r + 1)
                pool = [x for x in levend.get(r, []) if x != d]
                for _ in range(10):
                    if not pool:
                        break
                    bl_n += 1
                    bl_geholpen += gesteund(rng.choice(pool), r + 1)
        if not nederlagen:
            continue
        uit[c] = Result(
            value={"defeats_followed_by_support_pct": round(100 * geholpen / nederlagen, 1),
                   "defeats": nederlagen},
            n=len(paths), denominator=nederlagen, unit="defeats",
            baseline={"pct": round(100 * bl_geholpen / bl_n, 1) if bl_n else None,
                      "draws": bl_n,
                      "how": "a living agent drawn at random from the same round "
                             "of the same run, ten draws per defeat"},
            sensitivity={"window_rounds": VENSTER},
            note="a defeat is a fight the defender lost; support is a transfer "
                 "or a strengthen naming that agent within the window").as_dict()
    return uit


FIGURES["m:mutual-aid-after-defeat"] = support_after_defeat


# --- m:welfare-arrangements-redistribute -----------------------------------

def welfare_arrangements() -> dict:
    """Whether the arrangements that redistribute actually reach anyone.

    Three quantities the chapter states about L2, each against a baseline that
    holds the volume of giving fixed and destroys only the direction being
    claimed:

    - an agent who falls below a quarter of the living mean, and whether a
      transfer reaches it within three rounds;
    - what agents who lost a fight hold at round 60, against those who did not;
    - late transfers that run downhill --- giver richer than receiver --- in the
      last twenty rounds.

    The label these claims carried pointed at a model-arm figure in Section 4.4
    with no bearing on any of them.
    """
    import random
    VENSTER = 3
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        arm = arm_gered = 0
        bl_arm = bl_gered = 0
        verliezers, rest = [], []
        laat_af = laat_n = 0
        bl_af = bl_laat_n = 0
        for p in paths:
            rs = logs.rounds(p)
            rng = model.seeded(p.name)
            ontvangt: dict[int, set] = {}
            levend: dict[int, list] = {}
            for e in rs:
                r = e.get("round")
                ag = e.get("agents") or {}
                levend[r] = [nm for nm, a in ag.items() if (a.get("resources") or 0) > 0]
                for a in ag.values():
                    if a.get("action") == "transfer" and a.get("target"):
                        ontvangt.setdefault(r, set()).add(a["target"])

            def bereikt(wie, vanaf):
                return any(wie in ontvangt.get(vanaf + d, set()) for d in range(VENSTER))

            for e in rs:
                r = e.get("round")
                ag = e.get("agents") or {}
                lv = [(nm, a.get("resources") or 0) for nm, a in ag.items()
                      if (a.get("resources") or 0) > 0]
                if not lv:
                    continue
                m = sum(v for _, v in lv) / len(lv)
                for nm, v in lv:
                    if v < 0.25 * m:
                        arm += 1
                        arm_gered += bereikt(nm, r + 1)
                        bl_arm += 1
                        bl_gered += bereikt(rng.choice([x for x, _ in lv]), r + 1)

            # who lost a fight at any point
            gevallen = {f.get("defender") for _, f in combat.fights(p)
                        if f.get("winner") == "coalition" and f.get("defender")}
            eind = runstat.final(p)
            for nm, v in eind.items():
                (verliezers if nm in gevallen else rest).append(v)

            # late transfers, and whether they run downhill
            for e in rs:
                r = e.get("round") or 0
                if r <= 40:
                    continue
                ag = e.get("agents") or {}
                for nm, a in ag.items():
                    t = a.get("target")
                    if a.get("action") != "transfer" or not t or t not in ag:
                        continue
                    laat_n += 1
                    laat_af += (a.get("resources") or 0) > (ag[t].get("resources") or 0)
                    pool = [x for x in ag if x != nm]
                    if pool:
                        bl_laat_n += 1
                        bl_af += ((a.get("resources") or 0)
                                  > (ag[rng.choice(pool)].get("resources") or 0))

        if not arm and not laat_n:
            continue
        uit[c] = Result(
            value={"poor_reached_within_three_rounds_pct":
                       round(100 * arm_gered / arm, 1) if arm else None,
                   "agent_rounds_below_a_quarter_of_the_mean": arm,
                   "round_60_holding_of_those_who_lost_a_fight": _summary(verliezers, 1),
                   "round_60_holding_of_the_rest": _summary(rest, 1),
                   "late_transfers_running_downhill_pct":
                       round(100 * laat_af / laat_n, 1) if laat_n else None,
                   "late_transfers": laat_n},
            n=len(paths), denominator=arm or laat_n, unit="agent-rounds below a quarter",
            baseline={"poor_reached_pct": round(100 * bl_gered / bl_arm, 1) if bl_arm else None,
                      "late_downhill_pct": round(100 * bl_af / bl_laat_n, 1) if bl_laat_n else None,
                      "how": "a living agent drawn at random from the same round "
                             "of the same run, in place of the agent the claim names"},
            sensitivity={"window_rounds": VENSTER, "late_rounds": "41-60",
                         "poverty_line": "a quarter of the living mean"},
            note="the round-60 split counts every agent, including those "
                 "eliminated, at their final holding").as_dict()
    return uit


FIGURES["m:welfare-arrangements-redistribute"] = welfare_arrangements


# --- m:l4-attack-language --------------------------------------------------

def l4_peace_and_stock() -> dict:
    """How completely the fighting stops at L4, and what the stock does.

    Three things the chapter states together. Whether any fight occurs; how much
    of the talk is about attacking, against the same lexicon at L2 and L3 so the
    fall has something to be a fall from; and where the shared stock bottoms out
    and whether it comes back.

    The stock low is taken per run and summarised across the cell. A cell mean
    computed per round instead would average a run that bottomed in round 3 with
    one that bottomed in round 6 and report neither.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        zonder, dieptes, rondes, terug, met_voorraad = 0, [], [], 0, 0
        aanvalszinnen = zinnen = 0
        for p in paths:
            if not combat.fights(p):
                zonder += 1
            for _, _, _, s in text.public([p]):
                zinnen += 1
                aanvalszinnen += bool(LEX.ATTACK_TALK.search(s))
            # `stock_before` is the level going into the round, before that
            # round's harvest. The level *after* it is what the agents see when
            # they next decide, but the low point of the resource itself is the
            # pre-harvest series, and that is what the chapter describes.
            reeks = [(e.get("round"), (e.get("commons") or {}).get("stock_before"))
                     for e in logs.rounds(p)
                     if (e.get("commons") or {}).get("stock_before") is not None]
            if not reeks:
                continue
            met_voorraad += 1
            r, v = min(reeks, key=lambda rv: rv[1])
            rondes.append(r)
            dieptes.append(v)
            terug += reeks[-1][1] >= 119.5
        uit[c] = Result(
            value={"runs_with_no_fight": zonder, "runs": len(paths),
                   "attack_language_pct_of_sentences": round(100 * aanvalszinnen / zinnen, 3)
                       if zinnen else None,
                   "stock_low_round": _summary(rondes, 1),
                   "stock_low_level": _summary(dieptes, 1),
                   "runs_back_at_capacity_by_round_60": terug,
                   "runs_with_a_stock": met_voorraad},
            n=len(paths), denominator=zinnen, unit="sentences",
            note="the stock columns are empty outside L4, where there is no "
                 "commons; ATTACK_TALK is unvalidated and every share from it "
                 "is an upper bound").as_dict()
    return uit


FIGURES["m:l4-attack-language"] = l4_peace_and_stock


# --- m:membership-list -----------------------------------------------------

def name_knowledge() -> dict:
    """When an agent could know all thirty names, against when the rota is proposed.

    A rota over thirty names needs the names, and nobody is given them: an agent
    starts wired to about five neighbours and learns the rest from traffic. The
    claim the chapter makes is that the proposals arrive before the knowledge
    does, which needs both rounds measured on the same runs.

    An agent knows a name once it has appeared in a message that agent received
    --- as the sender, as a co-recipient, or written in the body. The round
    reported is the median over the thirty agents of the first round in which
    that set covers the whole population; agents who never get there are
    reported in `agents_never_reaching_thirty` rather than dropped.

    The proposal round is the first sentence matching the unvalidated ROTA
    pattern, so it is an upper bound on how late a proposal can be and still be
    counted --- which is the conservative direction for this comparison.
    """
    uit = {}
    for c in [f"prod_L4_{p}" for p in PAYOFFS]:
        paths = runset.cel(c)
        volledig, voorstel, nooit = [], [], 0
        for p in paths:
            rs = logs.rounds(p)
            allen = set((rs[0].get("agents") or {}).keys())
            bekend = {nm: {nm} for nm in allen}
            for nm, a in (rs[0].get("agents") or {}).items():
                pass
            bereikt: dict[str, int] = {}
            for e in rs:
                r = e.get("round")
                for m in (e.get("messages") or []):
                    afz = m.get("from")
                    ont = m.get("to") or []
                    tekst = str(m.get("text") or m.get("message") or "")
                    genoemd = {x for x in allen if x in tekst}
                    for wie in ont:
                        if wie not in bekend:
                            continue
                        bekend[wie] |= {afz} | set(ont) | genoemd
                for nm in allen:
                    if nm not in bereikt and bekend[nm] >= allen:
                        bereikt[nm] = r
            nooit += len(allen) - len(bereikt)
            if bereikt:
                volledig.append(_median(sorted(bereikt.values())))
            for _, r, _, s in text.public([p]):
                if LEX.ROTA.search(s):
                    voorstel.append(r)
                    break
        uit[c] = Result(
            value={"round_an_agent_knows_all_thirty": _summary(volledig, 1),
                   "round_the_rota_is_first_proposed": _summary(voorstel, 1),
                   "agents_never_reaching_thirty": nooit,
                   "agents": 30 * len(paths)},
            n=len(paths), denominator=len(paths), unit="runs",
            note="per run the median over its agents; the proposal round uses "
                 "the unvalidated ROTA pattern and is an upper bound").as_dict()
    return uit


FIGURES["m:membership-list"] = name_knowledge


# --- m:inequality-trajectory -----------------------------------------------

def inequality_trajectory() -> dict:
    """What the economy and the distribution do over a run, not just where they end.

    Four things the chapter states about L3 and none of which the final-state
    figures can answer: how long the run is quiet after the last fight, what the
    economy is worth at the end against its start and against its round-10 size,
    and whether the fall in inequality survives being computed over the living
    only.

    That last one is the point of the measure. A Gini over all thirty agents
    falls when the poorest are eliminated and their zeros leave the series,
    which is a fall in the wrong sense. Computing it over agents still holding
    something asks whether the survivors converged, and only that reading
    supports the claim.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        staart, tov_start, tov_r10 = [], [], []
        piek_g, eind_g = [], []
        for p in paths:
            rs = logs.rounds(p)
            tot = {e.get("round"): sum(a.get("resources") or 0.0
                                       for a in (e.get("agents") or {}).values())
                   for e in rs}
            if tot.get(1):
                tov_start.append(100 * tot[max(tot)] / tot[1])
            if tot.get(10):
                tov_r10.append(100 * tot[max(tot)] / tot[10])
            laatste = combat.last_round(p)
            staart.append(max(tot) - laatste if laatste else max(tot))

            # Gini over the living, round by round; the peak of that series
            # against its final value.
            reeks = []
            for e in rs:
                levend = [a.get("resources") or 0.0
                          for a in (e.get("agents") or {}).values()
                          if (a.get("resources") or 0) > 0]
                reeks.append(runstat.gini(levend) if levend else 0.0)
            if reeks:
                piek_g.append(max(reeks))
                eind_g.append(reeks[-1])
        uit[c] = Result(
            value={"rounds_after_the_last_fight": _summary(staart, 1),
                   "round_60_economy_as_pct_of_round_1": _summary(tov_start, 1),
                   "round_60_economy_as_pct_of_round_10": _summary(tov_r10, 1),
                   "peak_gini_among_the_living": _summary(piek_g, 3),
                   "round_60_gini_among_the_living": _summary(eind_g, 3)},
            n=len(paths), denominator=len(paths), unit="runs",
            note="the Gini here counts only agents still holding something, so "
                 "a fall cannot be produced by eliminations leaving zeros out "
                 "of the series; a run with no fight has its whole length as "
                 "its tail").as_dict()
    return uit


FIGURES["m:inequality-trajectory"] = inequality_trajectory


# --- m:naming-a-target ------------------------------------------------------

def naming_a_target() -> dict:
    """Coined terms that name an agent, which are not arrangements but calls.

    The coined-name detector drops any phrase containing an agent's name, on the
    ground that "Bronze and Cyan" is not an institution. Inverting that filter
    returns what it discards, and the discards are not noise: they are shared
    vocabulary reaching three or more speakers, of the form TAKE ONYX, Copper
    NOW, INVITE Dusk NOW. A collective coining those is not naming an
    arrangement; it is naming a person to act on.

    Which rungs produce them is the finding. The capacity to choose whom to
    reach is what makes a target a matter for the whole board rather than for
    whoever happens to be adjacent, and it is the rung where these appear.

    The same detector, the same threshold of three speakers, the same run set as
    the coined-name count it is the complement of: whatever inflates one deflates
    the other, so the two are read together.
    """
    uit = {}
    for c in _cells("L2", "L3", "L4"):
        paths = runset.cel(c)
        per_run, voorbeelden = [], []
        for p in paths:
            d = text.named_agreements(p, naming_agents=True)
            per_run.append(len(d))
            for naam, sprekers in sorted(d.items(), key=lambda kv: -kv[1]):
                if len(voorbeelden) < 4:
                    voorbeelden.append(f"{naam} ({sprekers} speakers)")
        uit[c] = Result(
            value={"terms_naming_an_agent": sum(per_run),
                   "per_run": round(sum(per_run) / len(paths), 2)},
            n=len(paths), denominator=sum(per_run), unit="coined terms",
            sensitivity={"min_users": text.MIN_USERS, "per_run_counts": per_run,
                         "examples": voorbeelden},
            note="the complement of the coined-name count: the phrases that "
                 "detector discards because they name an agent. Three speakers "
                 "minimum, so a single agent shouting a name does not count").as_dict()
    return uit


FIGURES["m:naming-a-target"] = naming_a_target


# --- m:how-long-a-name-lives ------------------------------------------------

def how_long_a_name_lives() -> dict:
    """How long a coined term stays in use, and how many circulate at once.

    The coined-name count says how many terms a cell produced and nothing about
    what became of them. Two cells can coin the same number and differ entirely:
    one settling on a single pact that runs to the last round, the other cycling
    through a dozen that each last a few rounds and are replaced.

    A term's life is measured from the round it is first said to the round it is
    last said, inclusive. That over-counts a term revived after a long silence,
    which is why the count of distinct rounds in which it is actually said is
    reported beside it --- where the two diverge, the term went quiet and came
    back rather than staying in use.

    `runs_with_one_term` is reported because a cell's mean is a poor guide to
    what its runs look like: six of the fifteen L2 knife-edge runs coin exactly
    one term, and the cell's median run is a single pact held all game.
    """
    uit = {}
    for c in _cells("L2", "L3"):
        paths = runset.cel(c)
        levens, gezegd, per_run, alleen_een = [], [], [], 0
        samengevoegd, starts, gestopt, loopt_door = [], [], [], 0
        for p in paths:
            d = text.named_agreements(p, with_rounds=True)
            per_run.append(len(d))
            alleen_een += len(d) == 1
            for naam, v in d.items():
                if len(v.get("variants", [naam])) > 1:
                    samengevoegd.append(" | ".join(v["variants"]))
                if v["rounds"]:
                    duur = max(v["rounds"]) - min(v["rounds"]) + 1
                    levens.append(duur)
                    gezegd.append(len(set(v["rounds"])))
                    starts.append(min(v["rounds"]))
                    if max(v["rounds"]) >= logs.HORIZON:
                        loopt_door += 1
                    else:
                        gestopt.append(duur)
        uit[c] = Result(
            value={"rounds_from_first_to_last": _summary(levens),
                   "rounds_it_is_actually_said": _summary(gezegd),
                   "round_first_said": _summary(starts),
                   # A name coined in round 58 cannot last more than three
                   # rounds whatever the collective does with it, so the span
                   # above is censored at the end of the game and the censoring
                   # is not even across the cells. These two separate it: how
                   # many names the game cut off, and how long the rest lasted.
                   "still_in_use_at_the_end": loopt_door,
                   "rounds_for_names_that_stopped": _summary(gestopt),
                   "terms_per_run": _summary(per_run),
                   "runs_with_one_term": alleen_een},
            n=len(paths), denominator=len(levens), unit="coined terms",
            sensitivity={"min_users": text.MIN_USERS, "terms_per_run": per_run,
                         "same_term": text.SAME_TERM,
                         "wordings_merged": sorted(samengevoegd)},
            note="a term coined and dropped in the same round counts as one "
                 "round, not zero; every group of wordings folded into one term "
                 "is listed, so the merge can be read rather than trusted").as_dict()
    return uit


FIGURES["m:how-long-a-name-lives"] = how_long_a_name_lives
