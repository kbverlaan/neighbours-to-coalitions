"""Section 4.1, The grid --- every reported figure, one function each.

Each function returns the numbers behind one claim in the text, and nothing
else. The counting lives in `core`; what is here is the definition of the claim:
which cells, which unit, which free parameters. If a function in this file
contains a loop over rounds, it belongs in `core` instead.

The key on each function is the `\\meth{}` label it answers in Chapter 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HIER.parent / _m))

import re                                                          # noqa: E402
import events, graph, lexicons as LEX, logs, model, runstat, text, turns  # noqa: E402
from result import Result                                           # noqa: E402
import runset                                                       # noqa: E402

L1 = ["prod_L1_scar", "prod_L1_knife", "prod_L1_abund"]
L2 = ["prod_L2_scar", "prod_L2_knife", "prod_L2_abund"]
L3 = ["prod_L3_scar", "prod_L3_knife", "prod_L3_abund"]
PAYOFFS = ("scar", "knife", "abund")
RUNGS = ("L1", "L2", "L3", "L4")


_summary = runstat.summary
_median = runstat.median


def _cells(namen):
    return {c: runset.cel(c) for c in namen}


# --- m:variance-within-between --------------------------------------------

def inequality_by_cell() -> dict:
    """Final Gini per cell, and whether the three payoffs order at each rung.

    The claim is an ordering, so the test is the ordering and not the means:
    two cells whose means differ by less than a tenth of a standard deviation
    are reported as not separating.
    """
    uit, ordening = {}, {}
    for r in RUNGS:
        rij = {}
        for p in PAYOFFS:
            res = runstat.summarise(runset.cel(f"prod_{r}_{p}"), runstat.final_gini)
            uit[f"prod_{r}_{p}"] = res.as_dict()
            rij[p] = (res.value, res.sensitivity["sd"])
        m = [rij[p][0] for p in PAYOFFS]
        gescheiden = []
        for i in range(2):
            verschil = m[i + 1] - m[i]
            pooled = (rij[PAYOFFS[i]][1] + rij[PAYOFFS[i + 1]][1]) / 2
            gescheiden.append(bool(verschil > 0.1 * pooled))
        ordening[r] = {"means": m, "monotone": m == sorted(m),
                       "separates_scar_knife": gescheiden[0],
                       "separates_knife_abund": gescheiden[1]}
    return {"per_cell": uit, "ordering": ordening}


# --- m:l1-message-collapse -------------------------------------------------

def l1_message_collapse() -> dict:
    """Transfer share, message variety and the single most repeated text, at L1.

    No agent is eliminated at this rung, which is checked rather than assumed:
    if any run ends below thirty living agents the turn denominator is not
    15 x 30 x 60 and the reported figure would be wrong.
    """
    uit = {}
    for c in L1:
        paths = runset.cel(c)
        levend = runstat.per_run(paths, runstat.alive)
        acties = turns.action_profile(paths)
        herhaling = text.repetition(paths)
        gini = runstat.summarise(paths, runstat.final_gini)
        uit[c] = {"transfer": acties["transfer"].as_dict(),
                  "hold": acties["hold"].as_dict(),
                  "repetition": herhaling.as_dict(),
                  "gini": gini.as_dict(),
                  "alive_min": min(levend), "alive_max": max(levend),
                  "turns": acties["transfer"].denominator}
    return uit


# --- m:scarce-transfers ----------------------------------------------------

# Vocabulary an agent uses when it states, in its own private trace, why it
# gave. Used only to check the behavioural classifier against what the giver
# said, never to classify. The three sets are deliberately narrow: a broad set
# would agree with everything and prove nothing.
STATED_REASON = {
    "repays_arming": re.compile(
        r"\b(repay\w*|repaid|owed?|debt|in return|as agreed|obligation|"
        r"promised|for (?:the )?(?:arming|strengthening|support he|support she))\b", re.I),
    "recent_attack": re.compile(
        r"\b(spoils|share of the|loot|proceeds|raid|strike|after the attack|"
        r"from the take)\b", re.I),
    "reciprocal": re.compile(
        r"\b(mutual\w*|reciproc\w*|swap|in exchange|back to me|each other|"
        r"keep it going|continuing the)\b", re.I),
}


def _classifier_agreement(cell: str, per_run: dict) -> dict:
    """How often the giver's own private trace states the reason we inferred.

    The classifier reads behaviour only: it never sees what an agent wrote. This
    checks it against an independent signal --- the giver's memory and thinking
    in the round it gave --- and reports, per assigned category, how often the
    text carries vocabulary for that category and how often it carries
    vocabulary for a different one instead.

    It is an agreement rate and not a precision: the stated reason can be absent
    without the inference being wrong, and an agent can say one thing and mean
    another. `silent` is the share where the trace says nothing either way, and
    it is large --- which is the honest limit of this check.
    """
    from collections import Counter
    regels = [("repays_arming", events.did_before("strengthen")),
              ("recent_attack", events.did_within("take", 5)),
              ("reciprocal", events.did_ever("transfer"))]
    tel: dict[str, Counter] = {k: Counter() for k in STATED_REASON}
    for p in runset.cel(cell):
        ix = events.index(p)
        tekst = {}
        for e in logs.rounds(p):
            for nm, a in (e.get("agents") or {}).items():
                tekst[(e.get("round"), nm)] = (
                    f"{a.get('thinking') or ''} {a.get('memory') or ''}")
        for r, actor, other, act in ix.events:
            if act != "transfer" or not other:
                continue
            raak = next((naam for naam, rule in regels if rule(ix, r, actor, other)), None)
            if raak is None:
                continue
            blob = tekst.get((r, actor), "")
            eigen = bool(STATED_REASON[raak].search(blob))
            ander = any(k != raak and pat.search(blob) for k, pat in STATED_REASON.items())
            tel[raak]["n"] += 1
            tel[raak]["agrees" if eigen else ("names_another" if ander else "silent")] += 1
    return {k: {"n": v["n"],
                "agrees_pct": round(100 * v["agrees"] / v["n"], 1) if v["n"] else None,
                "names_another_pct": round(100 * v["names_another"] / v["n"], 1) if v["n"] else None,
                "silent_pct": round(100 * v["silent"] / v["n"], 1) if v["n"] else None}
            for k, v in tel.items()}


def scarce_transfers() -> dict:
    """What a transfer is for at L2, from what the recipient did.

    Rules are ordered and exclusive, so reciprocity --- tested last --- is a
    lower bound and the two earlier categories are upper bounds. Both the
    pooled split and the per-run median are returned: pooling weights a run by
    how many transfers it made, and in the scarce cell one run supplies a
    quarter of them.
    """
    regels = [("repays_arming", events.did_before("strengthen")),
              ("recent_attack", events.did_within("take", 5)),
              ("reciprocal", events.did_ever("transfer"))]
    uit = {}
    for c in L2:
        res, _ = events.classify(runset.cel(c), "transfer", regels)
        # The narrower reading of the attack rule, as the sensitivity arm.
        smal = [("repays_arming", events.did_before("strengthen")),
                ("recent_attack_on_giver", events.did_within("take", 5, targeted=True)),
                ("reciprocal", events.did_ever("transfer"))]
        alt, _ = events.classify(runset.cel(c), "transfer", smal)
        res.sensitivity["attack_on_giver_only"] = alt.value
        res.sensitivity["agreement_with_stated_reason"] = _classifier_agreement(c, _)
        res.sensitivity["arms"] = {
            naam: events.classify(ps, "transfer", regels)[0].value
            for naam, ps in model.arms(c).items() if naam != "all"}
        uit[c] = res.as_dict()
    return uit


# --- m:pairing-capacity-level ---------------------------------------------

def pairing_and_floor() -> dict:
    """Mutual pairs at L1, and the poorest agent at L1 against L2.

    Ties are read from the settled flow rather than the declared action: a
    transfer that was announced but yielded nothing leaves no flow, and this
    figure is about who ended up connected. Mutuality is the free parameter and
    the one-directional reading is reported alongside.
    """
    uit = {"groups": {}, "floor": {}}
    for c in L1:
        paths = runset.cel(c)
        res = graph.group_profile(paths, flows=True, mutual=True)
        los = graph.group_profile(paths, flows=True, mutual=False)
        res.sensitivity["if_one_direction_suffices"] = los.value
        uit["groups"][c] = res.as_dict()
    for c in L1 + L2:
        res = runstat.summarise(runset.cel(c), runstat.floor, digits=2)
        res.baseline = round(runstat.DO_NOTHING, 2)
        res.note = ("mean over runs of each run's lowest round-60 holding; "
                    "the baseline is an agent that never acted")
        uit["floor"][c] = res.as_dict()
    return uit


# --- m:giving-more-make ----------------------------------------------------

def giving_and_equality() -> dict:
    """Whether giving more leaves a collective less unequal, at L1.

    Pooled across all three cells and then within each giving cell, because the
    scarce cell has no variance in transfer and drives the pooled sign on its
    own.
    """
    reeks = {}
    for c in L1:
        paths = runset.cel(c)
        aandeel = [turns.share([p], turns.is_action("transfer")).value for p in paths]
        gini = runstat.per_run(paths, runstat.final_gini)
        reeks[c] = (aandeel, gini)
    alle_x = [x for c in L1 for x in reeks[c][0]]
    alle_y = [y for c in L1 for y in reeks[c][1]]
    uit = {"pooled": model.correlate(alle_x, alle_y).as_dict()}
    for c in L1[1:]:
        uit[c] = model.correlate(*reeks[c]).as_dict()
    uit["note"] = ("the scarce cell has zero transfer in every run and no "
                   "variance, so it cannot carry a within-cell correlation")

    # Whether the association is behavioural at all. A transfer share is bounded
    # above, so a run whose mean sits near the ceiling forces its agents to
    # resemble each other, and it is that resemblance rather than the volume that
    # flattens the distribution. Where the dispersion predicts the Gini as well
    # as the mean does, and the mean predicts the dispersion, the finding is
    # largely arithmetic.
    for c in L1[1:]:
        paths = runset.cel(c)
        spreiding = []
        for p in paths:
            v = _per_agent_transfer(p)
            m = sum(v) / len(v)
            spreiding.append((sum((x - m) ** 2 for x in v) / len(v)) ** 0.5)
        gem, gini = reeks[c]
        uit[c + "_ceiling_check"] = {
            "mean_vs_gini": model.correlate(gem, gini).value,
            "dispersion_vs_gini": model.correlate(spreiding, gini).value,
            "mean_vs_dispersion": model.correlate(gem, spreiding).value,
            "note": "a bounded proportion near its ceiling couples its mean to "
                    "its dispersion; where all three are large the association "
                    "is arithmetic rather than behavioural"}
    return uit


def _per_agent_transfer(p) -> list[float]:
    """Share of its own acting turns each agent spent transferring."""
    from collections import Counter
    tel, tot = Counter(), Counter()
    for e in logs.rounds(p):
        for nm, a in (e.get("agents") or {}).items():
            if a.get("action"):
                tot[nm] += 1
                if a["action"] == "transfer":
                    tel[nm] += 1
    return [100 * tel[n] / tot[n] for n in tot]


# --- m:support-stays-inside ------------------------------------------------

MIN_LINK = 2   # interactions that make a tie; the free parameter of the camp


def _camp_thresholds(paths) -> dict:
    """What the split becomes at one and at three interactions per tie.

    The threshold decides how much of the population has a camp at all, and the
    unassigned share is the quantity the finding rests on, so the finding has to
    be shown to survive the choice.
    """
    from collections import Counter
    uit = {}
    for mc in (1, 2, 3):
        t: Counter = Counter()
        dekking = []
        for p in paths:
            g, _ = graph.build(p, actions=("transfer", "strengthen"),
                               rounds_in=(1, 20), min_count=mc)
            kamp = {m: i for i, gg in enumerate(g) for m in gg}
            dekking.append(len(kamp))
            for r, a, b in graph.interactions(p, actions=("transfer", "strengthen"),
                                              rounds_in=(41, 60)):
                t["total"] += 1
                if a not in kamp or b not in kamp:
                    t["unassigned"] += 1
                elif kamp[a] == kamp[b]:
                    t["inside"] += 1
                else:
                    t["between"] += 1
        toe = t["inside"] + t["between"]
        uit[f"min_count={mc}"] = {
            "agents_in_a_camp": round(sum(dekking) / len(dekking), 1),
            "unassigned_pct": round(100 * t["unassigned"] / t["total"], 1) if t["total"] else None,
            "inside_pct": round(100 * t["inside"] / toe, 1) if toe else None}
    return uit


def support_camps() -> dict:
    """Where late support goes, against camps fixed on the opening rounds.

    Camps come from rounds 1-20 and the scoring runs on rounds 21-60, so the
    structure is not defined by the actions being scored. Support involving an
    agent who never joined a camp is a third outcome and not a crossing; the
    reading that folded it into "crossing" is reported as the sensitivity arm
    because it produced the previously published figures.
    """
    from collections import Counter
    uit = {}
    # L3 as well as L2: the chapter compares the size of the block that forms
    # here against the one a level up, and a figure built on one level cannot
    # carry a comparison between two.
    for c in L2 + L3:
        paths = runset.cel(c)
        t: Counter = Counter()
        grootte, kampen = [], []
        for p in paths:
            groepen, leden = graph.build(p, actions=("transfer", "strengthen"),
                                         rounds_in=(1, 20), min_count=MIN_LINK)
            kamp = {m: i for i, g in enumerate(groepen) for m in g}
            grootte.append(max((len(g) for g in groepen), default=0))
            kampen.append(len(groepen))
            for r, a, b in graph.interactions(p, actions=("transfer", "strengthen"),
                                              rounds_in=(21, 60)):
                venster = "R21-40" if r <= 40 else "R41-60"
                t[f"{venster}|total"] += 1
                if a not in kamp or b not in kamp:
                    t[f"{venster}|unassigned"] += 1
                elif kamp[a] == kamp[b]:
                    t[f"{venster}|inside"] += 1
                else:
                    t[f"{venster}|between"] += 1
        vensters = {}
        for w in ("R21-40", "R41-60"):
            tot, ins = t[f"{w}|total"], t[f"{w}|inside"]
            tus, nie = t[f"{w}|between"], t[f"{w}|unassigned"]
            toe = ins + tus
            vensters[w] = {
                "inside_pct": round(100 * ins / toe, 1) if toe else None,
                "between_pct": round(100 * tus / toe, 1) if toe else None,
                "of_all_inside": round(100 * ins / tot, 1) if tot else None,
                "of_all_between": round(100 * tus / tot, 1) if tot else None,
                "of_all_unassigned": round(100 * nie / tot, 1) if tot else None,
                "support": tot, "assigned": toe}
        uit[c] = Result(
            value=vensters, n=len(paths), denominator=t["R41-60|total"],
            unit="support actions",
            sensitivity={"thresholds": _camp_thresholds(paths),
                         "largest_camp_mean": round(sum(grootte) / len(grootte), 1),
                         "camps_mean": round(sum(kampen) / len(kampen), 1),
                         "published_reading": "of_all_inside, which counts an "
                                              "unassigned pair as a crossing"},
            note="camps fixed on R1-20, scored on R21-60").as_dict()
    return uit


# --- severings ------------------------------------------------------------
# This count was once registered as m:rewiring-network, but the chapter reads
# five quantities from that measure and this one supplies a single drop count.
# The key belongs to capacities.rewiring_network(); severings() stays as a helper.

def severings() -> dict:
    """How often a tie is cut at L3, in absolute terms and as a share of turns."""
    uit = {}
    for c in L3:
        paths = runset.cel(c)
        res = turns.count(paths, lambda a: bool((a.get("rewire_intent") or {}).get("drop")))
        res.note = f"{res.value} drops in {res.denominator} agent-turns"
        uit[c] = res.as_dict()
    uit["total"] = sum(uit[c]["value"] for c in L3)
    return uit


# --- m:payoff-owns-capacity ------------------------------------------------

OUTCOMES = {
    "coalition_size": runstat.coalition_size,
    "attacks": lambda p: turns.share([p], turns.is_action("take")).value,
    "consensus_spread": runstat.consensus_spread,
    "wealth": runstat.economy_growth,
    "holding": lambda p: turns.share([p], turns.is_action("hold")).value,
    "transfer": lambda p: turns.share([p], turns.is_action("transfer")).value,
}

# Where an outcome has a rival operationalisation, it is computed too, so the
# reported split can be shown not to depend on the choice.
RIVALS = {
    "coalition_size": ("from_resolved_combat", runstat.coalition_size_resolved),
    "wealth": ("summed_holdings_at_R60", runstat.total_wealth),
}


def lever_split() -> dict:
    """How much of each outcome the payoff cell owns and how much the rung owns.

    The interaction is returned with the two main effects because an additive
    split is only a fair description of a design whose levers do not interact.
    """
    uit = {}
    for naam, scalar in OUTCOMES.items():
        cellen = {}
        for r in RUNGS:
            for p in PAYOFFS:
                cellen[(r, p)] = [scalar(x) for x in runset.cel(f"prod_{r}_{p}")]
        res = model.anova2(cellen)
        res.sensitivity["A_is"] = "capacity level"
        res.sensitivity["B_is"] = "payoff cell"
        # L1 has no `take` in its action schema, so five of the twelve cells
        # carry zero variance on the conflict outcomes and the capacity effect
        # there is partly definitional rather than behavioural. The same split
        # over L2-L4 only, where every action exists in every cell, separates
        # the two.
        res.sensitivity["excluding_L1"] = model.anova2(
            {k: v for k, v in cellen.items() if k[0] != "L1"}).value
        res.sensitivity["zero_variance_cells"] = [
            f"{a}_{b}" for (a, b), v in cellen.items()
            if len(set(round(x, 9) for x in v)) == 1]
        if naam in RIVALS:
            label, rival = RIVALS[naam]
            anders = {k: [rival(x) for x in runset.cel(f"prod_{k[0]}_{k[1]}")]
                      for k in cellen}
            res.sensitivity[label] = model.anova2(anders).value
        uit[naam] = res.as_dict()
    return uit


# --- m:institutions-patronage-welfare --------------------------------------

def institution_naming() -> dict:
    """Coined names per run, at L2 and L3, where the naming happens.

    A name counts when at least three distinct agents use the same capitalised
    multi-word phrase. That threshold is the free parameter; two and four are
    reported alongside, because a lower one admits any turn of phrase and a
    higher one admits only what the whole cell adopted.

    One arrangement written several ways counts once: phrases whose content
    words agree on half or more are one term, so "Core NAP", "Core Local NAP"
    and "Local NAPs" are one pact rather than three. Without that, a collective
    that cannot settle on a wording counts as one that keeps inventing.
    Fragments of shouted sentences --- IS DEPLETED, ONE ELSE --- are dropped for
    the same reason: a coined name is a noun phrase.

    The detector cannot tell an institution from a plan, and it counts a phrase
    coined once and echoed twice the same as one used for forty rounds. How long
    each one lasts is \\meth{m:how-long-a-name-lives}. It is a count of shared
    vocabulary, and the chapter reads it as no more than that.
    """
    uit = {}
    for c in L2 + L3:
        paths = runset.cel(c)
        per = {}
        for drempel in (2, 3, 4):
            aantal = [len(text.named_agreements(p, drempel)) for p in paths]
            per[f"min_users={drempel}"] = {
                "mean": round(sum(aantal) / len(aantal), 2),
                "per_run": sorted(aantal)}
        hoofd = per["min_users=3"]
        uit[c] = Result(value=hoofd["mean"], n=len(paths), denominator=len(paths),
                        unit="runs", sensitivity=per,
                        note="mean number of coined names per run").as_dict()
    return uit


# --- m:commons-capacity-level ----------------------------------------------

def commons_rota() -> dict:
    """Rota language at L4, and whether the harvest groups are alphabetical.

    Two questions in one figure because they answer each other. If the rounds in
    which a subset harvests are alphabetically contiguous blocks, the collective
    has not solved a coordination problem --- it has read an ordering off the
    agent list, which was never hidden from it. The chance expectation for a
    contiguous block of k out of n is (n-k+1)/C(n,k), which is negligible for
    any k above two, so a high share is not a weak result but a different
    finding from the one the rota language suggests.
    """
    from math import comb
    uit = {}
    # The no-channel control belongs here: the chapter claims no rota forms in
    # silence, and a claim about a cell needs that cell measured. Its harvest
    # rounds are scored the same way, so the two are comparable.
    for c in [f"prod_L4_{p}" for p in PAYOFFS] + ["prod_L4_knife_nocomm"]:
        paths = runset.cel(c)
        taal = text.share(paths, text.public, LEX.ROTA)
        blok = kans = rondes = 0
        oogsten = solo_rondes = 0
        for p in paths:
            for e in logs.rounds(p):
                oogst = sorted(nm for nm, a in (e.get("agents") or {}).items()
                               if a.get("action") == "harvest")
                oogsten += len(oogst)
                # A single harvester is a contiguous block by definition, so a
                # round with one is dropped rather than scored. It is a real
                # exclusion and it is counted, not passed over in silence.
                if len(oogst) < 2:
                    solo_rondes += len(oogst) > 0
                    continue
                allen = sorted((e.get("agents") or {}).keys())
                idx = sorted(allen.index(x) for x in oogst)
                rondes += 1
                if idx == list(range(idx[0], idx[0] + len(idx))):
                    blok += 1
                n, k = len(allen), len(oogst)
                kans += (n - k + 1) / comb(n, k) if k <= n else 0.0
        uit[c] = {
            "rota_language": taal.as_dict(),
            "harvests_per_run": Result(
                value=round(oogsten / len(paths), 1), n=len(paths),
                denominator=len(paths), unit="runs",
                note="harvest actions, not harvesting rounds").as_dict(),
            "alphabetical_blocks": Result(
                value=round(100 * blok / rondes, 1) if rondes else None,
                n=len(paths), denominator=rondes, unit="harvest rounds",
                baseline=round(100 * kans / rondes, 6) if rondes else None,
                sensitivity={"rounds_with_one_harvester_excluded": solo_rondes},
                note="share of harvest rounds whose harvesters are a contiguous "
                     "alphabetical block; the baseline is the chance of that "
                     "under a random subset of the same size. Rounds with a "
                     "single harvester are excluded --- one name is a block by "
                     "definition --- and the number excluded is reported "
                     "alongside").as_dict()}
    return uit


# --- m:harvest-rhythm ------------------------------------------------------

def _rhythm(gaps: list[int]) -> tuple[int, float]:
    """The commonest interval between one agent's harvests, and its share."""
    if not gaps:
        return 0, 0.0
    mode = max(set(gaps), key=gaps.count)
    return mode, 100 * gaps.count(mode) / len(gaps)


def _cohorts(path, from_round: int = 20):
    """The turn-taking structure of one run, read off the second half.

    A period alone does not establish a rota: thirty agents each harvesting
    every third round could still be thirty independent habits. What makes it
    one schedule is that the population divides exactly --- `k` harvesters a
    round on a period of `P` with `k * P` equal to the population, every agent
    in one phase and every phase filled.

    Read from round 20 on, because the opening rounds are where the pattern
    forms and including them would count the search as part of the result.
    """
    rounds_ = {}
    for e in logs.rounds(path):
        rounds_[e.get("round")] = sorted(
            n for n, a in (e.get("agents") or {}).items()
            if a.get("action") == "harvest")
    last, gaps = {}, []
    for r in sorted(rounds_):
        for n in rounds_[r]:
            if n in last and r > from_round:
                gaps.append(r - last[n])
            last[n] = r
    if not gaps:
        return None
    period = max(set(gaps), key=gaps.count)
    on_period = 100 * gaps.count(period) / len(gaps)

    phases = {}
    for r in sorted(rounds_):
        if r <= from_round:
            continue
        for n in rounds_[r]:
            phases.setdefault(n, []).append(r % period)
    if not phases:
        return None
    assigned = {n: max(set(v), key=v.count) for n, v in phases.items()}
    names = sorted(assigned)
    n_agents = len(names)
    kept = sum(v.count(assigned[n]) for n, v in phases.items())
    in_phase = 100 * kept / sum(len(v) for v in phases.values())
    harvesters = runstat.median([len(v) for v in rounds_.values() if v]) or 0
    covered = harvesters * period / n_agents if n_agents else 0

    # Whether the phases follow the alphabet, as contiguous blocks of n/P. The
    # best offset is taken, since which block goes first is arbitrary; the
    # chance level is 100/P and is reported with it.
    idx = {x: i for i, x in enumerate(names)}
    size = n_agents / period if period else 1
    best = 0
    for off in range(period):
        best = max(best, sum(1 for x in names
                             if assigned[x] == (int(idx[x] // size) + off) % period))
    alphabetical = 100 * best / n_agents if n_agents else 0

    # The first round from which every agent keeps to one phase for the rest of
    # the game: when the schedule stopped being negotiated.
    settled = None
    for start in range(1, 50):
        seen = {}
        for r in range(start, 61):
            for n in rounds_.get(r, []):
                seen.setdefault(n, set()).add(r % period)
        if seen and all(len(v) == 1 for v in seen.values()):
            settled = start
            break
    # With a period of one there is a single phase, so "in its own slot" and
    # "alphabetical" are 100% by construction and mean nothing. The control cell
    # has a period of one throughout, and reporting those two as findings there
    # would be the trivial number read as a result --- which is the failure this
    # package exists to prevent. They are undefined instead.
    trivial = period < 2
    return {"period": period, "on_period": round(on_period, 1),
            "harvesters": harvesters, "population_covered": round(covered, 2),
            "in_phase": None if trivial else round(in_phase, 1),
            "alphabetical": None if trivial else round(alphabetical, 1),
            "chance_alphabetical": None if trivial else round(100 / period, 1),
            "settled_by_round": None if trivial else settled}


def harvest_rhythm() -> dict:
    """Whether agents take turns at the stock, apart from whose turn it is.

    The alphabetical-block figure asks whether a round's harvesters are
    contiguous in the agent list, and reads as though it measured whether a rota
    formed at all. It does not, and the two come apart: a scarce run scoring no
    alphabetical block harvests ten agents a round on a three-round cycle in
    which every agent keeps its slot. That is a rota. It is not an alphabetical
    one.

    What is measured here is the schedule rather than its ordering. `period` is
    the commonest interval between one agent's own harvests; `population_covered`
    is `harvesters x period / agents`, which is 1.0 exactly when the population
    divides into `period` cohorts with nobody left over; `in_phase` is how often
    an agent harvests in its own slot. `alphabetical` then asks the separate
    question of whether those cohorts are blocks of the agent list, against a
    chance level of 100/period.

    The no-channel control is measured the same way, and it is the comparison
    that matters: the rota is what the channel is for.

    The baseline holds each agent's number of harvests fixed and scatters them
    over the sixty rounds, ten times per run --- what the same agents would
    produce with no turn-taking at all.
    """
    from collections import Counter
    uit = {}
    for c in [f"prod_L4_{p}" for p in PAYOFFS] + ["prod_L4_knife_nocomm"]:
        paths = runset.cel(c)
        rows, bl, stilgevallen = [], [], []
        for p in paths:
            d = _cohorts(p)
            if d:
                rows.append(d)
            else:
                # No two harvests by the same agent after round 20. That is not
                # a missing measurement but a finding --- the run has stopped
                # harvesting --- and dropping it from the denominator without
                # saying so would report the control cell on the two runs that
                # kept going.
                stilgevallen.append(p.name.split("__")[1].split("_")[0])
            counts: Counter = Counter()
            for e in logs.rounds(p):
                for nm, a in (e.get("agents") or {}).items():
                    if a.get("action") == "harvest":
                        counts[nm] += 1
            rng = model.seeded(p.name)
            for _ in range(10):
                gaps = []
                for nm, k in counts.items():
                    rs = sorted(rng.sample(range(1, 61), min(k, 60)))
                    gaps += [b - a for a, b in zip(rs, rs[1:])]
                if gaps:
                    top = max(set(gaps), key=gaps.count)
                    bl.append(100 * gaps.count(top) / len(gaps))
        if not rows:
            continue
        def col(k):
            return runstat.summary([r[k] for r in rows if r[k] is not None])
        uit[c] = Result(
            value={"period": col("period"),
                   "share_of_intervals_at_that_period": col("on_period"),
                   "population_covered": col("population_covered"),
                   "harvests_in_own_slot_pct": col("in_phase"),
                   "cohorts_are_alphabetical_blocks_pct": col("alphabetical"),
                   "settled_by_round": col("settled_by_round"),
                   "runs_with_a_period_above_one": sum(1 for r in rows if r["period"] > 1),
                   "runs_that_stopped_harvesting": len(stilgevallen)},
            n=len(paths), denominator=len(rows), unit="runs",
            baseline=round(sum(bl) / len(bl), 1) if bl else None,
            sensitivity={"per_run": rows, "read_from_round": 20,
                         "runs_with_no_second_harvest_after_round_20": stilgevallen},
            note="a period of one is every agent harvesting every round, which "
                 "is the absence of turn-taking; population_covered is 1.0 when "
                 "the agents divide exactly into that many cohorts. The "
                 "baseline keeps each agent's harvest count and scatters it "
                 "over the sixty rounds").as_dict()
    return uit


FIGURES = {
    "m:harvest-rhythm": harvest_rhythm,
    "m:institutions-patronage-welfare": institution_naming,
    "m:commons-capacity-level": commons_rota,
    "m:variance-within-between": inequality_by_cell,
    "m:l1-message-collapse": l1_message_collapse,
    "m:scarce-transfers": scarce_transfers,
    "m:pairing-capacity-level": pairing_and_floor,
    "m:giving-more-make": giving_and_equality,
    "m:support-stays-inside": support_camps,
    "m:payoff-owns-capacity": lever_split,
}


# --- m:pairing-timing ------------------------------------------------------

def pairing_timing() -> dict:
    """When an agent finds its partner at L1, and what that is worth.

    The pair is the whole economy at this level, so the question of who does
    well collapses into the question of who pairs and when. Agents are grouped
    by the round their first mutual tie completes --- the round the second
    direction of the exchange arrives --- and reported at their round-60
    holding.

    Also here because it answers the obvious objection: whether the starting
    position decides it. Round-1 degree against round-60 holding, per cell.
    Ties are read from settled flows rather than declared actions, as in
    Measure~\\ref{m:pairing-capacity-level}, so an announced transfer that moved
    nothing does not create a partner.
    """
    EMMERS = ((1, 3), (4, 5), (6, 10), (11, 60))
    uit = {}
    for c in L1:
        paths = runset.cel(c)
        emmer: dict[str, list] = {f"rounds {a}-{b}": [] for a, b in EMMERS}
        emmer["never"] = []
        laatste_paar, graad_x, graad_y = [], [], []
        for p in paths:
            gepaard: dict[str, int] = {}
            gezien: dict[tuple, set] = {}
            for r, a, b in graph.interactions(p, flows=True):
                sleutel = tuple(sorted((a, b)))
                gezien.setdefault(sleutel, set()).add((a, b))
                if len(gezien[sleutel]) == 2:
                    for x in sleutel:
                        gepaard.setdefault(x, r)
            eind = runstat.final(p)
            for nm, v in eind.items():
                r = gepaard.get(nm)
                if r is None:
                    emmer["never"].append(v)
                    continue
                for a, b in EMMERS:
                    if a <= r <= b:
                        emmer[f"rounds {a}-{b}"].append(v)
                        break
            if gepaard:
                laatste_paar.append(max(gepaard.values()))
            rs = logs.rounds(p)
            buren: dict[str, int] = {}
            for e in (rs[0].get("network") or {}).get("edges") or []:
                for x in e:
                    buren[x] = buren.get(x, 0) + 1
            for nm, v in eind.items():
                graad_x.append(buren.get(nm, 0))
                graad_y.append(v)
        uit[c] = Result(
            value={k: _summary(v, 1) for k, v in emmer.items()}
                  | {"last_pair_forms_in_round": _summary(laatste_paar, 1)},
            n=len(paths), denominator=sum(len(v) for v in emmer.values()),
            unit="agents",
            sensitivity={"round_1_degree_vs_round_60_holding":
                             model.correlate(graad_x, graad_y).value},
            note="an agent is paired from the round its first mutual flow "
                 "completes; agents who never pair are their own group and are "
                 "not dropped").as_dict()
    return uit


FIGURES["m:pairing-timing"] = pairing_timing


# --- m:registered-predictions ----------------------------------------------

# The registration's decision rule, verbatim from the frozen document:
#   P1  the ordinal order-type label rises monotonically across the ladder
#       (Jonckheere-Terpstra or an ordinal mixed model with RunID), one-sided
#   P2  within a rung the payoff sweep flips the regime, primarily on `mob`
#       and `consensus-std`, alpha .05, effect at or above SESOI d = 1.0
#   P3  persistence against collapse depends on both levers together --- a
#       significant affordance x payoff interaction, alpha .05, effect >= SESOI
#
# `mob` is read here as coalition size: the registration's own DV list glosses
# it as the number of agents attacking together, which is what that scalar is.
# Where a rung has no attack in its schema the outcome is undefined rather than
# zero, and the test is not run --- see the `not_testable` key.

SESOI = 1.0
ALPHA = 0.05


def registered_predictions() -> dict:
    """The registered confirmatory tests, run and reported whatever they say.

    Written after an external audit observed that the chapter names this
    analysis in Section 4.2 and never reports it. Until now every
    "separates", "holds" and "flat" in the chapter was descriptive, which made
    the whole of Section 4 a planned-description exercise with no confirmatory
    layer. That is a fact about the chapter and it belongs in it.

    P1 cannot be run here and is reported as not run rather than as failed. It
    needs the ordinal order-type label, and that label was downgraded during the
    DV revision: no function in this package produces it, so there is nothing to
    apply a trend test to. Reporting it as absent is the honest column.
    """
    uit = {}

    # --- P2: within each rung, does the payoff sweep move mob and consensus? --
    p2 = {}
    for r in RUNGS:
        for naam in ("coalition_size", "consensus_spread"):
            scalar = OUTCOMES[naam]
            groepen = {p: [scalar(x) for x in runset.cel(f"prod_{r}_{p}")]
                       for p in PAYOFFS}
            plat = [x for v in groepen.values() for x in v]
            if len(set(round(x, 9) for x in plat)) == 1:
                p2[f"{r}_{naam}"] = {"not_testable":
                                     "no variance in the rung: the outcome is "
                                     "undefined at this capacity level"}
                continue
            t = model.anova1(groepen)
            t["confirmed"] = bool(t["p"] is not None and t["p"] < ALPHA
                                  and t["largest_abs_d"] is not None
                                  and t["largest_abs_d"] >= SESOI)
            p2[f"{r}_{naam}"] = t

    # --- P3: the interaction, on the same two outcomes ------------------------
    p3 = {}
    for naam in ("consensus_spread", "coalition_size"):
        scalar = OUTCOMES[naam]
        cellen = {(r, p): [scalar(x) for x in runset.cel(f"prod_{r}_{p}")]
                  for r in RUNGS for p in PAYOFFS}
        v = model.anova2(cellen).value
        p3[naam] = {"interaction": v["AxB"], "capacity": v["A"], "payoff": v["B"],
                    "confirmed_on_alpha": bool(v["AxB"]["p"] < ALPHA)}

    uit["P2_payoff_within_rung"] = Result(
        value=p2, n=150, denominator=150, unit="runs",
        sensitivity={"alpha": ALPHA, "SESOI_cohen_d": SESOI,
                     "mob_read_as": "coalition size"},
        note="confirmation requires p < alpha AND a pairwise effect at or above "
             "SESOI, both fixed before the runs; cells with no variance are "
             "reported as not testable rather than as null").as_dict()

    uit["P3_interaction"] = Result(
        value=p3, n=150, denominator=150, unit="runs",
        sensitivity={"alpha": ALPHA, "SESOI_cohen_d": SESOI},
        note="the registration also names 'no commons collapse' as part of the "
             "P3 outcome; no Gemma production run collapses, so that half of "
             "the criterion has no variance to test and the interaction is "
             "reported on the continuous half alone").as_dict()

    uit["P1_ordinal_ladder"] = Result(
        value={"run": False,
               "reason": "the ordinal order-type label the trend test needs is "
                         "not produced by any function in this package; it was "
                         "downgraded during the DV revision and no replacement "
                         "was registered"},
        n=0, denominator=150, unit="runs",
        note="reported as not run. A prediction that cannot be tested is not a "
             "prediction that failed, and the chapter must not present it as "
             "either confirmed or refuted").as_dict()
    return uit


# Deliberately NOT in FIGURES. The pre-registration has been dropped as a frame;
# this function stays because the work is done and its result must remain
# reproducible, but the chapter no longer makes a confirmatory argument.


# --- m:serving-stack-split --------------------------------------------------

def serving_stack_split() -> dict:
    """Whether the OpenRouter runs differ from the cluster runs on each outcome.

    Thirty of the 150 production runs were served through OpenRouter to finish
    cells the cluster allocation could not, on the same model and configuration.
    The appendix has always said so, and has always said that three figures
    differ between the arms by more than the within-cell variation --- without
    naming them. This names them.

    Reported per mixed cell and per outcome as a Cohen's d between the arms,
    with the flag set where |d| >= 1.0, the same SESOI the registration fixes.
    That threshold is not a test: with four to eight runs a side these
    comparisons are underpowered and every one of them is post hoc and
    uncorrected. What the flag marks is where a reader should not treat the cell
    as one population, which is a different and weaker claim than a difference
    having been established.

    Cells with runs on only one stack are absent rather than reported as null.
    """
    UIT = dict(OUTCOMES, final_gini=runstat.final_gini)
    uit = {}
    for r in RUNGS:
        for pay in PAYOFFS:
            c = f"prod_{r}_{pay}"
            armen = runset.per_arm(c)
            if len(armen) < 2 or min(len(v) for v in armen.values()) < 2:
                continue
            kolommen = {}
            for naam, scalar in UIT.items():
                waarden = {a: [scalar(x) for x in paths] for a, paths in armen.items()}
                d = model.cohen_d(waarden.get("openrouter", []),
                                  waarden.get("cluster", []))
                kolommen[naam] = {
                    "means": {a: round(sum(v) / len(v), 3) for a, v in waarden.items()},
                    "cohen_d_openrouter_minus_cluster": None if d != d else round(d, 2),
                    "beyond_within_cell_spread": bool(d == d and abs(d) >= 1.0)}
            uit[c] = Result(
                value=kolommen,
                n=sum(len(v) for v in armen.values()),
                denominator=len(runset.cel(c)), unit="runs",
                sensitivity={"runs_per_arm": {a: len(v) for a, v in armen.items()},
                             "flag_threshold_cohen_d": 1.0},
                note="post hoc and uncorrected; underpowered at these cell sizes. "
                     "A flag marks where the cell should not be read as one "
                     "population, not a difference that has been established").as_dict()
    return uit


# Deliberately NOT in FIGURES: this is a check on the run set, not a figure in
# the chapter.
