"""Section 4.5, How the paths diverge --- what the design does not account for.

Everything here is exploratory and none of it was registered. That is stated in
the chapter and it changes what these figures are for: they describe the spread
that survives every manipulation, and any test among them is one of many.

The correction is carried in the figures rather than left to the prose.
`screen()` reports the size of its own family --- 120 tests on the current run
set, where an earlier draft said 48 --- and returns the Bonferroni threshold
alongside each p, so that a reader can see which survive without doing the
arithmetic and without taking the family size on trust.
"""
from __future__ import annotations

import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HIER.parent / _m))

import combat, graph, lexicons as LEX, logs, model, runstat, text, turns  # noqa: E402
from result import Result                                                # noqa: E402
import runset                                                            # noqa: E402

PAYOFFS = ("scar", "knife", "abund")
RUNGS = ("L1", "L2", "L3", "L4")
PRODUCTION = [f"prod_{r}_{p}" for r in RUNGS for p in PAYOFFS]


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    h = len(xs) // 2
    return xs[h] if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2


# --- m:path-divergence-run -------------------------------------------------

SPREAD = {
    "fights": combat.count,
    "final_gini": runstat.final_gini,
    "last_attack_round": combat.last_round,
    "first_attack_round": combat.first_round,
    "transfer_pct": lambda p: turns.share([p], turns.is_action("transfer")).value,
    "wealth": runstat.total_wealth,
}


def divergence_within_cells() -> dict:
    """How far apart runs sharing every parameter end up.

    Reported per cell as the range and the median, not as a standard deviation:
    the claim in the chapter is that identical settings produce different
    societies, and a range is what that claim is about. Runs in which a quantity
    is undefined --- a last attack in a run with no fights --- are counted
    separately rather than folded in as zero.
    """
    uit = {}
    for c in PRODUCTION:
        paths = runset.cel(c)
        per = {}
        for naam, scalar in SPREAD.items():
            xs = [scalar(p) for p in paths]
            geldig = [x for x in xs if x is not None]
            per[naam] = {"min": round(min(geldig), 3) if geldig else None,
                         "max": round(max(geldig), 3) if geldig else None,
                         "median": round(_median(geldig), 3) if geldig else None,
                         "defined_in": len(geldig), "runs": len(xs)}
        uit[c] = Result(value=per, n=len(paths), denominator=len(paths),
                        unit="runs", note="range across runs sharing every "
                                          "parameter").as_dict()
    return uit


def design_accounts_for() -> dict:
    """The share of each outcome's total spread the two levers account for.

    The complement is what remains between runs with identical settings, which
    is what this section is about. Reported over the twelve comms-on cells.
    """
    uit = {}
    for naam, scalar in SPREAD.items():
        cellen = {}
        bruikbaar = True
        for r in RUNGS:
            for p in PAYOFFS:
                xs = [scalar(x) for x in runset.cel(f"prod_{r}_{p}")]
                if any(x is None for x in xs):
                    bruikbaar = False
                cellen[(r, p)] = [x for x in xs if x is not None]
        if not bruikbaar:
            uit[naam] = {"skipped": "undefined in at least one run; a variance "
                                    "decomposition over a quantity that does not "
                                    "exist everywhere is not interpretable"}
            continue
        res = model.anova2(cellen)
        uit[naam] = {"design_share": res.value["model_r2"],
                     "between_identical_runs": res.value["residual_share"],
                     "n": res.n}
    return uit


# --- m:opening-predicts-run ------------------------------------------------

PREDICTORS = {
    "first_attack_round": combat.first_round,
    "attack_talk_before_first_blow": lambda p: _talk_before_first(p),
    "early_transfer_pct": lambda p: _early_share(p, "transfer"),
    "early_gini": lambda p: _gini_at(p, 10),
}

OUTCOMES = {
    "fights": combat.count,
    "final_gini": runstat.final_gini,
    "wealth": runstat.total_wealth,
    "transfer_pct": lambda p: turns.share([p], turns.is_action("transfer")).value,
}


def _talk_before_first(p) -> float | None:
    """Attack vocabulary, as a share of sentences spoken before the first fight.

    Undefined for a run with no fights: there is no "before" to measure. The
    lexicon is broad and matches the verb regardless of polarity, so "let us not
    attack" counts; it is an upper bound and the chapter says so.
    """
    grens = combat.first_round(p)
    if grens is None:
        return None
    raak = tot = 0
    for e in logs.rounds(p):
        if (e.get("round") or 0) >= grens:
            break
        for m in (e.get("messages") or []):
            for s in text.sentences(m.get("text") or ""):
                tot += 1
                if LEX.ATTACK_TALK.search(s):
                    raak += 1
    return 100 * raak / tot if tot else None


def _early_share(p, actie: str, tot_ronde: int = 10) -> float:
    raak = tot = 0
    for e in logs.rounds(p):
        if (e.get("round") or 0) > tot_ronde:
            break
        for a in (e.get("agents") or {}).values():
            if a.get("action"):
                tot += 1
                raak += a["action"] == actie
    return 100 * raak / tot if tot else 0.0


def _gini_at(p, ronde: int) -> float | None:
    s = combat.state_at(p, ronde)
    return runstat.gini(s.values()) if s else None


def screen() -> dict:
    """Every opening quantity against every outcome, per cell, with the correction.

    The family size is returned rather than asserted, which is the point of
    running the whole table: the denominator of the correction is the length of
    this list and not a number someone remembered. It has been 48 in an earlier
    draft and is 120 now, and only the returned value is ever right.
    """
    rijen = []
    reeksen = {}
    for c in PRODUCTION:
        paths = runset.cel(c)
        # Each series once per cell. The inner loop used to recompute every
        # outcome for every predictor --- twenty passes over a cell's runs where
        # eight are needed --- which made this figure take ten minutes and
        # looked like a hang rather than like arithmetic.
        # Memoised on disk: these are scalars over immutable logs, and computing
        # them costs a pass over 24 GB. Without it this figure did not finish on
        # this machine. See core/runstat.py::cached_scalar.
        pred = {pn: [runstat.cached_scalar(pn, p, pf) for p in paths]
                for pn, pf in PREDICTORS.items()}
        uitk = {on: [runstat.cached_scalar(on, p, of) for p in paths]
                for on, of in OUTCOMES.items()}
        runstat.memo_flush()
        reeksen[c] = pred
        for pn, xs in pred.items():
            for on, ys in uitk.items():
                paren = [(a, b) for a, b in zip(xs, ys)
                         if a is not None and b is not None]
                if len(paren) < 3:
                    continue
                r = model.correlate([a for a, _ in paren], [b for _, b in paren])
                if r.value is None:
                    continue
                rijen.append({"cell": c, "predictor": pn, "outcome": on,
                              "r": r.value, "p": r.sensitivity["p"], "n": r.n})
    # A pair that shares a term with its own outcome is not a finding. The early
    # transfer share predicting the full-run transfer share is one quantity
    # measured twice, and predicting wealth is the mechanism by which wealth is
    # made at these capacity levels. Marked rather than dropped, so the family
    # size the correction rests on stays the family that was actually run.
    def _deelt_een_term(x) -> bool:
        return bool(
            x["predictor"].startswith("early_") and (
                x["predictor"].replace("early_", "") in x["outcome"]
                or (x["predictor"] == "early_transfer_pct" and x["outcome"] == "wealth")
                or (x["predictor"] == "early_gini" and x["outcome"] == "final_gini")))

    for x in rijen:
        x["shares_a_term"] = _deelt_een_term(x)
    # How far the predictors are from being each other. `early_gini` looked
    # like an independent survivor at L4 until this was computed: inequality can
    # only appear early there if transfers have happened, so the two are close to
    # one variable and a survivor on one is a survivor on the other.
    collineair = {}
    for c, pred in reeksen.items():
        a, b = pred.get("early_gini") or [], pred.get("early_transfer_pct") or []
        paar = [(u, v) for u, v in zip(a, b) if u is not None and v is not None]
        if len(paar) >= 3:
            r = model.correlate([u for u, _ in paar], [v for _, v in paar])
            collineair[c] = r.value

    # Sharing a term is only half of it. At L4 `early_gini` shares no term with
    # `transfer_pct`, yet inequality can only appear there once transfers have
    # happened, so the two predictors are close to one variable and a survivor
    # on one is a survivor on the other. A row therefore also counts as
    # definitional when its predictor is collinear, in that cell, with a
    # predictor that does share a term with the same outcome.
    COLLINEAIR = 0.7
    for x in rijen:
        via = False
        if not x["shares_a_term"] and x["predictor"] == "early_gini":
            r = collineair.get(x["cell"])
            partner = dict(x, predictor="early_transfer_pct")
            via = (r is not None and abs(r) >= COLLINEAIR
                   and _deelt_een_term(partner))
        x["collinear_with_a_definitional_predictor"] = via
        x["definitional"] = x["shares_a_term"] or via

    k = len(rijen)
    drempel = 0.05 / k if k else None
    overleeft = [x for x in rijen if x["p"] is not None and x["p"] < drempel]

    def _tel(grens):
        n = 0
        for x in overleeft:
            if x["shares_a_term"]:
                continue
            r = collineair.get(x["cell"])
            if x["predictor"] == "early_gini" and r is not None and abs(r) >= grens \
               and _deelt_een_term(dict(x, predictor="early_transfer_pct")):
                continue
            n += 1
        return n

    return {"tests": k,
            "survives_and_is_not_definitional":
                [x for x in overleeft if not x["definitional"]],
            "collinearity_threshold": COLLINEAIR,
            "survivors_left_at_other_thresholds":
                {"0.6": _tel(0.6), "0.7": _tel(0.7), "0.8": _tel(0.8),
                 "1.0 (collinearity ignored)": _tel(1.01)},
            "bonferroni_threshold": round(drempel, 6) if drempel else None,
            "survives_correction": overleeft,
            "strongest_uncorrected": sorted(
                rijen, key=lambda x: abs(x["r"]), reverse=True)[:10],
            "predictor_collinearity_early_gini_vs_early_transfer": collineair,
            "all": rijen}


# --- m:state-blow-falls ----------------------------------------------------

def state_when_the_blow_falls() -> dict:
    """What the board looks like in the round of the first fight.

    Restricted to cells where fighting occurs. The comparison that gives the
    claim its force is the same cells at round 60, computed here rather than
    quoted from another figure so the two cannot drift apart.
    """
    uit = {}
    for c in PRODUCTION:
        paths = [p for p in runset.cel(c) if combat.first_round(p) is not None]
        if not paths:
            continue
        rondes, ginis, aanval, verdedig = [], [], [], []
        solo, ratio_solo, ratio_coal, winst_solo = [], [], [], []
        eerste_solo, eerste_ratio, eerste_winst = [], [], []
        for p in paths:
            r = combat.first_round(p)
            rondes.append(r)
            st = combat.state_before(p, r)  # going in, not after the blow
            if st:
                ginis.append(runstat.gini(st.values()))
                m = sum(st.values()) / len(st)
                c0 = combat.first(p)
                att = (c0.get("attackers") or [None])[0]
                dfd = c0.get("defender")
                if m and att in st:
                    aanval.append(st[att] / m)
                if m and dfd in st:
                    verdedig.append(st[dfd] / m)
            solo.append(combat.solo_share(p))
            # The opening blow only. The chapter attributes these three to "the
            # blow itself", and an earlier version of this function computed them
            # over every fight in the run, which is a different question with a
            # different answer: pooled over all fights the solo share at L2 is
            # around three quarters, and over first blows alone it is nearer a
            # half. Both are kept, and the one the sentence needs is named.
            c0 = combat.first(p)
            if c0 and c0.get("defender_power"):
                n0 = len(c0.get("attackers") or [])
                eerste_solo.append(n0 == 1)
                eerste_ratio.append((c0.get("coalition_power") or 0) / c0["defender_power"])
                eerste_winst.append(c0.get("winner") == "coalition")
            for _, cc in combat.fights(p):
                d = cc.get("defender_power")
                if not d:
                    continue
                n_att = len(cc.get("attackers") or [])
                (ratio_solo if n_att == 1 else ratio_coal).append(
                    (cc.get("coalition_power") or 0) / d)
                if n_att == 1:
                    winst_solo.append(cc.get("winner") == "coalition")
        uit[c] = Result(
            value={"first_fight_round": round(sum(rondes) / len(rondes), 2),
                   "gini_at_that_round": round(sum(ginis) / len(ginis), 3) if ginis else None,
                   "gini_at_round_60": round(sum(runstat.per_run(paths, runstat.final_gini)) / len(paths), 3),
                   "attacker_relative_to_mean": round(sum(aanval) / len(aanval), 2) if aanval else None,
                   "defender_relative_to_mean": round(sum(verdedig) / len(verdedig), 2) if verdedig else None,
                   "first_blow_solo_pct": round(100 * sum(eerste_solo) / len(eerste_solo), 1) if eerste_solo else None,
                   "first_blow_strength_ratio": round(sum(eerste_ratio) / len(eerste_ratio), 2) if eerste_ratio else None,
                   "first_blow_attacker_wins_pct": round(100 * sum(eerste_winst) / len(eerste_winst), 1) if eerste_winst else None,
                   "solo_share_of_all_fights": round(sum(solo) / len(solo), 1) if solo else None,
                   "solo_strength_ratio": round(sum(ratio_solo) / len(ratio_solo), 2) if ratio_solo else None,
                   "coalition_strength_ratio": round(sum(ratio_coal) / len(ratio_coal), 2) if ratio_coal else None,
                   "solo_win_rate": round(100 * sum(winst_solo) / len(winst_solo), 1) if winst_solo else None},
            n=len(paths), denominator=len(runset.cel(c)), unit="runs with a fight",
            note="cells with no fighting are absent, not zero").as_dict()
    return uit


def opening_defeat() -> dict:
    """Whether the run turns on the opening blow being lost.

    A count and not a test: five runs sit in the losing group in the cell the
    chapter discusses, which is too few for one. Reported as the two groups with
    every run's fight count listed, so a reader can see the whole basis.
    """
    uit = {}
    for c in PRODUCTION:
        verloren, gewonnen = [], []
        for p in runset.cel(c):
            uitk = combat.first_attacker_lost(p)
            if uitk is None:
                continue
            (verloren if uitk else gewonnen).append(combat.count(p))
        if not (verloren and gewonnen):
            continue
        uit[c] = Result(
            value={"opening_lost": {"runs": len(verloren), "fights": sorted(verloren),
                                    "min": min(verloren)},
                   "opening_won": {"runs": len(gewonnen), "fights": sorted(gewonnen),
                                   "min": min(gewonnen)}},
            n=len(verloren) + len(gewonnen), denominator=len(runset.cel(c)),
            unit="runs with a fight",
            note="an observation about these runs, not a test").as_dict()
    return uit


# --- m:paired-blow ---------------------------------------------------------

def paired_first_blow() -> dict:
    """Whether a shared seed fixes which agent takes the first blow.

    The Qwen arm re-uses production seeds, so the same seed can be followed
    across the two model arms. If the seed determined the target, the pairs
    would agree.
    """
    uit = {}
    for niveau in ("L2", "L3", "L4"):
        gemma = {r["seed"]: runset.WORTEL / f"prod_{niveau}_knife" / r["bestand"]
                 for r in runset.rijen() if r["cel"] == f"prod_{niveau}_knife"}
        qwen = {r["seed"]: runset.WORTEL / f"robust_qwen_{niveau}_knife" / r["bestand"]
                for r in runset.rijen() if r["cel"] == f"robust_qwen_{niveau}_knife"}
        paren = []
        for seed in sorted(set(gemma) & set(qwen)):
            a, b = combat.defender_of_first(gemma[seed]), combat.defender_of_first(qwen[seed])
            paren.append({"seed": seed, "gemma": a, "qwen": b,
                          "same": (a is not None and a == b)})
        gedeeld = [x for x in paren if x["gemma"] and x["qwen"]]
        uit[niveau] = Result(
            value={"pairs_with_a_fight_on_both_sides": len(gedeeld),
                   "same_target": sum(1 for x in gedeeld if x["same"]),
                   "seeds_matched": len(paren)},
            n=len(paren), denominator=len(gedeeld) or None, unit="seed pairs",
            sensitivity={"pairs": paren},
            note="a pair counts only where both arms produced a first fight").as_dict()
    return uit


# --- m:harvest-groups-form -------------------------------------------------

def harvest_group_formation() -> dict:
    """How the L4 harvest settles, round by round.

    The alphabetical share is given against its chance expectation in every
    round, because a contiguous block of any size above two is vanishingly
    unlikely by accident and the raw share alone invites the wrong reading.
    """
    from math import comb
    uit = {}
    for c in [f"prod_L4_{p}" for p in PAYOFFS] + ["prod_L4_knife_nocomm"]:
        paths = runset.cel(c)
        per_ronde = {}
        for p in paths:
            for e in logs.rounds(p):
                r = e.get("round")
                ag = e.get("agents") or {}
                oogst = sorted(nm for nm, a in ag.items() if a.get("action") == "harvest")
                allen = sorted(ag)
                stock = (e.get("commons") or {}).get("stock_before")
                d = per_ronde.setdefault(r, {"harvesters": [], "stock": [],
                                             "block": 0, "scored": 0, "chance": 0.0})
                d["harvesters"].append(len(oogst))
                if stock is not None:
                    d["stock"].append(float(stock))
                if len(oogst) >= 2:
                    idx = sorted(allen.index(x) for x in oogst)
                    d["scored"] += 1
                    d["block"] += idx == list(range(idx[0], idx[0] + len(idx)))
                    n, k = len(allen), len(oogst)
                    d["chance"] += (n - k + 1) / comb(n, k) if k <= n else 0.0
        rijen = {}
        for r in sorted(per_ronde)[:14]:
            d = per_ronde[r]
            rijen[r] = {"mean_harvesters": round(sum(d["harvesters"]) / len(d["harvesters"]), 1),
                        "mean_stock": round(sum(d["stock"]) / len(d["stock"]), 1) if d["stock"] else None,
                        "alphabetical_pct": round(100 * d["block"] / d["scored"], 1) if d["scored"] else None,
                        "chance_pct": round(100 * d["chance"] / d["scored"], 4) if d["scored"] else None}
        uit[c] = Result(value=rijen, n=len(paths), denominator=len(paths),
                        unit="runs", note="first fourteen rounds").as_dict()
    return uit


FIGURES = {
    "m:path-divergence-run": divergence_within_cells,
    "m:opening-predicts-run": screen,
    "m:state-blow-falls": state_when_the_blow_falls,
    "m:opening-defeat": opening_defeat,
    "m:paired-blow": paired_first_blow,
    "m:harvest-groups-form": harvest_group_formation,
    "m:design-accounts-for": design_accounts_for,
}
