"""Section 4.4, Swap the model --- Gemma against Qwen, and a one-run probe.

Five runs per level, one payoff cell, one second model. That is an existence
proof that a finding can depend on the player and carries no estimate of how far
the dependence goes, which the chapter states and these figures do not try to
improve on.

The arms share their seeds, so a difference is not a difference in starting
position. `seed_overlap()` checks that rather than assuming it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HIER.parent / _m))

from collections import Counter                                          # noqa: E402
import combat, graph, handlabels, logs, model, runstat, text, turns     # noqa: E402
from result import Result                                                # noqa: E402
import runstat  # noqa: E402
import runset                                                            # noqa: E402

LEVELS = ("L2", "L3", "L4")
ARMS = {"gemma": "prod_{}_knife",
        "qwen": "robust_qwen_{}_knife",
        "deepseek": "robust_deepseek_{}_knife"}


def _cells(arm: str):
    return {lvl: runset.cel(ARMS[arm].format(lvl)) for lvl in LEVELS}


# --- m:model-swap-arm ------------------------------------------------------

def seed_overlap() -> dict:
    """Which seeds the arms actually share, per level.

    The claim that the swap holds the starting position fixed rests on this, so
    it is computed rather than asserted. A seed present in one arm and not the
    other means the comparison at that level is between different draws.
    """
    uit = {}
    for lvl in LEVELS:
        per = {}
        for arm, patroon in ARMS.items():
            cel = patroon.format(lvl)
            per[arm] = {r["seed"] for r in runset.rijen() if r["cel"] == cel}
        gedeeld = per["gemma"] & per["qwen"]
        uit[lvl] = {"gemma": len(per["gemma"]), "qwen": len(per["qwen"]),
                    "shared": len(gedeeld),
                    "qwen_seeds_not_in_gemma": sorted(per["qwen"] - per["gemma"])}
    return uit


def inequality_by_arm() -> dict:
    """Final Gini per arm and level, with the two run-level ranges.

    The chapter claims the L2 cells do not overlap, which is a statement about
    ranges and not about means, so both are reported.
    """
    def samenvat(xs):
        return {"mean": round(sum(xs) / len(xs), 3), "n": len(xs),
                "min": round(min(xs), 3), "max": round(max(xs), 3)}

    uit = {}
    for lvl in LEVELS:
        per = {}
        for arm in ARMS:
            paths = runset.cel(ARMS[arm].format(lvl))
            per[arm] = samenvat(runstat.per_run(paths, runstat.final_gini))

        # The chapter compares the arms "on the same five seeds", and the whole
        # Gemma cell is fifteen or ten runs. Reporting only the full cell left
        # the sentence's own numbers unreproducible from this figure --- they
        # are the seed-matched ones, and they differ by up to six points.
        gedeeld = {r["seed"] for r in runset.rijen()
                   if r["cel"] == ARMS["gemma"].format(lvl)} & \
                  {r["seed"] for r in runset.rijen()
                   if r["cel"] == ARMS["qwen"].format(lvl)}
        op_seed = {r["bestand"]: r["seed"] for r in runset.rijen()
                   if r["cel"] == ARMS["gemma"].format(lvl)}
        paden = [p for p in runset.cel(ARMS["gemma"].format(lvl))
                 if op_seed.get(p.name) in gedeeld]
        if paden:
            per["gemma_on_shared_seeds"] = samenvat(
                runstat.per_run(paden, runstat.final_gini))

        per["overlap"] = not (per["gemma"]["max"] < per["qwen"]["min"]
                              or per["qwen"]["max"] < per["gemma"]["min"])
        g = per.get("gemma_on_shared_seeds")
        if g:
            per["overlap_on_shared_seeds"] = not (
                g["max"] < per["qwen"]["min"] or per["qwen"]["max"] < g["min"])
        uit[lvl] = per
    return uit


# --- m:address-each-other --------------------------------------------------

def _recipients_per_message(p) -> float | None:
    n = tot = 0
    for e in logs.rounds(p):
        for m in (e.get("messages") or []):
            n += 1
            tot += len(m.get("to") or [])
    return tot / n if n else None


def _mean_degree(p, laatste: bool = True) -> float | None:
    rs = logs.rounds(p)
    e = rs[-1] if laatste else rs[0]
    net = e.get("network") or {}
    edges = net.get("edges") or []
    leden = {x for a, b in edges for x in (a, b)} | set(e.get("agents") or {})
    return 2 * len(edges) / len(leden) if leden else None


def _invites(p) -> int:
    return sum(1 for e in logs.rounds(p)
               for a in (e.get("agents") or {}).values()
               if (a.get("rewire_intent") or {}).get("invite"))


def _severings(p) -> int:
    return sum(1 for e in logs.rounds(p)
               for a in (e.get("agents") or {}).values()
               if (a.get("rewire_intent") or {}).get("drop"))


def _largest_coalition(p) -> int | None:
    v = [len(c.get("attackers") or []) for _, c in combat.fights(p)]
    return max(v) if v else None


REACH = {
    "recipients_per_message": _recipients_per_message,
    "mean_degree_first_round": lambda p: _mean_degree(p, laatste=False),
    "mean_degree_last_round": _mean_degree,
    "invitations": _invites,
    "severings": _severings,
    "solo_share_of_attacks": combat.solo_share,
    "largest_coalition": _largest_coalition,
}


def how_broadly_they_coordinate() -> dict:
    """The reach axis: how many agents a model addresses, links to and attacks with.

    Every quantity is a per-run value summarised over five to ten runs, and the
    totals the chapter quotes for severings and invitations are sums rather than
    means, so both are given.
    """
    uit = {}
    for arm in ARMS:
        per_level = {}
        for lvl in LEVELS:
            paths = runset.cel(ARMS[arm].format(lvl))
            rij = {}
            for naam, scalar in REACH.items():
                xs = [scalar(p) for p in paths]
                geldig = [x for x in xs if x is not None]
                rij[naam] = {"mean": round(sum(geldig) / len(geldig), 2) if geldig else None,
                             "total": round(sum(geldig), 1) if geldig else None,
                             "max": round(max(geldig), 2) if geldig else None,
                             "defined_in": len(geldig), "runs": len(xs)}
            per_level[lvl] = rij
        uit[arm] = per_level
    return uit


# --- m:pact-institution-language -------------------------------------------

def coined_terms_by_arm() -> dict:
    """Shared names per arm and level, on the same three-adopter threshold.

    The threshold is what makes the comparison interesting and also what makes
    it partly circular: a name needs three adopters, and a model that addresses
    2.5 agents per message rarely reaches three. Both the count and the reach
    are reported so a reader can see the two together.

    Wordings of one arrangement are folded together before counting, which
    matters more for a comparison between arms than within one: a model that
    repeats a phrase verbatim and a model that rephrases it each time would
    otherwise differ in coined names without differing in what they coined.
    """
    uit = {}
    for arm in ARMS:
        per_level = {}
        for lvl in LEVELS:
            paths = runset.cel(ARMS[arm].format(lvl))
            per = {}
            for drempel in (2, 3, 4):
                aantal = [len(text.named_agreements(p, drempel)) for p in paths]
                per[f"min_users={drempel}"] = {"total": sum(aantal),
                                               "mean": round(sum(aantal) / len(aantal), 2),
                                               "per_run": sorted(aantal)}
            per_level[lvl] = per
        uit[arm] = per_level
    return uit


# --- m:welfare-arrangements-redistribute -----------------------------------

def nobody_rescues_the_dying() -> dict:
    """Transfers reaching an agent already below one resource.

    A structural count with no detector in it: an agent's holdings and the
    target of every transfer are both in the action log. The denominator is
    every agent-round spent below the threshold, so the figure answers "of all
    the chances there were to rescue someone, how many were taken".
    """
    uit = {}
    for arm in ARMS:
        onder = gered = 0
        for lvl in LEVELS:
            for p in runset.cel(ARMS[arm].format(lvl)):
                k, g = runstat.destitute_and_rescues(p)
                onder += k
                gered += g
        uit[arm] = Result(value=gered, n=len(LEVELS) * 5, denominator=onder,
                          unit="agent-rounds below one resource",
                          note=f"{gered} transfers reached an agent below 1.0 "
                               f"resource, out of {onder} agent-rounds spent there").as_dict()
    return uit


def model_swap_arm() -> dict:
    """The two halves of the swap: which seeds the arms share, and what each arm
    does to the distribution. A named function rather than a lambda, so the
    provenance line in every generated file can say where the figure came from."""
    return {"seeds": seed_overlap(), "inequality": inequality_by_arm()}


FIGURES = {
    "m:model-swap-arm": model_swap_arm,
    "m:address-each-other": how_broadly_they_coordinate,
    "m:pact-institution-language": coined_terms_by_arm,
    "m:rescues-the-dying": nobody_rescues_the_dying,
}


# --- m:second-order-reasoning ----------------------------------------------

def attribution_orders() -> dict:
    """How far the reasoning traces model other minds, per arm and level.

    The unit is the reasoning block, not the sentence. That departs from the
    rule the rest of the language family enforces, and it is the measure's own
    definition: an attribution can span two sentences and splitting would cut
    the nesting apart. These shares are therefore not comparable with any
    sentence-level figure elsewhere in the chapter.

    Two cell lists serve two purposes and they are not interchangeable. The
    between-arm comparison uses knife-edge only, because that is the sole cell
    the robustness arms ran and therefore the only matched column. The
    within-Gemma spread uses every production cell, because it is the noise
    floor: it says how large a difference has to be before it is worth reading,
    and a comparison narrower than that spread sits inside it.

    The probe is what the design is for. If the order of attribution tracked how
    broadly a model coordinates, the arm that writes to 1.4 to 2.4 recipients and
    coins no shared name should sit lowest. It is too small to confirm anything
    and exactly large enough to refute that.

    Three things are reported beside the rates because each answers an
    objection. The length control divides by block length, since a longer block
    has more chances to match. The plural-first-person share matters because one
    arm writes "we" where the others write "I", and counting that as another
    agent turns its own reasoning into an attribution. And the within-arm spread
    across capacity levels is given, because if it exceeds the difference
    between arms then the arm comparison sits inside its own noise.
    """
    uit = {}
    # Every production cell, not only knife-edge. The within-arm spread across
    # capacity levels and payoff cells is the figure that decides whether the
    # between-arm comparison means anything: if one model varies more against
    # itself than the two models differ from each other, the model difference
    # sits inside its own noise. That cannot be seen from the knife-edge column
    # alone, which is what an earlier version of this list measured.
    cellen = ([f"prod_{r}_{q}" for r in ("L1", "L2", "L3", "L4")
               for q in ("scar", "knife", "abund")]
              + [f"prod_{r}_knife_nocomm" for r in ("L1", "L2", "L3", "L4")]
              + [f"robust_{m}_{r}_knife" for m in ("qwen", "deepseek") for r in LEVELS])
    for c in cellen:
        try:
            paths = runset.cel(c)
        except runset.RunsetError:
            continue
        tel = Counter()
        lengtes = []
        for p in paths:
            namen = set()
            for e in logs.rounds(p):
                namen |= set((e.get("agents") or {}).keys())
            for e in logs.rounds(p):
                for a in (e.get("agents") or {}).values():
                    blok = str(a.get("thinking") or "")
                    if not blok.strip():
                        continue
                    tel["blocks"] += 1
                    tel[f"order{text.attribution_order(blok, namen)}"] += 1
                    tel["plural"] += text.plural_first_person(blok)
                    lengtes.append(len(blok.split()))
        n = tel["blocks"]
        if not n:
            continue
        mediaan_lengte = sorted(lengtes)[len(lengtes) // 2]
        kort = [l for l in lengtes if l <= mediaan_lengte]
        uit[c] = Result(
            value={"blocks": n,
                   "order_2_or_higher_pct": round(100 * (tel["order2"] + tel["order3"]) / n, 2),
                   "order_3_pct": round(100 * tel["order3"] / n, 3),
                   "order_3_count": tel["order3"]},
            n=len(paths), denominator=n, unit="reasoning blocks",
            sensitivity={"first_person_plural_pct": round(100 * tel["plural"] / n, 1),
                         "median_block_words": mediaan_lengte,
                         "blocks_at_or_below_median_length": len(kort)},
            note="unit is the reasoning block, not the sentence; not comparable "
                 "with sentence-level figures").as_dict()
    return uit


# --- m:second-order-reasoning ----------------------------------------------
#
# Replaces the classifier version. That one read an agent computing its own
# odds of winning as someone attributing a thought to another, and got three of
# twelve right on a hand check. This measure counts no detector hits but hand
# labels: a hundred traces, three blind labellers, codebook fixed in advance.
#
# No mean order is reported. A mean over a scale of three states suggests a
# distance between 0 and 1 equal to the one between 1 and 2, and it is moreover
# sensitive to the one codebook rule that on rereading was applied most loosely.
# The distribution and the hit rate per character carry the same comparison and
# are both checkable.

HANDLABELS = HIER.parent.parent / "handlabels" / "tom" / "gold_v2.jsonl"
FRAGMENTEN = HIER.parent.parent / "handlabels" / "tom" / "v2_blind.jsonl"
LABELSETS = ("v2_G", "v2_H", "v2_I")


def _lambda_mle(lengtes, raak):
    """Hit rate per thousand characters under P(order>=2) = 1 - exp(-lambda*L).

    The measure is a per-trace indicator over texts that differ a factor of five
    in length between arms, and a longer text has more chance to hit the second
    order somewhere. Without a denominator the comparison therefore partly
    measures volume. This model supplies that denominator and handles saturation
    correctly: a trace that hits twice still counts once, where a crude count per
    thousand characters would tell against the long arm.

    The log-likelihood is concave in lambda, so a bounded one-dimensional search
    suffices.
    """
    import numpy as np
    from scipy import optimize

    lengtes, raak = np.asarray(lengtes, float), np.asarray(raak, float)

    def neg(lam):
        if lam <= 0:
            return 1e9 if raak.any() else 0.0
        p = np.clip(1 - np.exp(-lam * lengtes), 1e-12, 1 - 1e-12)
        return -float(np.sum(raak * np.log(p) + (1 - raak) * np.log(1 - p)))

    return float(optimize.minimize_scalar(neg, bounds=(1e-6, 5.0),
                                          method="bounded").x)


def _loglik(lengtes, raak, lam):
    import numpy as np
    p = np.clip(1 - np.exp(-lam * np.asarray(lengtes, float)), 1e-12, 1 - 1e-12)
    raak = np.asarray(raak, float)
    return float(np.sum(raak * np.log(p) + (1 - raak) * np.log(1 - p)))


def attribution_order_handcoded() -> dict:
    """The order of mental-state attribution, hand-coded on a stratified sample.

    Not a detector over the corpus but a hand-labelled sample of it, which is
    why the numbers are small and why a reader can check them: the hundred
    traces, the codebook they were read against and the three label sets are in
    the repository, and every disagreement carries the span that decided it.

    Two things are reported and they answer different questions. The counts say
    what each arm puts on the page, which is what the swap was run to see. The
    rate per thousand characters says whether an arm reaches the second order
    more often than its own volume of text would produce anyway, and that is a
    harder question, because the arms write at medians five times apart.

    On the counts the three arms differ. On the rate only one of them does: a
    model in which DeepSeek and Gemma share a rate and Qwen has its own is not
    improved on by giving all three their own, while collapsing those two into
    one is decisively worse. What survives the correction is that Qwen reasons
    about how a neighbour reads it far less often per unit of text than either
    other arm.
    """
    import itertools
    import json
    import numpy as np
    from collections import Counter, defaultdict
    from scipy import stats

    for pad in (HANDLABELS, FRAGMENTEN):
        if not pad.exists():
            raise runset.RunsetError(f"{pad} is missing; label the sample first")

    goud = handlabels.op_id(HANDLABELS)
    tekst = {i: d["text"] for i, d in handlabels.op_id(FRAGMENTEN).items()}

    rng = np.random.default_rng(20260907)

    # --- agreement, counted from the label sets themselves and not copied over ---
    sets = {}
    for naam in LABELSETS:
        pad = HANDLABELS.parent / f"{naam}.jsonl"
        if not pad.exists():
            raise runset.RunsetError(f"{pad} is missing")
        sets[naam] = {i: str(d["order"]) for i, d in handlabels.op_id(pad).items()}
    ids = sorted(set.intersection(*(set(s) for s in sets.values())))

    def kappa(a, b):
        labels = sorted(set(a) | set(b))
        idx = {l: i for i, l in enumerate(labels)}
        m = np.zeros((len(labels), len(labels)))
        for x, y in zip(a, b):
            m[idx[x], idx[y]] += 1
        m /= m.sum()
        po, pe = np.trace(m), float(m.sum(axis=0) @ m.sum(axis=1))
        return round((po - pe) / (1 - pe), 2) if pe < 1 else 1.0

    kappas = [kappa([sets[a][i] for i in ids], [sets[b][i] for i in ids])
              for a, b in itertools.combinations(LABELSETS, 2)]
    stemmen = Counter(Counter(sets[n][i] for n in LABELSETS).most_common(1)[0][1]
                      for i in ids)

    # --- distribution per arm ---
    per_arm = defaultdict(list)
    for i, d in goud.items():
        per_arm[d["model"]].append((int(d["order"]) if str(d["order"]).isdigit()
                                    else None, len(tekst[i])))

    verdeling = {}
    for arm, rijen in per_arm.items():
        orden = [o for o, _ in rijen if o is not None]
        verdeling[arm] = {
            "n": len(rijen),
            "at_least_first_order": sum(o >= 1 for o in orden),
            "at_least_second_order": sum(o >= 2 for o in orden),
            "third_order_or_higher": sum(o >= 3 for o in orden),
            "unplaceable": len(rijen) - len(orden),
            "median_trace_characters": int(np.median([n for _, n in rijen]))}

    # --- hit rate per thousand characters ---
    L = {a: np.array([n / 1000 for _, n in r]) for a, r in per_arm.items()}
    Y = {a: np.array([1.0 if (o or 0) >= 2 else 0.0 for o, _ in r])
         for a, r in per_arm.items()}
    lam = {a: _lambda_mle(L[a], Y[a]) for a in L}

    def interval(a, R=10000):
        uit = [_lambda_mle(L[a][k], Y[a][k])
               for k in (rng.integers(0, len(L[a]), len(L[a])) for _ in range(R))]
        return [round(float(x), 3) for x in np.quantile(uit, [0.025, 0.975])]

    # Draw once per arm, not once per endpoint: two calls drain the same
    # generator and then return a lower and an upper endpoint from different
    # bootstraps, which is not an interval.
    grenzen = {a: interval(a) for a in lam}
    tarieven = {a: {"per_thousand_characters": round(lam[a], 3),
                    "ci_lo": grenzen[a][0], "ci_hi": grenzen[a][1]} for a in lam}

    paren = {}
    for a, b in itertools.combinations(sorted(L), 2):
        Lc, Yc = np.concatenate([L[a], L[b]]), np.concatenate([Y[a], Y[b]])
        gedeeld = _lambda_mle(Lc, Yc)
        chi2 = 2 * (_loglik(L[a], Y[a], lam[a]) + _loglik(L[b], Y[b], lam[b])
                    - _loglik(L[a], Y[a], gedeeld) - _loglik(L[b], Y[b], gedeeld))
        paren[f"{a}_over_{b}"] = {"ratio": round(lam[a] / lam[b], 2) if lam[b] else None,
                                  "chi2_1df": round(chi2, 2),
                                  "p": round(float(stats.chi2.sf(chi2, 1)), 4)}

    # Which model of the rates suffices: one shared, two, or three?
    allen = (np.concatenate(list(L.values())), np.concatenate(list(Y.values())))
    een = _lambda_mle(*allen)
    saam = _lambda_mle(np.concatenate([L["deepseek"], L["gemma"]]),
                       np.concatenate([Y["deepseek"], Y["gemma"]]))
    ll1 = sum(_loglik(L[a], Y[a], een) for a in L)
    ll2 = (_loglik(L["deepseek"], Y["deepseek"], saam)
           + _loglik(L["gemma"], Y["gemma"], saam)
           + _loglik(L["qwen"], Y["qwen"], lam["qwen"]))
    ll3 = sum(_loglik(L[a], Y[a], lam[a]) for a in L)

    return {
        "by_model": verdeling,
        "rate_by_model": tarieven,
        "rate_ratios": paren,
        "how_many_rates_are_needed": {
            "one_versus_two": {"chi2_1df": round(2 * (ll2 - ll1), 2),
                               "p": round(float(stats.chi2.sf(2 * (ll2 - ll1), 1)), 5)},
            "two_versus_three": {"chi2_1df": round(2 * (ll3 - ll2), 2),
                                 "p": round(float(stats.chi2.sf(2 * (ll3 - ll2), 1)), 3)},
            "reading": "deepseek and gemma share a rate; qwen has its own"},
        "agreement": {"labellers": len(LABELSETS),
                      "unanimous": stemmen[3], "two_of_three": stemmen[2],
                      "no_majority": stemmen[1],
                      "cohens_kappa_pairwise": kappas},
        "n": len(goud), "unit": "hand-coded reasoning traces",
        "note": "ten traces per model-by-level cell. The counts compare what "
                "the arms write; the rates compare how often they reach the "
                "second order per unit of text, which the counts cannot, "
                "because the arms write at very different lengths. No mean "
                "order is reported: a mean over three ordered classes asserts "
                "a spacing the codebook does not define."}


FIGURES["m:second-order-reasoning"] = attribution_order_handcoded


# --- m:attribution-and-capacity ---------------------------------------------

def attribution_order_capacity() -> dict:
    """Whether the capacity level or the price moves the order of attribution.

    A second hand-coded sample, and it exists because the first one cannot
    answer this. Those hundred traces were drawn ten per model and per capacity
    level, which is the right shape for comparing arms and the wrong one for
    comparing rungs: the two extra arms ran only at knife-edge, so they carry no
    price variation, and ten traces a cell leaves the rungs indistinguishable.

    This sample is the production arm alone, a hundred and sixty traces spread
    thirteen or fourteen over each of the twelve comms-on cells, so the rung can
    be read by pooling over the price and the price by pooling over the rung.
    The forty Gemma traces from the first sample are excluded by their text: they
    raised the hypothesis this sample tests, and a hypothesis tested on the
    observation that raised it is not tested at all.

    Two tests, and which of them applies is itself the finding. The trend
    imposes lambda_t = a * r^(t-1) and asks whether r = 1 --- one degree of
    freedom, so it has power, but it sees only a line and averages a peak away.
    The free form gives every rung its own rate --- three degrees of freedom, and
    the only one of the two that can see a peak in the middle.

    The exposure correction is the same as in attribution_order_handcoded(): a
    higher rung gives an agent more to write about, and the measure takes the
    highest order anywhere in a trace, so an uncorrected rung difference would
    partly be a difference in volume.
    """
    import numpy as np
    from collections import Counter
    from scipy import optimize, stats

    kern = handlabels.WORTEL / "capaciteit"

    stemmen = {r: {i: str(d["order"]).strip()
                   for i, d in handlabels.labelronde("capaciteit", r).items()}
               for r in ("G", "H", "I")}

    tekst = {i: d["text"] for i, d in handlabels.op_id(kern / "cap_blind.jsonl").items()}
    key = handlabels.op_id(kern / "cap_key.jsonl")
    ids = sorted(set(tekst) & set.intersection(*(set(v) for v in stemmen.values())))

    def kappa(a, b):
        labels = sorted(set(a) | set(b))
        n = len(a)
        eens = sum(x == y for x, y in zip(a, b)) / n
        kans = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
        return round((eens - kans) / (1 - kans), 3) if kans < 1 else 1.0

    paren = {}
    for a, b in (("G", "H"), ("G", "I"), ("H", "I")):
        paren[f"{a}-{b}"] = kappa([stemmen[a][i] for i in ids],
                                  [stemmen[b][i] for i in ids])

    gold, vorm = {}, Counter()
    for i in ids:
        c = Counter(stemmen[r][i] for r in ("G", "H", "I"))
        top, k = c.most_common(1)[0]
        vorm["unaniem" if k == 3 else "twee-van-drie" if k == 2 else "geen"] += 1
        if k > 1:
            gold[i] = top

    rijen = [{"rung": key[i]["rung"], "payoff": key[i]["payoff"],
              "L": len(tekst[i]) / 1000,
              "raak": 1 if gold[i].isdigit() and int(gold[i]) >= 2 else 0}
             for i in gold]

    def ll(lam, L, y):
        p = np.clip(1 - np.exp(-lam * L), 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))

    def mle(L, y):
        if y.sum() == 0:
            return 0.0
        return float(optimize.minimize_scalar(lambda x: -ll(x, L, y),
                                              bounds=(1e-9, 5), method="bounded").x)

    def groepeer(veld, niveaus):
        uit = {}
        for niv in niveaus:
            s = [r for r in rijen if r[veld] == niv]
            uit[niv] = (np.array([r["L"] for r in s]),
                        np.array([r["raak"] for r in s]))
        return uit

    def vrije_vorm(g):
        vrij = sum(ll(mle(L, y), L, y) for L, y in g.values())
        Ls = np.concatenate([L for L, _ in g.values()])
        ys = np.concatenate([y for _, y in g.values()])
        chi2 = 2 * (vrij - ll(mle(Ls, ys), Ls, ys))
        df = len(g) - 1
        return round(chi2, 2), df, round(float(stats.chi2.sf(max(chi2, 0), df)), 4)

    def trend(g):
        volgorde = list(g)

        def neg(p):
            return -sum(ll(np.exp(p[0]) * np.exp(p[1] * i), *g[t])
                        for i, t in enumerate(volgorde))

        r = optimize.minimize(neg, [np.log(0.05), 0.0], method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
        Ls = np.concatenate([L for L, _ in g.values()])
        ys = np.concatenate([y for _, y in g.values()])
        chi2 = 2 * (-r.fun - ll(mle(Ls, ys), Ls, ys))
        return (round(float(chi2), 2), round(float(stats.chi2.sf(max(chi2, 0), 1)), 4),
                round(float(np.exp(r.x[1])), 2))

    tredes = groepeer("rung", ("L1", "L2", "L3", "L4"))
    prijzen = groepeer("payoff", ("abund", "knife", "scar"))

    c_t, df_t, p_t = vrije_vorm(tredes)
    c_tr, p_tr, factor = trend(tredes)
    c_p, df_p, p_p = vrije_vorm(prijzen)

    return {
        "value": {
            "agreement": {"pairwise_kappa": paren, **dict(vorm)},
            "by_rung": {t: {"second_order": int(y.sum()), "traces": len(y),
                            "median_chars": int(np.median(L) * 1000),
                            "lambda": round(mle(L, y), 4)}
                        for t, (L, y) in tredes.items()},
            "by_price": {t: {"second_order": int(y.sum()), "traces": len(y),
                             "lambda": round(mle(L, y), 4)}
                         for t, (L, y) in prijzen.items()},
            "rung_free_form": {"chi2": c_t, "df": df_t, "p": p_t},
            "rung_trend": {"chi2": c_tr, "df": 1, "p": p_tr, "factor_per_rung": factor},
            "price_free_form": {"chi2": c_p, "df": df_p, "p": p_p},
            "above_second_order": sum(1 for o in gold.values()
                                      if o.isdigit() and int(o) > 2),
        },
        "n": len(rijen),
        "denominator": len(ids),
        "unit": "traces",
        "note": ("production arm only, 13-14 traces in each of the twelve comms-on "
                 "cells; the forty Gemma traces of m:second-order-reasoning are "
                 "excluded, since they raised the hypothesis this sample tests"),
    }


FIGURES["m:attribution-and-capacity"] = attribution_order_capacity
