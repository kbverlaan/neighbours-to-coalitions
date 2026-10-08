#!/usr/bin/env python3
"""
enforcement.py — NORM->DAAD marker (no-LLM, triage + subagent-haakjes).

Deontische dichtheid (deontic.py) meet of er een NORM is (taal). Deze marker
meet of die norm de ACTIES stuurt -> de norm->institutie-brug: worden woorden daden?

Twee lenzen (de twee kanten van de deontische logica):
  - LENS A  NORMHANDHAVING  : take/drop uitgevoerd als STRAF (handhaving van een verbod)
                              "punish the betrayer", "exile X", "X violated the pact"
  - LENS B  NORMVERVULLING  : transfer/arm uitgevoerd als NALEVING (uitvoering van een gebod)
                              "support our member", "mutual aid for anyone below 60"

Mechaniek (deterministisch): voor elke gerichte actie a->t in ronde r, kijk in de
PUBLIEKE berichten van venster [r-1, r] (berichten crossen 1 ronde, dus de recht-
vaardiging gaat meestal vooraf). Bevat een zin daar (i) het juiste lexicon EN
(ii) de naam van target t? -> de actie is GEFRAMED. Associatief, niet causaal —
maar genoeg om runs te TRIAGEREN en de (actie, citaat, ronde)-TRIPLES te leveren
waar een subagent in kan duiken.

Gebruik:
    python scripts/enforcement.py RUN.jsonl            # scores + triples van 1 run
    python scripts/enforcement.py DIR/                 # triage-ranking over een map
    python scripts/enforcement.py RUN.jsonl --top 30
"""
import json, re, sys, glob, os, argparse, random, statistics
from collections import defaultdict, Counter


def gini(xs):
    """Gini over niet-negatieve tellingen (0 = gelijk verdeeld, ->1 = geconcentreerd)."""
    xs = sorted(xs); n = len(xs); s = sum(xs)
    if n == 0 or s <= 0:
        return 0.0
    return (2 * sum((i + 1) * x for i, x in enumerate(xs))) / (n * s) - (n + 1) / n

# ── LEXICONS (Engels) ────────────────────────────────────────────────────────
# Lens A: straf / handhaving. SPLIT 2026-07-16 (CO-definition): Crawford &
# Ostrom's OR ELSE is "the sanction assigned to detected NONCOMPLIANCE with an
# institutional statement" (p.586). Handhaving telt dus alleen als het doelwit
# een gedeeld voorschrift SCHOND — niet als het gewoon het sterkste/rijkste
# doelwit is ("level the whale" = gecoördineerde predatie, conventie-niveau).
#
# VIOLATION = de framing impliceert een geschonden verbintenis (betrapte
# noncompliance) -> telt als CO OR-ELSE-handhaving.
VIOLATION = [r"betray\w*", r"traitor", r"backstab\w*", r"double[\-\s]?cross\w*",
             r"violat\w*", r"defect\w*", r"renege\w*", r"cheat\w*", r"two[\-\s]?faced",
             r"broke (?:the|our|his|her|their) (?:pact|nap|deal|word|agreement|promise|vow|trust)",
             r"broke faith", r"oath[\-\s]?break\w*", r"dishonest", r"liar", r"hypocri\w*",
             # straf-werkwoorden die een overtreding vooronderstellen
             r"punish", r"exile", r"expel", r"banish", r"ostrac\w*", r"cast out",
             r"sanction", r"make an example", r"hold (?:him|her|them) accountable",
             r"pay for (?:this|that|it|betray\w*)"]
# COORD_AGGRESSION = focal predatie zonder voorschrift -> NIET CO-handhaving,
# apart geteld als contrast (dit is wat de oude marker ten onrechte meetelde).
COORD_AGGRESSION = [r"bring (?:him|her|them)? ?down", r"must fall", r"take down",
                    r"level the", r"gang up", r"wolves?", r"deserv\w*", r"deal with",
                    r"target the", r"must be (?:stopped|removed)", r"cut (?:him|her|them) off"]
# Terugwaartse compat: SANCTION = de brede unie (oude gedrag, nu 'coordinated
# aggression'-triage). RX_VIOLATION draagt de CO-handhaving.
SANCTION = VIOLATION + COORD_AGGRESSION
RX_VIOLATION = re.compile(r"(?:" + "|".join(VIOLATION) + r")", re.I)
# Lens B: naleving / uitvoering van een gebod (institutioneel, niet zomaar 'aardig').
FULFILMENT = [r"mutual aid", r"support (?:our|the|each)", r"defend (?:our|the)",
              r"protect (?:our|the|each)", r"loyal\w*", r"reciprocat\w*",
              r"contribut\w*", r"reinforce", r"stand with", r"back (?:you|our)",
              r"reward", r"uphold", r"honou?r (?:our|the)", r"as (?:agreed|promised)",
              r"keep (?:our|my) word", r"fulfil\w*", r"our (?:duty|obligation|pact|core|member)",
              r"the (?:pact|council|core|alliance|vsc)", r"stability (?:pact|core)",
              r"rotation", r"aid (?:to|for) (?:the|our|any)"]
RX_SANC = re.compile(r"(?:" + "|".join(SANCTION) + r")", re.I)
RX_FULF = re.compile(r"(?:" + "|".join(FULFILMENT) + r")", re.I)
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def cat(info):
    """Geef (lens, target) voor de gerichte actie van deze agent, of (None,None).
    take/attack -> ('A', target); transfer/invest -> ('B', target); arm -> ('B', target).
    Drops worden apart afgehandeld (rewire_intent)."""
    a = str(info.get("action") or "hold").lower()
    tg = info.get("target")
    if not tg:
        return (None, None)
    if a.startswith("take") or a == "attack":
        return ("A", tg)
    if a.startswith("invest") or a == "transfer":
        return ("B", tg)
    if "arm" in a or "strengthen" in a:
        return ("B", tg)
    return (None, None)


ADDRESSEE = re.compile(r"^\s*[A-Z]\w+(?:,\s*[A-Z]\w+)*\s*:\s*")  # "Copper, Gold, Green: ..."


def make_framing(msgs_by_round):
    """Framing-checker over een run (gememoized op (r, target, lens-rx) zodat de
    permutatie-null geen 500x regex-sweep kost)."""
    cache = {}

    def framing(r, target, rx):
        """Zoek in venster [r-1, r] een publieke zin waar target het DOELWIT van de
        framing is. Twee precisie-eisen bovenop lexicon+naam:
          1. strip de adressaat-kop ("Copper, Gold:") -> target mag geen geadresseerde zijn;
          2. nabijheid: target-naam binnen ~40 tekens van de lexicon-trigger
             -> target is object/subject van de straf, geen losse vermelding.
        Naam case-SENSITIVE (agent-namen zijn Capitalized)."""
        key = (r, target, id(rx))
        if key in cache:
            return cache[key]
        name_rx = re.compile(r"\b" + re.escape(target) + r"\b")
        hit = None
        for rr in (r - 1, r):
            if hit:
                break
            for frm, txt in msgs_by_round.get(rr, []):
                if hit:
                    break
                for s in SENT.split(txt):
                    s2 = ADDRESSEE.sub("", s)            # adressaat-kop weg
                    for lm in rx.finditer(s2):
                        for nm in name_rx.finditer(s2):
                            if abs(lm.start() - nm.start()) <= 40:
                                hit = (frm, s2.strip())
                                break
                        if hit:
                            break
                    if hit:
                        break
        cache[key] = hit
        return hit

    return framing


def _load_actions(path):
    """Gedeeld door analyze() en permutation_null(): (nr, msgs_by_round, actions)."""
    rounds = [json.loads(l) for l in open(path) if l.strip()]
    nr = len(rounds)
    msgs_by_round = defaultdict(list)
    actions = []
    for d in rounds:
        r = d.get("round") or 0
        for m in d.get("messages", []) or []:
            if m.get("text"):
                msgs_by_round[r].append((m.get("from"), m["text"]))
        for aid, info in (d.get("agents") or {}).items():
            lens, tg = cat(info)
            if lens:
                actions.append((r, lens, aid, tg, str(info.get("action"))))
            drop = (info.get("rewire_intent") or {}).get("drop")
            if drop:
                actions.append((r, "A", aid, drop, "drop"))
    return nr, msgs_by_round, actions


def permutation_null(path, n_perm=500, seed=42):
    """Target-permutatie-null (DVs.md Laag B): wordt het ECHTE doelwit van een actie
    vaker als straf/naleving geframed dan een WILLEKEURIG doelwit? Permuteer per
    lens de target-kolom over de acties (rondes/actors blijven staan), herbereken
    de framed-rate, n_perm x. Rapporteert obs vs null (mean, sd, z, percentiel)."""
    _, msgs_by_round, actions = _load_actions(path)
    framing = make_framing(msgs_by_round)
    rng = random.Random(seed)
    out = {}
    for lens, rx in (("A", RX_SANC), ("B", RX_FULF)):
        acts = [(r, tg) for (r, l, a, tg, lab) in actions if l == lens]
        if not acts:
            out[lens] = None
            continue
        obs = sum(1 for r, tg in acts if framing(r, tg, rx)) / len(acts)
        targets = [tg for _, tg in acts]
        null = []
        for _ in range(n_perm):
            rng.shuffle(targets)
            null.append(sum(1 for (r, _), tg in zip(acts, targets)
                            if framing(r, tg, rx)) / len(acts))
        mu = statistics.fmean(null)
        sd = statistics.pstdev(null)
        z = (obs - mu) / sd if sd > 0 else float("inf") if obs > mu else 0.0
        pctl = 100 * sum(1 for v in null if v < obs) / n_perm
        out[lens] = dict(n_actions=len(acts), observed=obs, null_mean=mu,
                         null_sd=sd, z=z, pctl=pctl)
    return out


def analyze(path):
    rounds = [json.loads(l) for l in open(path) if l.strip()]
    nr = len(rounds)
    msgs_by_round = defaultdict(list)            # r -> [(from, text)]
    actions = []                                 # (r, lens, actor, target, action_label)
    agents = set()
    for d in rounds:
        r = d.get("round") or 0
        for m in d.get("messages", []) or []:
            if m.get("text"):
                msgs_by_round[r].append((m.get("from"), m["text"]))
        for aid, info in (d.get("agents") or {}).items():
            agents.add(aid)
            lens, tg = cat(info)
            if lens:
                actions.append((r, lens, aid, tg, str(info.get("action"))))
            drop = (info.get("rewire_intent") or {}).get("drop")
            if drop:                              # drop = sociale uitsluiting -> lens A
                actions.append((r, "A", aid, drop, "drop"))

    framing = make_framing(msgs_by_round)

    triples = {"A": [], "B": []}
    n_act = {"A": 0, "B": 0}
    n_framed = {"A": 0, "B": 0}
    # CO-handhaving (OR ELSE): sanctie-actie geframed als reactie op een OVERTREDING
    # (betrapte noncompliance), niet als losse agressie op een focal doelwit.
    n_co_enforce = 0
    co_triples = []
    for (r, lens, actor, tg, label) in actions:
        n_act[lens] += 1
        rx = RX_SANC if lens == "A" else RX_FULF
        f = framing(r, tg, rx)
        if f:
            n_framed[lens] += 1
            triples[lens].append((r, actor, label, tg, f[0], f[1][:150]))
        if lens == "A":
            fv = framing(r, tg, RX_VIOLATION)     # zelfde nabijheids-regel, violatie-lexicon
            if fv:
                n_co_enforce += 1
                co_triples.append((r, actor, label, tg, fv[0], fv[1][:150]))
    rate = lambda l: n_framed[l] / n_act[l] if n_act[l] else 0.0
    co_enforce_rate = n_co_enforce / n_act["A"] if n_act["A"] else 0.0
    # ── Governance-rol-attributie (rol-laag, 2026-07-07; Aoki-verfijning 2026-07-08) ──
    # WIE handhaaft, en OP WIE? Geconcentreerde geframede sanctie (lens A) op een
    # paar agents = een monitor/sanctioner-ROL is ontstaan. MAAR: concentratie van
    # sanctiecapaciteit ALLEEN volstaat niet om governance te herkennen. Aoki (2001)
    # Ch.6 laat zien dat dezelfde hoge top-share past op zowel een democratische/
    # rule-of-law-staat (een handhaver die een gedeelde regel volgt) als een
    # COLLUSIEVE/roof-staat (handhaver + kliek roven systematisch een minderheid):
    # "A government strong enough to protect property rights and enforce contracts is
    # also strong enough to confiscate the wealth of its citizens" (Weingast, in Aoki
    # p.151). Het onderscheidende is niet de concentratie maar (a) WIE het doelwit is
    # -- systematisch dezelfde minderheid (collusief) vs roterend onder gedeelde
    # dreiging (democratisch; Aoki p.156 "punished by the withdrawal of the support of
    # private agents ... nonvictims" / p.158 "lack of government's information about the
    # identity ... a commitment device ... to abstain from random extortion"); en (b) of
    # de tophandhaver ZELF disciplineerbaar is (ooit zelf doelwit van geframede sanctie;
    # Glorious-Revolution-mechaniek, Aoki p.161-162). Daarom meten we naast enforcer-
    # concentratie ook TARGET-concentratie en enforcer-disciplineerbaarheid.
    # Alle drempels ONGEKALIBREERD (nog geen bevestigde governance-run; voorspeld pas
    # op de commons-trede) -> conservatief + gevlagd.
    enf_by_actor = Counter(actor for (r, actor, label, tg, frm, s) in triples["A"])
    counts = [enf_by_actor.get(a, 0) for a in agents]
    total_enf = sum(counts)
    enforcer_gini = gini(counts)
    n_enforcers = sum(1 for c in counts if c > 0)
    top_enf = enf_by_actor.most_common(1)[0] if enf_by_actor else (None, 0)
    top_enf_share = top_enf[1] / total_enf if total_enf else 0.0
    # Target-laag: op wie valt de geframede sanctie? Vaste minderheid = roof-signatuur;
    # roterend = democratische-zelfbeperking-signatuur (Aoki p.156/p.158).
    tgt_by_target = Counter(tg for (r, actor, label, tg, frm, s) in triples["A"])
    tcounts = [tgt_by_target.get(a, 0) for a in agents]
    target_gini = gini(tcounts)
    n_targets = sum(1 for c in tcounts if c > 0)
    top_tgt = tgt_by_target.most_common(1)[0] if tgt_by_target else (None, 0)
    top_tgt_share = top_tgt[1] / total_enf if total_enf else 0.0
    # Disciplineerbaarheid: is de tophandhaver zélf ooit doelwit van geframede sanctie?
    # (democratisch = de handhaver-rol is zelf onderworpen aan de orde; Aoki p.152/p.161)
    top_enforcer_targeted = bool(top_enf[0]) and tgt_by_target.get(top_enf[0], 0) > 0
    return dict(name=os.path.basename(path), nr=nr,
                neg_rate=rate("A"), pos_rate=rate("B"),
                co_enforce_rate=co_enforce_rate, n_co_enforce=n_co_enforce,
                co_triples=co_triples,
                n_takedrop=n_act["A"], n_investarm=n_act["B"],
                n_neg=n_framed["A"], n_pos=n_framed["B"],
                enforcer_gini=enforcer_gini, n_enforcers=n_enforcers,
                top_enforcer=top_enf[0], top_enf_share=top_enf_share,
                target_gini=target_gini, n_targets=n_targets,
                top_target=top_tgt[0], top_tgt_share=top_tgt_share,
                top_enforcer_targeted=top_enforcer_targeted,
                triples=triples)


def print_single(r, top):
    print(f"\n{r['name']}  ({r['nr']} rondes)")
    print(f"  LENS A  normhandhaving (take/drop als straf) : {r['neg_rate']:.3f}"
          f"   ({r['n_neg']}/{r['n_takedrop']} geframed)")
    print(f"  LENS B  normvervulling (invest/arm als naleving): {r['pos_rate']:.3f}"
          f"   ({r['n_pos']}/{r['n_investarm']} geframed)")
    te = f"{r['top_enforcer']} ({r['top_enf_share']:.0%})" if r['top_enforcer'] else "-"
    tt = f"{r['top_target']} ({r['top_tgt_share']:.0%})" if r['top_target'] else "-"
    # Rol-verdict: top_enf_share (dominantie van EEN handhaver) is de betere
    # discriminator dan Gini -- Gini over de hele populatie vuurt al bij "niet
    # iedereen handhaaft". Maar concentratie alleen scheidt democratisch niet van
    # collusief (Aoki Ch.6, zie analyze()) -> gebruik de target-laag + enforcer-
    # disciplineerbaarheid als tweede as. Alle drempels ONGEKALIBREERD -> gevlagd.
    if not r['n_enforcers']:
        verdict = ""
    elif r['top_enf_share'] >= 0.40 and r['n_enforcers'] <= 4:
        # Er is een concentreerde handhaver-rol. Aoki-discriminator dem. vs collusief:
        if r['top_enforcer_targeted']:
            verdict = "-> concentr. handhaver MAAR zelf disciplineerbaar = rule-of-law/democratisch? [ongekalibreerd]"
        elif r['top_tgt_share'] >= 0.40:
            verdict = "-> concentr. handhaver + vast doelwit + niet zelf gesanctioneerd = COLLUSIEF/roof? [ongekalibreerd]"
        else:
            verdict = "-> concentr. sanctioner-rol; doelwit roteert = ambigu [ongekalibreerd]"
    else:
        verdict = "-> diffuse handhaving (geen duidelijke rol)"
    print(f"  GOV-ROL  handhaving: Gini {r['enforcer_gini']:.3f}, top-share {r['top_enf_share']:.0%}"
          f"   ({r['n_enforcers']} handhaver(s); top {te})   {verdict}")
    disc = "ja (disciplineerbaar)" if r['top_enforcer_targeted'] else "nee"
    print(f"  GOV-TGT  doelwit: Gini {r['target_gini']:.3f}, top-share {r['top_tgt_share']:.0%}"
          f"   ({r['n_targets']} doelwit(en); top {tt}); tophandhaver zelf gesanctioneerd: {disc}")
    for lens, tag in (("A", "NORMHANDHAVING (straf)"), ("B", "NORMVERVULLING (steun)")):
        tr = sorted(r["triples"][lens])
        print(f"\n  --- TOP {top} {tag}-TRIPLES (actie | citaat | ronde) ---")
        for (rd, actor, label, tg, frm, s) in tr[:top]:
            print(f"  R{rd:<2} {actor} {label}->{tg}  | [{frm}]: \"{s}\"")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--null", type=int, default=0, metavar="N",
                   help="target-permutatie-null met N permutaties (DVs.md: 500)")
    args = p.parse_args()
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*_reasoning_live.jsonl")))
        rows = [analyze(f) for f in files]
        rows.sort(key=lambda r: -(r["neg_rate"] + r["pos_rate"]))
        print(f"\nTRIAGE — norm->daad  ({len(rows)} runs in {args.path})\n")
        print(f"  {'run':<50} {'handhaving':>10} {'vervulling':>11}  (#take/drop, #inv/arm)")
        for r in rows:
            print(f"  {r['name'][:49]:<50} {r['neg_rate']:10.3f} {r['pos_rate']:11.3f}"
                  f"   ({r['n_takedrop']},{r['n_investarm']})")
    else:
        print_single(analyze(args.path), args.top)
        if args.null:
            print(f"\n  --- TARGET-PERMUTATIE-NULL ({args.null}x, seed 42) ---")
            allnull = permutation_null(args.path, n_perm=args.null)
            for lens, tag in (("A", "handhaving"), ("B", "vervulling")):
                nres = allnull.get(lens)
                if nres is None:
                    print(f"  LENS {lens} {tag}: geen gerichte acties")
                    continue
                print(f"  LENS {lens} {tag}: obs {nres['observed']:.3f} vs null "
                      f"{nres['null_mean']:.3f}+-{nres['null_sd']:.3f}  "
                      f"z={nres['z']:.1f}  obs>{nres['pctl']:.0f}% van de null "
                      f"(n={nres['n_actions']})")


if __name__ == "__main__":
    main()
