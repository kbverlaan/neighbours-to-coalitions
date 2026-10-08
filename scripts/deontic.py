#!/usr/bin/env python3
"""
deontic.py — NORM-dichtheid marker (no-LLM, snelle triage).

Doel: in EEN oogopslag zien of een run "goud" bevat voor het normen/instituties-
verhaal. Meet hoeveel COLLECTIEF-VOORSCHRIJVENDE taal ("we must hold", "any member
who attacks is exiled") er in de publieke berichten zit -- het taalkundige spoor
van normvorming (Bicchieri: een norm is een collectieve prescriptie, geen
individueel plan).

Kern-onderscheid:
  - "I should attack X"      -> deontisch van VORM, maar individueel EV-redeneren -> GEEN norm
  - "we must hold the line"  -> deontisch + COLLECTIEF referent                   -> WEL norm
  - "any member who attacks will be exiled" -> regel met sanctie                  -> sterke norm

Gemeten op MESSAGES (publiek). Notes zijn privE-intentie -> niet de juiste laag.

Gebruik:
    python scripts/deontic.py RUN.jsonl              # score + geExtraheerde regels van 1 run
    python scripts/deontic.py DIR/                   # triage-ranking over alle runs in een map
    python scripts/deontic.py RUN.jsonl --top 40     # meer regel-zinnen tonen

Het lexicon is bewust expliciet en aanpasbaar -- tune het naar smaak.
"""
import json, re, sys, glob, os, argparse
from collections import defaultdict

# ── LEXICON (Engels, lowercase; \b = woordgrens) ─────────────────────────────
# Deontische modaliteit: verplichting + verbod.
OBLIGATION = [r"must", r"should", r"shall", r"ought", r"have to", r"has to",
              r"need to", r"needs to", r"supposed to", r"required", r"obligated",
              r"mandatory"]
# NB (kalibratie 2026-06-21): bare instrumentele negaties "cannot/can't/can not"
# zijn GESCHRAPT -- ze vuren op niet-normatief EV-redeneren ("I cannot afford to
# attack") en inflateren de deontische telling. De prohibitieve sense ("you cannot
# attack a member") is zeldzaam in dit corpus en wordt door de duidelijker
# prohibitieve termen hieronder + de COLLECTIVE-filter alsnog opgevangen.
PROHIBITION = [r"must not", r"mustn't",
               r"should not", r"shouldn't", r"do not", r"don't", r"never",
               r"forbidden", r"prohibited", r"not allowed", r"may not",
               r"refuse to", r"cease", r"stop attacking", r"no longer"]
# Collectieve / 2e-persoons referent -> tilt een deontische zin naar een NORM.
COLLECTIVE = [r"we", r"us", r"our", r"all of us", r"everyone", r"every member",
              r"members?", r"anyone", r"any agent", r"any member", r"no one",
              r"nobody", r"you", r"each of us", r"the council", r"the pact",
              r"the alliance", r"the group", r"the core", r"the coalition"]
# Sanctie / handhaving (aparte teller -- markeert regels MET tanden).
SANCTION = [r"exile", r"expel", r"banish", r"excluded?", r"targeted by",
            r"punish", r"sanction", r"retaliat", r"cast out", r"enforce",
            r"declaration of war", r"traitor", r"betrayer", r"wolf", r"wolves",
            r"purge", r"kicked", r"war on"]
# MORAL / geInternaliseerde morele kracht (Sugden 1986, pp.147-161; Aoki 2001 Ch.3
# pp.77-78): de norm werkt via schuld/resentment/ought/afkeuring, NIET via externe
# sanctie. Aoki splitst naar handhavings-AGENT (Table 3.1: self / partner / community /
# third party):
#   - FIRST-PARTY = in het zelf, geInternaliseerd uit gewoonte: "violating the rules ...
#     should evoke autonomously negative moral sentiments and emotions, such as guilt,
#     shame, or anxiety, within the agent's mind" (p.78); "thou shalt not steal" als
#     ge-ought (p.77). Dit is de Sugden-internalisatie-lezing.
#   - SECOND/THIRD-PARTY = gericht op de overtreder: resentment/blame/censure door
#     partner of gemeenschap (Aoki Table 3.1; Sugden: resentment sustains the norm).
# Twee tellers -> scheidt geInternaliseerde morele norm (privE, zelf-referentieel) van
# gericht-veroordelende handhavingstaal. MORAL = union (backward-compat teller).
MORAL_FIRST = [r"ought", r"guilt\w*", r"ashamed", r"shame\w*", r"conscience",
               r"duty", r"obligation", r"morally", r"immoral", r"wrong to",
               r"dishonou?rable"]
MORAL_OTHER = [r"resent\w*", r"blame\w*", r"deserv\w*", r"condemn\w*", r"censure",
               r"disgrace\w*", r"unfair", r"unjust", r"betray\w*"]
MORAL = MORAL_FIRST + MORAL_OTHER
# NB: de aparte "regel-introductie"-teller (RULE_INTRO: pact/those who/threshold/
# protocol/...) is GESCHRAPT -- matchte te breed en bleek ruis (zie DVs.md). De
# betrouwbare maten zijn norm-dichtheid (deontisch + collectief, Bicchieri) en de
# sanctie-telling. Precisie van het lexicon is niet gevalideerd op een handgelabelde
# sample -> rapporteer deze als "lexicale markers", niet als directe norm-meting.


def _rx(words):
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I)

RX_OBL, RX_PRO = _rx(OBLIGATION), _rx(PROHIBITION)
RX_COL, RX_SAN = _rx(COLLECTIVE), _rx(SANCTION)
RX_MOR = _rx(MORAL)
RX_MOR_F, RX_MOR_O = _rx(MORAL_FIRST), _rx(MORAL_OTHER)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def classify(sent):
    """Geef terug welke labels op een zin van toepassing zijn."""
    deontic = bool(RX_OBL.search(sent) or RX_PRO.search(sent))
    collective = bool(RX_COL.search(sent))
    return {
        "deontic": deontic,                       # deontisch van vorm
        "norm": deontic and collective,           # KERN: collectieve prescriptie
        "sanction": bool(RX_SAN.search(sent)),
        "moral": bool(RX_MOR.search(sent)),       # Sugden: geInternaliseerde morele kracht (union)
        "moral_first": bool(RX_MOR_F.search(sent)),   # Aoki first-party (guilt/shame/ought, zelf)
        "moral_other": bool(RX_MOR_O.search(sent)),   # Aoki second/third-party (resent/blame/censure, gericht)
    }


def _scan(items, nr):
    """items = lijst van (round, from, text). Geeft tellers, per-ronde-norm en regels terug.
    Gebruikt voor zowel de PUBLIEKE laag (messages) als de PRIVE laag (memory-notes)."""
    n_unit = n_sent = 0
    c = defaultdict(int)
    per_round_norm = defaultdict(lambda: [0, 0])
    rules = []
    for rnum, frm, txt in items:
        txt = (txt or "").strip()
        if not txt:
            continue
        n_unit += 1
        for s in SENT_SPLIT.split(txt):
            s = s.strip()
            if len(s) < 4:
                continue
            n_sent += 1
            lab = classify(s)
            per_round_norm[rnum][1] += 1
            for k, v in lab.items():
                if v:
                    c[k] += 1
            if lab["norm"]:
                per_round_norm[rnum][0] += 1
            if lab["norm"] or (lab["sanction"] and lab["deontic"]):
                rules.append((lab["norm"] + lab["sanction"], rnum, frm, s))
    dens = lambda k: c[k] / n_sent if n_sent else 0.0
    thirds = [[0, 0], [0, 0], [0, 0]]
    for rnum, (nn, tt) in per_round_norm.items():
        idx = 0 if rnum <= nr / 3 else (1 if rnum <= 2 * nr / 3 else 2)
        thirds[idx][0] += nn; thirds[idx][1] += tt
    temporal = [(t[0] / t[1] if t[1] else 0.0) for t in thirds]
    return dict(n_unit=n_unit, n_sent=n_sent, deontic=dens("deontic"), norm=dens("norm"),
                sanction=c["sanction"], moral=dens("moral"),
                moral_first=dens("moral_first"), moral_other=dens("moral_other"),
                temporal=temporal, rules=sorted(rules, reverse=True))


def analyze(path):
    rounds = [json.loads(l) for l in open(path) if l.strip()]
    nr = len(rounds)
    # PUBLIEKE laag: messages (waar normen worden afgekondigd)
    msgs = [(d.get("round") or 0, m.get("from"), m.get("text"))
            for d in rounds for m in (d.get("messages") or [])]
    # PRIVE laag: memory-notes (waar normen worden geINTERNALISEERD)
    notes = [(d.get("round") or 0, aid, info.get("memory"))
             for d in rounds for aid, info in (d.get("agents") or {}).items()]
    pub = _scan(msgs, nr)
    priv = _scan(notes, nr)
    return dict(name=os.path.basename(path), nr=nr, pub=pub, priv=priv,
                # platte velden = publieke laag (triage-default, backwards-compatible)
                n_msg=pub["n_unit"], n_sent=pub["n_sent"],
                deontic=pub["deontic"], norm=pub["norm"],
                sanction=pub["sanction"],
                temporal=pub["temporal"], rules=pub["rules"])


def print_single(r, top):
    pub, priv = r['pub'], r['priv']
    print(f"\n{r['name']}  ({r['nr']} rondes)")
    print(f"  {'laag':<22}{'norm':>7}{'deon':>7}{'moral':>7}{'sanc':>6}   temporeel(v/m/l)")
    for tag, L in (("PUBLIEK (messages)", pub), ("PRIVE (memory-notes)", priv)):
        tv = L['temporal']
        print(f"  {tag:<22}{L['norm']:7.3f}{L['deontic']:7.3f}{L['moral']:7.3f}{L['sanction']:6d}"
              f"   {tv[0]:.2f}/{tv[1]:.2f}/{tv[2]:.2f}")
    # internalisering-ratio: privE-norm / publiek-norm -> wordt de afkondiging ook geleefd?
    ratio = priv['norm'] / pub['norm'] if pub['norm'] else 0.0
    print(f"  internalisering (privE-norm / publiek-norm): {ratio:.2f}"
          f"   {'(>=1: norm leeft in privE-redenering)' if ratio >= 0.8 else '(<1: vooral afgekondigd, minder geleefd)'}")
    # Sugden-internalisatie: moreel-evaluatieve taal in de PRIVE laag (schuld/ought/resentment)
    # = de geInternaliseerde-morele-kracht-lezing, los van externe sanctie.
    # Aoki-split (Ch.3 pp.77-78): first-party (guilt/shame/ought, geInternaliseerd) vs
    # second/third-party (resent/blame/censure, gericht op de overtreder).
    print(f"  Sugden-moraal (privE moral-dichtheid): {priv['moral']:.3f}"
          f"  [first-party {priv['moral_first']:.3f} / 2e-3e-party {priv['moral_other']:.3f}]"
          f"   (geInternaliseerde morele kracht; vgl publiek {pub['moral']:.3f})")
    print(f"\n  --- TOP {top} PUBLIEKE REGEL/NORM-ZINNEN (het 'wetboek') ---")
    for score, rnum, frm, s in pub['rules'][:top]:
        print(f"  R{rnum:<2} {frm}: \"{s[:140]}\"")
    print(f"\n  --- TOP {min(top, 10)} PRIVE NORM-NOTES (internalisering) ---")
    for score, rnum, frm, s in priv['rules'][:min(top, 10)]:
        print(f"  R{rnum:<2} {frm}: \"{s[:140]}\"")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="een RUN.jsonl of een map met *_reasoning_live.jsonl")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*_reasoning_live.jsonl")))
        rows = [analyze(f) for f in files]
        rows.sort(key=lambda r: -r["norm"])
        print(f"\nTRIAGE-ranking op NORM-dichtheid  ({len(rows)} runs in {args.path})\n")
        print(f"  {'run':<48} {'pub-norm':>8} {'priv-norm':>9} {'sanc':>5}  temporeel(v/m/l)")
        for r in rows:
            tv = r['temporal']
            print(f"  {r['name'][:47]:<48} {r['norm']:8.3f} {r['priv']['norm']:9.3f} "
                  f"{r['sanction']:5d}  {tv[0]:.2f}/{tv[1]:.2f}/{tv[2]:.2f}")
    else:
        print_single(analyze(args.path), args.top)


if __name__ == "__main__":
    main()
