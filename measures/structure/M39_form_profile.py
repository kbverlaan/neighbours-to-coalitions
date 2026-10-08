"""M39 · The form profile: four arrangements, each counted separately (ledger S37).

Scope   L1-L4, all twelve cells, per run over all sixty rounds.

Four forms, and the reason they are counted separately is substantive: none may
be derived from another. A cell that has dyads need not have cycles, and a named
agreement says nothing about whether it is enforced.

**Dyad** — an ordered pair A→B whose reverse B→A also occurs at least once.
Counted separately over `transfer` and over `strengthen`, then summed. This is a
count over the whole run and is not the same quantity as the late mutual pairs
in S19.

**Cycle** — a closed triangle A→B→C→A in the transfer graph, where only edges
used three times or more count. The threshold keeps accidental triangles out.

**Named shared representation** — a capitalised term of two to four words, used
by three or more different agents in public messages. The detector is taken
verbatim from the hand-validated original, including its cleanup: capitals from a
sentence start do not count, edge articles drop away, subsequences fold into the
longest form, and abbreviations link through the introduction sentence "Full
Name (ABBR)". That logic was hand-validated and is therefore preserved rather
than rebuilt.

**Executed collective sanction** — an agent named grammatically as the subject
of a violation verb, followed within three rounds by two or more agents
attacking it. Reported as a cell share, not as a per-run mean: it is an event
that occurs or does not.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import product

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

import runset
from base import log_path, rounds

AGENTS = {"Bronze", "Rust", "Silver", "Sage", "Indigo", "Cyan", "Violet", "Plum",
          "Pearl", "Mauve", "Ash", "Jade", "Onyx", "Storm", "Slate", "Olive",
          "Coral", "Dusk", "Teal", "Blue", "Cobalt", "Gold", "Green", "Maroon",
          "Copper", "Scarlet", "Amber", "Crimson", "Ivory", "Red"}
W = r"[A-Z][a-zA-Z-]*[a-zA-Z]"
NG = re.compile(rf"\b({W}(?:\s+(?:of\s+|the\s+)?{W}){{1,3}})\b")
ABBR = re.compile(rf"\b({W}(?:\s+{W}){{1,3}})\s*\(([A-Z]{{2,5}})\)")
SENT = re.compile(r"(?:^|[.!?;:]\s+|\n\s*|[-*]\s+|\"\s*)$")
EDGE = {"the", "a", "an", "to", "in", "of", "for", "on", "at", "and", "if"}
VIOL = re.compile(r"\b(broke|violat\w*|betray\w*|breach\w*|traitor|defect\w*)\b", re.I)
CYCLUS_DREMPEL = 3
SANCTIE_VENSTER = 3


def _clean(c: str) -> str:
    w = c.split()
    while w and w[0].lower() in EDGE:
        w = w[1:]
    while w and w[-1].lower() in EDGE:
        w = w[:-1]
    return " ".join(w)


def _benoemd(rs: list[dict]) -> dict:
    """Taken verbatim from the hand-validated original."""
    users, hits, first, abbr = defaultdict(set), defaultdict(int), {}, {}
    for d in rs:
        for m in d.get("messages") or []:
            for full, a in ABBR.findall(m.get("text") or ""):
                abbr.setdefault(a, _clean(full))
    for d in rs:
        r = d.get("round")
        for m in d.get("messages") or []:
            t, who = m.get("text") or "", m.get("from")
            found = set()
            for mt in NG.finditer(t):
                c = mt.group(1)
                if SENT.search(t[:mt.start()]):
                    c = " ".join(c.split()[1:])
                c = _clean(c)
                if len(c.split()) < 2:
                    continue
                if c in AGENTS or any(w in AGENTS for w in c.split()):
                    continue
                found.add(c)
            for a, full in abbr.items():
                if re.search(rf"\b{a}\b", t):
                    found.add(full)
            for c in found:
                users[c].add(who); hits[c] += 1; first.setdefault(c, r)
    for c in sorted(users, key=lambda x: -len(x.split())):
        if c not in users:
            continue
        host = next((k for k in users if k != c and f" {c} " in f" {k} "), None)
        if host:
            users[host] |= users[c]; hits[host] += hits[c]
            first[host] = min(first[host], first[c]); del users[c]
    return {c: len(u) for c, u in users.items() if len(u) >= 3}


def _run(p) -> dict:
    rs = rounds(log_path(p))
    tr, st = Counter(), Counter()
    aanval: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    beschuldigd: list[tuple[int, str]] = []
    for d in rs:
        r = d.get("round")
        for nm, a in (d.get("agents") or {}).items():
            act, tgt = a.get("action"), a.get("target")
            if act == "transfer" and tgt:
                tr[(nm, tgt)] += 1
            elif act == "strengthen" and tgt:
                st[(nm, tgt)] += 1
            elif act == "take" and tgt:
                aanval[r][tgt].add(nm)
        for m in d.get("messages") or []:
            t = m.get("text") or ""
            if VIOL.search(t):
                for naam in AGENTS:
                    if re.search(rf"\b{naam}\b[^.]{{0,40}}{VIOL.pattern}", t, re.I):
                        beschuldigd.append((r, naam))

    dyades = (sum(1 for (a, b) in tr if (b, a) in tr and a < b)
              + sum(1 for (a, b) in st if (b, a) in st and a < b))

    sterk = {(a, b) for (a, b), n in tr.items() if n >= CYCLUS_DREMPEL}
    cycli = sum(1 for a, b in sterk for c in {y for x, y in sterk if x == b}
                if (c, a) in sterk and a < b < c)

    sanctie = any(len(aanval.get(r + d, {}).get(naam, ())) >= 2
                  for r, naam in beschuldigd for d in range(SANCTIE_VENSTER + 1))

    return {"dyades": dyades, "cycli": cycli,
            "benoemd": len(_benoemd(rs)), "sanctie": bool(sanctie)}


def compute() -> dict:
    uit = {}
    for rung, payoff in product(("L1", "L2", "L3", "L4"), ("scar", "knife", "abund")):
        naam = f"prod_{rung}_{payoff}"
        rijen = [_run(p) for p in runset.cel(naam)]
        n = len(rijen)
        uit[naam] = {
            "n": n,
            "dyades": round(sum(r["dyades"] for r in rijen) / n, 1),
            "cycli": round(sum(r["cycli"] for r in rijen) / n, 1),
            "benoemd": round(sum(r["benoemd"] for r in rijen) / n, 1),
            "sanctie_pct": round(100 * sum(r["sanctie"] for r in rijen) / n, 0),
            "per_run": rijen,
        }
    return uit


if __name__ == "__main__":
    try:
        res = compute()
    except runset.RunsetError as e:
        sys.exit(f"RUNSET: {e}")
    if "--json" in sys.argv:
        print(json.dumps(res, indent=1)); raise SystemExit
    print("M39 · form profile, mean per run\n")
    print(f"{'cell':18}{'n':>3}{'dyads':>9}{'cycles':>8}{'named':>10}{'sanction':>10}")
    for naam, r in res.items():
        print(f"  {naam:16}{r['n']:3}{r['dyades']:9.1f}{r['cycli']:8.1f}"
              f"{r['benoemd']:10.1f}{r['sanctie_pct']:9.0f}%")
