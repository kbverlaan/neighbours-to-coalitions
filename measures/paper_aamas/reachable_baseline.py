"""Jacht op de rijksten tegen een kansbasis over BEREIKBARE doelwitten (AAMAS, 7 okt 2026; verkennend).

Waargenomen: aandeel gevechten waarin het doelwit bij de drie rijksten hoorde, gelezen van het bord
vóór de ronde (zelfde definitie als figures/channel.py::_hit_the_richest).
Kans: een aanval kan alleen op een buur. Per gevecht, per aanvaller: aandeel van de drie rijksten
onder diens levende buren op het bord vóór de ronde; gemiddeld over aanvallers en gevechten.
Ook de oude basis (3 / aantal levenden). Per trede, alle prijscellen, plus stille runs en Qwen.
  uv run --python 3.12 python paper_aamas/reachable_baseline.py
"""
import sys, json
from collections import defaultdict
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset, logs

def run(p):
    rs = logs.rounds(p); per = {e.get("round"): e for e in rs}
    hit = n = 0; reach = []; flat = []
    for e in rs:
        prev = per.get((e.get("round") or 0) - 1) or e
        ag = prev.get("agents") or e.get("agents") or {}
        alive = {nm for nm, a in ag.items() if (a.get("resources") or 0) > 0}
        top3 = {nm for nm, _ in sorted(ag.items(), key=lambda kv: -(kv[1].get("resources") or 0))[:3]}
        nb = defaultdict(set)
        for a, b in ((prev.get("network") or {}).get("edges") or []):
            nb[a].add(b); nb[b].add(a)
        for c in e.get("combat") or []:
            if not isinstance(c, dict) or not c.get("defender"): continue
            n += 1; hit += c["defender"] in top3
            ps = []
            for at in c.get("attackers") or []:
                cand = (nb[at] & alive) - {at}
                if cand: ps.append(len(cand & top3) / len(cand))
            if ps: reach.append(sum(ps) / len(ps))
            flat.append(len(top3 - set(c.get("attackers") or [])) / max(len(alive) - 1, 1))
    return hit, n, reach, flat

out = {}
for lab, cells in [(f"{L} Gemma", [f"prod_{L}_{c}" for c in ("scar", "knife", "abund")]) for L in ("L2", "L3", "L4")] + \
                  [(f"{L} silent", [f"prod_{L}_knife_nocomm"]) for L in ("L2", "L3")] + \
                  [(f"{L} Qwen", [f"robust_qwen_{L}_knife"]) for L in ("L2", "L3")]:
    H = N = 0; R = []; F = []
    for c in cells:
        for p in runset.cel(c):
            h, n, r, f = run(p); H += h; N += n; R += r; F += f
    if N:
        out[lab] = dict(fights=N, observed=round(100 * H / N, 1),
                        chance_reachable=round(100 * sum(R) / len(R), 1) if R else None,
                        chance_all=round(100 * sum(F) / len(F), 1))
        print(lab, out[lab])
(HIER / "reachable_baseline.json").write_text(json.dumps(out, indent=1))
