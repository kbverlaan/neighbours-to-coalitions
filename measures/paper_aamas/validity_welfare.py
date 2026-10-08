"""Geldigheid van de acties en welvaart per arm (AAMAS, 8 okt 2026; verkennend, na review).

Per agent-beurt uit de logs: fallback (engine viel terug op een standaardactie), errors (niet-lege foutlijst),
any_retry (minstens een nieuwe poging nodig), recovered_action (actie hersteld uit een kapotte respons).
Welvaart: gemiddelde eindholding per run (mediaan over runs). Per level (Gemma, alle prijzen) en per arm (knife-edge).
  uv run --python 3.12 --with numpy python paper_aamas/validity_welfare.py
"""
import sys, json
from pathlib import Path
import numpy as np
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset

def run(p):
    n = fb = er = rt = rc = 0; last = None
    for line in open(p):
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        last = d
        for x in d["agents"].values():
            n += 1; fb += bool(x.get("fallback")); er += bool(x.get("errors")); rt += bool(x.get("any_retry"))
            rc += x.get("recovered_action") not in (None, False, "")
    R = [x["resources"] for x in last["agents"].values()]
    return dict(turns=n, fallback=fb, errors=er, retry=rt, recovered=rc, mean_final=float(np.mean(R)))

def summ(paths):
    rs = [run(p) for p in paths]; T = sum(r["turns"] for r in rs)
    return dict(runs=len(rs), turns=T, **{k: sum(r[k] for r in rs) / T for k in ("fallback", "errors", "retry", "recovered")},
                mean_final_med=float(np.median([r["mean_final"] for r in rs])))
out = {"levels": {}, "arms": {}}
for L in ("L1", "L2", "L3", "L4"):
    out["levels"][L] = summ([p for c in ("scar", "knife", "abund") for p in runset.cel(f"prod_{L}_{c}")])
    a = {"gemma": summ(runset.cel(f"prod_{L}_knife")), "silent": summ(runset.cel(f"prod_{L}_knife_nocomm"))}
    if L != "L1": a["qwen"] = summ(runset.cel(f"robust_qwen_{L}_knife"))
    out["arms"][L] = a
(HIER / "validity_welfare.json").write_text(json.dumps(out, indent=1))
for k, v in out["levels"].items(): print("LEVEL", k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()})
for L, a in out["arms"].items():
    for k, v in a.items(): print(L, k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()})
