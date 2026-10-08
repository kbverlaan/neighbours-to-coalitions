"""Mob size gedeeld door het bereik (AAMAS, 7 okt 2026; verkennend). Take kan alleen op buren,
dus een grotere mob op L3 kan komen doordat herbedraden meer buren geeft. Per run: mob size
(prereg_per_run.json, batch_suite-definitie) en de gemiddelde graad over de rondes (network.edges);
verhouding mob / graad. Mediaan per trede.
  uv run --python 3.12 --with numpy python paper_aamas/mob_per_reach.py
"""
import sys, json
from pathlib import Path
import numpy as np
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset, logs
rows = json.loads((HIER / "prereg_per_run.json").read_text())
out = {}
for L in ("L2", "L3", "L4"):
    mob, deg = [], []
    for c in ("scar", "knife", "abund"):
        vals = rows[f"{L}|{c}"]; paths = runset.cel(f"prod_{L}_{c}")
        for v, p in zip(vals, paths):
            ds = []
            for e in logs.rounds(p):
                ed = (e.get("network") or {}).get("edges") or []; n = len(e.get("agents") or {}) or 30
                ds.append(2 * len(ed) / n)
            mob.append(v["mob"]); deg.append(float(np.mean(ds)) if ds else np.nan)
    mob, deg = np.array(mob), np.array(deg)
    out[L] = dict(runs=len(mob), mob=round(float(np.median(mob)), 2), degree=round(float(np.nanmedian(deg)), 1),
                  mob_per_degree=round(float(np.nanmedian(mob / deg)), 3))
    print(L, out[L])
(HIER / "mob_per_reach.json").write_text(json.dumps(out, indent=1))
