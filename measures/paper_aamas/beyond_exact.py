"""Exacte rekensom per gevecht (AAMAS, 8 okt 2026; verkennend, na review).

Twee lezingen van 'geen aanvaller kon dit doelwit alleen met winst aanvallen':
 (a) benchmark, Eq. 8: alpha*(r-1)/(r+1) - c*r, r = kracht aanvaller / kracht verdediger;
 (b) exact: alpha*(s_i - s_j)/(s_i + s_j) - c*R_i/R_j, met de kosten op de eigen holding (R = holding aan het
     eind van de vorige ronde) en kracht s inclusief bewapening.
Per arm: aandeel gevechten waarin elke aanvaller alleen verlies verwacht (alle gevechten), en het aandeel
gevechten met >= 2 aanvallers waarin elke aanvaller alleen verlies verwacht (gezamenlijk voorbij de rekensom).
  uv run --python 3.12 python paper_aamas/beyond_exact.py
"""
import sys, json, re
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset
A, C = 0.35, 0.02
def rounds(p):
    for line in open(p):
        try: yield json.loads(line)
        except json.JSONDecodeError: continue
def arm(paths):
    n = bench = exact = multi = jb = je = 0
    for p in paths:
        prev = None
        for d in rounds(p):
            for f in d.get("combat") or []:
                sj = f["defender_power"] or 1e-9; k = len(f["attackers"]); n += 1; multi += k >= 2
                b = all(A * (si / sj - 1) / (si / sj + 1) - C * si / sj < 0 for si in f["attacker_powers"].values())
                if prev:
                    Rj = prev.get(f["defender"], {}).get("resources") or 1e-9
                    e = all(A * (si - sj) / (si + sj) - C * (prev.get(a, {}).get("resources") or 0) / Rj < 0
                            for a, si in f["attacker_powers"].items())
                else: e = b
                bench += b; exact += e; jb += b and k >= 2; je += e and k >= 2
            prev = d["agents"]
    return dict(fights=n, bench=bench / n, exact=exact / n, multi=multi / n, joint_bench=jb / n, joint_exact=je / n) if n else dict(fights=0)
seed = lambda p: re.search(r"__s(\d+)", str(p)).group(1)
out = {}
for L in ("L2", "L3"):
    g = runset.cel(f"prod_{L}_knife"); q = runset.cel(f"robust_qwen_{L}_knife"); qs = {seed(p) for p in q}
    out[L] = {"gemma": arm(g), "gemma_shared": arm([p for p in g if seed(p) in qs]), "qwen": arm(q),
              "silent": arm(runset.cel(f"prod_{L}_knife_nocomm")),
              "gemma_all_prices": arm([p for c in ("scar", "knife", "abund") for p in runset.cel(f"prod_{L}_{c}")])}
(HIER / "beyond_exact.json").write_text(json.dumps(out, indent=1))
for L, a in out.items():
    for k, v in a.items(): print(L, f"{k:16}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})
