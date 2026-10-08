"""Verder dan de rekensom (AAMAS, 7 okt 2026). Verkennend, toegevoegd na de eerste figuur.

Per aanvaller-actie: zou deze aanval ALLEEN verlies opleveren? Verwachte waarde van een losse aanval
(Eq. take-ev): alpha*(r-1)/(r+1) - c_atk*r, met r = eigen kracht / kracht verdediger (bevroren vóór het gevecht).
Telt het aandeel alleen-verlieslatende aanvallen en welk deel daarvan in coalitie (>= 2 aanvallers) gebeurt.
Plus: 'whale' / 'whale hunt' in de tekst per trede, en per L4-run: gevechten en overleving van de voorraad.

Leest alle canonieke runs via runset.cel (reasoning_live, of de compacte _log als die er is).
  uv run --python 3.12 python paper_aamas/beyond_arithmetic.py
"""
import sys, json, re
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset

ALPHA, C_ATK = 0.35, 0.02

def rounds(p):
    for line in open(p):
        try: yield json.loads(line)
        except json.JSONDecodeError: continue  # afgebroken laatste regel

def attacks(paths):
    n = lose = lose_coal = coal = fights = fights_lose = 0
    for p in paths:
        for d in rounds(p):
            for f in d.get("combat") or []:
                fights += 1
                k = len(f["attackers"]); dp = f["defender_power"] or 1e-9
                fights_lose += all(ALPHA * (pw / dp - 1) / (pw / dp + 1) - C_ATK * pw / dp < 0 for pw in f["attacker_powers"].values())
                for _, pw in f["attacker_powers"].items():
                    r = pw / dp
                    n += 1; coal += k >= 2
                    if ALPHA * (r - 1) / (r + 1) - C_ATK * r < 0:
                        lose += 1; lose_coal += k >= 2
    return dict(runs=len(paths), acts=n, lone_losing=lose / max(n, 1),
                lone_losing_in_coalition=lose_coal / max(lose, 1), coalition=coal / max(n, 1),
                fights_per_run=fights / max(len(paths), 1), fights_lone_losing=fights_lose / max(fights, 1))

WHALE = re.compile(r"whale", re.I); HUNT = re.compile(r"whale[- ]?hunt", re.I)
def words(paths):
    w = h = 0
    for p in paths:
        t = Path(p).read_text(errors="ignore")
        w += bool(WHALE.search(t)); h += bool(HUNT.search(t))
    return dict(runs=len(paths), whale=w, whale_hunt=h)

def stock(paths):
    out = []
    for p in paths:
        fights = 0; smin = None
        for d in rounds(p):
            fights += len(d.get("combat") or [])
            s = (d.get("commons") or {}).get("stock") if isinstance(d.get("commons"), dict) else None
            if s is not None: smin = s if smin is None else min(smin, s)
        out.append((fights, smin))
    return out

res = {}
for L in ("L2", "L3", "L4"):
    prod = [x for c in ("scar", "knife", "abund") for x in runset.cel(f"prod_{L}_{c}")]
    sil = runset.cel(f"prod_{L}_knife_nocomm")
    qw = runset.cel(f"robust_qwen_{L}_knife")
    knife = runset.cel(f"prod_{L}_knife")
    res[L] = {"gemma": attacks(prod), "gemma_knife": attacks(knife), "silent": attacks(sil), "qwen": attacks(qw),
              "words_gemma": words(prod)}
for L, r in res.items():
    for k in ("gemma", "gemma_knife", "silent", "qwen"):
        a = r[k]
        print(f"{L} {k:6} runs {a['runs']:2} acts {a['acts']:5} fights/run {a['fights_per_run']:6.1f} "
              f"lone-losing acts {a['lone_losing']:4.0%} fights {a['fights_lone_losing']:4.0%}")
    print(f"{L} words {r['words_gemma']}")
(HIER / "beyond_arithmetic.json").write_text(json.dumps(res, indent=1))
