"""Cliff's delta voor de twee ablaties, op de geregistreerde fingerprint (AAMAS, 29 sep 2026).

Kanaal: sprekend (prod_Lk_knife) -> stil (prod_Lk_knife_nocomm), op gedeelde seeds waar mogelijk.
  Alleen maten waarvan de definitie geen publieke berichten gebruikt (constructieregel):
  publieke normtaal, sanctietaal en handhaving zijn zonder kanaal per definitie nul, en de
  clubdetector gebruikt het berichtengraaf voor de helft -> niet vergelijkbaar.
Model: Gemma -> Qwen op de vijf gedeelde seeds per trede (L2--L4); alle veertien maten.
Instrument: scripts/batch_suite.py, identiek aan tag prereg-2026-07-21.
"""
import sys, json, re
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent; SIM = M.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
sys.path.insert(0, str(SIM / "scripts"))
import numpy as np
import runset, batch_suite
RES, SEED = 2000, 20260817
rng = np.random.default_rng(SEED)
seed = lambda p: re.search(r"__s(\d+)", p.name).group(1)

ALL14 = [("Q","club modularity Q"),("K","number of clubs"),("coh","cohesion"),("cmean","consensus mean"),
         ("cstd","consensus spread"),("drift","action drift"),("mob","mob size"),("verr","intra-club takes (%)"),
         ("rw","rewiring rate"),("norm_pub","norm density, public"),("norm_priv","norm density, private"),
         ("sanc","sanction talk"),("enf_neg","enforcement (negative)"),("enf_pos","enforcement (positive)")]
CHANNEL_OK = {"cmean","cstd","drift","mob","rw","norm_priv"}
FROM = {"mob":"L2","verr":"L2","enf_neg":"L2","rw":"L3"}  # constructieregel voor treden
ORDER = ("L1","L2","L3","L4")

def cliff(a, b):
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[None, :]
    return float(((b > a).sum() - (b < a).sum()) / (a.size * b.size))
def boot(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [cliff(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(RES)]
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]

cache = HIER / "ablation_per_run.json"
C = json.loads(cache.read_text()) if cache.exists() else {}
def vals(p):
    if p.name not in C: C[p.name] = batch_suite.analyze(str(p))
    return C[p.name]

def compare(base, alt, keys, rungs):
    out = {}
    for k, lab in keys:
        out[k] = {"label": lab}
        for r in rungs:
            if ORDER.index(r) < ORDER.index(FROM.get(k, "L1")): continue
            b = [vals(p)[k] for p in base[r]]; a = [vals(p)[k] for p in alt[r]]
            out[k][r] = {"delta": round(cliff(b, a), 2), "ci": boot(b, a), "n": [len(b), len(a)]}
    return out

chan_base, chan_alt, mod_base, mod_alt = {}, {}, {}, {}
for r in ORDER:
    silent = runset.cel(f"prod_{r}_knife_nocomm"); ss = {seed(p) for p in silent}
    speak = runset.cel(f"prod_{r}_knife"); shared = [p for p in speak if seed(p) in ss]
    chan_base[r] = shared if len(shared) >= 5 else speak; chan_alt[r] = silent
for r in ("L2","L3","L4"):
    q = runset.cel(f"robust_qwen_{r}_knife"); qs = {seed(p) for p in q}
    mod_base[r] = [p for p in runset.cel(f"prod_{r}_knife") if seed(p) in qs]; mod_alt[r] = q

res = {"channel": compare(chan_base, chan_alt, [x for x in ALL14 if x[0] in CHANNEL_OK], ORDER),
       "model": compare(mod_base, mod_alt, ALL14, ("L2","L3","L4")),
       "n_channel_base": {r: len(v) for r, v in chan_base.items()}}
cache.write_text(json.dumps(C)); (HIER / "ablations.json").write_text(json.dumps(res, indent=1))
for name, rungs in (("channel", ORDER), ("model", ("L2","L3","L4"))):
    print(f"\n== {name} (delta: >0 = hoger na de ingreep)  basis-n {res.get('n_channel_base') if name=='channel' else 5}")
    for k, v in res[name].items():
        cells = "".join(f"{v[r]['delta']:>+7.2f}{'*' if not (v[r]['ci'][0] <= 0 <= v[r]['ci'][1]) else ' '}" if r in v else "      - " for r in rungs)
        print(f"  {v['label']:<26}{cells}")
