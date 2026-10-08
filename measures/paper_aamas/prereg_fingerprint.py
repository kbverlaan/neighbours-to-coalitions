"""Alle geregistreerde fingerprint-dv's door de hefboomdecompositie (AAMAS, 29 sep 2026).

Instrument: scripts/batch_suite.py + deontic.py + enforcement.py, byte-identiek aan
git-tag prereg-2026-07-21 (gecontroleerd). Geen klassen opgelegd: per dv de rang-eta2
voor capaciteit en prijs (zelfde methode, bootstrap en seed als figures/levers.py) en
per trede-overgang Cliff's delta, gestratificeerd (binnen elke prijscel, dan gemiddeld) met 95%-bootstrapinterval.

  uv run --python 3.12 --with igraph==1.0.0 --with leidenalg==0.11.0 \
    --with numpy==2.3.5 --with scipy==1.17.0 python paper_aamas/prereg_fingerprint.py
"""
import sys, json
from collections import defaultdict
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent; SIM = M.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
sys.path.insert(0, str(SIM / "scripts"))
import numpy as np
import runset, batch_suite
from model import anova2

RUNGS = ("L1", "L2", "L3", "L4"); PAYS = ("scar", "knife", "abund")
RES, SEED = 2000, 20260817
# geregistreerde familie 1 en 2 zoals batch_suite ze levert; defined = treden waar de DEFINITIE van de dv kan
# gelden: een maat die een actie vereist die pas vanaf trede k bestaat, wordt alleen vanaf k ontbonden.
DVS = [
 ("Q",        "club modularity Q",          "F1", RUNGS),
 ("K",        "number of clubs",            "F1", RUNGS),
 ("coh",      "cohesion",                   "F1", RUNGS),
 ("cmean",    "consensus mean",             "F1", RUNGS),
 ("cstd",     "consensus spread",           "F1", RUNGS),
 ("drift",    "action drift",               "F1", RUNGS),
 ("mob",      "mob size",                   "F1", ("L2", "L3", "L4")),
 ("verr",     "intra-club takes (%)",       "F1", ("L2", "L3", "L4")),
 ("rw",       "rewiring rate",              "F1", ("L3", "L4")),
 ("norm_pub", "norm density, public",       "F2", RUNGS),
 ("norm_priv","norm density, private",      "F2", RUNGS),
 ("sanc",     "sanction talk",              "F2", RUNGS),
 ("enf_neg",  "enforcement (negative)",     "F2", ("L2", "L3", "L4")),  # definitie vereist take/drop
 ("enf_pos",  "enforcement (positive)",     "F2", RUNGS),
]

def _rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0]*len(xs); i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[o[j+1]] == xs[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2 + 1
        i = j + 1
    return r

def reta(cells):
    pl = [(a, b, x) for (a, b), xs in cells.items() for x in xs]
    rk = _rank([x for *_, x in pl]); cr = defaultdict(list)
    for (a, b, _), r in zip(pl, rk): cr[(a, b)].append(r)
    v = anova2(dict(cr)).as_dict()["value"]; return v["A"]["eta2p"], v["B"]["eta2p"]

def cliff(a, b):
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[None, :]
    return float(((b > a).sum() - (b < a).sum()) / (a.size * b.size))  # >0: hogere trede hoger

rng = np.random.default_rng(SEED)
def q(xs): return [round(float(np.percentile(xs, 2.5)), 2), round(float(np.percentile(xs, 97.5)), 2)]

def boot_eta(cells):
    keys = list(cells); arr = {k: np.asarray(cells[k], float) for k in keys}; ca, pr = [], []
    for _ in range(RES):
        b = {k: list(arr[k][rng.integers(0, len(arr[k]), len(arr[k]))]) for k in keys}
        a, p = reta(b); ca.append(a); pr.append(p)
    return q(ca), q(pr)

def boot_cliff(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float); ds = []
    for _ in range(RES):
        ds.append(cliff(lo[rng.integers(0, len(lo), len(lo))], hi[rng.integers(0, len(hi), len(hi))]))
    return q(ds)

cache = HIER / "prereg_per_run.json"
if cache.exists():
    rows = {tuple(k.split("|")): v for k, v in json.loads(cache.read_text()).items()}
else:
    rows = {}
    for r in RUNGS:
        for p in PAYS:
            rows[(r, p)] = [batch_suite.analyze(str(x)) for x in runset.cel(f"prod_{r}_{p}")]
    cache.write_text(json.dumps({"|".join(k): v for k, v in rows.items()}))

uit = []
for key, label, fam, defined in DVS:
    cells = {(r, p): [v[key] for v in rows[(r, p)]] for r in defined for p in PAYS}
    a, b = reta(cells); ca, cb = boot_eta(cells)
    steps = {}
    for lo, hi in zip(RUNGS, RUNGS[1:]):
        if lo in defined and hi in defined:
            # gestratificeerd: delta binnen elke prijscel, dan het gemiddelde over de drie cellen
            per = [cliff([v[key] for v in rows[(lo, p)]], [v[key] for v in rows[(hi, p)]]) for p in PAYS]
            ds = []
            for _ in range(RES):
                bs = []
                for p in PAYS:
                    xa = np.asarray([v[key] for v in rows[(lo, p)]], float); xb = np.asarray([v[key] for v in rows[(hi, p)]], float)
                    bs.append(cliff(xa[rng.integers(0, len(xa), len(xa))], xb[rng.integers(0, len(xb), len(xb))]))
                ds.append(np.mean(bs))
            steps[f"{lo}-{hi}"] = {"delta": round(float(np.mean(per)), 2), "ci": q(ds),
                                   "per_price": {p: round(d, 2) for p, d in zip(PAYS, per)}}
    uit.append({"key": key, "label": label, "family": fam, "rungs": list(defined),
                "cap": round(a, 2), "price": round(b, 2), "cap_ci": ca, "price_ci": cb, "steps": steps})
(HIER / "prereg_fingerprint.json").write_text(json.dumps(uit, indent=1))
print(f"{'dv':<26}{'fam':<4}{'cap':>6}{'prijs':>7}   L1-L2   L2-L3   L3-L4")
for u in uit:
    st = "".join(f"{u['steps'].get(s, {'delta': float('nan')})['delta']:>8.2f}" for s in ("L1-L2", "L2-L3", "L3-L4"))
    print(f"{u['label']:<26}{u['family']:<4}{u['cap']:>6.2f}{u['price']:>7.2f}{st}")
