"""Permutatietoets voor het effect van de actieset per maat, Holm-gecorrigeerd (AAMAS, 7 okt 2026).

Toegevoegd na de eerste figuur. Per maat: de treden-labels worden geschud BINNEN elke prijscel
(de prijs blijft vast), toetsgrootheid = som over prijscellen van de Kruskal-Wallis-H over de treden.
5000 permutaties, p = (k+1)/(n+1). Holm over de veertien maten. Maten die pas vanaf trede k bestaan,
alleen over de treden waar ze bestaan (zelfde regel als de decompositie).
  uv run --python 3.12 --with numpy --with scipy python paper_aamas/perm_test.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HIER = Path(__file__).resolve().parent
src = open(HIER / "prereg_fingerprint.py").read()
i = src.index("DVS = ["); j = src.index("\n]", i) + 2
ns = {"RUNGS": ("L1", "L2", "L3", "L4")}; exec(src[i:j], ns)
PAYS = ("scar", "knife", "abund"); N, SEED = 5000, 20261007
rows = {tuple(k.split("|")): v for k, v in json.loads((HIER / "prereg_per_run.json").read_text()).items()}
rng = np.random.default_rng(SEED)

def H(x, g):
    r = rankdata(x); n = len(r); out = 0.0
    for u in np.unique(g):
        m = g == u; out += m.sum() * (r[m].mean() - (n + 1) / 2) ** 2
    return 12 / (n * (n + 1)) * out

res = []
for key, label, fam, defined in ns["DVS"]:
    strata = []
    for p in PAYS:
        x = []; g = []
        for k, r in enumerate(defined):
            v = [row[key] for row in rows[(r, p)]]; x += v; g += [k] * len(v)
        strata.append((np.array(x, float), np.array(g)))
    obs = sum(H(x, g) for x, g in strata)
    cnt = 0
    for _ in range(N):
        cnt += sum(H(x, rng.permutation(g)) for x, g in strata) >= obs - 1e-12
    res.append({"measure": label, "rungs": list(defined), "H": round(obs, 2), "p": (cnt + 1) / (N + 1)})
order = sorted(range(len(res)), key=lambda k: res[k]["p"]); m = len(res); run = 0.0
for rank, k in enumerate(order):
    run = max(run, min(1.0, (m - rank) * res[k]["p"])); res[k]["p_holm"] = round(run, 5)
for r in res: print(f"{r['measure']:<28} H {r['H']:7.2f}  p {r['p']:.4f}  Holm {r['p_holm']:.4f}")
print("significant after Holm (.05):", sum(r["p_holm"] < .05 for r in res), "of", m)
(HIER / "perm_test.json").write_text(json.dumps(res, indent=1))
