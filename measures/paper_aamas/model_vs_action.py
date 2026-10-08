"""Actieset tegen model: partiele eta2 op rangen (AAMAS, 7 okt 2026; verkennend, toegevoegd na de eerste figuur).

Gedeelde seeds: Gemma (prod_Lk_knife) en Qwen (robust_qwen_Lk_knife), L2--L4, knife-edge, 5 runs per cel.
Twee-weg rang-anova (factor A = level, factor B = model, plus interactie AxB), zelfde methode als prereg_fingerprint.reta;
Plus per model apart: Kruskal-Wallis over de levels (verkennend, n = 5 per level, ongecorrigeerd).
bootstrap 2000 binnen cellen. Maten die pas vanaf een level bestaan, alleen over die levels.
  uv run --python 3.12 --with numpy --with scipy python paper_aamas/model_vs_action.py
"""
import sys, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kruskal
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
from model import anova2

src = open(HIER / "prereg_fingerprint.py").read()
i = src.index("DVS = ["); j = src.index("\n]", i) + 2
ns = {"RUNGS": ("L1", "L2", "L3", "L4")}; exec(src[i:j], ns)
runs = json.loads((HIER / "ablation_per_run.json").read_text())
seed = lambda k: re.search(r"__s(\d+)", k).group(1)
LEVELS = ("L2", "L3", "L4")
cells = {}
for L in LEVELS:
    q = {seed(k): v for k, v in runs.items() if k.startswith(f"robust_qwen_{L}_knife")}
    g = {seed(k): v for k, v in runs.items() if k.startswith(f"prod_{L}_knife_r")}
    shared = sorted(set(q) & set(g))
    cells[(L, "gemma")] = [g[s] for s in shared]; cells[(L, "qwen")] = [q[s] for s in shared]

def _rank(xs):
    o = sorted(range(len(xs)), key=lambda k: xs[k]); r = [0.0] * len(xs); a = 0
    while a < len(xs):
        b = a
        while b + 1 < len(xs) and xs[o[b + 1]] == xs[o[a]]: b += 1
        for k in range(a, b + 1): r[o[k]] = (a + b) / 2 + 1
        a = b + 1
    return r

def reta(c):
    pl = [(a, b, x) for (a, b), xs in c.items() for x in xs]; rk = _rank([x for *_, x in pl]); cr = defaultdict(list)
    for (a, b, _), r in zip(pl, rk): cr[(a, b)].append(r)
    v = anova2(dict(cr)).as_dict()["value"]; return v["A"]["eta2p"], v["B"]["eta2p"], v["AxB"]["eta2p"]

rng = np.random.default_rng(20261007); out = []
for key, label, fam, defined in ns["DVS"]:
    lv = [L for L in LEVELS if L in defined]
    if len(lv) < 2: continue
    c = {(L, m): [r[key] for r in cells[(L, m)]] for L in lv for m in ("gemma", "qwen")}
    a, b, ab = reta(c); A, B = [], []
    for _ in range(2000):
        bc = {k: list(np.array(v)[rng.integers(0, len(v), len(v))]) for k, v in c.items()}
        x, y, _ = reta(bc); A.append(x); B.append(y)
    q = lambda xs: [round(float(np.percentile(xs, 2.5)), 2), round(float(np.percentile(xs, 97.5)), 2)]
    kw = {m: (float(kruskal(*[c[(L, m)] for L in lv]).pvalue) if len({x for L in lv for x in c[(L, m)]}) > 1 else 1.0)
          for m in ("gemma", "qwen")}
    out.append(dict(measure=label, levels=lv, action=round(a, 2), action_ci=q(A), model=round(b, 2), model_ci=q(B),
                    interaction=round(ab, 2), kw_level_p={m: round(v, 3) for m, v in kw.items()}))
    print(f"{label:<26} action {a:.2f} {q(A)}  model {b:.2f} {q(B)}  AxM {ab:.2f}  KW gemma {kw['gemma']:.3f} qwen {kw['qwen']:.3f}")
print("levels differ (KW p<.05):", {m: sum(o["kw_level_p"][m] < .05 for o in out) for m in ("gemma", "qwen")})
print("n per cell:", {f"{k[0]}|{k[1]}": len(v) for k, v in cells.items()})
(HIER / "model_vs_action.json").write_text(json.dumps(out, indent=1))
