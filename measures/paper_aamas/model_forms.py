"""Vormen onder het tweede model (AAMAS, 7 okt 2026; verkennend, toegevoegd na de eerste figuur).

Gedeelde seeds, knife-edge: Gemma (prod_Lk_knife) tegen Qwen (robust_qwen_Lk_knife), L2--L4.
Per run: giften, wederkerige paren (beide richtingen minstens één gift), gevechten, Gini van de eindbezit,
agents onder 10 aan het eind, en de mediaan van de buren van de rijkste agent. Gini ook over alle 15 Gemma-knife-runs.
  uv run --python 3.12 --with numpy python paper_aamas/model_forms.py
"""
import sys, json, re
from collections import Counter
from pathlib import Path
import numpy as np
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset

def rounds(p):
    for line in open(p):
        try: yield json.loads(line)
        except json.JSONDecodeError: continue

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float(((2 * np.arange(n) - n + 1) * x).sum() / (n * x.sum())) if x.sum() else 0.0

def run(p):
    dy = Counter(); fights = 0; last = None
    for d in rounds(p):
        last = d; fights += len(d.get("combat") or [])
        for a, x in d["agents"].items():
            if x["action"] in ("transfer", "invest_other") and x["target"]: dy[(a, x["target"])] += 1
    R = {a: x["resources"] for a, x in last["agents"].items()}
    top = max(R, key=R.get)
    nb = [R.get(b if a == top else a, 0) for a, b in last["network"]["edges"] if top in (a, b)]
    return dict(transfers=sum(dy.values()), pairs=sum(1 for (a, b) in dy if a < b and (b, a) in dy), fights=fights,
                gini=gini(list(R.values())), below10=sum(v < 10 for v in R.values()),
                top_nbr_median=float(np.median(nb)) if nb else 0.0)

seed = lambda p: re.search(r"__s(\d+)", str(p)).group(1)
out = {}
for L in ("L2", "L3", "L4"):
    q = {seed(p): p for p in runset.cel(f"robust_qwen_{L}_knife")}
    g = {seed(p): p for p in runset.cel(f"prod_{L}_knife")}
    shared = sorted(set(q) & set(g))
    out[L] = {"gemma": [run(g[s]) for s in shared], "qwen": [run(q[s]) for s in shared],
              "gemma_all15_gini": [run(p)["gini"] for p in runset.cel(f"prod_{L}_knife")]}
    for m in ("gemma", "qwen"):
        rs = out[L][m]; f = lambda k: [r[k] for r in rs]
        print(f"{L} {m:5} transfers {min(f('transfers'))}-{max(f('transfers'))}  runs with pair {sum(x > 0 for x in f('pairs'))}/5 "
              f"(median pairs {np.median(f('pairs')):.0f})  fights median {np.median(f('fights')):.0f}  gini median {np.median(f('gini')):.2f} "
              f"below10 {min(f('below10'))}-{max(f('below10'))}  top-nbr median {min(f('top_nbr_median')):.1f}-{max(f('top_nbr_median')):.1f}")
    print(f"{L} gemma all 15 knife: gini median {np.median(out[L]['gemma_all15_gini']):.2f} mean {np.mean(out[L]['gemma_all15_gini']):.2f}")
(HIER / "model_forms.json").write_text(json.dumps(out, indent=1))
