"""Herrekening van twee levers-maten naar hun intensieve vorm (AAMAS, 29 sep 2026).

names  (telling coined names)      -> names_any: 1 als de run minstens één coined name heeft
                                    -> names_per_kmsg: coined names per 1000 berichten
floor  (armste eindbezit, niveau)  -> floor_ratio: armste eindbezit / gemiddeld eindbezit
pairs_any                          -> 1 als de run minstens één wederkerig paar heeft (vorm-lezing van paren)
Zelfde rang-ANOVA, bootstrap (2000 resamples binnen de cel) en seed als figures/levers.py.
"""
import sys, json
from collections import defaultdict
from pathlib import Path
M = Path(__file__).resolve().parents[1]
for s in ("core", "_shared"):
    sys.path.insert(0, str(M / s))
import numpy as np
import logs, runstat, text, runset, graph
from base import log_path
from model import anova2

RUNGS = ("L1", "L2", "L3", "L4"); PAYS = ("scar", "knife", "abund")
RESAMPLES, SEED = 2000, 20260817

def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0]*len(xs); i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[order[j+1]] == xs[order[i]]: j += 1
        for k in range(i, j+1): r[order[k]] = (i+j)/2+1
        i = j+1
    return r

def rank_eta(cells):
    pooled = [(a, b, x) for (a, b), xs in cells.items() for x in xs]
    rk = _rank([x for *_, x in pooled]); cr = defaultdict(list)
    for (a, b, _), r in zip(pooled, rk): cr[(a, b)].append(r)
    v = anova2(dict(cr)).as_dict()["value"]
    return v["A"]["eta2p"], v["B"]["eta2p"], v["AxB"]["eta2p"]

def boot(cells):
    rng = np.random.default_rng(SEED); keys = list(cells)
    arr = {k: np.asarray(cells[k], float) for k in keys}; ca, pr = [], []
    for _ in range(RESAMPLES):
        b = {k: list(arr[k][rng.integers(0, len(arr[k]), len(arr[k]))]) for k in keys}
        a, p, _ = rank_eta(b); ca.append(a); pr.append(p)
    q = lambda xs: [round(float(np.percentile(xs, 2.5)), 3), round(float(np.percentile(xs, 97.5)), 3)]
    return q(ca), q(pr)

def run_values(p):
    rounds = logs.rounds(log_path(p))
    n_msg = sum(len(e.get("messages") or []) for e in rounds)
    n_names = len(text.named_agreements(p))
    fl, mean = runstat.floor(p), runstat.mean_holding(p)
    return {"names": float(n_names), "names_any": float(n_names > 0),
            "names_per_kmsg": 1000.0*n_names/n_msg if n_msg else 0.0,
            "floor": float(fl), "floor_ratio": fl/mean if mean > 0 else 0.0,
            "pairs_any": float(graph.mutual_dyads(p, min_count=1) > 0),
            "n_msg": n_msg, "mean": mean}

rows = {(r, p): [run_values(x) for x in runset.cel(f"prod_{r}_{p}")] for r in RUNGS for p in PAYS}
uit = {}
for dv in ("names", "names_any", "names_per_kmsg", "floor", "floor_ratio", "pairs_any"):
    cells = {k: [v[dv] for v in vs] for k, vs in rows.items()}
    a, b, ab = rank_eta(cells); ci_a, ci_b = boot(cells)
    uit[dv] = {"cap": a, "price": b, "inter": ab, "cap_ci": ci_a, "price_ci": ci_b,
               "cell_means": {f"{r}_{p}": round(float(np.mean(cells[(r, p)])), 3) for r, p in cells}}
    print(f"{dv:<16} cap {a:.2f} {ci_a}   prijs {b:.2f} {ci_b}   interactie {ab:.2f}")
n = sum(len(v) for v in rows.values()); print("runs:", n)
print("mean eindbezit <= 0 in", sum(1 for vs in rows.values() for v in vs if v["mean"] <= 0), "runs")
print("runs zonder berichten:", sum(1 for vs in rows.values() for v in vs if v["n_msg"] == 0))
json.dump(uit, open(Path(__file__).with_name("herreken.json"), "w"), indent=1)
