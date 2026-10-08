"""Hangt de hefboomsplitsing aan één trede? (AAMAS, 29 sep 2026)

Acht maten van §4.2 per run; rang-ANOVA (capaciteit x prijs) op het volledige
raster en met telkens één trede weggelaten. Zelfde rang-eta, bootstrap en seed
als figures/levers.py. Draaien met de gepinde omgeving:

  uv run --python 3.12 --with igraph==1.0.0 --with leidenalg==0.11.0 \
    --with numpy==2.3.5 --with scipy==1.17.0 python paper_aamas/leave_one_rung_out.py
"""
import sys, json
from collections import defaultdict
from pathlib import Path
M = Path(__file__).resolve().parent.parent
for s in ("core", "_shared"):
    sys.path.insert(0, str(M / s))
import numpy as np
import logs, runstat, text, runset, graph, community
from base import log_path
from model import anova2

RUNGS = ("L1", "L2", "L3", "L4"); PAYS = ("scar", "knife", "abund")
RES, SEED = 2000, 20260817
DVS = [("names_any", "form"), ("Q", "form"), ("cons", "form"), ("pairs_any", "form"),
       ("dyads", "volume"), ("econ", "volume"), ("floor_ratio", "distribution"), ("gini", "distribution")]

def per_run(p):
    rounds = logs.rounds(log_path(p))
    c = community.community_structure(rounds)
    fin = [a.get("resources") or 0.0 for a in (rounds[-1].get("agents") or {}).values()]
    mean = runstat.mean_holding(p)
    return {"names_any": float(len(text.named_agreements(p)) > 0), "Q": c["Q"],
            "cons": float(runstat.consensus_spread(p)),
            "pairs_any": float(graph.mutual_dyads(p, min_count=1) > 0),
            "dyads": float(graph.mutual_dyads(p, min_count=1)), "econ": float(sum(fin)),
            "floor_ratio": runstat.floor(p) / mean if mean > 0 else 0.0,
            "gini": float(runstat.final_gini(p))}

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

def boot(cells):
    rng = np.random.default_rng(SEED); keys = list(cells); arr = {k: np.asarray(cells[k], float) for k in keys}
    ca, pr = [], []
    for _ in range(RES):
        b = {k: list(arr[k][rng.integers(0, len(arr[k]), len(arr[k]))]) for k in keys}
        a, p = reta(b); ca.append(a); pr.append(p)
    q = lambda xs: [round(float(np.percentile(xs, 2.5)), 2), round(float(np.percentile(xs, 97.5)), 2)]
    return q(ca), q(pr)

cache = Path(__file__).with_name("per_run_8.json")
if cache.exists():
    rows = {tuple(k.split("|")): v for k, v in json.loads(cache.read_text()).items()}
else:
    rows = {(r, p): [per_run(x) for x in runset.cel(f"prod_{r}_{p}")] for r in RUNGS for p in PAYS}
    cache.write_text(json.dumps({"|".join(k): v for k, v in rows.items()}))

uit = {}
for weg in (None, "L1", "L2", "L3", "L4"):
    rungs = [r for r in RUNGS if r != weg]; tag = "alle" if weg is None else f"zonder {weg}"
    uit[tag] = {}
    for dv, klass in DVS:
        cells = {(r, p): [v[dv] for v in rows[(r, p)]] for r in rungs for p in PAYS}
        a, b = reta(cells); ci_a, ci_b = boot(cells)
        uit[tag][dv] = {"class": klass, "cap": round(a, 2), "price": round(b, 2), "cap_ci": ci_a, "price_ci": ci_b}
Path(__file__).with_name("leave_one_rung_out.json").write_text(json.dumps(uit, indent=1))
print(f"{'maat':<13}" + "".join(f"{t:>17}" for t in uit))
for dv, klass in DVS:
    print(f"{dv:<13}" + "".join(f"{uit[t][dv]['cap']:>9.2f}/{uit[t][dv]['price']:<7.2f}" for t in uit))
