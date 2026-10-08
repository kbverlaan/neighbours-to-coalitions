"""Getallen voor de hoofdtekst-tabellen en de review-controle (AAMAS, 7 okt 2026; verkennend waar aangegeven).

Tabel A (armen, knife-edge, L2--L4): Gemma alle runs | Gemma op de gedeelde seeds van Qwen | Qwen | zonder kanaal.
Tabel B (levels, Gemma, alle prijzen): per level de vorm-maten die de tekst noemt.
Plus controles: alleen-gevochten per level x prijs, stille L3-gevechten met >= 2 aanvallers, stille L2-gevechten,
minimum van de voorraad (na het oogsten, voor de aangroei), interactie actieset x prijs per maat (rang-anova, zelfde methode als herreken.py).
Per-run waarden; samenvattingen als mediaan [min, max] of mediaan over runs, en gepoold waar de tekst poolt.
  uv run --python 3.12 --with numpy --with scipy python paper_aamas/arms_levels.py
"""
import sys, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset, text, graph
from model import anova2

ALPHA, C_ATK = 0.35, 0.02
LEVELS = ("L1", "L2", "L3", "L4"); PAYS = ("scar", "knife", "abund")

def rounds(p):
    for line in open(p):
        try: yield json.loads(line)
        except json.JSONDecodeError: continue

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float(((2 * np.arange(n) - n + 1) * x).sum() / (n * x.sum())) if x.sum() else 0.0

def lone_losing(f):
    dp = f["defender_power"] or 1e-9
    return all(ALPHA * (pw / dp - 1) / (pw / dp + 1) - C_ATK * pw / dp < 0 for pw in f["attacker_powers"].values())

def run(p):
    deg = []
    fights = beyond = joint = multi = alone = transfers = 0; smin = None; emptied = False; last = None; rec = []; dy = Counter()
    for d in rounds(p):
        last = d
        E = (d.get("network") or {}).get("edges")
        if E is not None: deg.append(2 * len(E) / len(d["agents"]))
        for f in d.get("combat") or []:
            fights += 1; k = len(f["attackers"])
            ll = lone_losing(f); multi += k >= 2; alone += k == 1; beyond += ll; joint += ll and k >= 2
        for a, x in d["agents"].items():
            if x["action"] in ("transfer", "invest_other") and x["target"]:
                transfers += 1; dy[(a, x["target"])] += 1
        for m in d.get("messages") or []:
            to = m.get("to") or m.get("recipients") or m.get("message_to")
            if isinstance(to, list): rec.append(len(to))
        c = d.get("commons")
        if isinstance(c, dict) and c.get("stock_before") is not None:
            low = c["stock_before"] - (c.get("harvested") or 0.0)  # stand na het oogsten, voor de aangroei
            smin = low if smin is None else min(smin, low); emptied |= bool(c.get("collapsed"))
    R = [x["resources"] for x in last["agents"].values()]
    return dict(fights=fights, beyond=beyond, joint=joint, multi=multi, alone=alone, transfers=transfers,
                pair=float(any((b, a) in dy for (a, b) in dy)), names=float(len(text.named_agreements(p))),
                gini=gini(R), mean_final=float(np.mean(R)), tie_degree=float(np.mean(deg)) if deg else None, stock_min=smin, emptied=float(emptied), recipients=float(np.mean(rec)) if rec else None)

seed = lambda p: re.search(r"__s(\d+)", str(p)).group(1)
med = lambda xs: float(np.median(xs)) if xs else None
def summary(rs):
    F = sum(r["fights"] for r in rs)
    share = lambda k: [r[k] / r["fights"] for r in rs if r["fights"]]
    return dict(n=len(rs), fights_med=med([r["fights"] for r in rs]),
                beyond_pooled=sum(r["beyond"] for r in rs) / F if F else None, joint_pooled=sum(r["joint"] for r in rs) / F if F else None, beyond_run_med=med(share("beyond")),
                multi_pooled=sum(r["multi"] for r in rs) / F if F else None, alone_pooled=sum(r["alone"] for r in rs) / F if F else None,
                names_mean=float(np.mean([r["names"] for r in rs])), pair_runs=sum(r["pair"] for r in rs),
                transfers_min=min(r["transfers"] for r in rs), transfers_max=max(r["transfers"] for r in rs),
                gini_med=med([r["gini"] for r in rs]), mean_final_med=med([r["mean_final"] for r in rs]), tie_degree_med=med([r["tie_degree"] for r in rs if r["tie_degree"] is not None]), emptied=sum(r["emptied"] for r in rs),
                stock_min=min((r["stock_min"] for r in rs if r["stock_min"] is not None), default=None),
                recipients_med=med([r["recipients"] for r in rs if r["recipients"] is not None]))

out = {"arms": {}, "levels": {}, "checks": {}}
cache = {}
def R(p):
    if p not in cache: cache[p] = run(p)
    return cache[p]

for L in ("L2", "L3", "L4"):
    g = runset.cel(f"prod_{L}_knife"); q = runset.cel(f"robust_qwen_{L}_knife"); s = runset.cel(f"prod_{L}_knife_nocomm")
    qs = {seed(p) for p in q}
    out["arms"][L] = {"gemma_all": summary([R(p) for p in g]),
                      "gemma_shared": summary([R(p) for p in g if seed(p) in qs]),
                      "qwen": summary([R(p) for p in q]), "silent": summary([R(p) for p in s])}
s1 = runset.cel("prod_L1_knife_nocomm")
if s1: out["arms"]["L1"] = {"silent": summary([R(p) for p in s1]), "gemma_all": summary([R(p) for p in runset.cel("prod_L1_knife")])}

for L in LEVELS:
    rs = [R(p) for c in PAYS for p in runset.cel(f"prod_{L}_{c}")]
    out["levels"][L] = summary(rs)
    out["checks"][f"alone_{L}_per_price"] = {c: summary([R(p) for p in runset.cel(f"prod_{L}_{c}")])["alone_pooled"] for c in PAYS}

# interactie actieset x prijs per maat (rang-anova, zoals de decompositie)
src = open(HIER / "prereg_fingerprint.py").read(); i = src.index("DVS = ["); j = src.index("\n]", i) + 2
ns = {"RUNGS": LEVELS}; exec(src[i:j], ns)
rows = {tuple(k.split("|")): v for k, v in json.loads((HIER / "prereg_per_run.json").read_text()).items()}
def rk(xs):
    o = sorted(range(len(xs)), key=lambda k: xs[k]); r = [0.0] * len(xs); a = 0
    while a < len(xs):
        b = a
        while b + 1 < len(xs) and xs[o[b + 1]] == xs[o[a]]: b += 1
        for k in range(a, b + 1): r[o[k]] = (a + b) / 2 + 1
        a = b + 1
    return r
inter = {}
for key, label, fam, defined in ns["DVS"]:
    pl = [(L, c, row[key]) for L in defined for c in PAYS for row in rows[(L, c)]]
    rr = rk([x for *_, x in pl]); cr = defaultdict(list)
    for (L, c, _), v in zip(pl, rr): cr[(L, c)].append(v)
    inter[label] = round(anova2(dict(cr)).as_dict()["value"]["AxB"]["eta2p"], 2)
out["checks"]["interaction_action_price"] = inter
(HIER / "arms_levels.json").write_text(json.dumps(out, indent=1, default=float))

def fmt(x, pct=False):
    if x is None: return "-"
    return f"{x:.0%}" if pct else (f"{x:.2f}" if isinstance(x, float) else str(x))
for L, arms in out["arms"].items():
    for a, v in arms.items():
        print(f"{L} {a:12} n {v['n']:2} fights {fmt(v['fights_med'])} beyond pooled {fmt(v['beyond_pooled'],1)} run-med {fmt(v['beyond_run_med'],1)} multi {fmt(v['multi_pooled'],1)} names {v['names_mean']:.1f} pairs {v['pair_runs']:.0f} transf {v['transfers_min']}-{v['transfers_max']} gini {fmt(v['gini_med'])} emptied {v['emptied']:.0f} smin {fmt(v['stock_min'])} rec {fmt(v['recipients_med'])}")
for L, v in out["levels"].items():
    print(f"LEVEL {L} n {v['n']} fights {fmt(v['fights_med'])} alone {fmt(v['alone_pooled'],1)} beyond {fmt(v['beyond_pooled'],1)} names {v['names_mean']:.1f} pairs {v['pair_runs']:.0f} gini {fmt(v['gini_med'])} smin {fmt(v['stock_min'])} rec {fmt(v['recipients_med'])}")
print({k: v for k, v in out["checks"].items() if k.startswith("alone")})
print("interaction AxP:", inter)
