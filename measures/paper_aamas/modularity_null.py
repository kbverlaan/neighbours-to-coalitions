"""Clubs lossen op bij L3: echt, of een dichtheidsartefact? (AAMAS, 7 okt 2026; verkennend.)

Zelfde gewogen interactiegraaf als scripts/batch_suite.py (A = 0.5*ns(transfers) + 0.5*ns(berichten),
Leiden RBConfiguration, seed 42). Nulmodel per run: dezelfde graaf met behoud van gradenreeks
herschud (igraph rewire), de oorspronkelijke gewichten willekeurig over de nieuwe randen verdeeld;
20 herhalingen. Gerapporteerd per trede: mediaan Q, mediaan nul-Q, mediaan overschot Q - nul.
  uv run --python 3.12 --with numpy --with igraph==1.0.0 --with leidenalg==0.11.0 python paper_aamas/modularity_null.py
"""
import sys, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np, igraph as ig, leidenalg
HIER = Path(__file__).resolve().parent; M = HIER.parent; SIM = M.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
sys.path.insert(0, str(SIM / "scripts"))
import runset
from batch_suite import cat, effective_action, ns

REPS, SEED = 20, 20261007

def graph(path):
    T = defaultdict(float); Mm = defaultdict(float); agents = set()
    for l in open(path):
        if not l.strip(): continue
        try: d = json.loads(l)
        except json.JSONDecodeError: continue
        ag = d.get("agents", {}); agents.update(ag.keys()); bf = d.get("bilateral_flows") or {}
        for aid, info in ag.items():
            c = cat(effective_action(info)); tg = info.get("target")
            if not bf and c == "transfer" and tg:
                T[(aid, tg)] += (info.get("breakdown") or {}).get("invest_cost", 1.0) or 1.0
        for k, v in bf.items():
            if "→" in k: a, b = k.split("→"); T[(a, b)] += v
        for m in d.get("messages", []):
            to = m.get("to"); tos = to if isinstance(to, list) else ([] if to in (None, "all") else [to])
            for t in tos: Mm[(m.get("from"), t)] += 1
    agents = sorted(agents); idx = {a: i for i, a in enumerate(agents)}; n = len(agents)
    Tm = np.zeros((n, n)); Mx = np.zeros((n, n))
    for (a, b), v in T.items():
        if a in idx and b in idx: Tm[idx[a], idx[b]] += v
    for (a, b), v in Mm.items():
        if a in idx and b in idx: Mx[idx[a], idx[b]] += v
    A = 0.5 * ns(Tm) + 0.5 * ns(Mx)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]
    return n, edges, [A[i, j] for i, j in edges]

def q(n, edges, w):
    g = ig.Graph(n=n, edges=edges); g.es["weight"] = w
    return leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight", seed=42).modularity

rng = random.Random(SEED); out = {}
for L in ("L1", "L2", "L3", "L4"):
    obs, null, exc, deg = [], [], [], []
    for c in ("scar", "knife", "abund"):
        for p in runset.cel(f"prod_{L}_{c}"):
            n, e, w = graph(p)
            if not e: continue
            qo = q(n, e, w); qs = []
            for _ in range(REPS):
                g = ig.Graph(n=n, edges=e); g.rewire(n=10 * len(e))
                w2 = w[:]; rng.shuffle(w2)
                qs.append(q(n, [tuple(x.tuple) for x in g.es], w2))
            obs.append(qo); null.append(float(np.mean(qs))); exc.append(qo - float(np.mean(qs))); deg.append(2 * len(e) / n)
    out[L] = dict(runs=len(obs), Q=round(float(np.median(obs)), 3), Q_null=round(float(np.median(null)), 3),
                  excess=round(float(np.median(exc)), 3), mean_degree=round(float(np.median(deg)), 1))
    print(L, out[L], flush=True)
(HIER / "modularity_null.json").write_text(json.dumps(out, indent=1))
