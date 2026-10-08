"""Community structure of a run — the club layer of the run fingerprint.

Detects the subgroups of a run and reports the club signal: modularity Q, the number
of clubs, the cohesion, and the intra-club take ratio.

Method:
  - build a directed transfer matrix T (volume i->j) and message matrix M (count i->j);
  - normalise each by its max and symmetrise: ns(X) = (X/X.max() + (X/X.max()).T)/2;
  - combine A = alpha*ns(T) + (1-alpha)*ns(M), alpha=0.5 (transfers and talk weigh equal);
  - partition A with Leiden (RBConfigurationVertexPartition, weighted, seed=42);
  - Q = modularity, K = #clubs, cohesion = share of agents whose home-share >= 0.6.
Intra-take ratio (verr) uses the same membership: % of take-actions that stay inside a club.

Takes `rounds` (the list of per-round records from logs.rounds) so it is IO-free and
testable; a run path is read by the caller.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

try:
    import igraph as ig
    import leidenalg
    _HAVE = True
except Exception:                                                       # pragma: no cover
    _HAVE = False


def _is_transfer(a) -> bool:
    return a == "transfer" or (isinstance(a, str) and a.startswith("invest"))


def _ns(X):
    Y = X / X.max() if X.max() > 0 else X
    return (Y + Y.T) / 2


def community_structure(rounds, alpha: float = 0.5, seed: int = 42) -> dict:
    """{Q, K, coh, verr} for one run. Q=0,K=0,coh=0 when there are no flows."""
    if not _HAVE:
        raise ImportError("community_structure needs igraph + leidenalg")
    agents = set()
    T: dict = defaultdict(float)
    M: dict = defaultdict(float)
    takes = []
    for d in rounds:
        ag = d.get("agents") or {}
        agents.update(ag.keys())
        bf = d.get("bilateral_flows") or {}
        for aid, info in ag.items():
            a = info.get("action")
            tg = info.get("target")
            if a == "take" and tg:
                takes.append((aid, tg))
            if not bf and _is_transfer(a) and tg:
                T[(aid, tg)] += (info.get("breakdown") or {}).get("invest_cost", 1.0) or 1.0
        for k, v in bf.items():
            if "→" in k:
                x, y = k.split("→"); T[(x, y)] += v
        for m in (d.get("messages") or []):
            to = m.get("to")
            tos = to if isinstance(to, list) else ([] if to in (None, "all") else [to])
            for t in tos:
                M[(m.get("from"), t)] += 1
    agents = sorted(a for a in agents if a)
    idx = {a: i for i, a in enumerate(agents)}
    nn = len(agents)
    if nn < 2:
        return {"Q": 0.0, "K": 0, "coh": 0.0, "verr": 0.0}
    Tm = np.zeros((nn, nn)); Mm = np.zeros((nn, nn))
    for (a, b), v in T.items():
        if a in idx and b in idx:
            Tm[idx[a], idx[b]] += v
    for (a, b), v in M.items():
        if a in idx and b in idx:
            Mm[idx[a], idx[b]] += v
    A = alpha * _ns(Tm) + (1 - alpha) * _ns(Mm)
    edges = [(i, j) for i in range(nn) for j in range(i + 1, nn) if A[i, j] > 0]
    if not edges:
        return {"Q": 0.0, "K": 0, "coh": 0.0, "verr": 0.0}
    w = [A[i, j] for i, j in edges]
    gr = ig.Graph(n=nn, edges=edges); gr.es["weight"] = w
    part = leidenalg.find_partition(
        gr, leidenalg.RBConfigurationVertexPartition, weights="weight", seed=seed)
    memb = np.array(part.membership); Q = float(part.modularity); K = len(part)
    share = np.array([[A[i, memb == c].sum() for c in range(K)] for i in range(nn)])
    coh = sum(1 for i in range(nn)
              if share[i].sum() > 0 and share[i][memb[i]] / share[i].sum() >= 0.6) / nn
    membmap = {agents[i]: int(memb[i]) for i in range(nn)}
    intra = sum(1 for s, t in takes
                if s in membmap and t in membmap and membmap[s] == membmap[t])
    verr = 100 * intra / len(takes) if takes else 0.0
    return {"Q": Q, "K": int(K), "coh": float(coh), "verr": float(verr)}
