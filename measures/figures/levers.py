"""Which lever owns the structure of the order: capacity or price.

Two registered measures, both resting on registered kernels plus the ported club
detector (core/community.py):

  m:club-structure       per cell: Q (modularity), K (#clubs), cohesion, intra-take
                         ratio --- the club layer of the run fingerprint.
  m:levers-own-structure a two-way (capacity x price) ANOVA per structural DV, so the
                         claim "capacity selects the form, price sets the volume" rests
                         on a lever split and not on a single pooled number. The gated
                         conflict DVs (coalition size, takes) already live in
                         m:payoff-owns-capacity and are not recomputed here.

Provenance:
  - The club detector reproduces an earlier exploratory implementation exactly (pooled
    raw Q = 0.74 from both); its two knobs --- the Leiden resolution
    (RBConfigurationVertexPartition default) and the cohesion home-share threshold of
    0.6 --- were carried over unchanged, not tuned against the 150-run outcome. That
    reproduction is a port check, not a validation of Q as a construct.
  - Leiden with seed=42 is deterministic within a library version, not across versions;
    igraph and leidenalg are pinned in requirements.txt, and a different Q on another
    machine points at a version drift first.

DV classes (stated so the head claim is not circular):
  - clean, rung-general (definable and variable at every rung): Q, cohesion, consensus
    spread, Gini, floor. These carry the claim.
  - price-side volume: reciprocal-pair count.  magnitude: summed final economy.
  - gated over the ladder (take-based, only exist from L2): coalition, takes, intra-take
    --- read from m:payoff-owns-capacity; illustrative, not a main-effect argument.

    PYTHONPATH=../_shared:../core python3 -c "import levers,json;print(json.dumps(levers.levers_own_structure(),indent=1,default=float))"
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HIER.parent / _m))

import graph, logs, runstat                                             # noqa: E402
import community                                                        # noqa: E402
from base import log_path                                              # noqa: E402
from model import anova2                                               # noqa: E402
from result import Result                                             # noqa: E402
from runstat import summary as _summary                               # noqa: E402
import runset                                                          # noqa: E402

RUNGS = ("L1", "L2", "L3", "L4")
PAYS = ("scar", "knife", "abund")

# (key, label, class) — class drives the honest reading in the text.
DVS = (
    ("Q", "clubs modularity Q", "structure"),
    ("coh", "cohesion", "structure"),
    ("cons", "consensus spread", "structure"),
    ("gini", "final inequality (Gini)", "structure"),
    ("floor", "poorest finish", "structure"),
    ("dyads", "reciprocal pairs", "price-volume"),
    ("econ", "final economy (summed)", "magnitude"),
)


def _cells(*rungs):
    return [f"prod_{r}_{p}" for r in rungs for p in PAYS]


def _run_values(p) -> dict:
    """The structural DVs for one run, from registered kernels + the ported detector."""
    rounds = logs.rounds(log_path(p))
    c = community.community_structure(rounds)
    laatste = [a.get("resources") or 0.0
               for a in (rounds[-1].get("agents") or {}).values()] if rounds else []
    return {
        "Q": c["Q"], "coh": c["coh"], "verr": c["verr"], "K": float(c["K"]),
        "cons": float(runstat.consensus_spread(p)),
        "gini": float(runstat.final_gini(p)),
        "floor": float(runstat.floor(p)),
        "dyads": float(graph.mutual_dyads(p, min_count=1)),
        "econ": float(sum(laatste)),
    }


def _rank(xs: list[float]) -> list[float]:
    """Average ranks, so a five-orders-of-magnitude spread cannot drive the eta squared."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _rows() -> dict:
    """(rung, payoff) -> list of per-run value dicts, over the 12 comms-on cells."""
    per_cell = {}
    for rung in RUNGS:
        for payoff in PAYS:
            paths = runset.cel(f"prod_{rung}_{payoff}")
            per_cell[(rung, payoff)] = [_run_values(p) for p in paths]
    return per_cell


def club_structure() -> dict:
    """m:club-structure — Q, #clubs, cohesion, intra-take ratio per cell."""
    rows = _rows()
    uit = {}
    for rung in RUNGS:
        for payoff in PAYS:
            vs = rows[(rung, payoff)]
            naam = f"prod_{rung}_{payoff}"
            uit[naam] = Result(
                value={"Q": _summary([v["Q"] for v in vs]),
                       "clubs": _summary([v["K"] for v in vs]),
                       "cohesion": _summary([v["coh"] for v in vs]),
                       "intra_take_pct": _summary([v["verr"] for v in vs])},
                n=len(vs), denominator=len(vs), unit="runs",
                note="club detector (core/community.py); Q reproduces an earlier "
                     "implementation (port check, not construct validation)").as_dict()
    return uit


def levers_own_structure() -> dict:
    """m:levers-own-structure — two-way (capacity x price) ANOVA per structural DV.

    Reports partial eta squared for capacity (A), price (B) and their interaction,
    on the raw values and on the ranks. n is one per run; a small price effect is a
    small effect, never read as demonstrated equality.
    """
    rows = _rows()
    uit = {}
    for dv, label, klass in DVS:
        cells = {(a, b): [v[dv] for v in rows[(a, b)]] for a in RUNGS for b in PAYS}
        raw = anova2(cells).as_dict()
        pooled = [(a, b, x) for (a, b), xs in cells.items() for x in xs]
        rk = _rank([x for _, _, x in pooled])
        cells_r: dict = defaultdict(list)
        for (a, b, _), r in zip(pooled, rk):
            cells_r[(a, b)].append(r)
        ranked = anova2(dict(cells_r)).as_dict()
        uit[dv] = {
            "label": label, "class": klass,
            "capacity_eta2p": raw["value"]["A"]["eta2p"],
            "price_eta2p": raw["value"]["B"]["eta2p"],
            "interaction_eta2p": raw["value"]["AxB"]["eta2p"],
            "capacity_p": raw["value"]["A"]["p"], "price_p": raw["value"]["B"]["p"],
            "rank_capacity_eta2p": ranked["value"]["A"]["eta2p"],
            "rank_price_eta2p": ranked["value"]["B"]["eta2p"],
            "model_r2": raw["value"]["model_r2"], "n": raw["n"],
        }
    return uit


FIGURES = {
    "m:club-structure": club_structure,
    "m:levers-own-structure": levers_own_structure,
}
