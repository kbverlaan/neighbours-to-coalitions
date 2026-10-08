"""Action distribution and final Gini per cell — the backbone under Table 1.

This is the only measure without an S-number in the ledger: the action
distribution was recomputed per analysis instead of fixed once. That let two
tables in the same text drift apart without anything flagging it.

Columns follow Table 1: n, final Gini (M±SD), and the share of each action in
all agent-turns of the cell.

    python3 grid_profile.py            # all cells
    python3 grid_profile.py --json     # machine-readable, for make_tables
"""
from __future__ import annotations

import argparse
import json
import sys

import runset
from base import ACTIONS, action_shares, final_gini, alive, mean_sd, rewire_shares


def compute(namen=None) -> dict:
    namen = namen or (runset.PRODUCTION + runset.NOCOMM + runset.QWEN + runset.DEEPSEEK)
    uit = {}
    for naam in namen:
        paths = runset.cel(naam)
        shares, turns = action_shares(paths)
        rew, _ = rewire_shares(paths)
        ginis = [final_gini(p) for p in paths]
        m, sd = mean_sd(ginis)
        uit[naam] = {
            "n": len(paths),
            "agent_turns": turns,
            "gini_mean": round(m, 3),
            "gini_sd": round(sd, 3),
            "alive_mean": round(sum(alive(p) for p in paths) / len(paths), 1),
            "acties": {a: round(v, 1) for a, v in shares.items()},
            "rewire": {k: round(v, 1) for k, v in rew.items()},
            "bronnen": runset.source_count(paths),
            "overgeslagen": [],
        }
    return uit


def _tabel(res: dict) -> None:
    kop = f"{'cell':26}{'n':>3}{'Gini':>13}{'alive':>7}"
    for a in ACTIONS:
        kop += f"{a[:6]:>8}"
    kop += f"{'drop':>7}{'invite':>7}"
    print(kop)
    print("-" * len(kop))
    for naam, r in res.items():
        regel = (f"{naam:26}{r['n']:3}"
                 f"{r['gini_mean']:8.3f}±{r['gini_sd']:.3f}{r['alive_mean']:7.1f}")
        for a in ACTIONS:
            v = r["acties"][a]
            regel += f"{v:8.1f}" if v else f"{'·':>8}"
        for k in ("drop", "invite"):
            v = r["rewire"][k]
            regel += f"{v:7.1f}" if v else f"{'·':>7}"
        print(regel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    try:
        res = compute(a.cells)
    except runset.RunsetError as e:
        sys.exit(f"RUNSET: {e}")
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        _tabel(res)
