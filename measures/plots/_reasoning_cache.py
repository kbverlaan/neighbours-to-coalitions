"""The reasoning-trace embeddings, built once and read from disk after that.

    python3 plots/_reasoning_cache.py            # build, ~100 minutes on MPS
    python3 plots/_reasoning_cache.py --check    # verify the runs, embed nothing

Why this is a separate module. Embedding ninety thousand chains of thought takes
an hour and a half; redrawing the figure takes two seconds. Keeping the two
apart means the figure can be redrawn --- different panels, different colours,
a different trajectory --- without the model being loaded again. The cache is
three files in `plots/cache/fig8_reasoning/`: the matrix, one row per trace in
an index CSV, and a `meta.json` saying what was embedded and how.

**The resume trap, and why every file is checked.** A run interrupted on the
cluster and restarted leaves two reasoning files: the job-numbered one holds
only the resumed segment, and the `RESUMED_`-prefixed one holds all sixty
rounds. Both parse cleanly as JSON, so reading the wrong one analyses a sixth of
a run and says nothing about it. `data/thesis_final/` has already resolved that
--- its index points at the complete file under a canonical name --- but a
figure that trusts an earlier consolidation without checking is one rename away
from the same silent error. So each file is verified here before a single trace
is taken from it: sixty distinct rounds, the first numbered 1, the last numbered
60. A run that fails is an error and never a run with no traces.

One production run, `prod_L3_knife_r01`, carries seventy-two records for sixty
rounds: it was resumed at round 35 and rounds 35 to 46 appear twice. Rounds are
deduplicated on their number with the later record winning, which is the resumed
segment --- the same rule `core/logs.py` applies, so this figure and every
measure read that run identically.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(HERE.parent / _m))

import runset  # noqa: E402

CACHE = HERE / "cache" / "fig8_reasoning"
HORIZON = 60

# --- what is embedded, and what is left out ---------------------------------
#
# Every production run and every agent, but one round in three. The claim the
# figure makes is about the spread between runs, so runs are what may not be
# thinned; rounds within a run are strongly autocorrelated and the trajectory is
# a mean over thirty agents, which twenty rounds trace as well as sixty at a
# third of the cost. Rounds 1, 4, ... 58.
ROUND_STEP = 3
ROUNDS = list(range(1, HORIZON + 1, ROUND_STEP))

MODEL = "nomic-ai/nomic-embed-text-v1.5"
PREFIX = "search_document: "        # nomic wants a task prefix; without it the
                                    # model is off its training distribution
MAX_SEQ = 512                       # tokens
TAIL_CHARS = 2400                   # characters kept, from the end

# The tail and not the head. A trace opens with the agent restating its holdings,
# its neighbours and the prices --- text that is nearly identical across every
# agent in a round and would dominate the embedding with the scenario rather than
# the reasoning. The strategic argument and the decision are at the end. The
# median trace is 5,400 characters, so the tail keeps roughly the closing half.


def traces(path: Path) -> list[tuple[int, str, str]]:
    """(round, agent, thinking) for the sampled rounds of one verified run.

    Raises rather than returning a short list: an unreadable or incomplete
    source is a fault in the run set, not a run that happened to be quiet.
    """
    per: dict[int, dict] = {}
    for line in path.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError as exc:
            raise runset.RunsetError(f"{path.name}: unparseable line ({exc})") from exc
        r = e.get("round")
        if r is not None:
            per[r] = e                       # later record wins; see the docstring
    got = sorted(per)
    if len(got) != HORIZON or got[0] != 1 or got[-1] != HORIZON:
        raise runset.RunsetError(
            f"{path.name}: {len(got)} rounds, {got[0] if got else None} to "
            f"{got[-1] if got else None}; a run is sixty rounds from 1 to 60. "
            f"Check whether this is a resumed segment rather than the full run.")
    out = []
    for r in ROUNDS:
        for name, a in sorted((per[r].get("agents") or {}).items()):
            t = (a.get("thinking") or "").strip()
            if t:
                out.append((r, name, t))
    return out


def collect(cells=None) -> tuple[list[dict], list[str], list[str]]:
    """Every sampled trace of every production run, with its row of the index.

    Returns the rows, the texts, and the runs that could not be used. Nothing is
    skipped quietly: a run that fails verification is reported by name and the
    caller decides whether that is tolerable.
    """
    cells = cells or runset.PRODUCTION
    rows, texts, skipped = [], [], []
    for cell in cells:
        _, rung, payoff = cell.split("_")
        for path in runset.cel(cell):
            run_id = path.name.split("__")[0].rsplit("_", 1)[-1]
            seed = path.name.split("__")[1].split("_")[0]
            try:
                got = traces(path)
            except runset.RunsetError as exc:
                skipped.append(f"{cell}/{run_id}: {exc}")
                continue
            for r, agent, text in got:
                rows.append({"cell": cell, "rung": rung, "payoff": payoff,
                             "run": run_id, "seed": seed, "round": r,
                             "agent": agent, "chars": len(text)})
                texts.append(text)
    return rows, texts, skipped


def embed(texts: list[str], batch: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, trust_remote_code=True, device=device)
    if device == "mps":
        model = model.half()          # half the arithmetic, no visible loss on
                                      # unit-normalised vectors read by UMAP
    model.max_seq_length = MAX_SEQ
    prepared = [PREFIX + t[-TAIL_CHARS:] for t in texts]

    out = np.empty((len(prepared), 768), dtype=np.float16)
    step, t0 = 4096, time.time()
    for i in range(0, len(prepared), step):
        chunk = prepared[i:i + step]
        out[i:i + len(chunk)] = model.encode(
            chunk, batch_size=batch, normalize_embeddings=True,
            show_progress_bar=False).astype(np.float16)
        done, elapsed = i + len(chunk), time.time() - t0
        rate = done / elapsed
        print(f"  {done:6}/{len(prepared)}  {rate:5.1f} traces/s  "
              f"{(len(prepared) - done) / rate / 60:5.1f} min left", flush=True)
    return out


def build(cells=None) -> None:
    rows, texts, skipped = collect(cells)
    print(f"{len(rows)} traces from {len({(r['cell'], r['run']) for r in rows})} runs")
    if skipped:
        print(f"EXCLUDED {len(skipped)} run(s):")
        for s in skipped:
            print(f"   {s}")
    E = embed(texts)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "embeddings.npy", E)
    with (CACHE / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (CACHE / "meta.json").write_text(json.dumps({
        "model": MODEL, "prefix": PREFIX, "max_seq": MAX_SEQ,
        "tail_chars": TAIL_CHARS, "round_step": ROUND_STEP, "rounds": ROUNDS,
        "cells": list(cells or runset.PRODUCTION), "traces": len(rows),
        "runs": len({(r["cell"], r["run"]) for r in rows}),
        "dim": int(E.shape[1]), "dtype": "float16", "skipped": skipped,
        "built": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))
    print(f"-> {CACHE}/")


def load() -> tuple[np.ndarray, list[dict], dict]:
    """The cached matrix, its index and its metadata."""
    if not (CACHE / "embeddings.npy").exists():
        raise runset.RunsetError(
            f"{CACHE}/embeddings.npy is missing. Build it first:\n"
            f"    python3 plots/_reasoning_cache.py")
    E = np.load(CACHE / "embeddings.npy").astype(np.float32)
    with (CACHE / "index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["round"] = int(r["round"])
    meta = json.loads((CACHE / "meta.json").read_text())
    if len(rows) != len(E):
        raise runset.RunsetError(
            f"cache is inconsistent: {len(E)} vectors against {len(rows)} rows")
    return E, rows, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify every run and count the traces; embed nothing")
    args = ap.parse_args()
    if args.check:
        rows, texts, skipped = collect()
        runs = {(r["cell"], r["run"]) for r in rows}
        print(f"{len(rows)} traces from {len(runs)} runs over {len(ROUNDS)} rounds")
        print(f"excluded: {len(skipped)}")
        for s in skipped:
            print(f"   {s}")
    else:
        build()
