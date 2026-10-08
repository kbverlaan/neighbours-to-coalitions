"""The runset — one definition of which runs count, for all measures.

Why this exists: the old analysis scripts each kept their own idea of the
dataset — different directories, and three carried an inline manifest of
job IDs. Those sources diverged: an older copy missed the cleanup of rejected
runs and counted four fallback-polluted runs as valid, which in the T4 table
produced 66.6% where 64.4% belonged.

Here the truth is `_INDEX.csv` in `data/thesis_final/`: 188 runs, numbered
per cell, each with its seed as an irreducible marker.

Two hard rules, both learned the hard way:

1. **An unreadable source never counts as an empty source.** If the index is
   missing, or a cell cannot be read, that is an error and not n=0. Three times
   in one day something silently counted as empty: iCloud discarded 72 of the
   157 logs and left placeholders that read as b''; macOS revoked disk access
   during a tool update, after which glob returned nothing; and an old copy
   missed the cleanup of rejected runs.
2. **Whoever skips runs says so.** `skipped` belongs in the output of every
   measure, so that an exclusion can land in the text instead of in a README.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# _shared/ sits two levels below simulation/, hence parents[2].
WORTEL = Path(__file__).resolve().parents[2] / "data" / "thesis_final"
INDEX = WORTEL / "_INDEX.csv"

RUNGS = ("L1", "L2", "L3", "L4")
PAYOFFS = ("scar", "knife", "abund")
PRODUCTION = [f"prod_{r}_{p}" for r in RUNGS for p in PAYOFFS]
NOCOMM = [f"prod_{r}_knife_nocomm" for r in RUNGS]
QWEN = [f"robust_qwen_{r}_knife" for r in ("L2", "L3", "L4")]
DEEPSEEK = [f"robust_deepseek_{r}_knife" for r in ("L2", "L3", "L4")]


class RunsetError(RuntimeError):
    """Unreadable or missing — never silently treated as empty."""


def _index() -> list[dict]:
    if not INDEX.exists():
        raise RunsetError(
            f"{INDEX} is missing. The measures run on the consolidated set;\n"
            f"build it first or point WORTEL at the right directory.")
    with INDEX.open() as fh:
        rijen = list(csv.DictReader(fh))
    if not rijen:
        raise RunsetError(f"{INDEX} is empty.")
    return rijen


_RIJEN = None


def rijen() -> list[dict]:
    global _RIJEN
    if _RIJEN is None:
        _RIJEN = _index()
    return _RIJEN


def cel(naam: str) -> list[Path]:
    """All reasoning_live paths of one cell, sorted by number.

    Fails hard if the cell is not in the index or the directory is not readable.
    A typo in a cell name must raise an error, not return an empty list —
    otherwise a measure neatly reports n=0 over a cell that plainly exists.
    """
    hoort = [r for r in rijen() if r["cel"] == naam]
    if not hoort:
        raise RunsetError(
            f"cell '{naam}' is not in _INDEX.csv. Known: "
            f"{sorted({r['cel'] for r in rijen()})}")
    map_ = WORTEL / naam
    try:
        os.listdir(map_)
    except OSError as e:
        raise RunsetError(f"cell '{naam}' exists but cannot be read: {e}") from e
    paths = []
    for r in sorted(hoort, key=lambda r: r["nieuw_id"]):
        p = map_ / r["bestand"]
        if not p.exists():
            raise RunsetError(f"{p} is in the index but not on disk.")
        paths.append(p)
    return paths


def cells(namen) -> dict[str, list[Path]]:
    return {n: cel(n) for n in namen}


# --- which serving stack a run used ----------------------------------------
#
# Thirty of the 150 production runs ran through OpenRouter instead of the
# cluster engine, on the same model and the same configuration, to complete
# cells where the allocation fell short. The appendix reported that, and also
# reported that three figures diverge further between the arms than the spread
# within a cell — without saying which three. That question cannot be answered
# without this function: the source lived in the index and nowhere in the code.

def arm_van(p: Path) -> str:
    """`cluster`, `openrouter`, `resumed` or `onbekend` for one run path."""
    for r in rijen():
        if r["bestand"] == p.name:
            b = (r.get("bron") or "").strip().lower()
            if b.startswith("openrouter"):
                return "openrouter"
            if b.startswith("resumed"):
                return "resumed"
            if b.startswith("job"):
                return "cluster"
            return "onbekend"
    raise RunsetError(f"{p.name} staat niet in _INDEX.csv.")


def per_arm(naam: str) -> dict[str, list[Path]]:
    """The runs of a cell, grouped by serving stack.

    `resumed` counts here as cluster: those runs started on Snellius and resumed
    there after an interruption, so they shared the engine with the cluster
    runs. What is separated here is the stack, not the interruption; that is
    recorded separately in the index and has its own footnote in the appendix.
    """
    uit: dict[str, list[Path]] = {}
    for p in cel(naam):
        a = arm_van(p)
        uit.setdefault("cluster" if a in ("cluster", "resumed") else a, []).append(p)
    return uit


def log_path(p: Path) -> Path:
    """The cheapest file with actions and resources for this run.

    Preferably the compact `_log.jsonl`: the same fields as the
    `reasoning_live`, without the reasoning text, and ten times smaller.

    21 of the 188 runs have none — exactly the runs resumed on Snellius, where
    only the full file was kept. Those fall back to the `reasoning_live`, which
    is no concession: `action`, `resources`, `target` and `breakdown` appear
    there identically. It only costs reading time.

    The fallback is not silent: `source_count()` shows which runs used which
    file, so a measure can account for where it read from.
    """
    q = p.with_name(p.name.replace("_reasoning_live.jsonl", "_log.jsonl"))
    if q.exists():
        return q
    if p.exists():
        return p
    raise RunsetError(f"neither log nor reasoning_live for {p.name}")


def source_count(paths) -> dict[str, int]:
    """How many runs read from a `_log.jsonl` and how many from the full trace."""
    uit = {"log": 0, "reasoning_live": 0}
    for p in paths:
        uit["log" if log_path(p).name.endswith("_log.jsonl") else "reasoning_live"] += 1
    return uit


def clean(namen=None) -> dict[str, list[Path]]:
    """Only runs without fallbacks.

    Eight of the 188 runs do contain them, all eight on T2, at most 0.50%
    forced. That is well below the 1.0% at which a run is rejected, so they
    belong in the main analysis. This entry point exists to test whether a
    finding leans on them — not to drop them by default.
    """
    namen = namen or PRODUCTION
    uit = {}
    for n in namen:
        houd = []
        for r in sorted((r for r in rijen() if r["cel"] == n),
                        key=lambda r: r["nieuw_id"]):
            if int(r["forced"]) == 0 and int(r["hersteld"]) == 0:
                houd.append(WORTEL / n / r["bestand"])
        uit[n] = houd
    return uit


def count() -> dict[str, int]:
    uit: dict[str, int] = {}
    for r in rijen():
        uit[r["cel"]] = uit.get(r["cel"], 0) + 1
    return uit


if __name__ == "__main__":
    t = count()
    print(f"{sum(t.values())} runs in {len(t)} cells\n")
    for groep, namen in (("production", PRODUCTION), ("no-channel", NOCOMM),
                         ("Qwen", QWEN), ("DeepSeek", DEEPSEEK)):
        n = sum(t.get(c, 0) for c in namen)
        print(f"  {groep:12} {n:3}  ({', '.join(f'{c.split(chr(95))[-1]}={t.get(c,0)}' for c in namen)})")
    dirty = [r for r in rijen() if int(r["forced"]) or int(r["hersteld"])]
    print(f"\n  runs with fallbacks: {len(dirty)}")
    for r in dirty:
        print(f"    {r['cel']}/{r['nieuw_id']}  forced={r['forced']} hersteld={r['hersteld']}")
