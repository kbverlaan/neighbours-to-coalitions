"""Reading a run once, not eleven times.

Splitting the counting into primitives means several figures walk the same run
in one pass of `registry.py`: the lever split alone asks six questions of all
150 production runs. Re-parsing the log each time made that run take minutes for
no gain, since the file cannot change while the process is alive.

The cache is keyed on the path and holds the parsed rounds. Callers must treat
what they get back as read-only --- it is the same list object every time, and
mutating it would corrupt every later reader. Nothing in `core` or `figures`
writes to a round record, and the standards test would not catch it if something
did, so this is a convention rather than a guarantee.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

import runset  # noqa: E402

# A bounded cache, not an unbounded one. Several figures read the same run
# twice, which is what this is for; a figure that walks every cell reads ninety
# runs once each, and holding all of them took a process to sixteen gigabytes
# and put it into swap. Thirty-two is comfortably more than any single figure's
# working set and small enough to stay in memory.
#
# Nothing about any number changes: eviction decides what is reparsed, never
# what is returned.
_CACHE_MAX = 2
_CACHE: "OrderedDict[Path, list[dict]]" = OrderedDict()

# Every run in this study is sixty rounds. The index column counts records, not
# rounds, so it overshoots where segments overlap.
HORIZON = 60


def _parse(p: Path) -> list[dict]:
    """Every round in one file, ordered, deduplicated on round number.

    One run was assembled from two segments and carries twelve rounds twice;
    counting straight through would weight those rounds double.
    """
    per: dict[int, dict] = {}
    with p.open() as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = e.get("round")
            if r is not None:
                per[r] = e
    return [per[k] for k in sorted(per)]


def rounds(path: Path) -> list[dict]:
    """Every round of one run, read at most once, checked against the index.

    A truncated log is readable and silently short. One compact log in this set
    is exactly 36 MiB and stops at round 36 while its reasoning_live carries all
    sixty, so every final-state figure for that run read round 36 as the end:
    its wealth came out at 5,158 instead of 13,339.

    That is the failure the run set was built against --- an unreadable source
    counting as an empty one --- one layer deeper, in a file's contents rather
    than its presence. The index knows how many rounds a run has, so the check
    is cheap: fall back to the full trace, and fail loudly if that is short too
    rather than quietly returning a partial run.
    """
    p = runset.log_path(Path(path))
    hit = _CACHE.get(p)
    if hit is not None:
        _CACHE.move_to_end(p)
    if hit is not None:
        return hit
    uit = _parse(p)
    if not uit:
        raise runset.RunsetError(f"{p.name} parsed to zero rounds")
    # Compare the highest round reached, not the number of rounds. The index
    # counts raw records, and one run assembled from two overlapping segments
    # carries 72 records for 60 distinct rounds --- deduplication is correct
    # there and a count check would reject it. A truncated file is short in the
    # only sense that matters: it never reaches the end of the run.
    verwacht = min(_verwachte_rondes(Path(path)) or 0, HORIZON) or None
    if verwacht and uit[-1].get("round", 0) < verwacht:
        vol = Path(str(p).replace("_log.jsonl", "_reasoning_live.jsonl"))
        if vol != p and vol.exists():
            uit = _parse(vol)
        if not uit or uit[-1].get("round", 0) < verwacht:
            raise runset.RunsetError(
                f"{p.name} stops at round {uit[-1].get('round')} where the run "
                f"reaches {verwacht}; a truncated log is not a short run")
    _CACHE[p] = uit
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return uit


def _verwachte_rondes(path: Path) -> int | None:
    """How many rounds the index says this run has, or None if unknown."""
    naam = path.name.replace("_log.jsonl", "_reasoning_live.jsonl")
    for r in runset.rijen():
        if r["bestand"] == naam:
            try:
                return int(r["rondes"])
            except (ValueError, KeyError):
                return None
    return None


def clear():
    _CACHE.clear()


def stats() -> dict:
    return {"runs_cached": len(_CACHE), "cache_limit": _CACHE_MAX,
            "rounds_cached": sum(len(v) for v in _CACHE.values())}
