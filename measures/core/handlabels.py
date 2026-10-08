"""Reading the hand-coded label files.

The order of mental-state attribution is not detected over the corpus; it is
hand-coded on samples of it, and those labels live in `handlabels/tom/` as
JSONL. Reading them is file parsing, which belongs here and not in a figure
module: the standard that keeps log parsing out of `figures/` exists so that a
figure module states what a quantity is while a core module states how the
bytes are read, and a label file is no different from a log in that respect.

Nothing here interprets a label. The majority vote, the agreement and the
exposure model stay with the measure that reports them, because those are
claims about the data and this is only the reading of it.
"""
from __future__ import annotations

import json
from pathlib import Path

WORTEL = Path(__file__).resolve().parents[2] / "handlabels" / "tom"


def jsonl(pad: Path) -> list[dict]:
    """Every non-empty line of a JSONL file, in file order."""
    return [json.loads(r) for r in pad.read_text().splitlines() if r.strip()]


def op_id(pad: Path) -> dict:
    """The same, keyed by the `id` field.

    A duplicate id would silently overwrite its predecessor and shrink the
    sample without saying so, which is the failure this raises on instead.
    """
    uit = {}
    for d in jsonl(pad):
        i = int(d["id"])
        if i in uit:
            raise ValueError(f"{pad.name}: id {i} appears more than once")
        uit[i] = d
    return uit


def labelronde(map_naam: str, ronde: str) -> dict:
    """One labeller's verdicts, gathered from the batch files of one round.

    The rounds are written in batches, so a round is complete only when every
    batch is there; a gap would otherwise pass as a smaller sample.
    """
    uit = {}
    for pad in sorted((WORTEL / map_naam / "rondes").glob(f"{ronde}_b*.jsonl")):
        for d in jsonl(pad):
            uit[int(d["id"])] = d
    return uit
