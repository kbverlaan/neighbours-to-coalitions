"""The shape every figure returns.

A bare number cannot be checked. `Result` forces the three things that make one
checkable to travel with it: the value, the denominator it came out of, and the
runs it was computed over. A figure that cannot fill those in is a figure that
does not know what it measured.

`baseline` and `sensitivity` are optional in the type but not in practice --- a
count without a chance expectation and a definition with an unreported free
parameter both fail the standards test. They are optional here because some
quantities genuinely have neither: a mean holding has no chance baseline, and a
Gini has no free parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Result:
    value: Any                       # the reported quantity
    n: int = 0                       # runs behind it
    denominator: int | None = None   # what `value` was divided by, if a share
    unit: str = ""                   # "agent-turns", "sentences", "runs", ...
    baseline: Any = None             # what chance yields on the same corpus
    sensitivity: dict = field(default_factory=dict)  # value under other choices
    skipped: list = field(default_factory=list)      # runs left out, by name
    note: str = ""                   # anything a reader needs to not misread it

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if v not in (None, "", [], {}) or k == "value"}


def by_cell(per_cell: dict[str, Result]) -> dict:
    """Serialise a cell -> Result mapping for the figures file."""
    return {c: r.as_dict() for c, r in per_cell.items()}
