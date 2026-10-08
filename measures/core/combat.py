"""Fights: when they start, who is in them, and against whom.

Shared by the channel section, the paths section and the capacity sections,
which all ask the same questions of the combat record with different cells
attached. Written once here so that "the first attack" cannot mean the first
resolved fight in one figure and the first declared `take` in another.

The distinction matters and is kept explicit. A `take` action is an attempt; a
combat entry is an attempt that resolved. In cells where attempts are frequently
unresolved the two counts diverge, so every function here says which it reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from logs import rounds  # noqa: E402
import runset            # noqa: E402


def fights(path: Path) -> list[tuple[int, dict]]:
    """(round, entry) for every resolved fight, in order."""
    return [(e.get("round"), c) for e in rounds(path)
            for c in (e.get("combat") or []) if isinstance(c, dict)]


def count(path: Path) -> int:
    return len(fights(path))


def first_round(path: Path) -> int | None:
    fs = fights(path)
    return fs[0][0] if fs else None


def last_round(path: Path) -> int | None:
    fs = fights(path)
    return fs[-1][0] if fs else None


def first(path: Path) -> dict | None:
    fs = fights(path)
    return fs[0][1] if fs else None


def coalition_of_first(path: Path) -> int | None:
    c = first(path)
    return len(c.get("attackers") or []) if c else None


def defender_of_first(path: Path) -> str | None:
    c = first(path)
    return c.get("defender") if c else None


def first_attacker_lost(path: Path) -> bool | None:
    """Whether the opening blow failed.

    `winner` is a role label --- 'coalition' or 'defender' --- and never an
    agent name. An earlier version of this test compared it against the
    defender's name, which is true of every fight, and so recorded every opening
    as a defender victory.
    """
    c = first(path)
    return c.get("winner") == "defender" if c else None


def solo_share(path: Path) -> float | None:
    """Share of a run's fights struck by a single attacker."""
    fs = fights(path)
    return 100 * sum(1 for _, c in fs if len(c.get("attackers") or []) == 1) / len(fs) if fs else None


def strength_ratio(path: Path, per_attacker: bool = False) -> float | None:
    """Mean power advantage of the attacking side.

    By default the coalition's combined power over the defender's, which is at
    least 1.0 for a solo attack of equal strength and scales with the number of
    attackers. With `per_attacker` it is the mean individual attacker's power
    over the defender's, which does not. The chapter reports figures near 1.0
    and near 8.7 in different places; they are these two quantities.
    """
    v = []
    for _, c in fights(path):
        d = c.get("defender_power")
        if not d:
            continue
        if per_attacker:
            m = c.get("attacker_powers") or {}
            if m:
                v.append((sum(m.values()) / len(m)) / d)
        else:
            v.append((c.get("coalition_power") or 0) / d)
    return sum(v) / len(v) if v else None


def win_rate(path: Path) -> float | None:
    """Share of fights won by the attacking side."""
    fs = fights(path)
    return 100 * sum(1 for _, c in fs if c.get("winner") == "coalition") / len(fs) if fs else None


def state_at(path: Path, ronde: int | None) -> dict | None:
    """Holdings as recorded *at the end of* a given round.

    The round record is end-of-round state: a defender's `resources` in the
    round it was struck already has the loss taken out of it. Anything asking
    what the board looked like *when* something happened wants `state_before`.
    """
    if ronde is None:
        return None
    for e in rounds(path):
        if e.get("round") == ronde:
            return {nm: (a.get("resources") or 0.0)
                    for nm, a in (e.get("agents") or {}).items()}
    return None


def state_before(path: Path, ronde: int | None) -> dict | None:
    """Holdings as they stood going *into* a round.

    This is the state a decision was made against, and it is what every measure
    asking "where did the target stand when it was struck" needs. Reading the
    round itself gave the position after the blow: at L3 knife-edge the target's
    share of the top three read 35.3 per cent measured after and 49.7 before,
    and the Gini in the round of the first fight read 0.035 after and 0.005
    before --- which is to say the inequality being reported was the fight.

    Round 1 has no predecessor, so it returns the engine's opening state where
    that is recorded and None otherwise.
    """
    if ronde is None:
        return None
    return state_at(path, ronde - 1)
