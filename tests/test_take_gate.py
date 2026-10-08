"""take_enabled gate: below the predation rung, attacks do not exist —
the engine neutralises any ATTACK to a no-op and no combat resolves."""
import numpy as np
import pytest

from game.engine import GameEngine, Action, ActionType


def test_take_disabled_neutralises_attack():
    eng = GameEngine(["A", "B"], 100.0, delta_R=1.0, mu_arm=0.0, take_enabled=False)
    log = eng.resolve_round([Action("A", ActionType.ATTACK, "B"),
                             Action("B", ActionType.DO_NOTHING, None)])
    assert eng.state.resources["B"] == pytest.approx(100.0)   # defender untouched
    assert not log["combat_results"]                          # no combat resolved


def test_take_enabled_allows_attack():
    np.random.seed(0)
    eng = GameEngine(["A", "B"], 100.0, delta_R=1.0, mu_arm=0.0, take_enabled=True)
    log = eng.resolve_round([Action("A", ActionType.ATTACK, "B"),
                             Action("B", ActionType.DO_NOTHING, None)])
    assert log["combat_results"]                              # combat happened


def test_take_enabled_default_true():
    eng = GameEngine(["A", "B"], 100.0)
    assert eng.params["take_enabled"] is True                 # backward-compatible default
