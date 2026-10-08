"""Harvest-claim parsing and fraction arithmetic (production behaviour): harvest
is an ACTION; claim = continuous, unbounded fraction of OWN resources, parsed
from a percent field. Deterministic (no LLM)."""
from types import SimpleNamespace

import numpy as np
import pytest

from game.engine import GameEngine, Action, ActionType
from agents.llm_agent import LLMAgent


def _snap(game_params, raw):
    dummy = SimpleNamespace(game_params=game_params)
    return LLMAgent._snap_harvest(dummy, raw)


def _engine(agents, K=100.0, init=None, collapse_frac=0.05, R0=100.0):
    return GameEngine(agent_ids=agents, initial_resources=R0,
                      delta_R=1.0, mu_arm=0.0,
                      commons_enabled=True, commons_K=K, commons_init=init,
                      commons_collapse_frac=collapse_frac)


def _harvest_round(eng, harvests):
    acts = [Action(a, ActionType.HARVEST if a in harvests else ActionType.DO_NOTHING,
                   None, harvest=harvests.get(a, 0.0))
            for a in eng.state.agents]
    return eng.resolve_round(acts)


# --- fraction arithmetic: harvest = f_i x R_i from the pool -------------------

def test_fraction_harvest_math():
    a = ["A", "B"]
    eng = _engine(a, K=600.0, init=600.0)
    log = _harvest_round(eng, {"A": 0.10, "B": 0.10})
    assert log["commons"]["harvested"] == pytest.approx(20.0)
    assert eng.state.resources["A"] == pytest.approx(110.0)
    assert eng.state.commons_stock == pytest.approx(600.0)  # 580 -> regen capped K


def test_fraction_scales_with_own_resources():
    a = ["Rich", "Poor"]
    eng = GameEngine(agent_ids=a, initial_resources={"Rich": 200.0, "Poor": 50.0},
                     delta_R=1.0, mu_arm=0.0,
                     commons_enabled=True, commons_K=600.0, commons_init=600.0)
    _harvest_round(eng, {"Rich": 0.10, "Poor": 0.10})
    assert eng.state.resources["Rich"] == pytest.approx(220.0)  # +20
    assert eng.state.resources["Poor"] == pytest.approx(55.0)   # +5


def test_unbounded_grab_empties_pool_and_collapses():
    a = ["A", "B"]
    eng = _engine(a, K=100.0, init=100.0)
    _harvest_round(eng, {"A": 1.0})       # 100% of own 100 = the whole pool
    assert eng.state.commons_collapsed is True
    assert eng.state.commons_stock == pytest.approx(0.0)


def test_rationing_when_request_exceeds_stock():
    np.random.seed(0)
    a = ["A", "B", "C"]
    eng = _engine(a, K=100.0, init=25.0)
    log = _harvest_round(eng, {x: 0.20 for x in a})  # claims 60 > 25
    assert log["commons"]["harvested"] == pytest.approx(25.0)


def test_chosen_fraction_is_logged():
    a = ["A", "B"]
    eng = _engine(a, K=600.0, init=600.0)
    log = _harvest_round(eng, {"A": 0.07})
    assert log["commons"]["harvest_frac"]["A"] == pytest.approx(0.07)
    assert log["resource_breakdown"]["A"]["harvest_frac"] == pytest.approx(0.07)


def test_non_harvest_action_never_draws():
    # hold with harvest field > 0 draws NOTHING (harvest is the action itself)
    a = ["A", "B"]
    eng = _engine(a, K=600.0, init=600.0)
    log = eng.resolve_round([Action("A", ActionType.DO_NOTHING, None, harvest=0.5),
                             Action("B", ActionType.DO_NOTHING, None)])
    assert log["commons"]["harvested"] == pytest.approx(0.0)
    assert eng.state.resources["A"] == pytest.approx(100.0)


# --- parsing: percent field -> fraction ---------------------------------------

def test_parse_percent_semantics():
    gp = {"commons_enabled": True}
    assert _snap(gp, "5") == pytest.approx(0.05)
    assert _snap(gp, "5%") == pytest.approx(0.05)
    assert _snap(gp, 12.5) == pytest.approx(0.125)
    assert _snap(gp, "150") == pytest.approx(1.50)   # no upper bound


def test_negative_fraction_field_clamped_to_zero():
    gp = {"commons_enabled": True}
    assert _snap(gp, -3) == 0.0
    assert _snap(gp, "garbage") == 0.0
    assert _snap(gp, None) == 0.0


def test_parse_zero_when_commons_off():
    assert _snap({"commons_enabled": False}, "5") == 0.0
