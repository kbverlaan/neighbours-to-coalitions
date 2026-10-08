"""Complexity-ladder gating: each rung adds exactly one affordance set via an
independent boolean flag, and a disabled affordance is fully neutralised in
(a) the engine, (b) action parsing, and (c) the rendered prompt.

Flag matrix:
    rung  arm_enabled  take_enabled  assoc_enabled  commons_enabled  rewiring_prob
    L1    False        False         False          False            0.0
    L2    True         True          False          False            0.0
    L3    True         True          True           False            0.5
    L4    True         True          True           True             0.5
"""
import numpy as np
import pytest

from game.engine import GameEngine, Action, ActionType
from agents.prompts import BaselinePrompt
from agents.llm_agent import LLMAgent


# ── Flag matrices per rung (only the rung flags; numeric params left default) ──
RUNGS = {
    "L1": dict(arm_enabled=False, take_enabled=False, assoc_enabled=False,
               commons_enabled=False, rewiring_prob=0.0),
    "L2": dict(arm_enabled=True,  take_enabled=True,  assoc_enabled=False,
               commons_enabled=False, rewiring_prob=0.0),
    "L3": dict(arm_enabled=True,  take_enabled=True,  assoc_enabled=True,
               commons_enabled=False, rewiring_prob=0.5),
    "L4": dict(arm_enabled=True,  take_enabled=True,  assoc_enabled=True,
               commons_enabled=True,  rewiring_prob=0.5),
}


def _engine(flags):
    return GameEngine(
        ["A", "B"], 100.0, delta_R=1.0,
        arm_enabled=flags["arm_enabled"],
        take_enabled=flags["take_enabled"],
        assoc_enabled=flags["assoc_enabled"],
        commons_enabled=flags["commons_enabled"],
    )


def _game_params(flags):
    """Full game_params dict the prompt + agent read, with comms on (substrate)."""
    return dict(
        num_agents=2, mu_arm=3.0, c_arm=0.05, alpha=0.35, c_atk=0.02,
        delta_R=1.0, symmetric_stakes=True, network_enabled=True,
        comm_scope="choice", commons_harvest_pct=[0, 1, 2, 4, 8],
        **flags,
    )


def _prompt_text(flags):
    bp = BaselinePrompt(
        game_params=_game_params(flags),
        comm_scope="choice", network_enabled=True,
    )
    obs = {
        "agent_id": "A", "round": 1,
        "resources": {"A": 100.0, "B": 50.0},
        "visible_agents": ["B"],
        "arm_bonuses": {}, "broke_agents": [],
        "agent_memory": None,
        "commons": ({"stock_pct": 80.0} if flags["commons_enabled"] else None),
    }
    return bp.format_observation(obs, "A")


def _agent(flags):
    return LLMAgent(
        agent_id="A", api_key="none", model="test",
        game_params=_game_params(flags), memory_config={"enabled": False},
    )


# ── (a) engine exposes exactly the right action set (no-ops for disabled) ──────

def test_engine_arm_noop_when_disabled():
    eng = _engine(RUNGS["L1"])
    log = eng.resolve_round([Action("A", ActionType.ARM_OTHER, "B"),
                             Action("B", ActionType.DO_NOTHING, None)])
    # strengthen neutralised → no arm cost paid, no bonus granted
    assert eng.state.resources["A"] == pytest.approx(100.0)
    assert eng.state.arm_bonuses.get("B", 0.0) == pytest.approx(0.0)


def test_engine_arm_active_when_enabled():
    eng = _engine(RUNGS["L2"])
    eng.resolve_round([Action("A", ActionType.ARM_OTHER, "B"),
                       Action("B", ActionType.DO_NOTHING, None)])
    assert eng.state.resources["A"] < 100.0            # cost paid
    assert eng.state.arm_bonuses.get("B", 0.0) > 0.0   # bonus granted


def test_engine_take_noop_on_L1():
    eng = _engine(RUNGS["L1"])
    log = eng.resolve_round([Action("A", ActionType.ATTACK, "B"),
                             Action("B", ActionType.DO_NOTHING, None)])
    assert eng.state.resources["B"] == pytest.approx(100.0)
    assert not log["combat_results"]


def test_engine_take_active_from_L2():
    np.random.seed(0)
    eng = _engine(RUNGS["L2"])
    log = eng.resolve_round([Action("A", ActionType.ATTACK, "B"),
                             Action("B", ActionType.DO_NOTHING, None)])
    assert log["combat_results"]


def test_engine_flags_recorded():
    for name, flags in RUNGS.items():
        eng = _engine(flags)
        assert eng.params["arm_enabled"] == flags["arm_enabled"], name
        assert eng.params["take_enabled"] == flags["take_enabled"], name
        assert eng.params["assoc_enabled"] == flags["assoc_enabled"], name
        assert eng.params["commons_enabled"] == flags["commons_enabled"], name


def test_engine_defaults_backward_compatible():
    eng = GameEngine(["A", "B"], 100.0)
    assert eng.params["arm_enabled"] is True
    assert eng.params["take_enabled"] is True
    assert eng.params["assoc_enabled"] is True
    assert eng.params["commons_enabled"] is False


# ── (b) parsing: disabled actions snap to hold / drop the field ───────────────

def test_parse_strengthen_dropped_on_L1():
    agent = _agent(RUNGS["L1"])
    act = agent._action_dict_to_action({"action": "strengthen", "target": "B"})
    assert act.action_type == ActionType.DO_NOTHING


def test_parse_take_dropped_on_L1():
    agent = _agent(RUNGS["L1"])
    act = agent._action_dict_to_action({"action": "take", "target": "B"})
    assert act.action_type == ActionType.DO_NOTHING


def test_parse_strengthen_kept_on_L2():
    agent = _agent(RUNGS["L2"])
    act = agent._action_dict_to_action({"action": "strengthen", "target": "B"})
    assert act.action_type == ActionType.ARM_OTHER


def test_parse_rewire_dropped_when_assoc_off():
    # below L3 the drop/invite action does not exist -> parse fails (retry path)
    agent = _agent(RUNGS["L2"])
    assert agent._action_dict_to_action({"action": "invite", "target": "C"}) is None
    assert agent.get_last_rewire_nomination() is None


def test_parse_rewire_kept_when_assoc_on():
    agent = _agent(RUNGS["L3"])
    agent._visible_agents = ["B"]
    act = agent._action_dict_to_action({"action": "invite", "target": "C"})
    assert act.action_type == ActionType.DO_NOTHING
    assert agent.get_last_rewire_nomination() == {"drop": None, "invite": "C"}
    act = agent._action_dict_to_action({"action": "drop", "target": "B"})
    assert agent.get_last_rewire_nomination() == {"drop": "B", "invite": None}


def test_parse_harvest_snaps_to_zero_when_commons_off():
    agent = _agent(RUNGS["L3"])
    assert agent._snap_harvest(4) == 0.0
    agent4 = _agent(RUNGS["L4"])
    assert agent4._snap_harvest(4) == pytest.approx(0.04)  # "4" = 4% of own -> fraction


# ── (c) prompt: disabled mechanics' keywords absent, enabled ones present ──────

ARM_KW = ["strengthen", "arm bonus", "combat bonus"]
TAKE_KW = ["take:", "COMBAT", "Win probability", "BANKRUPTCY"]
ASSOC_KW = ["REWIRING", "- drop:", "- invite:"]
COMMONS_KW = ["SHARED STOCK", "HARVEST", "harvest"]


def _assert_absent(text, kws, label):
    for kw in kws:
        assert kw not in text, f"{label}: unexpected '{kw}' in prompt"


def _assert_present(text, kws, label):
    for kw in kws:
        assert kw in text, f"{label}: missing '{kw}' in prompt"


def test_prompt_L1_clean_cooperation_only():
    t = _prompt_text(RUNGS["L1"])
    _assert_absent(t, ARM_KW, "L1")
    _assert_absent(t, TAKE_KW, "L1")
    _assert_absent(t, ASSOC_KW, "L1")
    _assert_absent(t, COMMONS_KW, "L1")
    # enabled: transfer + hold + the action field
    assert "transfer:" in t
    assert "hold:" in t
    assert '"action"' in t


def test_prompt_L2_adds_arm_and_take_only():
    t = _prompt_text(RUNGS["L2"])
    _assert_present(t, ["strengthen", "take:", "COMBAT"], "L2")
    _assert_absent(t, ASSOC_KW, "L2")
    _assert_absent(t, COMMONS_KW, "L2")


def test_prompt_L3_adds_association_only():
    t = _prompt_text(RUNGS["L3"])
    _assert_present(t, ["strengthen", "COMBAT", "REWIRING", "- drop:", "- invite:"], "L3")
    _assert_absent(t, COMMONS_KW, "L3")


def test_prompt_L4_full_ladder():
    t = _prompt_text(RUNGS["L4"])
    _assert_present(t, ["strengthen", "COMBAT", "REWIRING", "SHARED STOCK", "harvest"], "L4")
