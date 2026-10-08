"""Action-vocabulary tests.

Canonical actions: transfer / strengthen / take / hold. Legacy names
(invest_other / arm_other / attack / do_nothing) are accepted on input and
normalized to canonical. arm_self is not an available action.
"""

from types import SimpleNamespace

from agents.llm_agent import LLMAgent


def _norm(s):
    dummy = SimpleNamespace(_VALID_ACTIONS=LLMAgent._VALID_ACTIONS,
                            _ACTION_ALIASES=LLMAgent._ACTION_ALIASES)
    return LLMAgent._normalize_action(dummy, s)


def test_canonical_action_set():
    # 'harvest' is only a real action under commons_harvest_mode="action_own"
    # (gated in _action_dict_to_action); it is in the canonical vocabulary so the
    # normaliser recognises it rather than dropping it.
    assert LLMAgent._VALID_ACTIONS == {'transfer', 'strengthen', 'take', 'hold', 'harvest',
                                        'drop', 'invite'}
    for a in ('transfer', 'strengthen', 'take', 'hold', 'harvest'):
        assert _norm(a) == a


def test_legacy_names_map_to_canonical():
    assert _norm('invest_other') == 'transfer'
    assert _norm('arm_other') == 'strengthen'
    assert _norm('attack') == 'take'
    assert _norm('do_nothing') == 'hold'


def test_shorthands_and_fuzz():
    assert _norm('invest') == 'transfer'
    assert _norm('arm') == 'strengthen'
    assert _norm('Take ') == 'take'
    assert _norm('transfers') == 'transfer'   # trailing-s strip
    assert _norm('do nothing') == 'hold'


def test_arm_self_not_available():
    assert 'arm_self' not in LLMAgent._VALID_ACTIONS
    assert _norm('arm_self') is None
    assert _norm('arm self') is None


# ── assoc_rewire_mode="action": drop/invite als actie (A/B-probe) ────────────

from game.engine import ActionType


def _agent_stub(visible=("B",)):
    from types import SimpleNamespace
    stub = SimpleNamespace(
        agent_id="A",
        game_params={"assoc_enabled": True,
                     "arm_enabled": True, "take_enabled": True},
        _visible_agents=list(visible),
        _last_rewire_nom=None,
        _VALID_ACTIONS=LLMAgent._VALID_ACTIONS,
        _ACTION_ALIASES=LLMAgent._ACTION_ALIASES,
    )
    stub._normalize_action = lambda s: LLMAgent._normalize_action(stub, s)
    stub._snap_harvest = lambda raw: 0.0
    return stub


def _to_action(stub, d):
    return LLMAgent._action_dict_to_action(stub, d)


def test_action_mode_drop_is_noop_plus_nomination():
    stub = _agent_stub()
    a = _to_action(stub, {"action": "drop", "target": "B"})
    assert a.action_type == ActionType.DO_NOTHING and a.target_id is None
    assert stub._last_rewire_nom == {"drop": "B", "invite": None}


def test_action_mode_invite_nonneighbor_allowed():
    stub = _agent_stub(visible=("B",))
    a = _to_action(stub, {"action": "invite", "target": "C"})
    assert a.action_type == ActionType.DO_NOTHING
    assert stub._last_rewire_nom == {"drop": None, "invite": "C"}


def test_action_mode_drop_requires_neighbor():
    stub = _agent_stub(visible=("B",))
    assert _to_action(stub, {"action": "drop", "target": "C"}) is None


