"""Lethal-pot spoils: phi = min(1, alpha * S_winner / S_loser), loser-sized.

The coalition winner is forced by patching np.random.random (0.0 -> coalition
wins; 1.0 -> defender wins), so the phi maths is tested deterministically off
the value logged in combat_results. No LLM, no randomness in the assertions.
"""
import numpy as np
import pytest

from game.engine import GameEngine, Action, ActionType

ALPHA = 0.33
R_DEF = 100.0


def _combat(attackers, lethal, force="coalition", alpha=ALPHA):
    """attackers: dict id->resources vs one 'Def' with R_DEF. Force winner, return combat log."""
    ids = list(attackers) + ["Def"]
    res = dict(attackers); res["Def"] = R_DEF
    eng = GameEngine(agent_ids=ids, initial_resources=res,
                     alpha=alpha, c_atk=0.0, mu_arm=0.0, delta_R=1.0,
                     lethal_pot=lethal)
    acts = [Action(a, ActionType.ATTACK, "Def") for a in attackers]
    acts.append(Action("Def", ActionType.DO_NOTHING, None))
    val = 0.0 if force == "coalition" else 1.0
    orig = np.random.random
    np.random.random = lambda *a, **k: val
    try:
        log = eng.resolve_round(acts)
    finally:
        np.random.random = orig
    return log["combat_results"][0]


@pytest.mark.parametrize("n_atk,exp_phi", [
    (1, ALPHA),                  # 1v1 equal -> phi = alpha (cell placement preserved)
    (3, min(1.0, ALPHA * 3)),    # 3v1 -> 0.99
    (4, 1.0),                    # 4v1 -> capped at 1.0 == KILL
])
def test_lethal_coalition_phi_scales(n_atk, exp_phi):
    cb = _combat({f"a{i}": 100.0 for i in range(n_atk)}, lethal=True)
    assert cb["phi"] == pytest.approx(exp_phi)
    assert cb["total_transfer"] == pytest.approx(R_DEF * exp_phi)


def test_lethal_kill_eliminates_defender():
    cb = _combat({f"a{i}": 100.0 for i in range(4)}, lethal=True)
    assert cb["phi"] == pytest.approx(1.0)          # drained to zero
    assert cb["total_transfer"] == pytest.approx(R_DEF)


def test_flat_pot_caps_at_alpha_no_kill():
    # 14-vs-1 overmatch, flat pot (the actual VSC R8 situation): still only alpha.
    cb = _combat({f"a{i}": 80.0 for i in range(14)}, lethal=False)
    assert cb["phi"] == pytest.approx(ALPHA)
    assert cb["total_transfer"] == pytest.approx(R_DEF * ALPHA)


def test_lethal_symmetric_defender_win_drains_attacker():
    # Defender wins: attacker loses phi of its OWN R (loser-sized, symmetric).
    cb = _combat({"a0": 100.0}, lethal=True, force="defender")
    assert cb["phi"] == pytest.approx(ALPHA)        # equal strength -> alpha
    assert cb["total_transfer"] == pytest.approx(100.0 * ALPHA)
