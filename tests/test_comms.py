"""Unit tests for target-list communication.

A message carries an explicit LIST of recipient ids (message_to). There is no
broadcast channel and no "all" keyword: to reach many agents the model lists
many ids. Delivery is gated to living agents in the runner; recipients see how
many others also received the message (n_recipients), not the full list.
"""

from types import SimpleNamespace

from agents.memory import AgentMemory
from agents.llm_agent import LLMAgent


# ── message_to normalization (LLMAgent._store_message) ──────────────────

def _store(agent_id, action):
    dummy = SimpleNamespace(agent_id=agent_id, _last_message=None)
    LLMAgent._store_message(dummy, action)
    return dummy._last_message


def test_message_to_list_normalized():
    msg = _store("Blue", {"message": "hi", "message_to": ["Red", "Green"]})
    assert msg["message_to"] == ["Red", "Green"]
    assert msg["from"] == "Blue"


def test_message_to_single_string_wrapped():
    msg = _store("Blue", {"message": "hi", "message_to": "Red"})
    assert msg["message_to"] == ["Red"]


def test_message_to_drops_all_self_blanks_dupes():
    msg = _store("Blue", {"message": "hi",
                          "message_to": ["Red", "all", "Blue", "", "Red", "Green"]})
    assert msg["message_to"] == ["Red", "Green"]


def test_silent_when_no_text_or_no_targets():
    assert _store("Blue", {"message": "", "message_to": ["Red"]}) is None
    assert _store("Blue", {"message": "hi", "message_to": []}) is None
    assert _store("Blue", {"message": "hi", "message_to": ["all", "Blue"]}) is None


# ── received / sent rendering ────────────────────────────────────────────

def test_received_label_private_vs_wide():
    mem = AgentMemory("Blue", window_size=10)
    mem.record_round(
        1, {"action": "do_nothing", "target": None, "outcome": {}}, [],
        received_messages=[
            {"from": "Red", "message": "secret", "n_recipients": 1},
            {"from": "Green", "message": "hello all", "n_recipients": 4},
        ],
    )
    out = mem.format_recent_rounds()
    assert 'Red → to you only: "secret"' in out
    assert 'Green → to you + 3 others: "hello all"' in out


def test_sent_label_lists_targets():
    mem = AgentMemory("Blue", window_size=10)
    mem.record_round(
        1, {"action": "do_nothing", "target": None, "outcome": {}}, [],
        sent_message={"message": "hi", "message_to": ["Red", "Green"]},
    )
    assert 'You sent (to Red, Green): "hi"' in mem.format_recent_rounds()


def test_clean_received_defaults_n_recipients():
    cleaned = AgentMemory._clean_received([{"from": "Red", "message": "x"}])
    assert cleaned[0]["n_recipients"] == 1
