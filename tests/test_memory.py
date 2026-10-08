"""Unit tests for the two-layer agent memory.

Layer 1 — the agent's own end-of-round notes persist for the whole game
(never windowed). Layer 2 — raw objective events use a recency window of
`window_size` rounds. `notes_persist=False` reverts to the legacy
single-window behaviour.
"""

from agents.memory import AgentMemory


def _fill(mem, n_rounds, note_prefix="note"):
    """Record n_rounds of a simple invest exchange, each with a note."""
    for r in range(1, n_rounds + 1):
        mem.record_round(
            r,
            {"action": "invest_other", "target": "Red", "outcome": {}},
            [{"agent": "Red", "action": "invest_other", "target": "Blue"}],
            all_resources={"Blue": 25.0, "Red": 30.0},
        )
        mem.record_memory(r, f"{note_prefix}-r{r}")


# ── empty / edge ────────────────────────────────────────────────────────

def test_empty_memory_returns_empty():
    assert AgentMemory("Blue").format_recent_rounds() == ""


def test_note_truncated_to_500():
    mem = AgentMemory("Blue")
    mem.record_memory(1, "x" * 600)
    assert len(mem.memory_stream[0].text) == 500


def test_empty_note_skipped():
    mem = AgentMemory("Blue")
    mem.record_round(1, {"action": "do_nothing", "target": None, "outcome": {}}, [])
    mem.record_memory(1, "")
    mem.record_memory(1, None)
    assert mem.memory_stream == []
    assert "Your note" not in mem.format_recent_rounds()


def test_notes_only_no_events():
    """Notes with no recorded events (e.g. before first resolve) still show."""
    mem = AgentMemory("Blue", window_size=10)
    for r in range(1, 4):
        mem.record_memory(r, f"note-r{r}")
    out = mem.format_recent_rounds()
    assert "YOUR NOTES SO FAR" in out
    assert "RECENT ROUNDS" not in out
    assert 'Round 2: "note-r2"' in out


# ── layer behaviour ─────────────────────────────────────────────────────

def test_notes_persist_beyond_window():
    mem = AgentMemory("Blue", window_size=10, notes_persist=True)
    _fill(mem, 15)
    out = mem.format_recent_rounds()
    assert "YOUR NOTES SO FAR" in out
    assert 'Round 1: "note-r1"' in out            # early note survives
    assert "RECENT ROUNDS (last 10" in out
    assert "Round 15:" in out                      # recent event detail present
    assert "Round 1:\n  You:" not in out           # round 1 has no event block
    assert out.count("You: invest other") == 10    # only windowed rounds detailed


def test_within_window_inline_only():
    mem = AgentMemory("Blue", window_size=10)
    _fill(mem, 5)
    out = mem.format_recent_rounds()
    assert "YOUR NOTES SO FAR" not in out          # nothing older than the window
    assert "RECENT ROUNDS (last 5" in out
    assert 'Your note: "note-r1"' in out           # notes shown inline


def test_notes_windowed_when_disabled():
    mem = AgentMemory("Blue", window_size=10, notes_persist=False)
    _fill(mem, 15)
    out = mem.format_recent_rounds()
    assert "YOUR NOTES SO FAR" not in out
    assert 'note-r1"' not in out                    # legacy: round-1 note windowed out
    assert 'note-r15"' in out


# ── serialization / checkpoint compat ───────────────────────────────────

def test_serialization_round_trip():
    mem = AgentMemory("Blue", window_size=7, notes_persist=True)
    _fill(mem, 12)
    mem2 = AgentMemory.from_dict(mem.to_dict())
    assert mem2.window_size == 7
    assert mem2.notes_persist is True
    assert len(mem2.event_log) == len(mem.event_log)
    assert len(mem2.memory_stream) == len(mem.memory_stream)
    assert mem2.format_recent_rounds() == mem.format_recent_rounds()


def test_from_dict_defaults_notes_persist_true():
    """A legacy checkpoint without the notes_persist key defaults to True."""
    mem = AgentMemory("Blue", window_size=5)
    _fill(mem, 3)
    d = mem.to_dict()
    d.pop("notes_persist")
    assert AgentMemory.from_dict(d).notes_persist is True
