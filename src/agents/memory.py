"""
Per-agent memory: a chronological event log (objective events per round
— actions, messages, network changes) plus a free-form memory stream
(the agent's own note about each round).

The agent's own end-of-round notes persist for the WHOLE game (never
windowed) so durable regularities — agreements, recurring patterns,
grudges — survive arbitrarily long. Detailed objective events (own
action, incoming/observed actions, incoming/outgoing messages, rewiring
outcomes, resources) are shown only for the last `window_size` rounds
(default 10). Full logs persist to the canonical `_log.jsonl` regardless
of what fell out of the event window.

Design references:
- Memory stream inspired by Generative Agents (Park et al., 2023),
  simplified: persistent agent-authored notes + a recency window for raw
  events; no embedding retrieval scoring.
- Local information follows Harsanyi (1967-68) incomplete-information
  framing: actions targeting the agent are always known; third-party
  actions only if actor or target is within the agent's visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RoundEvents:
    """Objective record of everything this agent witnessed in one round."""
    round: int
    own_action: Optional[dict] = None               # {action, target, outcome}
    incoming: List[dict] = field(default_factory=list)      # [{actor, action}]
    observed: List[dict] = field(default_factory=list)      # [{actor, action, target}]
    resources_snapshot: Dict[str, float] = field(default_factory=dict)  # {aid: resources} self + visible
    sent_message: Optional[dict] = None             # {'to': str|'all', 'text': str}
    received_messages: List[dict] = field(default_factory=list)  # [{'from', 'text', 'n_recipients'}]
    rewire: Optional[dict] = None                   # {'drop', 'invite', 'drop_outcome', 'invite_outcome'}

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "own_action": self.own_action,
            "incoming": list(self.incoming),
            "observed": list(self.observed),
            "resources_snapshot": dict(self.resources_snapshot),
            "sent_message": self.sent_message,
            "received_messages": list(self.received_messages),
            "rewire": self.rewire,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoundEvents":
        return cls(
            round=d.get("round", 0),
            own_action=d.get("own_action"),
            incoming=list(d.get("incoming", [])),
            observed=list(d.get("observed", [])),
            resources_snapshot=dict(d.get("resources_snapshot", {})),
            sent_message=d.get("sent_message"),
            received_messages=list(d.get("received_messages", [])),
            rewire=d.get("rewire"),
        )


@dataclass
class MemoryEntry:
    """A single free-form memory note, written by the agent at end of round."""
    round: int
    text: str

    def to_dict(self) -> dict:
        return {"round": self.round, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(round=d.get("round", 0), text=d.get("text", ""))


class AgentMemory:
    """Per-agent memory: integrated per-round event log + memory stream."""

    def __init__(self, agent_id: str, window_size: int = 10,
                 notes_persist: bool = True):
        self.agent_id = agent_id
        self.window_size = window_size
        # When True, the agent's own notes are never windowed (kept for the
        # whole game); raw events still use the recency window. When False,
        # notes fall back to the same window as events (legacy behaviour).
        self.notes_persist = notes_persist
        self.event_log: List[RoundEvents] = []
        self.memory_stream: List[MemoryEntry] = []

    # ── writes ─────────────────────────────────────────────────────────

    def record_round(self, round_num: int, own_action: Optional[dict],
                     round_actions: List[dict],
                     visible_agents: Optional[List[str]] = None,
                     all_resources: Optional[Dict[str, float]] = None,
                     sent_message: Optional[dict] = None,
                     received_messages: Optional[List[dict]] = None,
                     rewire: Optional[dict] = None):
        """Record everything this agent witnessed in one round.

        Args:
            round_num: Round number just resolved.
            own_action: Dict {action, target, outcome} for this agent, or None.
            round_actions: All actions from the round log (each with
                           {agent, action, target}).
            visible_agents: Agent IDs within this agent's visibility radius
                            this round (None = fully visible, no network).
            all_resources: Post-round resources for all agents. The snapshot
                           stored with this round keeps only self + visible.
            sent_message: {'to': agent_id|'all', 'text': str} for this agent's
                          outgoing message this round, or None.
            received_messages: List of {'from': agent_id, 'text': str,
                                        'n_recipients': int} received this round
                               (sent by others in the previous round).
            rewire: {'drop': agent_id|None, 'invite': agent_id|None,
                    'drop_outcome': str|None, 'invite_outcome': str|None} — the
                    agent's rewiring nomination + engine outcome for this round.
        """
        visible_set = set(visible_agents) if visible_agents is not None else None
        incoming: List[dict] = []
        observed: List[dict] = []

        for entry in round_actions:
            actor = entry.get("agent")
            act = entry.get("action", "")
            target = entry.get("target")

            if not act or act in ("no_action", ""):
                continue
            if actor == self.agent_id:
                continue  # own action captured separately

            if target == self.agent_id:
                incoming.append({"actor": actor, "action": act})
                continue

            # Visible if EITHER actor or target is a neighbour (Harsanyi
            # incomplete-information: you perceive events that touch your
            # neighbourhood). Agent IDs are globally known — resource
            # values for unseen agents are hidden via the snapshot.
            if visible_set is None:
                observed.append({"actor": actor, "action": act, "target": target})
            else:
                actor_vis = actor in visible_set
                target_vis = target is not None and target in visible_set
                if actor_vis or target_vis:
                    observed.append({"actor": actor, "action": act, "target": target})

        snapshot: Dict[str, float] = {}
        if all_resources:
            if self.agent_id in all_resources:
                snapshot[self.agent_id] = float(all_resources[self.agent_id])
            if visible_set is None:
                for aid, v in all_resources.items():
                    snapshot[aid] = float(v)
            else:
                for aid in visible_set:
                    if aid in all_resources:
                        snapshot[aid] = float(all_resources[aid])

        self.event_log.append(RoundEvents(
            round=round_num,
            own_action=own_action,
            incoming=incoming,
            observed=observed,
            resources_snapshot=snapshot,
            sent_message=self._clean_sent(sent_message),
            received_messages=self._clean_received(received_messages or []),
            rewire=self._clean_rewire(rewire),
        ))

    def record_memory(self, round_num: int, text: Optional[str]):
        """Record the agent's free-form memory note for this round.

        Empty or None is skipped — the block for that round simply won't
        show a 'Your note' line.
        """
        if not text or not isinstance(text, str):
            return
        trimmed = text.strip()[:500]
        if not trimmed:
            return
        self.memory_stream.append(MemoryEntry(round=round_num, text=trimmed))

    # ── reads (prompt-facing formatter) ────────────────────────────────

    def format_recent_rounds(self) -> str:
        """Format memory for the prompt, in two layers.

        - The agent's OWN notes persist for the whole game (never
          windowed): notes for rounds older than the event window are
          shown compactly up front, so durable regularities (agreements,
          recurring patterns, grudges) survive arbitrarily long.
        - Detailed objective events (actions, messages, network,
          resources) are shown only for the last `window_size` rounds.
        """
        if not self.event_log and not self.memory_stream:
            return ""

        recent_events = self.event_log[-self.window_size:]
        recent_rounds = {ev.round for ev in recent_events}
        notes_src = (
            self.memory_stream if self.notes_persist
            else self.memory_stream[-self.window_size:]
        )
        notes_by_round = {m.round: m.text for m in notes_src}

        out: List[str] = []

        # ── Layer 1: persistent note history (rounds before the window) ──
        earlier_notes = [
            m for m in notes_src if m.round not in recent_rounds
        ]
        if earlier_notes:
            nlines = [
                f"YOUR NOTES SO FAR (rounds {earlier_notes[0].round}-"
                f"{earlier_notes[-1].round}, kept for the whole game):"
            ]
            for m in earlier_notes:
                nlines.append(f"  Round {m.round}: \"{m.text}\"")
            out.append("\n".join(nlines))

        # ── Layer 2: recent rounds in full detail (events + that note) ───
        if recent_events:
            lines = [f"RECENT ROUNDS (last {len(recent_events)}, full detail):"]
            for ev in recent_events:
                lines.append("")
                lines.append(f"Round {ev.round}:")

                if ev.own_action:
                    lines.append(f"  You: {self._format_own_action(ev.own_action)}")

                if ev.incoming:
                    recv = ", ".join(
                        f"{e['actor']} {e['action'].replace('_', ' ')} you"
                        for e in ev.incoming
                    )
                    lines.append(f"  Received: {recv}")
                else:
                    lines.append("  Received: none")

                if ev.observed:
                    obs = ", ".join(self._format_observed(e) for e in ev.observed)
                    lines.append(f"  Observed: {obs}")

                if ev.resources_snapshot:
                    snap_parts = []
                    items = sorted(ev.resources_snapshot.items(),
                                   key=lambda kv: (kv[0] != self.agent_id, kv[0]))
                    for aid, val in items:
                        snap_parts.append(f"{aid} {val:.1f}")
                    lines.append(f"  Resources: {', '.join(snap_parts)}")

                if ev.sent_message:
                    to = ev.sent_message.get("to") or []
                    to_str = ", ".join(to) if isinstance(to, list) else str(to)
                    lines.append(
                        f"  You sent (to {to_str or '?'}): "
                        f"\"{ev.sent_message.get('text', '')}\""
                    )

                if ev.received_messages:
                    for m in ev.received_messages:
                        sender = m.get("from", "?")
                        n = int(m.get("n_recipients", 1) or 1)
                        label = "to you only" if n <= 1 else f"to you + {n - 1} others"
                        lines.append(
                            f"  {sender} → {label}: \"{m.get('text', '')}\""
                        )

                if ev.rewire:
                    lines.append(f"  Network: {self._format_rewire(ev.rewire)}")

                note = notes_by_round.get(ev.round)
                if note:
                    lines.append(f"  Your note: \"{note}\"")

            out.append("\n".join(lines))

        return "\n\n".join(out)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_own_action(own: dict) -> str:
        act = own.get("action", "").replace("_", " ")
        target = own.get("target")
        outcome = own.get("outcome") or {}
        parts = [act]
        if target:
            parts.append(f"→ {target}")
        details = []
        if "combat_won" in outcome and own.get("action") == "attack":
            details.append("won" if outcome["combat_won"] else "lost")
        rc = outcome.get("resource_change")
        if rc is not None and abs(rc) > 0.05:
            sign = "+" if rc >= 0 else ""
            details.append(f"{sign}{rc:.1f}")
        desc = " ".join(parts)
        if details:
            desc += f" ({', '.join(details)})"
        return desc

    @staticmethod
    def _format_observed(e: dict) -> str:
        actor = e.get("actor", "?")
        act = e.get("action", "").replace("_", " ")
        target = e.get("target")
        if target:
            return f"{actor} {act} → {target}"
        return f"{actor} {act}"

    @staticmethod
    def _format_rewire(r: dict) -> str:
        parts = []
        drop = r.get("drop")
        invite = r.get("invite")
        drop_out = r.get("drop_outcome")
        invite_out = r.get("invite_outcome")
        if drop:
            suffix = f" ({drop_out})" if drop_out else ""
            parts.append(f"drop {drop}{suffix}")
        if invite:
            suffix = f" ({invite_out})" if invite_out else ""
            parts.append(f"invite {invite}{suffix}")
        return ", ".join(parts) if parts else "no change"

    @staticmethod
    def _clean_sent(sent: Optional[dict]) -> Optional[dict]:
        if not sent:
            return None
        text = (sent.get("message") or sent.get("text") or "").strip()
        if not text:
            return None
        to = sent.get("message_to")
        if to is None:
            to = sent.get("to")
        if isinstance(to, str):
            to = [to]
        return {"to": list(to) if to else [], "text": text[:200]}

    @staticmethod
    def _clean_received(received: List[dict]) -> List[dict]:
        cleaned = []
        for m in received:
            text = (m.get("message") or m.get("text") or "").strip()
            if not text:
                continue
            cleaned.append({
                "from": m.get("from", "?"),
                "text": text[:200],
                "n_recipients": int(m.get("n_recipients", 1) or 1),
            })
        return cleaned

    @staticmethod
    def _clean_rewire(r: Optional[dict]) -> Optional[dict]:
        if not r:
            return None
        drop = r.get("drop") or r.get("drop_intent")
        invite = r.get("invite") or r.get("invite_intent")
        if not drop and not invite:
            return None
        return {
            "drop": drop,
            "invite": invite,
            "drop_outcome": r.get("drop_outcome"),
            "invite_outcome": r.get("invite_outcome"),
        }

    # ── accessors ──────────────────────────────────────────────────────

    def last_note(self) -> Optional[str]:
        if not self.memory_stream:
            return None
        return self.memory_stream[-1].text

    # ── serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "window_size": self.window_size,
            "notes_persist": self.notes_persist,
            "event_log": [e.to_dict() for e in self.event_log],
            "memory_stream": [m.to_dict() for m in self.memory_stream],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMemory":
        mem = cls(d["agent_id"], d.get("window_size", 10),
                  d.get("notes_persist", True))
        mem.event_log = [RoundEvents.from_dict(e) for e in d.get("event_log", [])]
        mem.memory_stream = [
            MemoryEntry.from_dict(m) for m in d.get("memory_stream", [])
        ]
        return mem
