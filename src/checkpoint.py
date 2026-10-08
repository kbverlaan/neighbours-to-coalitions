"""
Checkpoint save/load and JSONL-replay reconstruction.

A checkpoint captures everything needed to resume a run: engine state,
agent memories, network edges, pending messages, bilateral-flow history,
and the full round-log list. `reconstruct_from_jsonl` rebuilds the same
structure from a run's `_log.jsonl` when no checkpoint file is available.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.memory import AgentMemory


def save_checkpoint(path, engine, agents, network, pending_messages,
                    bilateral_flows_history, all_round_logs):
    """Persist full simulation state for later resume."""
    checkpoint = {
        'round_number': engine.state.round_number,
        'resources': dict(engine.state.resources),
        'arm_bonuses': dict(engine.state.arm_bonuses),
        'history': [],
        'agent_memories': {},
        'pending_messages': pending_messages,
        'bilateral_flows_history': [],
        'all_round_logs': all_round_logs,
    }

    for rd in engine.state.history:
        rd_copy = dict(rd)
        if 'bilateral_flows' in rd_copy:
            rd_copy['bilateral_flows'] = {
                f"{k[0]}→{k[1]}" if isinstance(k, tuple) else k: v
                for k, v in rd_copy['bilateral_flows'].items()
            }
        checkpoint['history'].append(rd_copy)

    for aid, agent in agents.items():
        if agent.memory is not None:
            checkpoint['agent_memories'][aid] = agent.memory.to_dict()

    if network:
        checkpoint['network_edges'] = network.get_edge_list()

    for bf in bilateral_flows_history:
        checkpoint['bilateral_flows_history'].append({
            f"{k[0]}→{k[1]}" if isinstance(k, tuple) else k: v
            for k, v in bf.items()
        })

    with open(path, 'w') as f:
        json.dump(checkpoint, f, default=str)


def load_checkpoint(path):
    """Load a checkpoint dict from disk."""
    with open(path) as f:
        return json.load(f)


def reconstruct_from_jsonl(jsonl_path: str, game_params: dict) -> dict:
    """Rebuild a checkpoint-compatible dict by replaying a run's `_log.jsonl`."""
    rounds = []
    with open(jsonl_path) as f:
        for line in f:
            rounds.append(json.loads(line))
    if not rounds:
        raise ValueError(f"Empty JSONL: {jsonl_path}")

    # Drop trailing crashed rounds (>80% do_nothing = likely API failure)
    while len(rounds) > 1:
        last = rounds[-1]
        agents_data = last['agents']
        do_nothing_count = sum(
            1 for a in agents_data.values() if a.get('action') in ('hold', 'do_nothing', None, '')
        )
        if do_nothing_count / len(agents_data) > 0.8:
            print(f"  Skipping round {last['round']} "
                  f"(likely API failure: {do_nothing_count}/{len(agents_data)} do_nothing)")
            rounds.pop()
        else:
            break

    last_round = rounds[-1]
    agent_ids = list(last_round['agents'].keys())

    _mem_cfg = game_params.get('memory', {})
    memories = {aid: AgentMemory(aid, _mem_cfg.get('window_size', 10),
                                 _mem_cfg.get('notes_persist', True))
                for aid in agent_ids}

    all_round_logs = []
    pending_messages = {aid: [] for aid in agent_ids}
    engine_history = []

    for rd in rounds:
        rnd = rd['round']
        agents_data = rd['agents']
        network_edges = rd.get('network', {}).get('edges', [])

        visible = {aid: set() for aid in agent_ids}
        for edge in network_edges:
            a, b = edge[0], edge[1]
            if a in visible and b in visible:
                visible[a].add(b)
                visible[b].add(a)

        round_actions = []
        for aid, adata in agents_data.items():
            act = adata.get('action', 'no_action')
            if act and act != 'no_action':
                round_actions.append({'agent': aid, 'action': act, 'target': adata.get('target')})
            else:
                round_actions.append({'agent': aid, 'action': 'no_action'})

        combat_results = rd.get('combat', [])
        post_resources = {aid: adata.get('resources', 0) for aid, adata in agents_data.items()}

        # Sent messages per agent (for this round's memory entry)
        msgs = rd.get('messages', [])
        sent_by = {}
        for msg in msgs:
            sender = msg.get('from') or msg.get('agent_id')
            msg_to = msg.get('to') or msg.get('message_to')
            text = msg.get('text') or msg.get('message', '')
            if sender and text:
                sent_by[sender] = {'message_to': msg_to, 'message': text}

        for aid in agent_ids:
            adata = agents_data.get(aid, {})
            act = adata.get('action', 'no_action')
            target = adata.get('target')
            breakdown = adata.get('breakdown', {}) or {}
            outcome = {}
            rc = breakdown.get('resource_change') if isinstance(breakdown, dict) else None
            if rc is not None:
                outcome['resource_change'] = rc
            # Combat outcome: is this agent an attacker in any combat?
            for c in combat_results:
                if aid in (c.get('attackers') or []):
                    outcome['combat_won'] = (c.get('winner') == 'coalition')
                    break

            own_action = {'action': act, 'target': target, 'outcome': outcome}
            rewire_intent = adata.get('rewire_intent') or None

            memories[aid].record_round(
                round_num=rnd,
                own_action=own_action,
                round_actions=round_actions,
                visible_agents=list(visible.get(aid, [])) if visible.get(aid) else None,
                all_resources=post_resources,
                sent_message=sent_by.get(aid),
                received_messages=pending_messages.get(aid, []),
                rewire=rewire_intent,
            )
            mem_text = adata.get('memory') or adata.get('note_to_self')
            if mem_text:
                memories[aid].record_memory(rnd, mem_text)

        next_messages = {aid: [] for aid in agent_ids}
        for msg in msgs:
            sender = msg.get('from') or msg.get('agent_id')
            text = msg.get('text') or msg.get('message', '')
            if not text:
                continue
            msg_to = msg.get('to')
            if msg_to is None:
                msg_to = msg.get('message_to')
            # Normalize to a recipient list (handle legacy 'all' / single id).
            if msg_to == 'all':
                recipients = [t for t in agent_ids if t != sender]
            elif isinstance(msg_to, str):
                recipients = [msg_to] if msg_to in agent_ids and msg_to != sender else []
            elif isinstance(msg_to, (list, tuple)):
                recipients = [t for t in msg_to if t in agent_ids and t != sender]
            else:
                recipients = []
            n = len(recipients)
            for target in recipients:
                next_messages[target].append(
                    {'from': sender, 'message': text, 'n_recipients': n}
                )
        pending_messages = next_messages

        engine_history.append({'actions': round_actions, 'round': rnd})
        all_round_logs.append(rd)

    checkpoint = {
        'round_number': last_round['round'] + 1,
        'resources': {aid: last_round['agents'][aid].get('resources', 0) for aid in agent_ids},
        'arm_bonuses': {aid: last_round['agents'][aid].get('arm_bonus', 0) for aid in agent_ids},
        'history': engine_history,
        'agent_memories': {aid: memories[aid].to_dict() for aid in agent_ids},
        'network_edges': last_round.get('network', {}).get('edges', []),
        'pending_messages': pending_messages,
        'bilateral_flows_history': [],
        'all_round_logs': all_round_logs,
    }

    print(f"Reconstructed state from {len(rounds)} rounds of {jsonl_path}")
    print(f"  Agents: {agent_ids}")
    print(f"  Resuming from round {checkpoint['round_number']}")
    return checkpoint
