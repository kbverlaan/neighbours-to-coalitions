"""
Single-run execution: `run_simulation()` is the hot loop that advances the
game round-by-round. Every round it produces one canonical round-log dict
(the SSOT — written to JSONL live). `save_results()` writes the final
`_log.jsonl` + `_meta.json` artifacts. `load_config()` reads a YAML config.
"""

from __future__ import annotations
import os
import sys
import json
import time
import platform
import random
import hashlib
import subprocess
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv  # noqa: F401  (available for callers that need it)

sys.path.insert(0, str(Path(__file__).parent))

from game.engine import GameEngine, GameState
from game.network import NetworkTopology
from agents.llm_agent import LLMAgent
from agents.memory import AgentMemory
from analysis.metrics import (
    gini, cooperation_rate,
    action_stability,
)
from checkpoint import save_checkpoint, load_checkpoint, reconstruct_from_jsonl
import display as d

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _git_info() -> dict:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL, text=True).strip())
        return {'commit': sha, 'dirty': dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': None, 'dirty': None}


def _package_versions() -> dict:
    versions = {}
    for pkg in ('numpy', 'networkx', 'openai', 'yaml'):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass
    return versions


def _modularity_from_edges(edges: list) -> float:
    """Greedy-modularity Q(t) on current network topology G(t) — §3.5.1."""
    if not _HAS_NX or not edges:
        return 0.0
    G = nx.Graph()
    G.add_edges_from([(a, b) for a, b in edges])
    if G.number_of_edges() < 2 or G.number_of_nodes() < 3:
        return 0.0
    try:
        communities = nx.community.greedy_modularity_communities(G, resolution=1.0)
        return nx.community.modularity(G, communities)
    except Exception:
        return 0.0


def save_results(game_state: GameState,
                 reasoning_traces: list,
                 round_logs: list,
                 output_dir: Path,
                 run_id: str,
                 run_metadata: Optional[dict] = None):
    """Persist a run's final artifacts. The JSONL is the SSOT for analyses."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{run_id}_log.jsonl"
    with open(log_file, 'w') as f:
        for rd in round_logs:
            f.write(json.dumps(rd, default=str) + '\n')

    if run_metadata:
        meta_file = output_dir / f"{run_id}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(run_metadata, f, indent=2)

    print(f"Results saved to {output_dir}")
    print(f"  - Log:  {log_file.name}")
    if run_metadata:
        print(f"  - Meta: {run_id}_meta.json")


def run_simulation(game_params: dict,
                   openrouter_config: dict,
                   run_id: Optional[str] = None,
                   resume_path: Optional[str] = None,
                   log_dir_override: Optional[Path] = None) -> tuple:
    """Run one full simulation from start/checkpoint to max_rounds."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    seed = game_params.get('random_seed')
    if seed is None:
        # Deterministic per-run seed derived from the run_id: reproducible from the
        # run_id alone (which is logged to meta + filename), while distinct run_ids
        # stay independent. Replaces the old os.urandom fallback (irreproducible).
        seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    random.seed(seed)
    # Stash the resolved seed so the prompt builder can shuffle deterministically
    # per (seed, agent_id, round) instead of the thread-shared global RNG (C1).
    game_params['random_seed'] = seed

    print(f"\n{'='*60}")
    print(f"Starting simulation: {run_id}")
    print(f"{'='*60}\n")
    print(f"Random seed: {seed}")

    api_key_env = openrouter_config.get('api_key_env_var', '')
    api_key = os.getenv(api_key_env) if api_key_env else None
    base_url = os.getenv('VLLM_BASE_URL') or openrouter_config.get(
        'base_url', 'https://openrouter.ai/api/v1'
    )
    if not api_key and 'openrouter.ai' in base_url:
        raise ValueError(f"API key not found in environment variable: {api_key_env}")
    if not api_key:
        api_key = "none"

    AGENT_NAMES = [
        "Red", "Blue", "Green", "Gold", "Silver",
        "Coral", "Jade", "Amber", "Ivory", "Slate",
        "Crimson", "Teal", "Copper", "Violet", "Pearl",
        "Bronze", "Scarlet", "Indigo", "Onyx", "Cobalt",
        "Maroon", "Olive", "Cyan", "Rust", "Mauve",
        "Sage", "Plum", "Dusk", "Ash", "Storm",
    ]
    n_agents = game_params['num_agents']

    _resume_ckpt = None
    if resume_path:
        _resume_ckpt = (reconstruct_from_jsonl(resume_path, game_params)
                        if resume_path.endswith('.jsonl')
                        else load_checkpoint(resume_path))
        names = list(_resume_ckpt['resources'].keys())
        n_agents = len(names)
    elif n_agents <= len(AGENT_NAMES):
        names = random.sample(AGENT_NAMES, n_agents)
    else:
        names = [f"agent_{i+1}" for i in range(n_agents)]
    agent_ids = names

    initial_resources = game_params['initial_resources']

    if isinstance(initial_resources, dict) and 'type' in initial_resources:
        import random as _random
        dtype = initial_resources['type']
        shuffled = list(agent_ids)
        _random.shuffle(shuffled)
        if dtype == 'bimodal':
            n_high = initial_resources.get('n_high', len(shuffled) // 2)
            high = initial_resources['high']
            low = initial_resources['low']
            initial_resources = {n: (high if i < n_high else low) for i, n in enumerate(shuffled)}
        elif dtype == 'linear':
            lo = initial_resources['min']; hi = initial_resources['max']
            n = len(shuffled)
            initial_resources = {n_: lo + (hi - lo) * i / max(1, n - 1) for i, n_ in enumerate(shuffled)}

    engine = GameEngine(
        agent_ids=agent_ids,
        initial_resources=initial_resources,
        c_inv=game_params.get('c_inv', 0.10),
        g_inv=game_params.get('g_inv', 0.15),
        c_arm=game_params.get('c_arm', 0.10),
        mu_arm=game_params.get('mu_arm', 3.0),
        delta_B=game_params.get('delta_B', 0.50),
        alpha=game_params.get('alpha', 0.20),
        c_atk=game_params.get('c_atk', 0.01),
        eta_atk=game_params.get('eta_atk', 1.0),
        tau_atk=game_params.get('tau_atk', 5),
        delta_R=game_params.get('delta_R', 1.0),
        gamma_sat=game_params.get('gamma_sat', 1.0),
        tau_sat=game_params.get('tau_sat', 5),
        max_rounds=game_params['max_rounds'],
        symmetric_stakes=game_params.get('symmetric_stakes', False),
        lethal_pot=game_params.get('lethal_pot', False),
        arm_enabled=game_params.get('arm_enabled', True),
        take_enabled=game_params.get('take_enabled', True),
        assoc_enabled=game_params.get('assoc_enabled', True),
        commons_enabled=game_params.get('commons_enabled', False),
        commons_K=game_params.get('commons_K', 600.0),
        commons_init=game_params.get('commons_init', None),
        commons_collapse_frac=game_params.get('commons_collapse_frac', 0.05),
        commons_regen=game_params.get('commons_regen', 2.0),
        commons_open_round=game_params.get('commons_open_round', 1),
        c_harvest=game_params.get('c_harvest', 0.0),
        harvest_frac_cap=game_params.get('harvest_frac_cap', 0.0),
    )

    network_enabled = game_params.get('network_enabled', game_params.get('spatial_enabled', False))
    network = None
    if network_enabled:
        mean_degree = game_params.get('mean_degree', 5.0)
        rewiring_prob = game_params.get('rewiring_prob', 0.0)
        network = NetworkTopology(agent_ids, mean_degree, rewiring_prob)
        degree_stats = network.get_degree_stats()
        print(f"Network topology: ER graph, ⟨k⟩={degree_stats['mean']:.1f} "
              f"(target {mean_degree}), w={rewiring_prob}, "
              f"degree range [{degree_stats['min']}, {degree_stats['max']}]")

    agents: Dict[str, LLMAgent] = {}
    prompt_config = openrouter_config.get('prompt_config', {})
    memory_config = game_params.get('memory', {})
    for agent_id in agent_ids:
        agents[agent_id] = LLMAgent(
            agent_id=agent_id,
            api_key=api_key,
            model=openrouter_config['model'],
            prompt_config=prompt_config,
            game_params=game_params,
            temperature=openrouter_config['temperature'],
            max_tokens=openrouter_config['max_tokens'],
            timeout=openrouter_config['timeout'],
            retry_attempts=openrouter_config['retry_attempts'],
            retry_delay=openrouter_config['retry_delay'],
            base_url=base_url,
            memory_config=memory_config,
        )

    print(f"Initialized {len(agents)} LLM agents")
    print(f"Model: {openrouter_config['model']}")
    print(f"Prompt config: {prompt_config}")
    print(f"Max rounds: {game_params['max_rounds']}\n")

    resumed_logs = []
    resumed_bilateral = []
    resumed_pending = {}
    if resume_path and _resume_ckpt:
        ckpt = _resume_ckpt
        print(f"Resuming from round {ckpt['round_number']} (source: {resume_path})")
        engine.state.round_number = ckpt['round_number']
        engine.state.resources = {aid: float(v) for aid, v in ckpt['resources'].items()}
        engine.state.arm_bonuses = {aid: float(v) for aid, v in ckpt.get('arm_bonuses', {}).items()}
        engine.state.history = []
        for rd in ckpt.get('history', []):
            if 'bilateral_flows' in rd:
                restored_bf = {}
                for k, v in rd['bilateral_flows'].items():
                    parts = k.split('→')
                    if len(parts) == 2:
                        restored_bf[(parts[0], parts[1])] = v
                    else:
                        restored_bf[k] = v
                rd['bilateral_flows'] = restored_bf
            engine.state.history.append(rd)
        for aid, mem_dict in ckpt.get('agent_memories', {}).items():
            if aid in agents:
                agents[aid].memory = AgentMemory.from_dict(mem_dict)
        if network and 'network_edges' in ckpt:
            network.restore_edges(ckpt['network_edges'])
        resumed_pending = ckpt.get('pending_messages', {})
        resumed_logs = ckpt.get('all_round_logs', ckpt.get('all_round_metrics', []))
        for bf_str in ckpt.get('bilateral_flows_history', []):
            restored = {}
            for k, v in bf_str.items():
                parts = k.split('→')
                if len(parts) == 2:
                    restored[(parts[0], parts[1])] = v
                else:
                    restored[k] = v
            resumed_bilateral.append(restored)

    def can_afford_any_action(resources: float, _game_params: Dict) -> bool:
        return resources > 0.01

    max_rounds = game_params['max_rounds']

    start_time = time.time()
    all_round_logs = resumed_logs if resumed_logs else []
    bilateral_flows_history = resumed_bilateral if resumed_bilateral else []
    comm_scope = game_params.get('comm_scope', 'none')
    pending_messages = resumed_pending if resumed_pending else {aid: [] for aid in agent_ids}

    log_dir = log_dir_override if log_dir_override else Path(__file__).parent.parent / "data" / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}_checkpoint.json"

    reasoning_log_path = log_dir / f"{run_id}_reasoning_live.jsonl"
    d.p(f"{d.C('Live reasoning log:', 'dim')} {reasoning_log_path}")

    start_round = engine.get_state().round_number

    # Initial network edges (pre any rewiring). Needed so round 1's agents'
    # decisions can be replayed from the log.
    initial_edges = network.get_edge_list() if network else []

    while not engine.is_game_over(max_rounds):
        state = engine.get_state()
        round_num = state.round_number

        # Snapshot incoming messages before pending_messages gets overwritten
        # with next round's outgoing traffic later in the loop.
        received_this_round = {aid: list(msgs) for aid, msgs in pending_messages.items()}

        d.print_round_header(round_num, max_rounds)
        d.print_resource_bars(state.resources, state.arm_bonuses, agent_ids)
        if network:
            d.print_network(agent_ids, network.get_neighbors)

        if network:
            valid_targets = {aid: network.get_neighbors(aid) for aid in agent_ids}
            engine.set_valid_targets(valid_targets)
        else:
            engine.set_valid_targets(None)

        history_length = game_params.get('history_length', 10)

        def get_agent_action(agent_id, current_state):
            observation = current_state.get_observation(agent_id, history_length)
            observation['broke_agents'] = [
                aid for aid in agent_ids
                if not can_afford_any_action(current_state.resources[aid], game_params)
            ]
            if network:
                observation['visible_agents'] = network.get_neighbors(agent_id)
            if pending_messages.get(agent_id):
                observation['received_messages'] = pending_messages[agent_id]
            if agents[agent_id].memory is not None:
                observation['agent_memory'] = agents[agent_id].memory

            action = agents[agent_id].select_action(observation)
            return agent_id, action

        actions = []
        broke_agents = [
            aid for aid in agent_ids
            if not can_afford_any_action(state.resources[aid], game_params)
        ]
        active_agents = [aid for aid in agent_ids if aid not in broke_agents]

        if active_agents:
            with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
                futures = {executor.submit(get_agent_action, aid, state): aid for aid in active_agents}
                for future in as_completed(futures):
                    agent_id, action = future.result()
                    actions.append(action)

        display_action_map = {
            a.agent_id: {'action': a.action_type.value, 'target': a.target_id}
            for a in actions
        }
        round_result = engine.resolve_round(actions)

        round_notes = {}
        for aid in agent_ids:
            agent = agents[aid]
            mem_text = getattr(agent, '_last_memory', None)
            if not mem_text and agent.memory:
                mem_text = agent.memory.last_note()
            if mem_text:
                round_notes[aid] = mem_text

        d.print_agent_round_summary(display_action_map, round_notes, agent_ids)
        d.print_combat_results(round_result.get('combat_results', []))

        # Collect + route messages. Language is decoupled from the 1-hop
        # action graph: a message carries an explicit LIST of target ids and
        # is delivered to each named target that is currently alive — no
        # neighbour restriction, no broadcast channel. Agents only know ids
        # they have encountered (neighbours, prior senders, observed actors),
        # so there is no global directory.
        if comm_scope != 'none':
            next_messages = {aid: [] for aid in agent_ids}
            round_messages = []
            resources_now = engine.get_state().resources
            living = {a for a in agent_ids if resources_now[a] > 0.01}
            for aid in agent_ids:
                if aid not in living:
                    agents[aid]._last_message = None
                    continue
                msg = agents[aid].get_last_message()
                if not (msg and msg.get('message')):
                    continue
                targets = msg.get('message_to') or []
                recipients = [t for t in targets if t in living and t != aid]
                n = len(recipients)
                for target in recipients:
                    next_messages[target].append({
                        'from': aid, 'message': msg['message'], 'n_recipients': n,
                    })
                if recipients:
                    round_messages.append(msg)
            pending_messages = next_messages
            if round_messages:
                round_result['messages'] = round_messages
                d.print_messages(round_messages)

        updated_state = engine.get_state()

        # Rewiring (§3.1 step 7). Gated by assoc_enabled (L3 rung): below the
        # association rung the topology is frozen — no drop/invite is applied.
        rewire_stats = None
        if network and game_params.get('assoc_enabled', True):
            bf = dict(round_result.get('bilateral_flows', {}))
            bilateral_flows_history.append(bf)
            nominations = {}
            for aid in agent_ids:
                nom = agents[aid].get_last_rewire_nomination()
                if nom:
                    nominations[aid] = nom
            rewire_stats = network.rewire(nominations, round_num=round_num)
            d.print_network_rewire(rewire_stats)

        # Per-agent reasoning traces for the round, incl. retry/fallback data
        # so we can compute truncation-count (§3.5.2) post-hoc.
        agent_traces = {}
        for aid in agent_ids:
            raw = agents[aid].reasoning_traces
            this_round_entries = [t for t in raw if t.get('round') == round_num]
            if not this_round_entries:
                continue
            last = this_round_entries[-1]
            any_retry = any(t.get('is_retry') for t in this_round_entries)
            fallback = last.get('fallback')  # 'thinking_recovery' | 'default' | None
            errors = last.get('errors') or []

            thinking = last.get('thinking', '') or ''
            response = last.get('response', '') or ''

            agent_traces[aid] = {
                'thinking': thinking or None,
                'memory': getattr(agents[aid], '_last_memory', None),
                'tokens': last.get('usage', {}).get('total_tokens', 0),
                'latency_s': last.get('latency_s') or last.get('latency', 0),
                'prompt': last.get('prompt', ''),
                'response': response,
                'attempts': len(this_round_entries),
                'any_retry': any_retry,
                'fallback': fallback,
                'errors': errors,
                'recovered_action': last.get('recovered_action'),
                'recovered_target': last.get('recovered_target'),
            }

        action_by_agent = {a['agent']: a for a in round_result.get('actions', [])}
        rewire_by_agent = {}
        if rewire_stats:
            for entry in rewire_stats.get('intents', []):
                rewire_by_agent[entry['agent']] = entry

        # ─── Canonical round log (= JSONL entry = SSOT) ────────────────────
        round_log = {
            'round': round_num,
            **({'config': {
                # Full reproducibility bundle — dumped at run start
                'game_params': dict(game_params),
                'prompt_config': dict(prompt_config),
                'model': openrouter_config.get('model', 'unknown'),
                'temperature': openrouter_config.get('temperature'),
                'max_tokens': openrouter_config.get('max_tokens'),
                'base_url': openrouter_config.get('base_url', ''),
                'agent_ids': list(agent_ids),
                'seed': seed,
                'initial_edges': initial_edges,
            }} if round_num == 1 or round_num == start_round else {}),
            'agents': {
                aid: {
                    'resources': updated_state.resources[aid],
                    'arm_bonus': updated_state.arm_bonuses.get(aid, 0.0),
                    'action': action_by_agent.get(aid, {}).get('action', 'no_action'),
                    'target': action_by_agent.get(aid, {}).get('target'),
                    'breakdown': round_result.get('resource_breakdown', {}).get(aid, {}),
                    'rewire_intent': (
                        {
                            'drop': rewire_by_agent.get(aid, {}).get('drop_intent'),
                            'invite': rewire_by_agent.get(aid, {}).get('invite_intent'),
                            'eligible': rewire_by_agent.get(aid, {}).get('eligible', False),
                            'drop_outcome': rewire_by_agent.get(aid, {}).get('drop_outcome'),
                            'invite_outcome': rewire_by_agent.get(aid, {}).get('invite_outcome'),
                        } if rewire_by_agent else None
                    ),
                    **agent_traces.get(aid, {}),
                }
                for aid in agent_ids
            },
            # Commons block (stock/grants/collapse) = primary T4 DV; without this
            # line it never reached the SSOT JSONL (found in the ground-truth
            # test of commons_dv).
            **({'commons': round_result['commons']} if round_result.get('commons') else {}),
            'combat': round_result.get('combat_results', []),
            'messages': [
                {'from': m.get('from'), 'to': m.get('message_to'), 'text': m.get('message', '')}
                for m in round_result.get('messages', [])
            ],
            'bilateral_flows': {
                f"{k[0]}→{k[1]}" if isinstance(k, tuple) else k: v
                for k, v in round_result.get('bilateral_flows', {}).items()
            },
            'network': {
                'edges': network.get_edge_list() if network else [],
                'rewire_stats': rewire_stats,
            },
        }

        with open(reasoning_log_path, 'a') as f:
            f.write(json.dumps(round_log, default=str) + '\n')
        all_round_logs.append(round_log)

        # Live display values derived from the round log
        prev_log = all_round_logs[-2] if len(all_round_logs) >= 2 else None
        d.print_metrics_dashboard(
            {
                'cooperation_rate': cooperation_rate(round_log),
                'gini': gini(round_log),
                'modularity': _modularity_from_edges(round_log['network']['edges']),
                'action_stability': action_stability(round_log, prev_log),
            },
            {'cooperation_rate': cooperation_rate(prev_log), 'gini': gini(prev_log)} if prev_log else None,
        )

        # Memory update
        if memory_config.get('enabled', False):
            post_resources = dict(updated_state.resources)
            round_actions = round_result.get('actions', [])
            resource_changes = round_result.get('resource_changes', {})
            combat_results = round_result.get('combat_results', [])
            agent_outcomes = {}
            for a in round_actions:
                aid = a['agent']
                outcome = {}
                rc = resource_changes.get(aid, 0.0)
                if abs(rc) > 0.001:
                    outcome['resource_change'] = rc
                for combat in combat_results:
                    if aid in combat.get('attackers', []):
                        outcome['combat_won'] = (combat['winner'] == 'coalition')
                        break
                agent_outcomes[aid] = outcome
            # Rewire info per agent (may be None if no rewiring this round or no nomination)
            rewire_per_agent = {}
            if rewire_stats:
                for entry in rewire_stats.get('intents', []):
                    rewire_per_agent[entry['agent']] = {
                        'drop': entry.get('drop_intent'),
                        'invite': entry.get('invite_intent'),
                        'drop_outcome': entry.get('drop_outcome'),
                        'invite_outcome': entry.get('invite_outcome'),
                    }
            for aid in agent_ids:
                action_str = action_by_agent.get(aid, {}).get('action', 'no_action')
                target = action_by_agent.get(aid, {}).get('target')
                visible = network.get_neighbors(aid) if network else None
                agents[aid].update_memory(
                    round_num=round_num,
                    action_str=action_str,
                    target=target,
                    outcome=agent_outcomes.get(aid, {}),
                    visible_agents=visible,
                    round_actions=round_actions,
                    all_resources=post_resources,
                    received_messages=received_this_round.get(aid, []),
                    rewire=rewire_per_agent.get(aid),
                )

        save_checkpoint(checkpoint_path, engine, agents, network,
                        pending_messages, bilateral_flows_history, all_round_logs)

    elapsed = time.time() - start_time

    all_traces = []
    for agent in agents.values():
        all_traces.extend(agent.get_reasoning_traces())
    total_prompt_tokens = sum(t.get('usage', {}).get('prompt_tokens', 0) for t in all_traces)
    total_completion_tokens = sum(t.get('usage', {}).get('completion_tokens', 0) for t in all_traces)

    state = engine.get_state()
    rounds_played = state.round_number - 1
    metrics_trajectory = [
        {'cooperation_rate': cooperation_rate(rd), 'gini': gini(rd)}
        for rd in all_round_logs
    ]
    d.print_final_summary(state.resources, state.arm_bonuses, agent_ids, elapsed, metrics_trajectory)
    d.p(f"  Rounds: {rounds_played}  ({elapsed/max(rounds_played,1):.1f}s/round)")

    round_summaries = []
    for round_data in state.history:
        rs = {}
        for action in round_data['actions']:
            rs[action['agent']] = {
                'action': action.get('action', 'unknown'),
                'target': action.get('target'),
            }
        round_summaries.append(rs)
    try:
        d.print_action_distribution(round_summaries)
        d.print_agent_profiles(round_summaries, agent_ids, state.resources)
    except Exception as e:
        import traceback
        print(f"[warning] post-run display crashed: {e}", flush=True)
        traceback.print_exc()

    run_metadata = {
        "run_id": run_id,
        "random_seed": seed,
        "model": openrouter_config.get('model', 'unknown'),
        "temperature": openrouter_config.get('temperature'),
        "max_tokens": openrouter_config.get('max_tokens'),
        "base_url": openrouter_config.get('base_url', ''),
        "elapsed_seconds": round(elapsed, 1),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "node": platform.node(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_nodelist": os.getenv("SLURM_NODELIST"),
            "cuda_visible": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        "git": _git_info(),
        "packages": _package_versions(),
    }

    print()
    return state, all_traces, all_round_logs, run_metadata
