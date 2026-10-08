"""
Round-log metric functions (§3.5). All functions are pure readers over the
canonical per-round log schema, which is ALSO the JSONL entry. Logs are the
single source of truth; every metric here can be recomputed post-hoc from
the raw JSONL without re-running the simulation.

Round log schema (also the JSONL entry):
  {
    'round': int,
    'agents': {
      agent_id: {
        'resources': float, 'arm_bonus': float,
        'action': str, 'target': str|None,
        'breakdown': {...}, 'rewire_intent': {...}, ...
      }
    },
    'combat': [combat_result, ...],
    'messages': [msg, ...],
    'network': {
      'edges': [...],
      'rewire_stats': {'edges_added', 'edges_dropped', 'intents', ...}
    },
    'bilateral_flows': {...},
  }
"""

from __future__ import annotations
from typing import Dict, List, Optional, Iterable
import json
import numpy as np


POSITIVE_SUM_ACTIONS = ('transfer', 'strengthen', 'invest_other', 'arm_other')
MEANINGFUL_EXCLUDE = ('no_action', 'hold', 'do_nothing', '')


# ─── loaders ────────────────────────────────────────────────────────────────

def load_run(jsonl_path: str) -> List[Dict]:
    """Load a run's round logs from JSONL (one object per line)."""
    rounds = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    return rounds


def _resources(round_log: Dict) -> Dict[str, float]:
    return {aid: a.get('resources', 0.0) for aid, a in round_log.get('agents', {}).items()}


def _actions(round_log: Dict) -> List[Dict]:
    return [
        {'agent': aid, 'action': a.get('action', ''), 'target': a.get('target')}
        for aid, a in round_log.get('agents', {}).items()
    ]


# ─── primaries (§3.5.1) ─────────────────────────────────────────────────────

def gini(round_log: Dict) -> float:
    """Gini of post-round resources."""
    values = np.array(sorted(_resources(round_log).values()))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * values) - (n + 1) * values.sum()) / (n * values.sum())


def cooperation_rate(round_log: Dict) -> float:
    """f_C(t) — fraction of positive-sum (invest_other + arm_other) among meaningful actions."""
    pos, meaningful = 0, 0
    for a in _actions(round_log):
        act = a['action']
        if act in MEANINGFUL_EXCLUDE:
            continue
        meaningful += 1
        if act in POSITIVE_SUM_ACTIONS:
            pos += 1
    return pos / meaningful if meaningful > 0 else 0.0


# modularity lives in main.py (needs networkx + history window); also derivable
# post-hoc from a run's edge snapshots — see analysis/network.py.


# ─── secondaries (§3.5.2) ───────────────────────────────────────────────────

def action_distribution(round_log: Dict) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in _actions(round_log):
        counts[a['action']] = counts.get(a['action'], 0) + 1
    return counts


def action_stability(cur: Dict, prev: Optional[Dict]) -> Optional[float]:
    if prev is None:
        return None
    cur_map = {a['agent']: a['action'] for a in _actions(cur)}
    prv_map = {a['agent']: a['action'] for a in _actions(prev)}
    common = set(cur_map) & set(prv_map)
    if not common:
        return 0.0
    return sum(1 for a in common if cur_map[a] == prv_map[a]) / len(common)


def network_churn(round_log: Dict) -> int:
    rw = round_log.get('network', {}).get('rewire_stats') or {}
    return int(rw.get('edges_added', 0) + rw.get('edges_dropped', 0))


def first_attack_round(run_rounds: Iterable[Dict]) -> Optional[int]:
    for r in run_rounds:
        for a in _actions(r):
            if a['action'] in ('take', 'attack'):
                return r.get('round')
    return None


def coalition_sizes(round_log: Dict) -> List[int]:
    """Size of each attack coalition in the round (len(attackers))."""
    return [len(c.get('attackers', [])) for c in round_log.get('combat', [])]


def rewire_intent_outcome(round_log: Dict) -> List[Dict]:
    """Per-agent rewire intent + outcome this round (empty list if no rewire)."""
    rw = round_log.get('network', {}).get('rewire_stats') or {}
    return rw.get('intents', [])


def truncation_count(round_log: Dict) -> int:
    """Number of agents whose JSON parsing fell back this round (§3.5.2)."""
    return sum(
        1 for a in round_log.get('agents', {}).values()
        if a.get('fallback') in ('thinking_recovery', 'default')
    )


def retry_count(round_log: Dict) -> int:
    """Number of agents whose response needed a JSON-retry this round."""
    return sum(
        1 for a in round_log.get('agents', {}).values()
        if a.get('any_retry')
    )


# ─── timeseries helpers (post-hoc) ─────────────────────────────────────────

def cooperation_rate_timeseries(run_rounds: Iterable[Dict]) -> List[float]:
    return [cooperation_rate(r) for r in run_rounds]


def gini_timeseries(run_rounds: Iterable[Dict]) -> List[float]:
    return [gini(r) for r in run_rounds]


def action_distribution_timeseries(run_rounds: Iterable[Dict]) -> List[Dict[str, int]]:
    return [action_distribution(r) for r in run_rounds]
