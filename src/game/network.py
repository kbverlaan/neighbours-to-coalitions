"""
Dynamic social network with agent-nominated preferential rewiring (§3.1.2).

Each round, every agent with probability w is eligible to rewire. When
eligible, they may nominate up to one neighbour to break with (`drop`) and
up to one non-neighbour to connect with (`invite`). Both execute unilaterally
— no consent from the counterparty is required.

Order within the rewiring phase is BREAKS first, then CONNECTS. This makes
lock-in emerge naturally: if A drops B while B invites A, the edge is severed
in the break phase and re-added in the connect phase. The lock-in costs B
their connect-slot — they cannot also invite a new neighbour in the same round.
At most one lock-in per agent per round (each agent has one connect-slot).

Full isolation is permitted: there is no minimum-degree constraint.
"""

import numpy as np
from typing import Dict, List, Optional


class NetworkTopology:
    """Undirected social graph with preferential rewiring.

    Interface:
        get_neighbors(agent_id) -> List[str]
        rewire(nominations, round_num) -> Dict  # intent/outcome log + edge stats
        get_edge_list() -> List[tuple]
        get_adjacency() -> Dict[str, set]
        restore_edges(edges)
        get_degree_stats() -> Dict
    """

    def __init__(self, agent_ids: List[str],
                 mean_degree: float = 5.0,
                 rewiring_prob: float = 0.0):
        self.agent_ids = list(agent_ids)
        self.n = len(self.agent_ids)
        self.rewiring_prob = rewiring_prob
        self.adj: Dict[str, set] = {aid: set() for aid in self.agent_ids}
        self._init_er_graph(mean_degree)

    def _init_er_graph(self, mean_degree: float):
        """Initialize Erdős-Rényi G(n, p) with connected-component guarantee."""
        if self.n <= 1:
            return
        p = min(mean_degree / (self.n - 1), 1.0)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.random.random() < p:
                    self.adj[self.agent_ids[i]].add(self.agent_ids[j])
                    self.adj[self.agent_ids[j]].add(self.agent_ids[i])
        self._ensure_connected()

    def _ensure_connected(self):
        """Link disconnected components at init via a random spanning tree."""
        visited = set()
        components = []
        for aid in self.agent_ids:
            if aid in visited:
                continue
            component = set()
            queue = [aid]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(self.adj[node] - visited)
            components.append(component)
        for i in range(len(components) - 1):
            a = list(components[i])[np.random.randint(len(components[i]))]
            b = list(components[i + 1])[np.random.randint(len(components[i + 1]))]
            self.adj[a].add(b)
            self.adj[b].add(a)

    def get_neighbors(self, agent_id: str) -> List[str]:
        return sorted(self.adj[agent_id])

    def rewire(self, nominations: Dict[str, Dict[str, Optional[str]]],
               round_num: int = 0) -> Dict:
        """Apply one round of preferential rewiring.

        Args:
            nominations: {agent_id: {'drop': neighbour_or_None, 'invite': other_or_None}}.
                Missing agents default to (None, None).
            round_num: for logging only.

        Returns:
            {
                'round', 'edges_dropped', 'edges_added',
                'agents_eligible', 'agents_rewired',
                'intents': [{'agent','eligible',
                             'drop_intent','invite_intent',
                             'drop_outcome','invite_outcome'}, ...]
            }
        drop_outcome:    'severed' | 'no_edge' | 'none' | 'invalid_target'
        invite_outcome:  'added' | 'relocked' | 'already_present' | 'none' | 'invalid_target'
        """
        w = self.rewiring_prob
        agent_ids_set = set(self.agent_ids)

        eligible = [aid for aid in self.agent_ids if np.random.random() < w]
        eligible_set = set(eligible)

        active = {}
        for aid in eligible:
            nom = nominations.get(aid, {}) or {}
            active[aid] = {'drop': nom.get('drop'), 'invite': nom.get('invite')}

        # Phase 1: BREAKS
        severed_this_round = set()
        edges_dropped = 0
        break_outcomes: Dict[str, str] = {}
        for aid in eligible:
            drop = active[aid]['drop']
            if drop is None:
                break_outcomes[aid] = 'none'
                continue
            if drop == aid or drop not in agent_ids_set:
                break_outcomes[aid] = 'invalid_target'
                continue
            if drop in self.adj[aid]:
                self.adj[aid].discard(drop)
                self.adj[drop].discard(aid)
                severed_this_round.add(frozenset((aid, drop)))
                edges_dropped += 1
                break_outcomes[aid] = 'severed'
            else:
                break_outcomes[aid] = 'no_edge'

        # Phase 2: CONNECTS
        edges_added = 0
        connect_outcomes: Dict[str, str] = {}
        for aid in eligible:
            invite = active[aid]['invite']
            if invite is None:
                connect_outcomes[aid] = 'none'
                continue
            if invite == aid or invite not in agent_ids_set:
                connect_outcomes[aid] = 'invalid_target'
                continue
            if invite in self.adj[aid]:
                connect_outcomes[aid] = 'already_present'
                continue
            self.adj[aid].add(invite)
            self.adj[invite].add(aid)
            edges_added += 1
            edge = frozenset((aid, invite))
            connect_outcomes[aid] = 'relocked' if edge in severed_this_round else 'added'

        intents = []
        for aid in self.agent_ids:
            intents.append({
                'agent': aid,
                'eligible': aid in eligible_set,
                'drop_intent': active.get(aid, {}).get('drop'),
                'invite_intent': active.get(aid, {}).get('invite'),
                'drop_outcome': break_outcomes.get(aid),
                'invite_outcome': connect_outcomes.get(aid),
            })

        agents_rewired = sum(
            1 for aid in eligible
            if break_outcomes.get(aid) == 'severed'
            or connect_outcomes.get(aid) in ('added', 'relocked')
        )

        return {
            'round': round_num,
            'edges_dropped': edges_dropped,
            'edges_added': edges_added,
            'agents_eligible': len(eligible),
            'agents_rewired': agents_rewired,
            'intents': intents,
        }

    def get_edge_list(self) -> List[tuple]:
        edges = set()
        for aid, neighbors in self.adj.items():
            for nbr in neighbors:
                edges.add(tuple(sorted([aid, nbr])))
        return sorted(edges)

    def restore_edges(self, edge_list: List):
        for aid in self.adj:
            self.adj[aid] = set()
        for edge in edge_list:
            a, b = edge[0], edge[1]
            if a in self.adj and b in self.adj:
                self.adj[a].add(b)
                self.adj[b].add(a)

    def get_adjacency(self) -> Dict[str, set]:
        return {aid: set(nbrs) for aid, nbrs in self.adj.items()}

    def get_degree_stats(self) -> dict:
        degrees = [len(self.adj[aid]) for aid in self.agent_ids]
        return {
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees)),
        }
