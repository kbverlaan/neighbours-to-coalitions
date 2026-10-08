"""
Game engine for multi-agent coordination game.
Handles game state, action resolution, and combat mechanics.

All costs/returns are percentages of the acting agent's current resources.
Arm bonuses are additive (combat strength = resources + arm bonus) and
decay by a configurable factor each round.

Design references:
- Game structure: N-agent resource game with invest/arm/attack actions,
  inspired by Sugarscape (Epstein & Axtell, 1996) and the cooperation-conflict
  dilemma in Axelrod (1984) and Skyrms (2004, Stag Hunt).
- Combat resolution: probabilistic, strength-proportional (Lanchester-type).
  Win probability = attacker_strength / total_strength. Coalition attacks:
  when multiple agents attack the same target, strengths combine (snapshot-based
  simultaneous resolution).
- Arm decay: exponential decay models transient military advantage,
  creating pressure for repeated investment (cf. arms race dynamics,
  Baliga & Sjöström, 2004).
- Parameter space: cost/benefit ratios (theta) determine the cooperation-conflict
  gradient. invest_other_return > invest_other_cost creates a social dilemma:
  cooperation is collectively optimal but individually costly.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    # Member names kept stable internally; the .value strings are the canonical
    # action vocabulary used in prompts and logs.
    INVEST_OTHER = "transfer"     # give resources to another
    ARM_OTHER = "strengthen"      # give a combat bonus to another
    ATTACK = "take"               # take resources by force
    DO_NOTHING = "hold"           # do nothing
    HARVEST = "harvest"           # take from the shared stock AS your action;
                                  # mutually exclusive with the dyadic actions, so harvesting forgoes cooperating


@dataclass
class Action:
    """Represents a single agent action."""
    agent_id: str
    action_type: ActionType
    target_id: Optional[str] = None  # None for self-actions, agent_id for other-actions
    harvest: float = 0.0  # commons harvest claim. category mode: % of carrying capacity K.
                          # fraction_own mode: fraction (in [0, inf)) of the actor's OWN current
                          # resources. 0 = none / commons off in both modes.
    harvest_raw: Optional[str] = None  # the model's RAW harvest field, kept verbatim for
                                       # post-hoc audit of percent→fraction scaling (C2).


@dataclass
class GameState:
    """Complete game state with resources and history."""
    agents: List[str]
    resources: Dict[str, float]
    round_number: int = 1
    max_rounds: Optional[int] = None
    history: List[Dict] = field(default_factory=list)
    arm_bonuses: Dict[str, float] = field(default_factory=dict)  # agent_id -> combat bonus (decays each round)
    commons_stock: float = 0.0     # shared common-pool resource level (absolute units); 0 if commons off
    commons_K: float = 0.0         # carrying capacity (absolute); >0 signals commons is active
    commons_collapsed: bool = False  # absorbing state once stock falls below the collapse threshold
    commons_open_round: int = 1    # first round harvesting is possible (announced from R1)

    def get_observation(self, agent_id: str, history_length: int = 10) -> Dict:
        """Get complete information observation for an agent."""
        obs = {
            "agent_id": agent_id,
            "round": self.round_number,
            "max_rounds": self.max_rounds,
            "resources": self.resources.copy(),
            "arm_bonuses": self.arm_bonuses.copy(),
            "recent_history": self.history[-history_length:] if len(self.history) > history_length else self.history.copy()
        }
        if self.commons_K > 0:
            # Agents see the stock as a PERCENTAGE of capacity (K stays hidden), plus
            # everyone's harvest from the previous round (Ostrom-style monitoring).
            last_harvests = {}
            if self.history:
                lc = self.history[-1].get("commons")
                if lc:
                    last_harvests = lc.get("grants_pct", {})
            obs["commons"] = {
                "stock_pct": 100.0 * self.commons_stock / self.commons_K,
                "collapsed": self.commons_collapsed,
                "last_harvests_pct": last_harvests,
            }
            if self.round_number < self.commons_open_round:
                obs["commons"]["opens_in_round"] = self.commons_open_round
        return obs


class GameEngine:
    """Core game engine managing state transitions and action resolution.

    Parameter names match the prompt exactly (see prompts.py _format_actions).
    All costs are percentages of the acting agent's current resources.
    """

    def __init__(self,
                 agent_ids: List[str],
                 initial_resources,  # float (equal) or Dict[str, float] (per-agent)
                 c_inv: float = 0.10,
                 g_inv: float = 0.15,
                 c_arm: float = 0.10,
                 mu_arm: float = 3.0,
                 delta_B: float = 0.50,
                 alpha: float = 0.20,
                 c_atk: float = 0.01,
                 eta_atk: float = 1.0,
                 tau_atk: int = 5,
                 delta_R: float = 1.0,
                 gamma_sat: float = 1.0,
                 tau_sat: int = 5,
                 max_rounds: int = None,
                 symmetric_stakes: bool = False,
                 lethal_pot: bool = False,
                 arm_enabled: bool = True,
                 take_enabled: bool = True,
                 assoc_enabled: bool = True,
                 commons_enabled: bool = False,
                 commons_K: float = 600.0,
                 commons_init: Optional[float] = None,
                 commons_collapse_frac: float = 0.05,
                 commons_regen: float = 2.0,
                 commons_open_round: int = 1,
                 c_harvest: float = 0.0,
                 harvest_frac_cap: float = 0.0):
        """All proportional parameters are fractions in [0, 1] (§3.1 symbols):
        c_inv, g_inv = invest cost / return (fraction of actor's R)
        c_arm = arm cost (fraction of actor's R), mu_arm = arm multiplier (scalar)
        delta_B = arm decay factor, delta_R = resource decay factor
        alpha = spoils fraction of loser's R
        c_atk = base conflict cost fraction; eta_atk / tau_atk = escalation
        gamma_sat / tau_sat = invest saturation factor and window
        """
        self.max_rounds = max_rounds
        self.params = {
            "c_inv": c_inv,
            "g_inv": g_inv,
            "c_arm": c_arm,
            "mu_arm": mu_arm,
            "delta_B": delta_B,
            "alpha": alpha,
            "c_atk": c_atk,
            "eta_atk": eta_atk,
            "tau_atk": tau_atk,
            "delta_R": delta_R,
            "gamma_sat": gamma_sat,
            "tau_sat": tau_sat,
            "symmetric_stakes": symmetric_stakes,
            "lethal_pot": lethal_pot,
            "arm_enabled": arm_enabled,
            "take_enabled": take_enabled,
            "assoc_enabled": assoc_enabled,
            "commons_enabled": commons_enabled,
            "commons_K": commons_K,
            "commons_collapse_frac": commons_collapse_frac,
            "commons_regen": commons_regen,
            "commons_open_round": commons_open_round,
            "c_harvest": c_harvest,
            "harvest_frac_cap": harvest_frac_cap,
        }

        if isinstance(initial_resources, dict):
            resources = {aid: initial_resources[aid] for aid in agent_ids}
        else:
            resources = {aid: float(initial_resources) for aid in agent_ids}

        self.state = GameState(
            agents=agent_ids,
            resources=resources,
            max_rounds=max_rounds
        )
        if commons_enabled:
            self.state.commons_K = commons_K
            self.state.commons_stock = commons_init if commons_init is not None else commons_K
            self.state.commons_open_round = commons_open_round
        self._valid_targets: Optional[Dict[str, List[str]]] = None

    def set_valid_targets(self, valid_targets: Optional[Dict[str, List[str]]]):
        """Set per-agent valid targets for network-restricted actions.

        Args:
            valid_targets: Dict mapping agent_id -> list of neighbor IDs, or None to disable.
        """
        self._valid_targets = valid_targets

    def _frac(self, agent_id: str, fraction: float) -> float:
        """Compute fraction * agent's current resources. Fraction is in [0, 1]."""
        return self.state.resources[agent_id] * fraction

    def can_afford_action(self, agent_id: str, action_type: ActionType) -> bool:
        """Check if agent has sufficient resources for action.

        With percentage-based costs, any action is affordable as long as
        resources are above a minimum threshold.
        """
        if action_type in (ActionType.DO_NOTHING, ActionType.HARVEST):
            return True  # harvesting from the pool is free (lets the poor recover)
        return self.state.resources[agent_id] > 0.01

    def resolve_round(self, actions: List[Action]) -> Dict:
        """Resolve all actions for a single round (simultaneous)."""
        round_log = {
            "round": self.state.round_number,
            "actions": [],
            "resource_changes": {},
            "bilateral_flows": defaultdict(float),  # (from, to) -> net resource flow
            "combat_results": [],
            "resource_breakdown": {},  # agent -> {invest_received, invest_cost, arm_cost, conflict_cost, combat_transfer, decay}
        }

        # Validate actions. When taking is disabled (rung below predation), any
        # attack is treated as hold — the affordance does not exist on this rung.
        # Same for arming/strengthen when arm_enabled is off.
        take_enabled = self.params.get("take_enabled", True)
        arm_enabled = self.params.get("arm_enabled", True)
        valid_actions = []
        for action in actions:
            if action.action_type == ActionType.ATTACK and not take_enabled:
                action = Action(action.agent_id, ActionType.DO_NOTHING, None,
                                harvest=getattr(action, "harvest", 0.0))
            if action.action_type == ActionType.ARM_OTHER and not arm_enabled:
                action = Action(action.agent_id, ActionType.DO_NOTHING, None,
                                harvest=getattr(action, "harvest", 0.0))
            # HARVEST exists only on the commons rung; below it, neutralise to hold.
            if action.action_type == ActionType.HARVEST and not self.params.get("commons_enabled", False):
                action = Action(action.agent_id, ActionType.DO_NOTHING, None,
                                harvest=getattr(action, "harvest", 0.0))
            if self.can_afford_action(action.agent_id, action.action_type):
                valid_actions.append(action)
                round_log["actions"].append({
                    "agent": action.agent_id,
                    "action": action.action_type.value,
                    "target": action.target_id
                })
            else:
                round_log["actions"].append({
                    "agent": action.agent_id,
                    "action": "no_action",
                    "reason": "insufficient_resources"
                })

        # Initialize resource changes and breakdown
        for agent_id in self.state.agents:
            round_log["resource_changes"][agent_id] = 0.0
            round_log["resource_breakdown"][agent_id] = {
                "invest_received": 0.0, "invest_cost": 0.0,
                "arm_cost": 0.0, "conflict_cost": 0.0, "combat_transfer": 0.0,
                "harvest": 0.0, "harvest_cost": 0.0, "decay": 0.0,
            }

        # Process non-attack actions first
        for action in valid_actions:
            if action.action_type == ActionType.DO_NOTHING:
                pass
            elif action.action_type == ActionType.INVEST_OTHER:
                self._resolve_invest_other(action, round_log)
            elif action.action_type == ActionType.ARM_OTHER:
                self._resolve_arm_other(action, round_log)

        # Process attacks last (after arms are updated) — coalition grouping
        attack_actions = [a for a in valid_actions if a.action_type == ActionType.ATTACK]
        if attack_actions:
            self._resolve_attacks_grouped(attack_actions, round_log)

        # Commons harvest (adds harvested units to resource_changes, draws down the stock)
        self._resolve_commons(valid_actions, round_log)

        # Resource transition: R(t+1) = max(0, delta_R * (R(t) + ΔR))
        delta_R = self.params["delta_R"]
        for agent_id in self.state.agents:
            before = self.state.resources[agent_id]
            change = round_log["resource_changes"].get(agent_id, 0.0)
            after_changes = before + change
            after_decay = after_changes * delta_R
            decay_amount = after_changes - after_decay  # positive when delta_R < 1
            self.state.resources[agent_id] = max(0, after_decay)
            round_log["resource_changes"][agent_id] -= decay_amount
            round_log["resource_breakdown"][agent_id]["decay"] += decay_amount

        # Decay arm bonuses
        self._decay_arms()

        # Commons regeneration (GovSim doubling, capped at K; collapse is absorbing)
        self._regenerate_commons(round_log)

        # Add to history and increment round
        self.state.history.append(round_log)
        self.state.round_number += 1

        return round_log

    def _resolve_commons(self, valid_actions: List[Action], round_log: Dict):
        """Harvest the shared stock (production behaviour).

        Harvesting is the agent's ACTION (ActionType.HARVEST) — mutually
        exclusive with transfer/strengthen/take/hold, so harvesting forgoes
        cooperating that round. The claim is f_i × (agent's own current
        resources), where f_i is a continuous, unbounded fraction the agent
        chose. If total claims exceed the stock, allocate in random order
        until depleted (GovSim rationing); harvested units are added to
        resources. Regeneration and collapse follow downstream."""
        if not self.params.get("commons_enabled", False):
            return
        K = self.params["commons_K"]
        S = self.state.commons_stock
        round_log["commons"] = {"stock_before": S, "K": K,
                                "collapsed": self.state.commons_collapsed}
        open_round = int(self.params.get("commons_open_round", 1))
        if self.state.round_number < open_round:
            # Commons not yet open: announced in the prompt from R1, harvestable
            # from open_round. Attempts draw nothing (the action is still spent).
            round_log["commons"]["closed_until"] = open_round
            round_log["commons"]["grants_pct"] = {}
            round_log["commons"]["harvested"] = 0.0
            return
        if self.state.commons_collapsed or S <= 0:
            round_log["commons"]["grants_pct"] = {}
            round_log["commons"]["harvested"] = 0.0
            return
        claims = {}
        chosen_frac = {}  # agent_id -> chosen fraction of own resources
        harvest_raw = {}  # agent_id -> the model's RAW harvest string, for scaling audit (C2)
        for a in valid_actions:
            # Harvesting is its OWN action — only HARVEST actions draw.
            if a.action_type != ActionType.HARVEST:
                continue
            raw = max(0.0, float(getattr(a, "harvest", 0.0) or 0.0))
            if raw <= 0:
                continue
            raw_str = getattr(a, "harvest_raw", None)
            if raw_str is not None:
                harvest_raw[a.agent_id] = raw_str
                round_log["resource_breakdown"][a.agent_id]["harvest_raw"] = raw_str
            # Per-round harvest cap: clamp the chosen fraction at
            # harvest_frac_cap so no single agent can strip the pool in one grab.
            # Makes depletion GRADUAL when many harvest (governance/moratorium has
            # time to form) and turns "stripping" into a SUSTAINED coordinated act.
            # 0 = uncapped (frozen behaviour). The sustainable harvester count is
            # ~MSY/(cap*R): fewer than the population -> rival access -> enclosure.
            cap = self.params.get("harvest_frac_cap", 0.0)
            if cap > 0 and raw > cap:
                raw = cap
            # raw is a fraction (>=0) of the agent's OWN current resources.
            chosen_frac[a.agent_id] = raw
            amt = raw * self.state.resources[a.agent_id]
            if amt > 0:
                claims[a.agent_id] = amt
        grants = self._ration_commons(claims, S) if claims else {}
        harvested = 0.0
        for aid, amt in grants.items():
            round_log["resource_changes"][aid] += amt
            round_log["resource_breakdown"][aid]["harvest"] += amt
            harvested += amt
        self.state.commons_stock = max(0.0, S - harvested)
        round_log["commons"]["grants_pct"] = {aid: 100.0 * amt / K for aid, amt in grants.items()}
        round_log["commons"]["harvested"] = harvested
        round_log["commons"]["stock_after_harvest"] = self.state.commons_stock
        if harvest_raw:
            round_log["commons"]["harvest_raw"] = harvest_raw  # raw model strings (C2 audit)
        # Log the chosen fraction per agent (the decision variable) alongside
        # the realized absolute harvest, for the comeback/inequality analysis.
        round_log["commons"]["harvest_frac"] = dict(chosen_frac)
        for aid, f in chosen_frac.items():
            round_log["resource_breakdown"][aid]["harvest_frac"] = f

        # Participation cost: choosing HARVEST costs
        # c_harvest x own resources, symmetric with take's c_atk and transfer's
        # c_inv. Paid on CHOICE (raw>0, commons live), like c_atk — whether or
        # not rationing granted much. Turns harvest into a priced action:
        # UNIFORM exploitation dissipates to ~0 (the cost eats the per-capita
        # sustainable share MSY/N ~= 1.3% of R), while ASYMMETRIC / low-congestion
        # access still pays -> common-pool rivalry + emergent enclosure, not a
        # free sustainable sip. c_harvest=0 reproduces the frozen (free) behaviour.
        c_harvest = self.params.get("c_harvest", 0.0)
        if c_harvest > 0:
            total_cost = 0.0
            for aid in chosen_frac:
                cost = self._frac(aid, c_harvest)
                round_log["resource_changes"][aid] -= cost
                round_log["resource_breakdown"][aid]["harvest_cost"] += cost
                total_cost += cost
            round_log["commons"]["harvest_cost_total"] = total_cost

    def _ration_commons(self, claims: Dict[str, float], stock: float) -> Dict[str, float]:
        """Grant all claims if the stock covers them; otherwise allocate in random
        order until the stock is depleted (the last grantee may get a partial)."""
        if sum(claims.values()) <= stock:
            return dict(claims)
        order = list(claims.keys())
        np.random.shuffle(order)
        grants, remaining = {}, stock
        for aid in order:
            amt = min(claims[aid], remaining)
            grants[aid] = amt
            remaining -= amt
            if remaining <= 0:
                break
        return grants

    def _regenerate_commons(self, round_log: Dict):
        """End-of-round stock dynamics: if the remaining stock is below the collapse
        threshold it collapses permanently (absorbing 0); otherwise it grows by
        commons_regen× (GovSim doubling = 2.0), capped at K."""
        if not self.params.get("commons_enabled", False):
            return
        if self.state.round_number < int(self.params.get("commons_open_round", 1)):
            # Stock frozen while the commons is closed: no regen, no collapse
            # check — it opens exactly at the configured level (e.g. MSY).
            return
        S = self.state.commons_stock
        K = self.params["commons_K"]
        C = K * self.params["commons_collapse_frac"]
        regen = self.params.get("commons_regen", 2.0)
        if self.state.commons_collapsed:
            new_S = 0.0
        elif S < C:
            new_S = 0.0
            self.state.commons_collapsed = True
        else:
            new_S = min(K, regen * S)
        self.state.commons_stock = new_S
        if "commons" in round_log:
            round_log["commons"]["stock_after_regen"] = new_S
            round_log["commons"]["collapsed"] = self.state.commons_collapsed

    def _count_recent_investments(self, investor: str, target: str) -> int:
        """Count how many times investor→target occurred in recent history."""
        window = self.params["tau_sat"]
        recent = self.state.history[-window:] if self.state.history else []
        count = 0
        for rd in recent:
            for act in rd.get("actions", []):
                if (act.get("agent") == investor and
                    act.get("action") == ActionType.INVEST_OTHER.value and
                    act.get("target") == target):
                    count += 1
        return count

    def _count_recent_attacks(self, aggressor: str) -> int:
        """Count aggressor's attack actions in the last tau_atk rounds (strictly prior)."""
        window = self.params["tau_atk"]
        recent = self.state.history[-window:] if self.state.history else []
        count = 0
        for rd in recent:
            for act in rd.get("actions", []):
                if act.get("agent") == aggressor and act.get("action") == ActionType.ATTACK.value:
                    count += 1
        return count

    def _resolve_invest_other(self, action: Action, round_log: Dict):
        """Resolve invest-other action.

        Investor pays invest_other_cost_pct% of own resources (always full cost).
        Target receives invest_other_return_pct% × saturation_decay^prior_investments.
        Diminishing returns per pair — diversification restores full returns.
        """
        agent_id = action.agent_id
        target_id = action.target_id

        if target_id not in self.state.agents:
            return
        # Network restriction: target must be a neighbor
        if self._valid_targets and target_id not in self._valid_targets.get(agent_id, []):
            return

        # Cost is always full (you pay the same regardless of saturation)
        cost = self._frac(agent_id, self.params["c_inv"])
        round_log["resource_changes"][agent_id] -= cost
        round_log["resource_breakdown"][agent_id]["invest_cost"] += cost

        # Return diminishes with repeated investment in same pair
        prior = self._count_recent_investments(agent_id, target_id)
        effective_return = self.params["g_inv"] * (self.params["gamma_sat"] ** prior)
        transfer = self.state.resources[agent_id] * effective_return

        round_log["resource_changes"][target_id] += transfer
        round_log["resource_breakdown"][target_id]["invest_received"] += transfer
        round_log["bilateral_flows"][(agent_id, target_id)] += transfer

    def _resolve_arm_other(self, action: Action, round_log: Dict):
        """Resolve arm-other: pay c_arm · R, target gains cost × mu_arm as combat bonus."""
        agent_id = action.agent_id
        target_id = action.target_id

        if target_id not in self.state.agents:
            return
        if self._valid_targets and target_id not in self._valid_targets.get(agent_id, []):
            return

        cost = self._frac(agent_id, self.params["c_arm"])
        bonus = cost * self.params["mu_arm"]
        round_log["resource_changes"][agent_id] -= cost
        round_log["resource_breakdown"][agent_id]["arm_cost"] += cost
        self.state.arm_bonuses[target_id] = self.state.arm_bonuses.get(target_id, 0.0) + bonus

    def _resolve_attacks_grouped(self, attack_actions: List[Action], round_log: Dict):
        """Group attacks by target and resolve as coalition combats.

        Snapshot-based simultaneous resolution: all combat strengths are frozen
        before any combat resolves, so an agent involved in multiple combats
        (e.g., attacking one target while being attacked by others) uses the
        same strength everywhere. No order dependence.
        """
        # Network restriction: filter out attacks on non-neighbors
        if self._valid_targets:
            attack_actions = [
                a for a in attack_actions
                if a.target_id in self._valid_targets.get(a.agent_id, [])
            ]
            if not attack_actions:
                return

        # Snapshot combat strengths for all agents involved in any attack
        involved = set()
        for action in attack_actions:
            involved.add(action.agent_id)
            involved.add(action.target_id)

        snapshots = {}
        for aid in involved:
            if aid in self.state.agents:
                snapshots[aid] = self.state.resources[aid] + round_log["resource_changes"].get(aid, 0.0) + self.state.arm_bonuses.get(aid, 0.0)

        # Group attacks by target
        attack_map = {}  # attacker → target
        groups = defaultdict(list)  # target → [Action]
        for action in attack_actions:
            if action.target_id in self.state.agents:
                attack_map[action.agent_id] = action.target_id
                groups[action.target_id].append(action)

        # Merge mutual/connected conflicts:
        # If defender D also attacks one of their own attackers, D's attack
        # is absorbed — it's the same fight, not a separate engagement.
        # Process larger groups first (coalition = primary conflict).
        absorbed = set()  # agents whose attack action is absorbed (they're defending)
        for target in sorted(groups.keys(), key=lambda t: len(groups[t]), reverse=True):
            if target in absorbed:
                continue
            attacker_ids = {a.agent_id for a in groups[target]} - absorbed
            if not attacker_ids:
                continue
            if target in attack_map:
                counter_target = attack_map[target]
                if counter_target in attacker_ids:
                    # Defender counter-attacks one of their attackers — same fight
                    absorbed.add(target)

        # Build final groups, filtering out absorbed attackers
        # Note: absorbed agents can still be DEFENDERS — only their attack
        # action is absorbed, not their role as target.
        final_groups = {}
        for target, actions in groups.items():
            remaining = [a for a in actions if a.agent_id not in absorbed]
            if remaining:
                final_groups[target] = remaining

        # Resolve each coalition attack
        for defender_id, attacker_actions in final_groups.items():
            self._resolve_coalition_attack(attacker_actions, defender_id, snapshots, round_log)

    def _resolve_coalition_attack(self, attacker_actions: List[Action],
                                   defender_id: str, snapshots: Dict[str, float],
                                   round_log: Dict):
        """Resolve coalition attack: multiple attackers vs one defender.

        When a single attacker targets the defender, this is equivalent to
        the old 1v1 combat. When multiple attackers target the same defender,
        their combat strengths combine. Spoils are split proportionally.
        """
        attacker_ids = [a.agent_id for a in attacker_actions]

        # Combat strengths from snapshot
        attacker_powers = {aid: snapshots.get(aid, 0.0) for aid in attacker_ids}
        coalition_power = sum(attacker_powers.values())
        defender_power = snapshots.get(defender_id, 0.0)

        # Probabilistic combat resolution (single roll for the group)
        total_power = coalition_power + defender_power
        if total_power > 0:
            attacker_win_prob = coalition_power / total_power
        else:
            attacker_win_prob = 0.5

        coalition_wins = np.random.random() < attacker_win_prob

        # Conflict costs: each participant pays c_atk% scaled by eta_atk ^ prior_attacks
        # within the last tau_atk rounds. Scaling is per-participant — defenders with
        # their own aggressive history also pay more.
        c_atk = self.params["c_atk"]
        eta = self.params["eta_atk"]
        attacker_cost_scales = {}
        for aid in attacker_ids:
            prior = self._count_recent_attacks(aid)
            scale = eta ** prior
            attacker_cost_scales[aid] = {"prior_attacks": prior, "scale": scale}
            cc = self._frac(aid, c_atk * scale)
            round_log["resource_changes"][aid] -= cc
            round_log["resource_breakdown"][aid]["conflict_cost"] += cc
        def_prior = self._count_recent_attacks(defender_id)
        def_scale = eta ** def_prior
        def_cc = self._frac(defender_id, c_atk * def_scale)
        round_log["resource_changes"][defender_id] -= def_cc
        round_log["resource_breakdown"][defender_id]["conflict_cost"] += def_cc

        # Spoils. phi = fraction of the LOSER's resources the winner takes.
        #   lethal_pot: phi = min(1, alpha * S_winner / S_loser) — scales with the
        #     strength ratio and is sized on the loser, so the stronger side can
        #     fully drain it (phi=1 => loser emptied => bankruptcy). Symmetric: a
        #     winning defender drains attackers the same way. At the equal-strength
        #     slice (S_win=S_lose) phi reduces to alpha, so cell placement is unchanged.
        #   else (flat pot): phi = alpha; symmetric_stakes governs the defender-win branch.
        alpha = self.params["alpha"]
        lethal = self.params.get("lethal_pot", False)
        total_transfer = 0.0
        phi = alpha  # logged below

        if coalition_wins:
            defender_effective = max(0, self.state.resources[defender_id] + round_log["resource_changes"][defender_id])
            if lethal:
                phi = min(1.0, alpha * coalition_power / defender_power) if defender_power > 0 else 1.0
            pot = defender_effective * phi
            total_transfer = pot
            round_log["resource_changes"][defender_id] -= pot
            round_log["resource_breakdown"][defender_id]["combat_transfer"] -= pot
            for aid in attacker_ids:
                share = (attacker_powers[aid] / coalition_power) if coalition_power > 0 else (1.0 / len(attacker_ids))
                aid_share = pot * share
                round_log["resource_changes"][aid] += aid_share
                round_log["resource_breakdown"][aid]["combat_transfer"] += aid_share
                round_log["bilateral_flows"][(defender_id, aid)] += aid_share
        else:
            if lethal:
                # Defender wins: each attacker loses phi of its OWN R; defender gains the sum.
                phi = min(1.0, alpha * defender_power / coalition_power) if coalition_power > 0 else 1.0
                for aid in attacker_ids:
                    aid_effective = max(0, self.state.resources[aid] + round_log["resource_changes"][aid])
                    aid_loss = aid_effective * phi
                    round_log["resource_changes"][aid] -= aid_loss
                    round_log["resource_breakdown"][aid]["combat_transfer"] -= aid_loss
                    round_log["resource_changes"][defender_id] += aid_loss
                    round_log["resource_breakdown"][defender_id]["combat_transfer"] += aid_loss
                    round_log["bilateral_flows"][(aid, defender_id)] += aid_loss
                    total_transfer += aid_loss
            elif self.params.get("symmetric_stakes", False):
                defender_effective = max(0, self.state.resources[defender_id] + round_log["resource_changes"][defender_id])
                pot = defender_effective * alpha
                per_attacker = pot / len(attacker_ids) if attacker_ids else 0.0
                for aid in attacker_ids:
                    aid_effective = max(0, self.state.resources[aid] + round_log["resource_changes"][aid])
                    actual_loss = min(per_attacker, aid_effective)
                    round_log["resource_changes"][aid] -= actual_loss
                    round_log["resource_breakdown"][aid]["combat_transfer"] -= actual_loss
                    round_log["resource_changes"][defender_id] += actual_loss
                    round_log["resource_breakdown"][defender_id]["combat_transfer"] += actual_loss
                    round_log["bilateral_flows"][(aid, defender_id)] += actual_loss
                    total_transfer += actual_loss
            else:
                for aid in attacker_ids:
                    aid_effective = max(0, self.state.resources[aid] + round_log["resource_changes"][aid])
                    aid_loss = aid_effective * alpha
                    round_log["resource_changes"][aid] -= aid_loss
                    round_log["resource_breakdown"][aid]["combat_transfer"] -= aid_loss
                    round_log["resource_changes"][defender_id] += aid_loss
                    round_log["resource_breakdown"][defender_id]["combat_transfer"] += aid_loss
                    round_log["bilateral_flows"][(aid, defender_id)] += aid_loss
                    total_transfer += aid_loss

        round_log["combat_results"].append({
            "attackers": attacker_ids,
            "defender": defender_id,
            "attacker_powers": attacker_powers,
            "coalition_power": coalition_power,
            "defender_power": defender_power,
            "winner": "coalition" if coalition_wins else "defender",
            "attacker_win_prob": attacker_win_prob,
            "lethal_pot": lethal,
            "phi": phi,
            "total_transfer": total_transfer,
            "attacker_cost_scales": attacker_cost_scales,
            "defender_cost_scale": {"prior_attacks": def_prior, "scale": def_scale},
        })

    def _decay_arms(self):
        """Decay all arm bonuses by delta_B factor. Remove when negligible."""
        decay = self.params["delta_B"]
        expired = []
        for agent_id in self.state.arm_bonuses:
            self.state.arm_bonuses[agent_id] *= decay
            if self.state.arm_bonuses[agent_id] < 0.01:
                expired.append(agent_id)
        for agent_id in expired:
            del self.state.arm_bonuses[agent_id]

    def get_state(self) -> GameState:
        """Get current game state."""
        return self.state

    def is_game_over(self, max_rounds: int) -> bool:
        """Check if game should end."""
        return self.state.round_number > max_rounds
