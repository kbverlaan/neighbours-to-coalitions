"""Standardized rich terminal output for simulation runs.

Stateless functions that format and print game state, actions, combat results,
reasoning traces, and network topology. Supports ANSI colors with auto-detection
(disabled when piped or on SLURM, respects NO_COLOR env var).
"""

import os
import sys

# ---------------------------------------------------------------------------
# Color handling
# ---------------------------------------------------------------------------

# ANSI color codes
_COLORS = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'gray': '\033[90m',
}

# 6 distinct agent colors, cycled by agent index
_AGENT_COLORS = ['cyan', 'magenta', 'yellow', 'green', 'blue', 'red']

# Action type → color (canonical new names + legacy aliases)
_ACTION_COLORS = {
    'transfer': 'green',   'invest_other': 'green',
    'strengthen': 'yellow', 'arm_other': 'yellow',
    'take': 'red',         'attack': 'red',
    'hold': 'gray',        'do_nothing': 'gray',
}

# Action type → icon (canonical new names + legacy aliases)
_ACTION_ICONS = {
    'transfer': '🤝',   'invest_other': '🤝',
    'strengthen': '🛡️', 'arm_other': '🛡️',
    'take': '⚔️',       'attack': '⚔️',
    'hold': '💤',       'do_nothing': '💤',
}


def _color_enabled():
    """Check if color output should be used."""
    if os.environ.get('NO_COLOR'):
        return False
    return sys.stdout.isatty()


def C(text, color):
    """Wrap text in ANSI color if enabled."""
    if not _color_enabled():
        return str(text)
    code = _COLORS.get(color, '')
    reset = _COLORS['reset']
    return f"{code}{text}{reset}" if code else str(text)


def _agent_idx(agent_id):
    """Extract numeric index from agent_id like 'agent_3' → 2."""
    try:
        return int(agent_id.split('_')[-1]) - 1
    except (ValueError, IndexError):
        return hash(agent_id) % len(_AGENT_COLORS)


def _agent_color(agent_id):
    """Get stable color name for an agent."""
    return _AGENT_COLORS[_agent_idx(agent_id) % len(_AGENT_COLORS)]


def _short_id(agent_id):
    """Shorten 'agent_3' → '3', 'Atlas' → 'Atlas'."""
    if agent_id.startswith('agent_'):
        return agent_id.split('_')[-1]
    return agent_id


def _ca(agent_id):
    """Color an agent ID."""
    return C(agent_id, _agent_color(agent_id))


def p(msg=''):
    """Print with immediate flush."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Per-round output
# ---------------------------------------------------------------------------

def print_round_header(round_num, max_rounds):
    pct = round_num / max_rounds
    filled = int(20 * pct)
    bar = '━' * filled + C('╺' + '─' * (19 - filled), 'dim')
    p(f"\n{C('━' * 80, 'dim')}")
    p(f"  {C('ROUND', 'bold')} {C(f'{round_num}', 'white')}{C(f'/{max_rounds}', 'dim')}  {bar}  {C(f'{pct:.0%}', 'dim')}")
    p(C('━' * 80, 'dim'))


def print_resource_bars(resources, arm_bonuses, agent_ids, bar_width=40):
    """Print resource bars with arm bonus as lighter blocks."""
    # Compute total strengths for sorting and scaling
    totals = {}
    for aid in agent_ids:
        r = resources.get(aid, 0)
        b = arm_bonuses.get(aid, 0)
        totals[aid] = r + b

    max_total = max(totals.values()) if totals else 1
    max_total = max(max_total, 0.01)  # avoid division by zero

    # Sort by total strength descending
    sorted_ids = sorted(agent_ids, key=lambda a: totals[a], reverse=True)

    for aid in sorted_ids:
        r = resources.get(aid, 0)
        b = arm_bonuses.get(aid, 0)

        # Scale bars — compute total length first, then split
        total_len = int(bar_width * (r + b) / max_total)
        res_len = int(bar_width * r / max_total)
        arm_len = total_len - res_len

        res_bar = C('█' * res_len, _agent_color(aid))
        arm_bar = C('░' * arm_len, _agent_color(aid)) if arm_len > 0 else ''
        arm_tag = f"  {C(f'(+{b:.1f})', 'dim')}" if b > 0.5 else ''
        broke = f"  {C('BROKE', 'red')}" if r < 0.01 else ''

        p(f"  {_ca(aid):>20s}: {r:6.1f} {res_bar}{arm_bar}{arm_tag}{broke}")


def print_actions(action_map, agent_ids):
    """Print agent actions with icons. action_map = {aid: {'action': str, 'target': str|None}}."""
    p()
    for aid in agent_ids:
        info = action_map.get(aid, {})
        action = info.get('action', 'no_action')
        target = info.get('target')
        color = _ACTION_COLORS.get(action, 'white')
        icon = _ACTION_ICONS.get(action, '  ')

        desc = C(action, color)
        if target:
            desc += f" → {_ca(target)}"

        p(f"  {icon} {_ca(aid):>20s}: {desc}")


# Keywords that signal interesting note content (matched in context)
_INTERESTING_KEYWORDS = [
    'attack', 'defect', 'betray', 'retali', 'punish',
    'coalition', 'coordinate', 'together',
    'distrust', 'broke promise', 'lied', 'deceiv',
    'switch strategy', 'change plan', 'new strategy',
    'target', 'vulnerable', 'exploit',
]

# Negated/boring contexts — if these appear near a keyword, skip it
_BORING_CONTEXTS = [
    'no threat', 'no attack', 'no conflict', 'not attack',
    'avoid attack', 'safe', 'steady', 'continue',
]

# Actions that are always "interesting"
_INTERESTING_ACTIONS = {'take', 'attack', 'strengthen', 'arm_other', 'transfer', 'invest_other'}


def _is_interesting(action, note):
    """Check if an agent's round is worth highlighting.

    Interesting = non-default action OR note with strategic content.
    A default action (hold) with a boring note is collapsed.
    """
    if action in _INTERESTING_ACTIONS:
        return True
    if action in ('do_nothing', 'hold') and note:
        note_lower = note.lower()
        # Check for boring contexts first — these negate keywords
        if any(ctx in note_lower for ctx in _BORING_CONTEXTS):
            return False
        return any(kw in note_lower for kw in _INTERESTING_KEYWORDS)
    return False


def print_agent_round_summary(action_map, notes, agent_ids):
    """Print smart per-agent summary: highlight interesting, collapse boring.

    Combines action + note on one line for interesting agents.
    Groups boring agents (hold, no interesting note) into a compact line.
    """
    interesting = []
    boring = []

    for aid in agent_ids:
        info = action_map.get(aid, {})
        action = info.get('action', 'no_action')
        target = info.get('target')
        note = notes.get(aid, '')

        if _is_interesting(action, note):
            interesting.append((aid, action, target, note))
        else:
            boring.append((aid, action))

    p()

    # Print interesting agents with full detail
    for aid, action, target, note in interesting:
        color = _ACTION_COLORS.get(action, 'white')
        icon = _ACTION_ICONS.get(action, '  ')

        desc = C(action, color)
        if target:
            desc += f" → {_ca(target)}"

        # Append note snippet on same line
        if note:
            text = note.strip().replace('\n', ' ')
            if len(text) > 65:
                text = text[:62] + '...'
            p(f"  {icon} {_ca(aid):>20s}: {desc}")
            p(f"     {C('└─', 'dim')} {C(text, 'dim')}")
        else:
            p(f"  {icon} {_ca(aid):>20s}: {desc}")

    # Collapse boring agents into one line
    if boring:
        boring_parts = []
        for aid, action in boring:
            boring_parts.append(C(_short_id(aid), _agent_color(aid)))
        boring_str = ', '.join(boring_parts)
        action_str = boring[0][1] if len(set(a for _, a in boring)) == 1 else 'safe'
        p(f"  {C('···', 'dim')} {boring_str}: {C(action_str, 'dim')}")


def print_combat_results(combat_results):
    """Print combat outcomes inline after actions."""
    for combat in combat_results:
        attackers = combat.get('attackers', [])
        defender = combat.get('defender', '?')
        winner = combat.get('winner', '?')
        win_prob = combat.get('attacker_win_prob', 0)
        mutual = combat.get('mutual', False)

        if mutual:
            # Mutual attack — symmetric duel
            a_str = _ca(attackers[0])
            b_str = _ca(defender)
            if winner == 'coalition':
                result = f"{a_str} {C('wins', 'red')}"
            else:
                result = f"{b_str} {C('wins', 'red')}"
            p(f"  ⚔️ {a_str} vs {b_str} (mutual): {result} ({win_prob:.0%})")
        else:
            atk_str = ','.join(_ca(a) for a in attackers)
            def_str = _ca(defender)

            if winner == 'coalition':
                icon = '⚔️'
                result = C('coalition wins', 'red')
            else:
                icon = '🛡️'
                result = C('defender wins', 'blue')

            p(f"  {icon} [{atk_str}] vs {def_str}: {result} ({win_prob:.0%})")


def print_messages(messages):
    """Print communication messages (future: comms phase)."""
    if not messages:
        return
    for msg in messages:
        sender = msg.get('from', '?')
        targets = msg.get('message_to') or msg.get('to') or []
        if isinstance(targets, str):
            targets = [targets]
        text = msg.get('message', '')

        icon = '📢' if len(targets) > 1 else '📨'
        target_str = ", ".join(_ca(t) for t in targets) if targets else C('?', 'white')

        # Truncate long messages
        if len(text) > 80:
            text = text[:77] + '...'

        p(f"  {_ca(sender)} {icon} → {target_str}: \"{C(text, 'dim')}\"")


def print_reasoning(agent_id, thinking, max_chars=150):
    """Print condensed reasoning for one agent."""
    if not thinking:
        return
    # Extract the end of thinking (usually the conclusion)
    text = thinking.strip()
    if len(text) > max_chars:
        text = '...' + text[-(max_chars - 3):]
    # Clean up newlines
    text = ' '.join(text.split())
    p(f"  {_ca(agent_id)} 💭 \"{C(text, 'dim')}\"")


def print_network(agent_ids, get_neighbors_fn):
    """Print compact network adjacency."""
    parts = []
    for aid in agent_ids:
        neighbors = get_neighbors_fn(aid)
        short_neighbors = ','.join(_short_id(n) for n in neighbors)
        parts.append(f"{C(_short_id(aid), _agent_color(aid))}→{short_neighbors}")
    p(f"  {C('Net:', 'dim')} {' | '.join(parts)}")


def print_network_rewire(stats):
    """Print network rewiring notification."""
    n = stats.get('agents_rewired', 0)
    if n > 0:
        p(f"  {C(f'🔗 Rewired: {n} agents', 'dim')}")


def print_round_time(seconds):
    """Print round elapsed time."""
    p(f"  {C(f'⏱ {seconds:.0f}s', 'dim')}")


def print_metrics_dashboard(metrics, prev_metrics=None):
    """Print colorful metrics dashboard with trend arrows."""
    coop = metrics.get('cooperation_rate', metrics.get('cooperation_ratio', 0))
    gini = metrics.get('gini', 0)
    mod = metrics.get('modularity', 0)
    stab = metrics.get('action_stability')

    def _trend(curr, prev_val):
        if prev_val is None:
            return ''
        diff = curr - prev_val
        if abs(diff) < 0.005:
            return C(' ─', 'dim')
        return C(' ▲', 'green') if diff > 0 else C(' ▼', 'red')

    prev = prev_metrics or {}

    # Cooperation: green when high, red when low
    coop_color = 'green' if coop >= 0.4 else 'yellow' if coop >= 0.15 else 'red'
    coop_bar = C('▮' * int(10 * coop), coop_color) + C('▯' * (10 - int(10 * coop)), 'dim')
    coop_trend = _trend(coop, prev.get('cooperation_rate', prev.get('cooperation_ratio')))

    # Gini: green when low (equal), red when high (unequal)
    gini_color = 'green' if gini < 0.15 else 'yellow' if gini < 0.35 else 'red'
    gini_trend = _trend(gini, prev.get('gini'))

    # Modularity: blue/cyan for structure
    mod_color = 'cyan' if mod > 0.3 else 'blue' if mod > 0.1 else 'dim'
    mod_trend = _trend(mod, prev.get('modularity'))

    # Stability
    stab_str = f"{stab:.0%}" if stab is not None else "n/a"
    stab_color = 'white' if stab is not None and stab >= 0.7 else 'dim'

    p()
    p(f"  {C('f_C', 'bold')} {C(f'{coop:.2f}', coop_color)} {coop_bar}{coop_trend}"
      f"   {C('Gini', 'bold')} {C(f'{gini:.3f}', gini_color)}{gini_trend}"
      f"   {C('Q', 'bold')} {C(f'{mod:.2f}', mod_color)}{mod_trend}"
      f"   {C('Stab', 'bold')} {C(stab_str, stab_color)}")


def print_notes(notes, agent_ids):
    """Print compact note-to-self previews per agent."""
    has_notes = any(notes.get(aid) for aid in agent_ids)
    if not has_notes:
        return
    p(f"  {C('Notes:', 'dim')}")
    for aid in agent_ids:
        note = notes.get(aid)
        if not note:
            continue
        # Truncate to ~60 chars
        text = note.strip().replace('\n', ' ')
        if len(text) > 60:
            text = text[:57] + '...'
        p(f"    {_ca(aid)}: {C(text, 'dim')}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_run_header(config_str):
    """Print run configuration header."""
    p(f"\n{C('=' * 80, 'bold')}")
    p(config_str)
    p(C('=' * 80, 'bold'))


def print_final_summary(resources, arm_bonuses, agent_ids, elapsed=None,
                        all_metrics=None):
    """Print final resource ranking with optional metrics summary."""
    p(f"\n{C('━' * 80, 'bold')}")
    title = '  FINAL STATE'
    if elapsed is not None:
        title += f'  {C(f"({elapsed:.0f}s)", "dim")}'
    p(C(title, 'bold'))
    p(C('━' * 80, 'bold'))

    print_resource_bars(resources, arm_bonuses, agent_ids)

    # Print metrics trajectory sparkline if available
    if all_metrics and len(all_metrics) > 1:
        _print_sparklines(all_metrics)


def print_action_distribution(round_summaries, total_label=True):
    """Print action distribution across all rounds."""
    action_counts = {}
    for rs in round_summaries:
        for aid, info in rs.items():
            a = info['action']
            action_counts[a] = action_counts.get(a, 0) + 1

    total = sum(action_counts.values())
    if total == 0:
        return

    p(f"\n  {C('Action distribution', 'bold')} ({total} total):")
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = c / total
        color = _ACTION_COLORS.get(a, 'white')
        icon = _ACTION_ICONS.get(a, '  ')
        bar = C('█' * int(25 * pct), color) + C('░' * (25 - int(25 * pct)), 'dim')
        p(f"  {icon} {a:15s} {bar} {C(f'{pct:4.0%}', color)} ({c})")


def print_agent_profiles(round_summaries, agent_ids, final_resources):
    """Print per-agent action profiles."""
    p(f"\n{C('Agent profiles:', 'bold')}")
    for aid in agent_ids:
        profile = {}
        inactive = 0
        for rs in round_summaries:
            entry = rs.get(aid)
            if entry is None:
                inactive += 1
                continue
            a = entry['action']
            profile[a] = profile.get(a, 0) + 1
        if inactive > 0:
            profile['broke'] = inactive
        profile_str = ', '.join(
            f"{C(a, _ACTION_COLORS.get(a, 'white'))}={c}"
            for a, c in sorted(profile.items(), key=lambda x: -x[1])
        )
        r = final_resources.get(aid, 0)
        p(f"  {_ca(aid):>20s}: {profile_str} | final={r:.1f}")


def _print_sparklines(all_metrics):
    """Print sparkline trajectories for key metrics."""
    sparks = '▁▂▃▄▅▆▇█'

    def _spark(values):
        if not values:
            return ''
        lo, hi = min(values), max(values)
        rng = hi - lo if hi > lo else 1
        return ''.join(sparks[min(int((v - lo) / rng * 7), 7)] for v in values)

    coop_vals = [m.get('cooperation_rate', m.get('cooperation_ratio', 0)) for m in all_metrics]
    gini_vals = [m.get('gini', 0) for m in all_metrics]

    p()
    p(f"  {C('f_C trajectory:', 'bold')}  {C(_spark(coop_vals), 'green')}  "
      f"{C(f'{coop_vals[0]:.2f}', 'dim')} → {C(f'{coop_vals[-1]:.2f}', 'green')}")
    p(f"  {C('Gini trajectory:', 'bold')} {C(_spark(gini_vals), 'yellow')}  "
      f"{C(f'{gini_vals[0]:.3f}', 'dim')} → {C(f'{gini_vals[-1]:.3f}', 'yellow')}")


def print_gini(gini, resources):
    """Print Gini coefficient and resource range."""
    vals = sorted(resources.values())
    p(f"\n  Gini: {C(f'{gini:.3f}', 'bold')}  "
      f"Range: {vals[0]:.1f} – {vals[-1]:.1f}")
