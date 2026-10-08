"""Calibration guard (M3): the complexity ladder must NEVER re-price the economy
across rungs. ladder_L1..L4 and commons_probe* are one unchanged engine — only
the rung FLAGS and comm scope may differ. This test asserts every numeric economy
parameter is byte-identical across all configs, catching silent calibration drift
(a per-action payoff that quietly changed between rungs would invalidate the
"action space is nested, payoffs never re-priced" claim).

Does NOT refactor configs to inheritance — just asserts equality on the loaded
dicts.
"""
import os

import pytest
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

# Configs that share one frozen economy (the ladder + the commons probes).
PARITY_CONFIGS = [
    "ladder_L1.yaml", "ladder_L2.yaml", "ladder_L3.yaml", "ladder_L4.yaml",
]

# Keys allowed to differ between rungs: the rung FLAGS (which affordances are on),
# rewiring magnitude, the harvest mechanic, and comm scope. Everything else is
# the frozen economy and must match.
ALLOWED_TO_DIFFER = {
    "arm_enabled", "take_enabled", "assoc_enabled", "commons_enabled",
    "rewiring_prob", "commons_harvest_mode", "commons_harvest_pct", "comm_scope",
    # commons affordance config (only active at L4), not a re-priced economy payoff:
    "commons_init", "commons_regen",
    "memory",  # substrate config block, not an economy price
}


def _load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_ladder_economy_params_identical():
    """Every numeric economy param shared across the parity configs is identical."""
    configs = {name: _load(name) for name in PARITY_CONFIGS}

    # Union of all economy keys (excluding the flags/mechanic keys allowed to vary).
    economy_keys = set()
    for params in configs.values():
        economy_keys |= set(params) - ALLOWED_TO_DIFFER

    reference_name = PARITY_CONFIGS[0]
    reference = configs[reference_name]
    mismatches = []
    for key in sorted(economy_keys):
        ref_val = reference.get(key, "<MISSING>")
        for name, params in configs.items():
            val = params.get(key, "<MISSING>")
            if val != ref_val:
                mismatches.append(
                    f"{key}: {reference_name}={ref_val!r} vs {name}={val!r}"
                )

    assert not mismatches, (
        "Economy params drifted across ladder rungs (calibration NOT frozen):\n  "
        + "\n  ".join(mismatches)
    )
