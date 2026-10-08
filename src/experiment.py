"""
Experiment orchestration: turns a sweep YAML spec into many runs.

A spec defines `name`, `reps`, `base_params`, and a `sweep` (either a single
`param`/`values` axis or a `grid` of multiple axes). `run_sweep` iterates
every (condition × rep) sequentially, `run_single` runs one flattened index
(the SLURM-array entrypoint), and `run_batch` runs a consecutive chunk.
"""

from __future__ import annotations
import sys
import re
import copy
import json
import time
from itertools import product
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from runner import run_simulation, save_results, load_config
from analysis.metrics import (
    gini, first_attack_round, cooperation_rate_timeseries,
)


def load_experiment(spec_path: str) -> dict:
    with open(spec_path, 'r') as f:
        import yaml
        spec = yaml.safe_load(f)
    for key in ('name', 'reps', 'base_params', 'sweep'):
        if key not in spec:
            raise ValueError(f"Experiment spec missing required key: '{key}'")
    sweep = spec['sweep']
    if 'grid' not in sweep and ('param' not in sweep or 'values' not in sweep):
        raise ValueError("sweep must have either 'grid' or 'param' + 'values'")
    return spec


def _sanitize_value(value) -> str:
    s = str(value)
    s = s.split('/')[-1] if '/' in s else s
    return re.sub(r'[^a-zA-Z0-9._-]', '_', s)


def generate_conditions(sweep: dict) -> list:
    if 'grid' in sweep:
        axes = sweep['grid']
        axis_values = [
            [{'param': ax['param'], 'target': ax.get('target', 'game_params'), 'value': v}
             for v in ax['values']]
            for ax in axes
        ]
        return [list(combo) for combo in product(*axis_values)]
    param = sweep['param']
    target = sweep.get('target', 'game_params')
    return [
        [{'param': param, 'target': target, 'value': v}]
        for v in sweep['values']
    ]


def condition_label(condition: list) -> str:
    return ", ".join(f"{e['param']}={e['value']}" for e in condition)


def condition_run_id(condition: list, rep: int) -> str:
    parts = [f"{e['param']}_{_sanitize_value(e['value'])}" for e in condition]
    return "_".join(parts) + f"_rep{rep}"


def apply_condition(condition: list, base_params: dict, or_config: dict) -> tuple:
    params = copy.deepcopy(base_params)
    cfg = copy.deepcopy(or_config)
    for e in condition:
        t, p, v = e['target'], e['param'], e['value']
        if t == 'game_params':
            params[p] = v
        elif t == 'openrouter':
            cfg[p] = v
        elif t == 'prompt_config':
            cfg.setdefault('prompt_config', {})[p] = v
        else:
            raise ValueError(f"Unknown target '{t}' for param '{p}'")
    return params, cfg


def _run_cooperation_rate(round_logs):
    pos, meaningful = 0, 0
    for rd in round_logs:
        for a in rd.get('agents', {}).values():
            act = a.get('action', '')
            if act in ('', 'no_action', 'hold', 'do_nothing'):
                continue
            meaningful += 1
            if act in ('transfer', 'strengthen', 'invest_other', 'arm_other'):
                pos += 1
    return pos / meaningful if meaningful > 0 else 0.0


def _summarize_run(round_logs, run_elapsed, condition, rep, run_id):
    final_gini = gini(round_logs[-1]) if round_logs else None
    cond_value = (condition[0]['value'] if len(condition) == 1
                  else {e['param']: e['value'] for e in condition})
    return {
        'run_id': run_id,
        'condition': {e['param']: e['value'] for e in condition},
        'condition_value': cond_value,
        'rep': rep,
        'final_gini': final_gini,
        'cooperation_rate': _run_cooperation_rate(round_logs),
        'first_attack_round': first_attack_round(round_logs),
        'fc_timeseries': cooperation_rate_timeseries(round_logs),
        'elapsed_seconds': round(run_elapsed, 1),
        'timestamp': datetime.now().isoformat(),
    }


def _load_sweep_config(spec):
    project_root = Path(__file__).parent.parent
    api_config_file = spec.get('api_config', 'openrouter_config.yaml')
    or_config = load_config(project_root / "config" / api_config_file)
    for key, value in (spec.get('base_openrouter') or {}).items():
        or_config[key] = value
    return project_root, or_config


def run_sweep(spec_path: str):
    """Run every (condition × rep) in the spec sequentially."""
    load_dotenv()
    spec = load_experiment(spec_path)
    project_root, or_config = _load_sweep_config(spec)

    name = spec['name']
    reps = spec['reps']
    base_params = spec['base_params']

    conditions = generate_conditions(spec['sweep'])
    output_dir = project_root / "data" / "runs" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(conditions) * reps

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}\nDescription: {spec.get('description','')}\n"
          f"Conditions: {len(conditions)}  Reps: {reps}  Total runs: {total_runs}")
    print(f"Output: {output_dir}\n{'='*70}\n")

    sweep_info = spec['sweep']
    manifest = {
        'name': name,
        'description': spec.get('description', ''),
        'sweep_param': sweep_info.get('param', 'grid'),
        'sweep_values': (sweep_info.get('values')
                         or [condition_label(c) for c in conditions]),
        'sweep_spec': sweep_info,
        'reps': reps,
        'base_params': base_params,
        'started_at': datetime.now().isoformat(),
        'runs': [],
    }
    t0 = time.time()

    for condition in conditions:
        label = condition_label(condition)
        print(f"\n{'#'*70}\n# CONDITION: {label}\n{'#'*70}")

        for rep in range(1, reps + 1):
            run_id = condition_run_id(condition, rep)
            params, cfg = apply_condition(condition, base_params, or_config)

            print(f"\n--- {run_id} ---")
            run_start = time.time()
            state, traces, round_logs, run_metadata = run_simulation(params, cfg, run_id)
            run_elapsed = time.time() - run_start

            save_results(state, traces, round_logs, output_dir, run_id, run_metadata)
            run_results = _summarize_run(round_logs, run_elapsed, condition, rep, run_id)
            manifest['runs'].append(run_results)

    manifest['completed_at'] = datetime.now().isoformat()
    manifest['total_elapsed_seconds'] = round(time.time() - t0, 1)
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}\nEXPERIMENT COMPLETE: {name}\n"
          f"Total time: {manifest['total_elapsed_seconds']:.0f}s\n{'='*70}\n")


def run_single(spec_path: str, run_index: int):
    """Run one (condition, rep) by flattened index — SLURM array entrypoint."""
    load_dotenv()
    spec = load_experiment(spec_path)
    project_root, or_config = _load_sweep_config(spec)

    name = spec['name']
    reps = spec['reps']
    base_params = spec['base_params']
    conditions = generate_conditions(spec['sweep'])
    total_runs = len(conditions) * reps

    if run_index < 0 or run_index >= total_runs:
        print(f"ERROR: run_index {run_index} out of range [0, {total_runs})")
        sys.exit(1)

    condition = conditions[run_index // reps]
    rep = (run_index % reps) + 1
    run_id = condition_run_id(condition, rep)

    output_dir = project_root / "data" / "runs" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{run_id}_log.jsonl"
    if log_file.exists():
        print(f"SKIP: {run_id} already exists at {log_file}")
        sys.exit(0)

    print(f"\n{'='*70}\nSINGLE RUN: {name} [{run_index}/{total_runs}]\n"
          f"Condition: {condition_label(condition)}  Rep: {rep}  Run ID: {run_id}\n{'='*70}\n")

    params, cfg = apply_condition(condition, base_params, or_config)
    run_start = time.time()
    state, traces, round_logs, run_metadata = run_simulation(params, cfg, run_id)
    run_elapsed = time.time() - run_start
    save_results(state, traces, round_logs, output_dir, run_id, run_metadata)

    coop_rate = _run_cooperation_rate(round_logs)
    final_gini = gini(round_logs[-1]) if round_logs else None
    print(f"\n--- Result: {run_id} ---")
    print(f"  Gini: {final_gini:.3f}" if final_gini else "  Gini: N/A")
    print(f"  f_C:  {coop_rate:.3f}")
    print(f"  Elapsed: {run_elapsed:.0f}s")


def run_batch(spec_path: str, start_index: int, count: int):
    """Run several consecutive (condition, rep) pairs in one job."""
    for i in range(count):
        try:
            run_single(spec_path, start_index + i)
        except SystemExit:
            continue
