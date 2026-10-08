"""
Thin CLI dispatcher.

Single-run mode:
    python src/main.py
    python src/main.py --game config/game_params.yaml --api config/openrouter_config.yaml
    python src/main.py --resume path/to/checkpoint.json
    python src/main.py --resume path/to/run_log.jsonl

Sweep mode (uses experiment.py):
    python src/main.py --sweep path/to/sweep.yaml
    python src/main.py --sweep ... --run-index $SLURM_ARRAY_TASK_ID
    python src/main.py --sweep ... --run-index N --batch-size K
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from runner import run_simulation, save_results, load_config
from experiment import run_sweep, run_single, run_batch


def main():
    parser = argparse.ArgumentParser(description="Run a simulation or parameter sweep")
    parser.add_argument('--game', type=str, default=None,
                        help='Game params YAML (default: config/game_params.yaml)')
    parser.add_argument('--api', type=str, default=None,
                        help='API config YAML (default: config/openrouter_config.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data/runs)')
    parser.add_argument('--sweep', type=str, default=None,
                        help='Experiment YAML for parameter sweep')
    parser.add_argument('--run-index', type=int, default=None,
                        help='Flat (condition × rep) index — use with --sweep for SLURM array jobs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Consecutive runs per job when --run-index is set')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint JSON or a _log.jsonl file')
    parser.add_argument('--label', type=str, default=None,
                        help='Cell/condition label — creates data/runs/{label}/ subdirectory')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Explicit run id (filename stem); overrides the timestamp default. Used for SLURM batch run labelling.')
    args = parser.parse_args()

    if args.sweep:
        if args.run_index is not None:
            if args.batch_size > 1:
                run_batch(args.sweep, args.run_index, args.batch_size)
            else:
                run_single(args.sweep, args.run_index)
        else:
            run_sweep(args.sweep)
        return

    # Single run
    load_dotenv()
    project_root = Path(__file__).parent.parent
    game_path = Path(args.game) if args.game else project_root / "config" / "game_params.yaml"
    api_path = Path(args.api) if args.api else project_root / "config" / "openrouter_config.yaml"
    game_params = load_config(game_path)
    openrouter_config = load_config(api_path)

    # When resuming, keep the original run_id so logs stay in one file.
    # Pattern: data/runs/<run_id>_log.jsonl  or  <run_id>_reasoning_live.jsonl
    run_id = args.run_id
    if run_id is None and args.resume:
        stem = Path(args.resume).stem  # e.g. "20260421_221222_log" or "..._reasoning_live"
        for suffix in ('_reasoning_live', '_log', '_checkpoint'):
            if stem.endswith(suffix):
                run_id = stem[: -len(suffix)]
                break
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir_override = None
    if args.label:
        log_dir_override = project_root / "data" / "runs" / args.label

    state, traces, round_logs, run_metadata = run_simulation(
        game_params, openrouter_config, run_id, resume_path=args.resume,
        log_dir_override=log_dir_override,
    )

    if args.output:
        output_dir = Path(args.output)
    elif log_dir_override:
        output_dir = log_dir_override
    else:
        output_dir = project_root / "data" / "runs"
    save_results(state, traces, round_logs, output_dir, run_id, run_metadata)


if __name__ == "__main__":
    main()
