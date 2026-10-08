# From Neighbours to Coalitions

Code for the AAMAS 2027 submission *From Neighbours to Coalitions: What Shapes Social
Structure in Language-Agent Societies?*

Thirty language agents play a repeated resource game. The design varies the action set in
four nested levels (giving, predation, rewiring, harvesting from a shared stock) and the
price of giving, and compares a second model and runs without a message channel. This
repository holds the simulation engine, the measures and the scripts that produce every
table, figure and reported number in the paper and its supplement.

## Layout

```
src/                 simulation engine: agents, prompts, game, runner
config/              run configurations, one per cell (prod_L{1-4}_{scar,knife,abund}, ...)
scripts/             the fourteen measures (batch_suite.py, deontic.py, enforcement.py),
                     frozen before the first production run
measures/
  core/, _shared/    primitives over run logs; run-set loader
  plots/             shared plotting style and the reasoning-trace projection
  paper_aamas/       one script per analysis in the paper (see below), with its JSON output
tests/               engine unit tests
handlabels/          hand-coded label sets used for validation
repo-audit.sh        hygiene checks (no data, no secrets, no local paths)
```

## From script to paper

Scripts run from `measures/`, e.g. `uv run --python 3.12 python paper_aamas/arms_levels.py`;
each script states its dependencies in its docstring. Tables and figures are written to
`measures/paper_aamas/out/`, or to the folder given in `AAMAS_PAPER_DIR`.

| Paper | Script(s) |
|---|---|
| Table 1 (parameters) | `config/` |
| Table 2 (measures), Table 3 (measures per level), Fig. 2 | `scripts/batch_suite.py`, `paper_aamas/prereg_fingerprint.py`, `main_figure.py`, `main_figure_col.py`, `supplementary_tables.py` |
| Table 4 (society per level), Table 5 (arms) | `paper_aamas/arms_levels.py`, `tables_main.py` |
| Fig. 1 (reasoning traces) | `paper_aamas/teaser_traces.py`, `trace_masking.py` |
| Permutation test, effect sizes | `paper_aamas/perm_test.py`, `prereg_fingerprint.py` |
| Attack arithmetic, joint attacks | `paper_aamas/beyond_arithmetic.py`, `beyond_exact.py`, `arms_levels.py` |
| Rewiring sequence | `paper_aamas/rewire_sequence.py` |
| Null model, mob per neighbour, richest targets | `paper_aamas/modularity_null.py`, `mob_per_reach.py`, `reachable_baseline.py` |
| Action set against model, forms under the second model | `paper_aamas/model_vs_action.py`, `model_forms.py` |
| Channel use, validity, welfare | `paper_aamas/channel_use.py`, `validity_welfare.py` |
| Supplement Table S1, Fig. S1, Fig. S2 | `paper_aamas/ablations.py`, `supplementary_tables.py`, `main_figure_col.py`, `ablation_figure.py` |

## Data

The per-turn run logs (about 33 GB of JSONL, one folder per run) are not in this
repository; the JSON files next to each script hold the aggregates the paper reports. The
logs will be released upon acceptance. To regenerate from logs, place them under
`data/thesis_final/` at the repository root.

## Running a simulation

```bash
pip install -r requirements.txt
# provide a model key in the environment (OPENROUTER_API_KEY),
# or point --api at a local vLLM server (config/vllm_config.yaml).
python3 src/main.py --game config/game_params.yaml --api config/vllm_config.yaml
```

The main-grid cells are `config/prod_L{1-4}_{scar,knife,abund}.yaml`, the no-channel controls `config/prod_L{1-4}_knife_nocomm.yaml`, and the model backends `config/api_gemma.yaml` and `config/api_qwen.yaml` (the second-model arm uses the knife-edge game files with `api_qwen.yaml`). The full configuration
of every run, including the second-model and no-channel arms, is recorded in the first line of
its log (model, temperature, token limit, seed, initial graph and all game parameters).

## License

MIT (see `LICENSE`).
