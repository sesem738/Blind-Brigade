# Blind Brigade

RL-based obstacle navigation for ROSbot in Isaac Lab.

## Prerequisites

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (Isaac Sim 4.5+)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 3.3.0
- Python >= 3.10

## Installation

With a Python environment that has Isaac Lab installed:

```bash
pip install -e source/BlindBrigade_tasks
pip install -e source/BlindBrigade_assets
```

## Environments

| Task ID | Description |
|---|---|
| `BB-rosbot-flat-v0` | Flat terrain point-to-point |
| `BB-rosbot-box-v0` | Box obstacle navigation |
| `BB-rosbot-maze-v0` | Procedural maze navigation |
| `BB-rosbot-coop-flat-v0` | 2-agent cooperative (flat) |
| `BB-rosbot-coop-box-v0` | 2-agent cooperative (box obstacles) |
| `BB-rosbot-diff-flat-v0` | Differential drive, flat terrain (WIP — not fully working) |
| `BB-rosbot-diff-box-v0` | Differential drive, box obstacles (WIP — not fully working) |
| `BB-mushr-flat-v0` | MuSHR (Ackermann steering), flat terrain (working, untested in training) |
| `BB-mushr-box-v0` | MuSHR (Ackermann steering), box obstacles (working, untested in training) |

Each task has a `-PLAY-v0` variant (e.g. `BB-rosbot-box-PLAY-v0`) for visualization with a single small environment.

## Quick Start

```bash
# Train
python scripts/rsl_rl/train.py --task BB-rosbot-box-v0 --num_envs 4096 --headless

# Play back a trained policy
python scripts/rsl_rl/play.py --task BB-rosbot-box-PLAY-v0 --num_envs 8

# Resume training from a checkpoint
python scripts/rsl_rl/train.py --task BB-rosbot-box-v0 --resume --load_run <timestamp>

# Teleop
python scripts/agents/teleop_agent.py --task BB-rosbot-box-PLAY-v0
```

Common flags: `--seed`, `--max_iterations`, `--video`, `--checkpoint`, `--logger wandb`

## Scripts

| Directory | Contents |
|---|---|
| `scripts/rsl_rl/` | `train.py`, `play.py`, CNN feature / Grad-CAM visualization, occupancy grid export |
| `scripts/agents/` | Random, zero, teleop, and potential-field controllers |
| `scripts/viz/` | Reward function visualization (heading alignment, SRU rewards) |
| `scripts/debug/` | Step response, lateral dynamics test, USD prim dump |

## Policy Architectures

Available agent configs for `BB-rosbot-box-v0` (selected via `--agent_id`):

| Architecture | Entry Point Key | Description |
|---|---|---|
| MLP | `rsl_rl_cfg_entry_point` | Default feedforward actor-critic |
| CNN + MLP | `rsl_rl_cnn_cfg_entry_point` | Height-map CNN encoder + MLP head |
| GRU + Raycaster | `rsl_rl_gru_cfg_entry_point` | 1-D height-scan encoder with GRU memory |
| LSTM + CNN | `rsl_rl_recurrent_cnn_cfg_entry_point` | Per-step CNN encoder with LSTM memory |
| Student-Teacher (MLP) | `rsl_rl_distil_cfg_entry_point` | MLP student distilled from MLP teacher |
| Student-Teacher (CNN) | `rsl_rl_distil_cnn_cfg_entry_point` | CNN student distilled from MLP teacher |
| Student-Teacher (MLP enc.) | `rsl_rl_distil_mlp_cfg_entry_point` | Flat-image MLP student from MLP teacher |

Custom network modules live in `source/BlindBrigade_tasks/BlindBrigade_tasks/modules/`.

## Project Structure

```
BlindBrigade_IsaacLab/
├── scripts/
│   ├── rsl_rl/          # Training & playback
│   ├── agents/          # Non-learned controllers
│   ├── viz/             # Reward visualization
│   └── debug/           # Dynamics testing & introspection
├── source/
│   ├── BlindBrigade_tasks/
│   │   └── BlindBrigade_tasks/
│   │       ├── tasks/
│   │       │   ├── rosbot/              # Single-agent tasks (flat, box)
│   │       │   ├── rosbot_differential/ # Differential drive tasks (WIP)
│   │       │   ├── mushr/              # MuSHR Ackermann steering tasks (untested)
│   │       │   ├── rosbot_maze/         # Procedural maze task
│   │       │   └── rosbot_cooperative/  # Multi-agent tasks
│   │       ├── common/mdp/              # Shared observations, rewards, actions, commands
│   │       └── modules/                 # Custom policy networks
│   └── BlindBrigade_assets/
│       └── BlindBrigade_assets/
│           └── robot/                   # Robot asset configs
└── logs/rsl_rl/                         # Training outputs
```

## VS Code Extension: Isaac Lab Runner

A sidebar extension for launching train/play sessions without typing CLI commands. Provides dropdowns for task, agent config, experiment, run, and checkpoint selection — all auto-detected from the codebase.

```bash
bash tools/isaaclab-runner/install.sh
# then restart VS Code
```

See [`tools/isaaclab-runner/README.md`](tools/isaaclab-runner/README.md) for details.

## Troubleshooting

### Pylance Missing Indexing

If VS Code / Pylance cannot resolve imports, add the extension paths to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "source/BlindBrigade_tasks",
        "source/BlindBrigade_assets"
    ]
}
```
