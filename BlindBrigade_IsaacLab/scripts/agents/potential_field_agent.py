"""Simple potential-field agent for the ROSbot box-navigation environment.

Drives toward the goal while repelling from obstacles detected via the
height-scan observation.  No learning — purely reactive.  Useful as a
sanity-check that the env loop, observations, and actions work end-to-end.

Usage:
    isaaclab -p scripts/agents/potential_field_agent.py --task BB-rosbot-box-PLAY-v0 --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Potential-field agent for ROSbot box nav.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="BB-rosbot-box-PLAY-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require the sim to be running ──────────────────────────
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import BlindBrigade_tasks.tasks  # noqa: F401

# ── observation layout (concatenated "policy" group) ────────────────────
# pose_command  : 2   [goal_x_body, goal_y_body]
# base_lin_vel  : 3   [vx, vy, vz] body-frame
# base_yaw_rate : 1
# height_scan   : 256 (16x16 grid, 3m×3m, res 0.2m)
# last_action   : 3
OBS_GOAL = slice(0, 2)
OBS_HSCAN = slice(6, 262)
GRID_N = 16


def build_grid_coords(device: torch.device):
    """Pre-compute (16,16) cell positions in body frame [x=fwd, y=left]."""
    # GridPatternCfg ordering='xy': x varies fastest → reshape (row=y, col=x)
    xs = torch.linspace(-1.5, 1.5, GRID_N, device=device)
    ys = torch.linspace(-1.5, 1.5, GRID_N, device=device)
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt(cx ** 2 + cy ** 2)
    return cx, cy, dist


def compute_action(
    obs: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    cell_dist: torch.Tensor,
) -> torch.Tensor:
    B = obs.shape[0]
    device = obs.device

    goal_xy = obs[:, OBS_GOAL]          # (B, 2)
    hscan = obs[:, OBS_HSCAN]           # (B, 256)

    # ── attractive force toward goal ────────────────────────────────────
    goal_dist = torch.norm(goal_xy, dim=1, keepdim=True).clamp(min=0.01)
    goal_dir = goal_xy / goal_dist
    # Ramp up speed with distance, saturate at 1
    speed = goal_dist.clamp(max=1.5)
    f_att = goal_dir * speed             # (B, 2)

    # ── repulsive force from obstacles ──────────────────────────────────
    grid = hscan.reshape(B, GRID_N//2, GRID_N)

    # Anything above ~9 cm (normalised) is an obstacle
    is_obstacle = (grid > 0.03).float()

    # Only care about cells within 1.2 m
    influence = (cell_dist < 1.2).float().unsqueeze(0)   # (1,16,16)
    weight = influence * is_obstacle / (cell_dist.unsqueeze(0) ** 2 + 0.05)

    # Direction: obstacle→robot = −cell_pos / |cell_pos|
    inv_cx = -cx / (cell_dist + 1e-3)    # (16,16)
    inv_cy = -cy / (cell_dist + 1e-3)

    f_rep_x = (weight * inv_cx.unsqueeze(0)).sum(dim=(1, 2))
    f_rep_y = (weight * inv_cy.unsqueeze(0)).sum(dim=(1, 2))
    f_rep = torch.stack([f_rep_x, f_rep_y], dim=1) * 0.05   # (B, 2)

    # ── combine ─────────────────────────────────────────────────────────
    f = f_att + f_rep

    vx = f[:, 0].clamp(-1, 1)
    vy = f[:, 1].clamp(-1, 1)
    heading_err = torch.atan2(f[:, 1], f[:, 0])
    wz = (heading_err * 1.5).clamp(-1, 1)

    # Slow down when not facing the goal (avoid side-sliding into walls)
    alignment = torch.cos(heading_err).clamp(min=0.0)
    vx = vx * (0.3 + 0.7 * alignment)

    return torch.stack([vx, vy, wz], dim=1)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] observation space: {env.observation_space}")
    print(f"[INFO] action space:      {env.action_space}")

    obs_dict, _ = env.reset()
    device = env.unwrapped.device
    cx, cy, cell_dist = build_grid_coords(device)

    while simulation_app.is_running():
        with torch.inference_mode():
            obs = obs_dict["policy"]
            actions = compute_action(obs, cx, cy, cell_dist)
            obs_dict, _, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
