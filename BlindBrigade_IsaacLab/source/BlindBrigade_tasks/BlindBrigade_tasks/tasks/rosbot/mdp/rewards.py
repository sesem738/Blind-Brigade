"""Reward terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation

def track_goal_reached(env: ManagerBasedRLEnv, promote_threshold: float = 0.05):
    """Call this every step to check if robot is close to goal."""
    terrain = env.scene.terrain
    if not hasattr(terrain, "_success_count"):
        terrain._success_count = torch.zeros(env.num_envs, device=env.device)

    goal_dist = torch.norm(env.command_manager.get_command("goal_pose")[:, :2], dim=1)
    terrain._success_count += (goal_dist < promote_threshold).float()

def lateral_velocity_penalty_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_vy: float = 1.0
    ) -> torch.Tensor:
    """Penalize lateral (vy) body velocity."""
    asset: Articulation = env.scene[asset_cfg.name]  
    return torch.square(torch.abs(asset.data.root_lin_vel_b[:, 1]) / max_vy)
