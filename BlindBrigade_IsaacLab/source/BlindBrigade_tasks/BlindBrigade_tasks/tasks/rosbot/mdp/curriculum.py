"""Curriculum terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_nav(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor, 
        promote_threshold: float = 0.1, 
        demote_threshold: float = 0.2
    ):
    terrain = env.scene.terrain

    # distance to goal at episode end
    goal_dist = torch.norm(env.command_manager.get_command("goal_pose")[env_ids, :2], dim=1)

    # promote: got close enough to goal
    move_up = goal_dist < promote_threshold
    # demote: stayed far from goal
    move_down = goal_dist > demote_threshold
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
