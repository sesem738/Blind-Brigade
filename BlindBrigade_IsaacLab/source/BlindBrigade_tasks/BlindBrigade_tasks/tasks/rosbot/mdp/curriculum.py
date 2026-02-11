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

def terrain_levels_nav_success_based(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor, 
    ):                                                                                 
    terrain = env.scene.terrain
                                                                                                                    
    # on first call, initialize buffers
    if not hasattr(terrain, "_success_count"):
        terrain._success_count = torch.zeros(env.num_envs, device=env.device)

    # promote/demote based on accumulated successes this episode
    move_up = terrain._success_count[env_ids] >= 1  # reached near a goal
    move_down = terrain._success_count[env_ids] == 0  # reached nothing
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    # reset counter for envs that just terminated
    terrain._success_count[env_ids] = 0

    return torch.mean(terrain.terrain_levels.float())
