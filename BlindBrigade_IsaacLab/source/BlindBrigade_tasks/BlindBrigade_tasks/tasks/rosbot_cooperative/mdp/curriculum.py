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
    min_successes: int = 3,
    ):
    """Curriculum based on goal-reaching success count.

    Promote: reached enough goals AND survived (no collision death).
    Demote: collision death OR reached no goals.
    """
    terrain = env.scene.terrain

    # on first call, initialize buffers
    if not hasattr(terrain, "_success_count"):
        terrain._success_count = torch.zeros(env.num_envs, device=env.device)
        terrain._was_near_goal = torch.zeros(env.num_envs, device=env.device)

    # check if terminated by collision (not timeout)
    collision_death = (
        env.termination_manager.terminated[env_ids]
        & ~env.termination_manager.time_outs[env_ids]
    )

    # promote: reached enough goals AND did not die from collision
    move_up = (terrain._success_count[env_ids] >= min_successes) & ~collision_death
    # demote: collision death OR reached nothing
    move_down = collision_death | (terrain._success_count[env_ids] == 0)
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    # reset counters only on collision or promotion — keep counting across timeouts
    reset_mask = collision_death | move_up
    reset_ids = env_ids[reset_mask]
    terrain._success_count[reset_ids] = 0
    terrain._was_near_goal[reset_ids] = 0

    return torch.mean(terrain.terrain_levels.float())
