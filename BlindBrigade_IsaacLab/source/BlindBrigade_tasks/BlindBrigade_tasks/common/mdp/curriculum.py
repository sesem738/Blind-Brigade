"""Curriculum terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_nav_success_based(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_successes: int = 3,
    ):
    """Curriculum that adjusts terrain difficulty based on goal-reaching success.

    Promotes an environment to a harder terrain level when the agent reaches at
    least ``min_successes`` goals in an episode without a collision termination.
    Demotes on collision death or zero goals reached. Success counts persist across
    timeouts and reset only on promotion or collision.

    Requires ``terrain._success_count`` to be incremented externally (e.g. in a
    reward or event term) whenever the agent reaches a goal.

    Returns the mean terrain level across all environments.
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
