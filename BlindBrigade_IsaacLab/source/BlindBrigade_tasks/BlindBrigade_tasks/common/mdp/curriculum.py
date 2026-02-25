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


def goal_ramp_resample_time_by_iteration(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "goal_pose",
    min_resample_time_s: float = 5.0,
    max_resample_time_s: float = 20.0,
    schedule_iterations: int = 1000,
    num_steps_per_env: int = 64,
    ):
    """Shrinks the goal resampling time range linearly over training iterations.

    At iteration 0 the resampling time equals ``max_resample_time_s`` (one goal per
    episode). It decays linearly to ``min_time_s`` at ``schedule_iterations``,
    then stays there.

    Args:
        command_name:        Name of the command term in the command manager.
        min_resample_time_s: Minimum resampling time at the end of annealing.
        max_resample_time_s: Starting resampling time — should match episode length.
        schedule_iterations: Number of training iterations to reach ``min_time_s``.
        num_steps_per_env:   Steps per env per iteration (from runner config).

    Returns:
        The current progress (%) for logging.
    """
    current_iteration = env.common_step_counter // num_steps_per_env
    progress = min(current_iteration / max(schedule_iterations, 1), 1.0)
    t = max_resample_time_s - progress * (max_resample_time_s - min_resample_time_s)
    
    term = env.command_manager.get_term(command_name)
    term.cfg.resampling_time_range = (t, max_resample_time_s)

    return progress


def goal_step_resample_time_at_iteration(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "goal_pose",
    schedule_iterations: int = 150,
    num_steps_per_env: int = 64,
    min_resample_time_s: float = 5.0,
    max_resample_time_s: float = 20.0,
    ):
    """Sets the goal resampling time to a fixed value."""
    term = env.command_manager.get_term(command_name)

    current_iteration = env.common_step_counter // num_steps_per_env
    condition = current_iteration >= schedule_iterations

    if condition:
        term.cfg.resampling_time_range = (min_resample_time_s, max_resample_time_s)
    
    return condition
