"""Reward terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation

def track_goal_reached(env: ManagerBasedRLEnv, promote_threshold: float = 0.05) -> torch.Tensor:
    """Call this every step to check if robot is close to goal.

    Counts once per goal reached (transition into threshold zone), not per step.
    Resets _was_near_goal when a new goal is sampled (command resamples).
    """

    terrain = env.scene.terrain
    if not hasattr(terrain, "_success_count"):
        terrain._success_count = torch.zeros(env.num_envs, device=env.device)
        terrain._was_near_goal = torch.zeros(env.num_envs, device=env.device)

    goal_dist = torch.norm(env.command_manager.get_command("goal_pose")[:, :2], dim=1)
    is_near = (goal_dist < promote_threshold).float()

    # only count the transition from not-near to near (once per goal)
    newly_reached = is_near * (1.0 - terrain._was_near_goal)
    terrain._success_count += newly_reached
    terrain._was_near_goal = is_near

    # store for logging
    env.extras["log"]["goal_dist_min"]  = goal_dist.min()
    env.extras["log"]["goal_dist_mean"] = goal_dist.mean()
    env.extras["log"]["success_count"]  = terrain._success_count.mean()

    return torch.zeros(env.num_envs, device=env.device)

def lateral_velocity_penalty_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_vy: float = 1.0
    ) -> torch.Tensor:
    """Penalize lateral (vy) body velocity."""

    asset: Articulation = env.scene[asset_cfg.name]  
    return torch.square(torch.abs(asset.data.root_lin_vel_b[:, 1]) / max_vy)

def blind_spot_velocity_penalty(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_vx: float = 1.0,
        max_vy: float = 1.0
    ) -> torch.Tensor:
      """Penalize moving fast in directions the robot can't see."""

      asset: Articulation = env.scene[asset_cfg.name] 
      vel_b = asset.data.root_lin_vel_b[:, :2]  # (vx, vy) in body frame

      # heading of velocity vector in body frame
      vx = vel_b[:, 0]
      vy = vel_b[:, 1]

      # penalize backward movement
      backward_penalty = torch.clamp(-vx, min=0.0) / max_vx  # only when vx < 0

      # penalize lateral movement (scaled by speed)
      lateral_penalty = torch.abs(vy) / max_vy

      # combine
      return backward_penalty + lateral_penalty

def goal_distance_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:                                                                                
    """Penalize distance to goal. Normalized by terrain tile size."""                                                                             
    command = env.command_manager.get_command("goal_pose")                                                                                        
    return torch.norm(command[:, :2], dim=1)


def heading_velocity_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_speed: float = 0.05,
) -> torch.Tensor:
    """Penalize misalignment between heading and velocity direction.

    In body frame, perfect alignment means all velocity is along +x.
    The penalty is |atan2(vy, vx)| / pi, normalized to [0, 1].
    Only applied when the robot is actually moving (above min_speed).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :2]
    vx = vel_b[:, 0]
    vy = vel_b[:, 1]

    speed = torch.norm(vel_b, dim=1)
    angle_error = torch.abs(torch.atan2(vy, vx))  # 0 when moving forward, pi when backward

    # only penalize when actually moving
    moving = (speed > min_speed).float()
    return moving * (angle_error / torch.pi)
