"""Reward terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster

def track_goal_reached(env: ManagerBasedRLEnv, promote_threshold: float = 0.05) -> torch.Tensor:
    """
    Increments ``terrain._success_count`` on goal reach; returns zero reward.

    Counts once per goal on transition into ``promote_threshold``, not per step.
    The count is consumed by the curriculum term to decide promotion/demotion.
    Also logs ``goal_dist_min/mean`` and ``success_count`` to ``env.extras``.
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


def blind_spot_velocity_penalty(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_vx: float = 1.0,
        max_vy: float = 1.0
    ) -> torch.Tensor:
      """
      Penalizes backward and lateral movement, normalized by max velocities.

      Assumes a forward-facing sensor — backward (vx < 0) and lateral (vy) motion
      move the robot into unseen space. Returns ``clamp(-vx, 0)/max_vx + |vy|/max_vy``.
      """

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


def obstacle_approach_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster_cam"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    danger_radius: float = 0.7,
    safe_dist_normalized: float = 0.85,
) -> torch.Tensor:
    """Penalize velocity components directed toward detected obstacles.

    Unlike a static proximity penalty, this formulation allows the robot to
    squeeze through narrow gaps: obstacles at constant clearance on both sides
    (i.e. a corridor) contribute zero gradient. Only movement that actively
    closes the gap to a detected obstacle is penalized.

    The contribution per ray is:
        clamp(vel_toward_obstacle, 0) * obstacle_severity
    summed across all rays within ``danger_radius`` of the robot center.

    Obstacle severity scales linearly from 0 (flat ground) to 1 (max obstacle
    height), so taller / closer obstacles generate stronger gradients.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    ray_hits   = sensor.data.ray_hits_w           # (B, N, 3)
    sensor_pos = sensor.data.pos_w                # (B, 3)

    # world-frame xy velocity
    vel_w = asset.data.root_lin_vel_w[:, :2]      # (B, 2)

    # vector and distance from robot center to each ray hit (horizontal only)
    robot_xy   = sensor_pos[:, :2]                                        # (B, 2)
    horiz_vec  = ray_hits[:, :, :2] - robot_xy.unsqueeze(1)              # (B, N, 2)
    horiz_dist = torch.norm(horiz_vec, dim=-1)                           # (B, N)

    # normalized ray length — small value indicates a tall obstacle
    raw_dist  = torch.norm(ray_hits - sensor_pos.unsqueeze(1), dim=-1)  # (B, N)
    raw_dist  = torch.nan_to_num(raw_dist, nan=sensor.cfg.max_distance)
    norm_dist = raw_dist / (sensor.cfg.max_distance + 0.01)

    # obstacle severity: 0 when clear, approaching 1 for max obstacle height
    obstacle_weight = torch.clamp(safe_dist_normalized - norm_dist, min=0.0) / safe_dist_normalized

    # unit vector pointing from robot toward each ray hit
    hit_dir    = horiz_vec / horiz_dist.clamp(min=1e-6).unsqueeze(-1)   # (B, N, 2)

    # scalar velocity component in the direction of each obstacle (positive = approaching)
    vel_toward = (vel_w.unsqueeze(1) * hit_dir).sum(dim=-1)             # (B, N)
    approach   = torch.clamp(vel_toward, min=0.0)

    in_zone = (horiz_dist < danger_radius)
    return (approach * obstacle_weight * in_zone.float()).sum(dim=1)


def heading_velocity_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_speed: float = 0.5,
) -> torch.Tensor:
    """Penalize misalignment between heading and velocity direction.

    In body frame, perfect alignment means all velocity is along +x.
    The penalty is |atan2(vy, vx)| / pi, normalized to [0, 1], weighted
    continuously by speed so that slow strafing (fine positioning) is nearly
    free while fast travel requires the robot to face its direction of motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :2]
    vx = vel_b[:, 0]
    vy = vel_b[:, 1]

    speed = torch.norm(vel_b, dim=1)
    angle_error = torch.abs(torch.atan2(vy, vx))  # 0 when moving forward, pi when backward

    # scale penalty linearly with speed — strafing at low speed is cheap,
    # heading misalignment at high speed is fully penalized
    speed_weight = torch.clamp(speed / max_speed, 0.0, 1.0)
    return speed_weight * (angle_error / torch.pi)
