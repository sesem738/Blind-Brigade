"""Reward terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster


def track_goal_reached(env: ManagerBasedRLEnv, goal_dist_threshold: float = 0.05) -> torch.Tensor:
    """
    Increments ``terrain._success_count`` on goal reach; returns zero reward.

    Counts once per goal on transition into ``promote_threshold``, not per step.
    The count is consumed by the curriculum term to decide promotion/demotion.
    Also tracks cumulative time (steps) spent at the goal in
    ``terrain._time_at_goal`` — every step within ``promote_threshold`` counts,
    regardless of whether the robot left and came back.
    Logs ``goal_dist_min/mean``, ``success_count``, and ``time_at_goal`` to
    ``env.extras``.
    """

    terrain = env.scene.terrain
    if not hasattr(terrain, "_success_count"):
        terrain._success_count = torch.zeros(env.num_envs, device=env.device)
        terrain._was_near_goal = torch.zeros(env.num_envs, device=env.device)
        terrain._time_at_goal = torch.zeros(env.num_envs, device=env.device)

    # Check if close to goal
    goal_dist = torch.norm(env.command_manager.get_command("goal_pose")[:, :2], dim=1)
    is_near = (goal_dist < goal_dist_threshold).float()

    # only count the transition from not-near to near (once per goal)
    newly_reached = is_near * (1.0 - terrain._was_near_goal)
    terrain._success_count += newly_reached
    terrain._was_near_goal = is_near

    # accumulate every step spent at goal (not necessarily continuous)
    terrain._time_at_goal[is_near.bool()] += env.step_dt

    # store for logging
    env.extras["log"]["goal_dist_min"]  = goal_dist.min()
    env.extras["log"]["goal_dist_mean"] = goal_dist.mean()
    env.extras["log"]["success_count"]  = terrain._success_count.mean()
    env.extras["log"]["time_at_goal"]   = terrain._time_at_goal.mean()

    return torch.zeros(env.num_envs, device=env.device)


def lateral_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for moving laterally using L1-Kernel.
    https://github.com/leggedrobotics/sru-navigation-sim

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    reward = torch.abs(lateral_velocity)
    return reward


def rot_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for rotating around the z-axis using an L2-Kernel.
    https://github.com/leggedrobotics/sru-navigation-sim
    
    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the rotational velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rot_vel_norm = torch.norm(asset.data.root_ang_vel_b, dim=1)
    return rot_vel_norm


def action_rate_l1(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L1 kernel.
    https://github.com/leggedrobotics/sru-navigation-sim
    """
    return torch.sum(torch.abs(env.action_manager.action - env.action_manager.prev_action), dim=1)


def goal_distance_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize distance to goal. Normalized by terrain tile size."""                                                                             
    command = env.command_manager.get_command("goal_pose")                                                                                        
    return torch.norm(command[:, :2], dim=1)


def reach_goal_xyz(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigmoid: float,
    T_r: float,
    probability: float,
    ) -> torch.Tensor:
    """Reward goal reaching with configurable sigmoid shaping.
    https://github.com/leggedrobotics/sru-navigation-sim

    Args:
        env: The learning environment.
        command_name: Name of the goal command.
        sigmoid: Sigmoid parameter for shaping.
        T_r: Time reward scaling factor.
        probability: Probability of random sampling.

    Returns:
        Dense reward based on distance to goal.
    """
    terrain = env.scene.terrain
    goal_cmd_generator = env.command_manager._terms[command_name]
    command = env.command_manager.get_command("goal_pose")

    t = env.episode_length_buf
    T = env.max_episode_length

    xyz_error = torch.norm(command[:, :2], dim=1)
    reward = 1 / (1 + torch.square(xyz_error / sigmoid)) / T_r

    timeup_mask = t > (T - T_r) # ABS does this, SRU does it a little differently 
    random_mask = torch.rand_like(t.float()) < probability
    timeup_mask = torch.logical_or(timeup_mask, random_mask)

    arrive_mask = terrain._time_at_goal > 0.0
    reward_mask = torch.logical_or(timeup_mask, arrive_mask)

    reward = reward * reward_mask.float()

    return reward


def backward_movement_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Small penalty for backward movement as a regularization term.
    https://github.com/leggedrobotics/sru-navigation-sim

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Penalty [0, +1] based on backward velocity (to be used with negative weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the penalty
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    # Only penalize negative forward velocity (backward movement)
    backward_velocity = torch.clamp(-forward_velocity, min=0.0, max=1.0)
    return backward_velocity


def goal_reached_bonus(env: ManagerBasedRLEnv, threshold: float = 0.5) -> torch.Tensor:
    """Smooth bonus that ramps linearly from 0 at *threshold* to 1.0 at the goal.

    Provides a strong, localised incentive for the final approach without
    discontinuous jumps that destabilise value-function learning.
    """
    command = env.command_manager.get_command("goal_pose")
    distance = torch.norm(command[:, :2], dim=1)
    return torch.clamp(1.0 - distance / threshold, min=0.0)


def obstacle_approach_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster_cam"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    danger_radius: float = 0.4,
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

    # Rays that fall into pits or miss return inf/nan — mark them invalid
    valid = torch.isfinite(ray_hits).all(dim=-1)  # (B, N)

    # world-frame xy velocity
    vel_w = asset.data.root_lin_vel_w[:, :2]      # (B, 2)

    # vector and distance from robot center to each ray hit (horizontal only)
    robot_xy   = sensor_pos[:, :2]                                        # (B, 2)
    horiz_vec  = ray_hits[:, :, :2] - robot_xy.unsqueeze(1)              # (B, N, 2)
    horiz_dist = torch.norm(horiz_vec, dim=-1)                           # (B, N)

    # normalized ray length — small value indicates a tall obstacle
    raw_dist  = torch.norm(ray_hits - sensor_pos.unsqueeze(1), dim=-1)  # (B, N)
    raw_dist  = torch.nan_to_num(raw_dist, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance, neginf=-sensor.cfg.max_distance)
    norm_dist = raw_dist / (sensor.cfg.max_distance + 0.01)

    # obstacle severity: 0 when clear, approaching 1 for max obstacle height
    obstacle_weight = torch.clamp(safe_dist_normalized - norm_dist, min=0.0) / safe_dist_normalized

    # unit vector pointing from robot toward each ray hit
    # inf/inf produces NaN — replace with zero (these rays are masked out below)
    hit_dir    = horiz_vec / horiz_dist.clamp(min=1e-6).unsqueeze(-1)   # (B, N, 2)
    hit_dir    = torch.nan_to_num(hit_dir, nan=0.0)

    # scalar velocity component in the direction of each obstacle (positive = approaching)
    vel_toward = (vel_w.unsqueeze(1) * hit_dir).sum(dim=-1)             # (B, N)
    approach   = torch.clamp(vel_toward, min=0.0)

    in_zone = (horiz_dist < danger_radius) & valid
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
