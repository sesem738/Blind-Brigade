from __future__ import annotations                                                                                    
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:   
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
import matplotlib.pyplot as plt  

def base_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor: 
    """
    Returns yaw rate in body frame
    
    :param env: Description
    :type env: ManagerBasedRLEnv
    :return: Description
    :rtype: Tensor
    """                                             
    asset: Articulation = env.scene[asset_cfg.name]              
    return asset.data.root_ang_vel_b[:, 2:3] 


def ray_caster_depth(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster_cam")) -> torch.Tensor:
    """Alternative:
    ObsTerm(
        func=mdp.image, clip=(0.0,1.0), 
        params={
            "sensor_cfg": SceneEntityCfg("ray_caster_cam"),
            "data_type":"distance_to_image_plane",
            "normalize":False
        }
    )
    * Needs to be flattened if using MLP instead of CNN
    """
    asset: Articulation = env.scene[asset_cfg.name] 
    depth = asset.data.output["distance_to_image_plane"]
    depth = torch.nan_to_num(depth, nan=asset.cfg.max_distance) / asset.cfg.max_distance
    return depth.reshape(env.num_envs, -1)

def ray_caster_depth_cropped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster_cam")) -> torch.Tensor:
      asset = env.scene[asset_cfg.name]
      depth = asset.data.output["distance_to_image_plane"]
      depth = torch.nan_to_num(depth, nan=asset.cfg.max_distance) / asset.cfg.max_distance
      # depth shape: (num_envs, H, W, 1) — keep bottom half
      h = depth.shape[1]
      depth = depth[:, h // 2:, :, :]  # bottom half only
      return depth.reshape(env.num_envs, -1)


def blind_goal_relative_to_guide(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    guide_cfg: SceneEntityCfg = SceneEntityCfg("guide"),
) -> torch.Tensor:
    """Returns the blind robot's goal position expressed in the guide's body frame.

    The command is generated for the blind (asset_name="blind" in CommandCfg),
    but transformed into the guide's local frame so the guide policy can navigate toward it.

    Returns:
        Tensor of shape (num_envs, 3) — [dx, dy, dz] in guide's body frame.
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    # goal in world frame (set by the command targeting the blind)
    command = env.command_manager.get_term(command_name)
    goal_pos_w = command.pos_command_w  # (num_envs, 3)

    # guide's root state
    guide: Articulation = env.scene[guide_cfg.name]
    guide_pos_w = guide.data.root_pos_w[:, :3]
    guide_quat_w = guide.data.root_quat_w

    # transform goal into guide's body frame
    delta_w = goal_pos_w - guide_pos_w
    delta_b = quat_apply_inverse(yaw_quat(guide_quat_w), delta_w)

    return delta_b


def ray_caster_lidar(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene[asset_cfg.name]
    distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    distances = torch.nan_to_num(distances, nan=sensor.cfg.max_distance) / sensor.cfg.max_distance
    return distances.reshape(env.num_envs, -1)
