from __future__ import annotations                                                                                    
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:   
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


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


def ray_caster_lidar(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene[asset_cfg.name]
    distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    distances = torch.nan_to_num(distances, nan=sensor.cfg.max_distance) / sensor.cfg.max_distance
    return distances.reshape(env.num_envs, -1)
