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

def ray_caster_depth(env: ManagerBasedRLEnv) -> torch.Tensor:
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
    cam = env.scene["ray_caster_cam"]
    depth = cam.data.output["distance_to_image_plane"]
    depth = torch.nan_to_num(depth, nan=cam.cfg.max_distance) / cam.cfg.max_distance
    return depth.reshape(env.num_envs, -1)
