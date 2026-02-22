from __future__ import annotations                                                                                    
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

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


def ray_caster_image(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster_cam"),
    grid_h: int = 64,
    grid_w: int = 64,
) -> torch.Tensor:
    """Returns a (B, 1, H, W) depth image from a GridPattern RayCaster.

    Requires GridPatternCfg with ordering='xy' (default) and
    size/resolution configured so total rays == grid_h * grid_w.
    """
    sensor = env.scene[asset_cfg.name]
    distances = torch.norm(
        sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1
    )
    distances = torch.nan_to_num(distances, nan=sensor.cfg.max_distance)
    distances = distances / sensor.cfg.max_distance  # normalize [0, 1]
    return distances.reshape(env.num_envs, 1, grid_h, grid_w)


def camera_billinear_interpolation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("guide_zed2_camera"),
    max_depth: float = 20.0,
    output_shape: tuple = (64, 36),
) -> torch.Tensor:
    """Normalized, downsampled depth image from the ZED2 pinhole camera for MLP policies.

    Depth is clamped to [0, max_depth], normalized to [0, 1], bilinearly downsampled
    to output_shape (W, H), then flattened. Output size: output_shape[0] * output_shape[1].
    """
    sensor = env.scene[asset_cfg.name]
    # shape: (num_envs, H, W, 1)
    depth = sensor.data.output["distance_to_image_plane"].float()
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth)
    depth = depth.clamp(0.0, max_depth) / max_depth
    # F.interpolate expects (N, C, H, W)
    depth = depth.permute(0, 3, 1, 2)
    depth = F.interpolate(depth, size=output_shape, mode="bilinear", align_corners=False)
    return depth.reshape(env.num_envs, -1)


def camera_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    flatten: bool = False,
    nan_fill_value: float | None = None,
) -> torch.Tensor:
    """Camera image Observations.

    The camera image observation from the given sensor w.r.t. the asset's root frame.
    Also removes nan/inf values and sets them to the maximum distance of the sensor

    Args:
        env: The environment object.
        sensor_cfg: The name of the sensor.
        data_type: The type of data to extract from the sensor. Default is "distance_to_image_plane".
        flatten: If True, the image will be flattened to 1D. Default is False.
        nan_fill_value: The value to fill nan/inf values with. If None, the maximum distance of the sensor will be used.

    Returns:
        The image data."""
    # extract the used quantities (to enable type-hinting)
    sensor: Camera | RayCasterCamera | TiledCamera = env.scene.sensors[sensor_cfg.name]

    img = sensor.data.output[data_type].clone()

    if data_type == "distance_to_image_plane":
        if nan_fill_value is None:
            nan_fill_value = (
                sensor.cfg.max_distance if isinstance(sensor, RayCasterCamera) else sensor.cfg.spawn.clipping_range[1]
            )
        img = torch.nan_to_num(img, nan=nan_fill_value, posinf=nan_fill_value, neginf=0.0)

    # if type torch.uint8, convert to float and scale between 0 and 1
    if img.dtype == torch.uint8:
        img = img.to(torch.float32) / 255.0

    if flatten:
        return img.flatten(start_dim=1)
    else:
        # reorder the image to [BS, C, H, W] if it is not already in that shape
        if img.shape[-1] == 1 or img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)

        return img
