from __future__ import annotations                                                                                    
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera
from isaaclab.envs.mdp import height_scan

if TYPE_CHECKING:   
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster


def base_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Body-frame yaw rate of the robot root. Returns shape (B, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]              
    return asset.data.root_ang_vel_b[:, 2:3] 


def height_scan_normalized(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Height scan from a RayCaster, normalized to [-1, 1] by ``max_distance`` and flattened to (B, N).
    """
    distances = height_scan(env=env, sensor_cfg=sensor_cfg, offset=0.0)
    distances = torch.nan_to_num(
        input=distances, 
        nan=-env.scene.sensors[sensor_cfg.name].cfg.max_distance
    )
    distances = torch.clamp(
        input=distances, 
        min=-(env.scene.sensors[sensor_cfg.name].cfg.max_distance),
        max= (env.scene.sensors[sensor_cfg.name].cfg.max_distance)
    ) / (env.scene.sensors[sensor_cfg.name].cfg.max_distance + 1e-6)
    
    return distances


def ray_caster_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster_cam"),
    grid_h: int = 64,
    grid_w: int = 64,
) -> torch.Tensor:
    """Depth image from a GridPattern RayCaster, normalized to [0, 1]. Returns shape (B, 1, H, W).

    A RayCaster outputs flat ray hits with no 2D structure, so ``grid_h`` and ``grid_w``
    must be provided explicitly and satisfy ``grid_h * grid_w == total rays``.
    Requires ``GridPatternCfg`` with ``ordering='xy'`` (default).
    """

    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    distances = torch.nan_to_num(distances, nan=sensor.cfg.max_distance)
    distances = distances / (sensor.cfg.max_distance + 0.01)  # normalize [0, 1]
    return distances.reshape(env.num_envs, 1, grid_h, grid_w)


def camera_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    flatten: bool = False,
    nan_fill_value: float | None = None,
) -> torch.Tensor:
    """Image observation from a Camera, RayCasterCamera, or TiledCamera sensor.

    Retrieves ``data_type`` from the sensor output. For depth images, NaN/inf values
    are replaced with the sensor's clipping range max (or ``nan_fill_value`` if provided),
    and uint8 images are scaled to [0, 1]. Returns shape (B, C, H, W), or (B, N) if
    ``flatten=True``.
    """
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

    # Safeguard from inf's
    img[img == float("inf")] = 0

    if flatten:
        return img.flatten(start_dim=1)
    else:
        # reorder the image to [BS, C, H, W] if it is not already in that shape
        if img.shape[-1] == 1 or img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)

        return img
