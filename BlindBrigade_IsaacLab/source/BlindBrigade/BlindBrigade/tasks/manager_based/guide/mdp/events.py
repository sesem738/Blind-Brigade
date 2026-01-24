from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def reset_root_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    blind_asset_cfg: SceneEntityCfg = SceneEntityCfg("blind"),
    guide_asset_cfg: SceneEntityCfg = SceneEntityCfg("guide"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    """
    # extract the used quantities (to enable type-hinting)
    blind_asset: RigidObject | Articulation = env.scene[blind_asset_cfg.name]
    guide_asset: RigidObject | Articulation = env.scene[guide_asset_cfg.name]

    # get default root states
    blind_root_states = blind_asset.data.default_root_state[env_ids].clone()
    guide_root_states = guide_asset.data.default_root_state[env_ids].clone()

    blind_patch_view = env.scene.terrain.flat_patches['root_blind_spawn'].view(-1,3)
    guide_patch_view = env.scene.terrain.flat_patches['root_priv_spawn'].view(-1,3)
    goal_patch_view = env.scene.terrain.flat_patches['target_spawn'].view(-1,3)

    select_ids = torch.randint(0, blind_patch_view.shape[0], (len(env_ids),), device=env_ids.device)

    # Reset blind robot
    blind_positions = blind_patch_view[select_ids,:]
    blind_orientations = blind_root_states[:, 3:7]
    blind_velocities = blind_root_states[:, 7:13]

    # set blind robot into the physics simulation
    blind_asset.write_root_pose_to_sim(torch.cat([blind_positions, blind_orientations], dim=-1), env_ids=env_ids)
    blind_asset.write_root_velocity_to_sim(blind_velocities, env_ids=env_ids)

    # Reset guide robot
    guide_positions = guide_patch_view[select_ids,:]
    guide_orientations = guide_root_states[:, 3:7]
    guide_velocities = guide_root_states[:, 7:13]

    # set guide robot into the physics simulation
    guide_asset.write_root_pose_to_sim(torch.cat([guide_positions, guide_orientations], dim=-1), env_ids=env_ids)
    guide_asset.write_root_velocity_to_sim(guide_velocities, env_ids=env_ids)

    