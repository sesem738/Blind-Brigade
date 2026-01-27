from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    blind_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    priv_asset_cfg: SceneEntityCfg = SceneEntityCfg("priv"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[blind_asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    blind_patch_view = env.scene.terrain.flat_patches['root_blind_spawn'].view(-1,3)
    priv_patch_view = env.scene.terrain.flat_patches['root_priv_spawn'].view(-1,3)
    goal_patch_view = env.scene.terrain.flat_patches['target_spawn'].view(-1,3)

    select_ids = torch.randint(0, blind_patch_view.shape[0], (len(env_ids),), device=env_ids.device)

    # pos
    positions = blind_patch_view[select_ids,:]
    orientations = root_states[:, 3:7]
    # velocities
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)