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


def reset_two_robots(                                                                                         
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,                                                                                    
    robot1_cfg: SceneEntityCfg,
    robot2_cfg: SceneEntityCfg,
    min_distance: float = 0.15,
    max_distance: float = 2.0,
):
    robot1 = env.scene[robot1_cfg.name]
    robot2 = env.scene[robot2_cfg.name]
    terrain = env.scene.terrain

    init_patches = terrain.flat_patches["init_pos"]
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]

    tile_patches = init_patches[levels, types]  # (len(env_ids), num_patches, 3)

    # robot 1: random patch
    ids1 = torch.randint(0, tile_patches.shape[1], (len(env_ids),), device=env.device)
    pos1 = tile_patches[torch.arange(len(env_ids), device=env.device), ids1]

    # robot 2: pick random valid patch within [min_distance, max_distance] of robot 1
    dists = torch.norm(tile_patches - pos1.unsqueeze(1), dim=-1)
    invalid = (dists < min_distance) | (dists > max_distance)
    dists[invalid] = float('inf')

    # if no valid patch, fall back to closest patch outside min_distance
    all_invalid = dists.isinf().all(dim=1)
    if all_invalid.any():
        fallback_dists = torch.norm(tile_patches[all_invalid] - pos1[all_invalid].unsqueeze(1), dim=-1)
        fallback_dists[fallback_dists < min_distance] = float('inf')
        ids_fallback = fallback_dists.argmin(dim=1)
        # will handle below

    # among valid candidates, pick randomly (not just argmin)
    valid_mask = ~dists.isinf()
    # sample one valid index per env using Gumbel-max trick for batched random argmin
    rand_scores = torch.rand_like(dists)
    rand_scores[~valid_mask] = -float('inf')
    ids2 = rand_scores.argmax(dim=1)

    # fallback for envs with no valid patches
    if all_invalid.any():
        ids2[all_invalid] = ids_fallback

    pos2 = tile_patches[torch.arange(len(env_ids), device=env.device), ids2]

    # write poses
    default1 = robot1.data.default_root_state[env_ids].clone()
    default2 = robot2.data.default_root_state[env_ids].clone()
    default1[:, :3] = pos1
    default2[:, :3] = pos2

    robot1.write_root_pose_to_sim(default1[:, :7], env_ids=env_ids)
    robot1.write_root_velocity_to_sim(default1[:, 7:13], env_ids=env_ids)
    robot2.write_root_pose_to_sim(default2[:, :7], env_ids=env_ids)
    robot2.write_root_velocity_to_sim(default2[:, 7:13], env_ids=env_ids)
