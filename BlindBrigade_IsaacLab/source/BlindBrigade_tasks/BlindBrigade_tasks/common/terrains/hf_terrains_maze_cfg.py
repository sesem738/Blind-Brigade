# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for maze height field terrains."""

from dataclasses import MISSING
from typing import Any, Optional

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg

from . import hf_terrains_maze


@configclass
class HfMazeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a maze height field terrain.

    Height Field Data (set during terrain generation):
        - height_field_visual: Heights for Z-lookup (num_terrains, W, H)
        - height_field_valid_mask: Valid goal positions with safety padding
        - height_field_platform_mask: Platform positions for curriculum
        - height_field_spawn_mask: Valid spawn positions with larger padding
    """

    function = hf_terrains_maze.maze_terrain

    # Height Field Storage (populated during terrain generation)
    height_field_visual: torch.Tensor = None
    height_field_valid_mask: torch.Tensor = None
    height_field_platform_mask: torch.Tensor = None
    height_field_spawn_mask: torch.Tensor = None

    # Maze Generation Parameters
    maze: bool = True
    open_probability: float = None
    grid_size: tuple[int, int] = (15, 15)
    cell_size: float = 2.0
    wall_height: float = 1.5

    # Terrain Features
    add_goal: Any = MISSING
    add_noise_to_flat: Any = MISSING
    randomize_wall: Any = MISSING
    random_wall_ratio: float = 0.5
    non_maze_terrain: bool = False
    stairs: bool = False
    add_stairs_to_maze: bool = False
    dynamic_obstacles: bool = False

    # Random Number Generator
    rng: Optional[np.random.Generator] = None
