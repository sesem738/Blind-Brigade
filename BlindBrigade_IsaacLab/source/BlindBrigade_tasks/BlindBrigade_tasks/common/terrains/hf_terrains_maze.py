# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Maze terrain generation for navigation tasks.

This module generates terrain height fields with explicit valid position masks.
The key simplification is that terrain generation directly outputs:
- `heights`: Actual terrain heights for rendering/physics
- `valid_mask`: Boolean mask of valid goal/spawn positions

This eliminates the need for complex height-based classification in goal sampling.

The terrain data is stored on the config during generation, then picked up by
MazeTerrainGenerator._add_sub_terrain and stored on MazeTerrainImporter for access via:
- self.env.scene.terrain._height_field_visual
- self.env.scene.terrain._height_field_valid_mask
- self.env.scene.terrain._height_field_platform_mask
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import binary_dilation, rotate, shift
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.terrains.height_field.utils import height_field_to_mesh

from .terrain_constants import HEIGHTS, PADDING, STAIRS, OBSTACLES, ObstacleType

if TYPE_CHECKING:
    from . import hf_terrains_maze_cfg


# =============================================================================
# Terrain Data Container
# =============================================================================

@dataclass
class TerrainData:
    """Container for terrain height field and valid position mask."""
    heights: np.ndarray
    valid_mask: np.ndarray
    platform_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def create(cls, width: int, height: int) -> "TerrainData":
        """Create empty terrain data with ground-level heights."""
        return cls(
            heights=np.zeros((width, height), dtype=np.int16),
            valid_mask=np.ones((width, height), dtype=bool),
            platform_mask=np.zeros((width, height), dtype=bool),
        )

    def set_obstacle(self, x_start: int, x_end: int, y_start: int, y_end: int, height_value: int):
        """Set a region as an obstacle (invalid for goals)."""
        self.heights[x_start:x_end, y_start:y_end] = height_value
        self.valid_mask[x_start:x_end, y_start:y_end] = False

    def set_platform(self, x_start: int, x_end: int, y_start: int, y_end: int, height_value: int):
        """Set a region as a platform (valid for goals, elevated)."""
        self.heights[x_start:x_end, y_start:y_end] = height_value
        self.valid_mask[x_start:x_end, y_start:y_end] = True
        self.platform_mask[x_start:x_end, y_start:y_end] = True

    def set_ground(self, x_start: int, x_end: int, y_start: int, y_end: int):
        """Set a region as flat ground (valid for goals)."""
        self.heights[x_start:x_end, y_start:y_end] = HEIGHTS.GROUND
        self.valid_mask[x_start:x_end, y_start:y_end] = True
        self.platform_mask[x_start:x_end, y_start:y_end] = False

    def apply_padding(self, padding_cells: int):
        """Dilate invalid regions by padding cells for safety margin."""
        obstacles = ~self.valid_mask
        kernel = np.ones((2 * padding_cells + 1, 2 * padding_cells + 1), dtype=bool)
        dilated = binary_dilation(obstacles, structure=kernel)
        self.valid_mask = ~dilated

    def create_spawn_mask(self, spawn_padding_cells: int) -> np.ndarray:
        """Create a mask for spawn positions with larger padding than goals."""
        extra_padding = spawn_padding_cells - PADDING.GOAL_PADDING
        if extra_padding > 0:
            obstacles = ~self.valid_mask
            kernel = np.ones((2 * extra_padding + 1, 2 * extra_padding + 1), dtype=bool)
            dilated = binary_dilation(obstacles, structure=kernel)
            return ~dilated
        return self.valid_mask.copy()

    def exclude_borders(self, border_cells: int = 2):
        """Mark terrain borders as invalid."""
        self.valid_mask[:border_cells, :] = False
        self.valid_mask[-border_cells:, :] = False
        self.valid_mask[:, :border_cells] = False
        self.valid_mask[:, -border_cells:] = False

    def apply_height_transition_padding(self, height_threshold: int, padding_cells: int):
        """Mark cells near height transitions as invalid."""
        grad_x = np.abs(np.diff(self.heights, axis=0, prepend=self.heights[:1, :]))
        grad_y = np.abs(np.diff(self.heights, axis=1, prepend=self.heights[:, :1]))
        grad_x_back = np.abs(np.diff(self.heights, axis=0, append=self.heights[-1:, :]))
        grad_y_back = np.abs(np.diff(self.heights, axis=1, append=self.heights[:, -1:]))

        max_grad = np.maximum.reduce([grad_x, grad_y, grad_x_back, grad_y_back])
        transition_mask = (max_grad >= height_threshold).astype(bool)

        if padding_cells > 0:
            kernel = np.ones((2 * padding_cells + 1, 2 * padding_cells + 1), dtype=bool)
            transition_mask = binary_dilation(transition_mask, structure=kernel).astype(bool)

        self.valid_mask = self.valid_mask & ~transition_mask


def get_cell_bounds(
    cell_x: int, cell_y: int, cell_pixels: int, max_x: int, max_y: int
) -> Tuple[int, int, int, int]:
    """Get pixel bounds for a maze cell with clamping."""
    return (
        max(0, cell_x * cell_pixels),
        min(max_x, (cell_x + 1) * cell_pixels),
        max(0, cell_y * cell_pixels),
        min(max_y, (cell_y + 1) * cell_pixels),
    )


# =============================================================================
# Maze Generation
# =============================================================================

def generate_maze(rng: np.random.Generator, width: int, height: int, open_prob: float) -> np.ndarray:
    """Generate maze using DFS with random openings."""
    maze = np.ones((width, height), dtype=np.uint8)
    stack = [(0, 0)]
    maze[0, 0] = 0

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[nx, ny] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            idx = rng.integers(len(neighbors))
            nx, ny = neighbors[idx]
            maze[(x + nx) // 2, (y + ny) // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Random openings
    maze[rng.random((width, height)) < open_prob] = 0
    return maze


def clear_center(maze: np.ndarray, terrain: TerrainData, cell_pixels: int):
    """Clear the center area for spawning."""
    cx, cy = maze.shape[0] // 2, maze.shape[1] // 2

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if abs(dx) + abs(dy) <= 1:  # Plus shape
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                    maze[nx, ny] = 0

    x_start = (cx - 1) * cell_pixels
    x_end = (cx + 2) * cell_pixels
    y_start = (cy - 1) * cell_pixels
    y_end = (cy + 2) * cell_pixels
    terrain.set_ground(x_start, x_end, y_start, y_end)


# =============================================================================
# Obstacle Generators
# =============================================================================

def make_pillar(_rng, size, wall_height, scale, is_pit, thickness):
    """Generate a centered pillar obstacle."""
    grid = np.zeros((size, size), dtype=np.int16)
    h = int(wall_height * scale) * (-1 if is_pit else 1)
    grid[thickness:size-thickness, thickness:size-thickness] = h
    return grid


def make_bar(rng, size, wall_height, scale, is_pit, thickness):
    """Generate a rotated bar obstacle."""
    grid = np.zeros((size, size), dtype=np.int16)
    center = size // 2
    h = int(wall_height * scale)
    grid[center - thickness//2:center + thickness//2, :] = h
    angle = rng.uniform(-180, 180)
    grid = rotate(grid, angle, reshape=False, order=1).astype(np.int16)
    if is_pit:
        grid = -grid
    return grid


def make_cross(rng, size, wall_height, scale, is_pit, thickness):
    """Generate a cross-shaped obstacle."""
    grid = np.zeros((size, size), dtype=np.int16)
    center = size // 2
    h = int(wall_height * scale)
    grid[center - thickness//2:center + thickness//2, :] = h
    grid[:, center - thickness//2:center + thickness//2] = h
    angle = rng.uniform(-180, 180)
    grid = rotate(grid, angle, reshape=False, order=1).astype(np.int16)
    if is_pit:
        grid = -grid
    return grid


def make_shifted_block(rng, size, wall_height, scale, is_pit, thickness):
    """Generate a randomly shifted block."""
    grid = np.zeros((size, size), dtype=np.int16)
    h = int(wall_height * scale)
    grid[thickness:size-thickness, thickness:size-thickness] = h
    room = size // 2 - thickness
    shift_amt = (rng.integers(-room, room + 1), rng.integers(-room, room + 1))
    grid = shift(grid, shift=shift_amt, cval=0).astype(np.int16)
    if is_pit:
        grid = -grid
    return grid


_OBSTACLE_GENERATORS = {
    ObstacleType.PILLAR: make_pillar,
    ObstacleType.BAR: make_bar,
    ObstacleType.CROSS: make_cross,
    ObstacleType.SHIFTED_BLOCK: make_shifted_block,
}


def make_random_obstacle(rng, size, wall_height, is_pit=None, pillar_weight=None):
    """Generate a random obstacle type."""
    scale = rng.uniform(OBSTACLES.SCALE_MIN, OBSTACLES.SCALE_MAX)
    if is_pit is None:
        is_pit = rng.random() < OBSTACLES.DEFAULT_PIT_PROB
    thickness = rng.integers(OBSTACLES.THICKNESS_MIN, OBSTACLES.THICKNESS_MAX)

    if pillar_weight is not None and pillar_weight > 0:
        other_weight = (1.0 - pillar_weight) / (ObstacleType.NUM_TYPES - 1)
        weights = [other_weight] * ObstacleType.NUM_TYPES
        weights[ObstacleType.PILLAR] = pillar_weight
        obstacle_type = rng.choice(ObstacleType.NUM_TYPES, p=weights)
    else:
        obstacle_type = rng.integers(ObstacleType.NUM_TYPES)

    generator = _OBSTACLE_GENERATORS[obstacle_type]
    return generator(rng, size, wall_height, scale, is_pit, thickness)


# =============================================================================
# Stair/Platform Generator
# =============================================================================

class StairGenerator:
    """Generates stair structures with platforms."""

    LAYOUTS = [
        {"platforms": [(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)],
         "stairs": [(0, 0, "n"), (2, 2, "s"), (0, 2, "s"), (2, 0, "w")]},
        {"platforms": [(0, 1), (1, 1), (2, 1)],
         "stairs": [(0, 0, "n"), (2, 2, "s"), (0, 2, "s"), (2, 0, "n")]},
        {"platforms": [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)],
         "stairs": [(0, 0, "n"), (2, 2, "s"), (0, 2, "s"), (2, 0, "n")]},
    ]

    def __init__(self, wall_height: float, vertical_scale: float):
        self.wall_height = wall_height
        self.platform_height = int(wall_height - 0.5 / vertical_scale)
        self.vertical_scale = vertical_scale
        self._make_stair_templates()

    def _make_stair_templates(self):
        cell_px = STAIRS.SINGLE_CELL_PIXELS
        step_res = cell_px // STAIRS.NUM_STEPS
        east = np.zeros((cell_px, cell_px), dtype=np.float32)
        for i in range(STAIRS.NUM_STEPS):
            h = STAIRS.STEP_HEIGHT_METERS * (i + 1) / self.vertical_scale
            east[i * step_res:(i + 1) * step_res, :] = h
        self.templates = {
            "e": east,
            "n": rotate(east, 90),
            "w": rotate(east, 180),
            "s": rotate(east, 270),
        }

    def generate(self, rng):
        layout = self.LAYOUTS[rng.integers(len(self.LAYOUTS))]
        size = STAIRS.STAIR_GRID_SIZE * STAIRS.SINGLE_CELL_PIXELS
        cell_px = STAIRS.SINGLE_CELL_PIXELS
        heights = np.zeros((size, size), dtype=np.float32)
        valid_mask = np.zeros((size, size), dtype=bool)
        platform_mask = np.zeros((size, size), dtype=bool)

        for gx, gy in layout["platforms"]:
            xs, xe = gx * cell_px, (gx + 1) * cell_px
            ys, ye = gy * cell_px, (gy + 1) * cell_px
            heights[xs:xe, ys:ye] = self.platform_height
            valid_mask[xs:xe, ys:ye] = True
            platform_mask[xs:xe, ys:ye] = True

        for gx, gy, direction in layout["stairs"]:
            xs, xe = gx * cell_px, (gx + 1) * cell_px
            ys, ye = gy * cell_px, (gy + 1) * cell_px
            heights[xs:xe, ys:ye] = self.templates[direction]

        return heights, valid_mask, platform_mask


# =============================================================================
# Main Terrain Generation
# =============================================================================

def _get_rng(cfg: "hf_terrains_maze_cfg.HfMazeTerrainCfg") -> np.random.Generator:
    """Get RNG from config or create a new one."""
    if cfg.rng is not None:
        return cfg.rng
    return np.random.default_rng()


@height_field_to_mesh
def maze_terrain(difficulty: float, cfg: "hf_terrains_maze_cfg.HfMazeTerrainCfg") -> np.ndarray:
    """Generate maze terrain with obstacles and valid position mask."""
    rng = _get_rng(cfg)

    cell_pixels = int(cfg.cell_size / cfg.horizontal_scale)
    wall_height = int(cfg.wall_height / cfg.vertical_scale)
    terrain_w = int(cfg.size[0] / cfg.horizontal_scale)
    terrain_h = int(cfg.size[1] / cfg.horizontal_scale)

    terrain = TerrainData.create(terrain_w, terrain_h)
    stair_gen = StairGenerator(wall_height, cfg.vertical_scale)

    if cfg.non_maze_terrain:
        maze = np.zeros(cfg.grid_size, dtype=np.uint8)
        obstacle_prob = difficulty * OBSTACLES.NON_MAZE_DENSITY
        maze[rng.random(cfg.grid_size) < obstacle_prob] = 1
    else:
        maze = generate_maze(rng, cfg.grid_size[0], cfg.grid_size[1], 1 - difficulty)

    clear_center(maze, terrain, cell_pixels)

    if cfg.dynamic_obstacles:
        _add_pits(rng, terrain, cfg, difficulty, wall_height, cell_pixels)
    elif cfg.stairs:
        _add_stairs(rng, terrain, cfg, difficulty, wall_height, cell_pixels, stair_gen)
    else:
        _add_walls(rng, maze, terrain, cfg, wall_height, cell_pixels)

    clear_center(maze, terrain, cell_pixels)

    if cfg.stairs:
        terrain.apply_height_transition_padding(
            height_threshold=PADDING.HEIGHT_TRANSITION_THRESHOLD,
            padding_cells=PADDING.HEIGHT_TRANSITION_PADDING
        )

    terrain.apply_padding(PADDING.GOAL_PADDING)
    terrain.exclude_borders(PADDING.BORDER_CELLS)

    spawn_mask = terrain.create_spawn_mask(PADDING.SPAWN_PADDING)
    spawn_mask[:PADDING.BORDER_CELLS, :] = False
    spawn_mask[-PADDING.BORDER_CELLS:, :] = False
    spawn_mask[:, :PADDING.BORDER_CELLS] = False
    spawn_mask[:, -PADDING.BORDER_CELLS:] = False

    if cfg.add_goal:
        cfg.height_field_visual = torch.from_numpy(terrain.heights.copy()).unsqueeze(0)
        cfg.height_field_valid_mask = torch.from_numpy(terrain.valid_mask.copy()).unsqueeze(0)
        cfg.height_field_platform_mask = torch.from_numpy(terrain.platform_mask.copy()).unsqueeze(0)
        cfg.height_field_spawn_mask = torch.from_numpy(spawn_mask.copy()).unsqueeze(0)

    return terrain.heights


# =============================================================================
# Terrain Type Generators
# =============================================================================

def _add_walls(rng, maze, terrain, cfg, wall_height, cell_pixels):
    """Add wall obstacles to terrain based on maze pattern."""
    pillar_weight = OBSTACLES.NON_MAZE_PILLAR_WEIGHT if cfg.non_maze_terrain else None

    for x in range(cfg.grid_size[0]):
        for y in range(cfg.grid_size[1]):
            if maze[x, y] != 1:
                continue

            xs, xe, ys, ye = get_cell_bounds(
                x, y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
            )

            if cfg.randomize_wall and rng.random() < cfg.random_wall_ratio:
                obs = make_random_obstacle(rng, cell_pixels, wall_height, pillar_weight=pillar_weight)
                terrain.heights[xs:xe, ys:ye] = obs[:xe - xs, :ye - ys]
                terrain.valid_mask[xs:xe, ys:ye] = False
            else:
                h = int(wall_height * rng.uniform(OBSTACLES.SCALE_MIN, OBSTACLES.SCALE_MAX))
                terrain.set_obstacle(xs, xe, ys, ye, h)


def _add_stairs(rng, terrain, cfg, difficulty, wall_height, cell_pixels, stair_gen):
    """Add stair/platform structures to terrain."""
    grid_w, grid_h = cfg.grid_size
    grid_middle = grid_w // 2
    excluded = set(range(grid_middle - 1, grid_middle + 1))

    stair_margin = 1
    max_x = grid_w - STAIRS.STAIR_GRID_SIZE - stair_margin
    max_y = grid_h - STAIRS.STAIR_GRID_SIZE - stair_margin
    num_locations = 6
    x_locs = set(np.round(np.linspace(stair_margin, max_x, num_locations)).astype(int)) - excluded
    y_locs = set(np.round(np.linspace(stair_margin, max_y, num_locations)).astype(int)) - excluded

    processed = set()
    stair_size = STAIRS.STAIR_GRID_SIZE * STAIRS.SINGLE_CELL_PIXELS
    stair_prob = difficulty * OBSTACLES.STAIRS_PLACEMENT_PROB
    obstacle_prob = difficulty * OBSTACLES.STAIRS_OBSTACLE_DENSITY

    for x in range(grid_w):
        for y in range(grid_h):
            if (x, y) in processed:
                continue

            if x in x_locs and y in y_locs and rng.random() < stair_prob:
                heights, valid, platform = stair_gen.generate(rng)
                xs = x * cell_pixels
                xe = min(terrain.heights.shape[0], xs + stair_size)
                ys = y * cell_pixels
                ye = min(terrain.heights.shape[1], ys + stair_size)
                sx, sy = xe - xs, ye - ys
                terrain.heights[xs:xe, ys:ye] = heights[:sx, :sy]
                terrain.valid_mask[xs:xe, ys:ye] = valid[:sx, :sy]
                terrain.platform_mask[xs:xe, ys:ye] = platform[:sx, :sy]
                for dx in range(3):
                    for dy in range(3):
                        processed.add((x + dx, y + dy))
            elif rng.random() < obstacle_prob:
                xs, xe, ys, ye = get_cell_bounds(
                    x, y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
                )
                if terrain.valid_mask[xs + 1:xe - 1, ys + 1:ye - 1].all():
                    obs = make_random_obstacle(rng, cell_pixels, wall_height)
                    terrain.heights[xs:xe, ys:ye] = obs[:xe - xs, :ye - ys]
                    terrain.valid_mask[xs:xe, ys:ye] = False


def _add_pits(rng, terrain, cfg, difficulty, wall_height, cell_pixels):
    """Add pit/trough obstacles to terrain."""
    grid_w, grid_h = cfg.grid_size
    trench_offset = OBSTACLES.PITS_TRENCH_ROW_OFFSET
    pit_rows = {trench_offset, grid_h - trench_offset - 1}
    bridges = _generate_bridges(rng, grid_w)

    for pit_y in pit_rows:
        for x in range(grid_w):
            if x in bridges:
                continue
            xs, xe, ys, ye = get_cell_bounds(
                x, pit_y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
            )
            terrain.set_obstacle(xs, xe, ys, ye, -wall_height)

    _add_middle_obstacles(rng, terrain, cfg, difficulty, wall_height, cell_pixels, pit_rows)


def _generate_bridges(rng, grid_width):
    """Generate bridge positions across pit rows."""
    num_bridges = rng.integers(OBSTACLES.BRIDGE_COUNT_MIN, OBSTACLES.BRIDGE_COUNT_MAX)
    margin = OBSTACLES.PITS_EDGE_MARGIN
    available = list(range(margin, grid_width - margin))
    rng.shuffle(available)
    bridges = set()
    for i in range(min(num_bridges, len(available))):
        pos = available[i]
        bridges.add(pos)
        if pos + 1 < grid_width - margin:
            bridges.add(pos + 1)
    return bridges


def _add_middle_obstacles(rng, terrain, cfg, difficulty, wall_height, cell_pixels, pit_rows):
    """Add random obstacles in the middle area between pit rows."""
    grid_w, grid_h = cfg.grid_size
    obstacle_prob = difficulty * OBSTACLES.PITS_DENSITY
    margin = OBSTACLES.PITS_EDGE_MARGIN
    trench_offset = OBSTACLES.PITS_TRENCH_ROW_OFFSET
    x_range = range(margin, grid_w - margin)
    y_range = range(trench_offset + 1, grid_h - trench_offset - 1)

    for x in x_range:
        for y in y_range:
            if y in pit_rows:
                continue
            if rng.random() < obstacle_prob:
                xs, xe, ys, ye = get_cell_bounds(
                    x, y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
                )
                obs = _generate_pit_obstacle(rng, cell_pixels, wall_height)
                terrain.heights[xs:xe, ys:ye] = obs[:xe - xs, :ye - ys]
                terrain.valid_mask[xs:xe, ys:ye] = False


def _generate_pit_obstacle(rng, cell_pixels, wall_height):
    """Generate an obstacle for pit terrain with high pit probability."""
    if rng.random() < OBSTACLES.PITS_BAR_RATIO:
        is_pit = rng.random() < OBSTACLES.PITS_BAR_PIT_PROB
        scale = rng.uniform(OBSTACLES.SCALE_MIN, OBSTACLES.SCALE_MAX)
        thickness = rng.integers(OBSTACLES.THICKNESS_MIN, OBSTACLES.THICKNESS_MAX)
        return make_bar(rng, cell_pixels, wall_height, scale, is_pit, thickness)
    else:
        is_pit = rng.random() < OBSTACLES.PITS_RANDOM_PIT_PROB
        return make_random_obstacle(rng, cell_pixels, wall_height, is_pit=is_pit)
