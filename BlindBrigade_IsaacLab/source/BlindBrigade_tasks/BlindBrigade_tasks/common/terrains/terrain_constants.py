# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Constants for terrain generation and goal sampling.

This module centralizes all height field values and terrain parameters used across
terrain generation and goal sampling code. Using these constants ensures consistency
and makes the height semantics explicit.

Height Field Value Semantics (in discretized units):
====================================================
The height field stores heights as integers where:
    actual_height_meters = height_value * VERTICAL_SCALE

Key height values:
- GROUND: 0 (flat walkable ground at z=0)
- PLATFORM: ~200 (flat raised platforms at ~1.0m, valid for goals)
- WALL: ~300 (obstacles/walls at ~1.5m, always excluded from goals)
- PIT: ~-300 (negative obstacles/troughs at ~-1.5m, always excluded)

Goal Sampling Valid Ranges:
- Ground: -10 to 50 (allows small noise/variation)
- Platform: 150 to 250 (captures platform height with margin)
- Excluded: < -10 (pits) or > 250 (walls)
"""

from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Scale Factors
# =============================================================================

HORIZONTAL_SCALE: float = 0.1
"""Horizontal resolution of height field in meters per cell."""

VERTICAL_SCALE: float = 0.005
"""Vertical resolution of height field (height_meters = height_value * VERTICAL_SCALE)."""

CELL_SIZE: float = 2.0
"""Size of each maze cell in meters (default: 2m x 2m)."""

CELL_PIXELS: int = int(CELL_SIZE / HORIZONTAL_SCALE)
"""Number of height field pixels per maze cell (20 at default scale)."""


# =============================================================================
# Height Values (in discretized units, multiply by VERTICAL_SCALE for meters)
# =============================================================================

@dataclass(frozen=True)
class HeightValues:
    """Standard height values for terrain features."""

    # Ground level (valid for goals/spawn)
    GROUND: int = 0

    # Platform height: wall_height - 0.5m in vertical scale units
    # = 1.5/0.005 - 0.5/0.005 = 300 - 100 = 200
    PLATFORM: int = 200

    # Wall/obstacle height: 1.5m in vertical scale units
    # = 1.5/0.005 = 300
    WALL: int = 300

    # Pit/trough height (negative obstacle)
    PIT: int = -300

    @property
    def platform_meters(self) -> float:
        """Platform height in meters."""
        return self.PLATFORM * VERTICAL_SCALE

    @property
    def wall_meters(self) -> float:
        """Wall height in meters."""
        return self.WALL * VERTICAL_SCALE


HEIGHTS = HeightValues()
"""Singleton instance of standard height values."""


# =============================================================================
# Goal Sampling Thresholds
# =============================================================================

@dataclass(frozen=True)
class SamplingThresholds:
    """Thresholds for classifying height values during goal/spawn sampling."""

    GROUND_MIN: int = 0
    GROUND_MAX: int = 10
    PLATFORM_MIN: int = 195
    PLATFORM_MAX: int = 205
    WALL_THRESHOLD: int = 10
    PIT_THRESHOLD: int = 0
    EDGE_THRESHOLD: float = 0.0
    OBSTACLE_MARKER: int = 110
    PLATFORM_EDGE_MIN: int = 180
    PLATFORM_EDGE_MAX: int = 220


THRESHOLDS = SamplingThresholds()
"""Singleton instance of sampling thresholds."""


# =============================================================================
# Padding and Border Configuration
# =============================================================================

@dataclass(frozen=True)
class PaddingConfig:
    """Configuration for obstacle padding and borders."""

    GOAL_PADDING: int = 5
    SPAWN_PADDING: int = 6
    BORDER_CELLS: int = 2
    PILLAR_FOOTPRINT: int = 5
    PILLAR_EDGE_MARGIN: int = 2
    HEIGHT_TRANSITION_THRESHOLD: int = 100
    HEIGHT_TRANSITION_PADDING: int = 1


PADDING = PaddingConfig()
"""Singleton instance of padding configuration."""


# =============================================================================
# Stair Configuration
# =============================================================================

@dataclass(frozen=True)
class StairConfig:
    """Configuration for stair generation."""

    NUM_STEPS: int = 5
    STEP_HEIGHT_METERS: float = 0.2
    STAIR_GRID_SIZE: int = 3
    SINGLE_CELL_PIXELS: int = 20

    @property
    def step_height_units(self) -> float:
        """Step height in discretized units."""
        return self.STEP_HEIGHT_METERS / VERTICAL_SCALE

    @property
    def step_resolution(self) -> int:
        """Pixels per step (cell_pixels / num_steps)."""
        return self.SINGLE_CELL_PIXELS // self.NUM_STEPS


STAIRS = StairConfig()
"""Singleton instance of stair configuration."""


# =============================================================================
# Obstacle Structure Types
# =============================================================================

class ObstacleType:
    """Enumeration of obstacle structure types for random generation."""

    PILLAR = 0
    BAR = 1
    CROSS = 2
    SHIFTED_BLOCK = 3
    NUM_TYPES = 4


# =============================================================================
# Obstacle Generation Parameters
# =============================================================================

@dataclass(frozen=True)
class ObstacleConfig:
    """Configuration for random obstacle generation."""

    SCALE_MIN: float = 0.5
    SCALE_MAX: float = 1.5
    THICKNESS_MIN: int = 7
    THICKNESS_MAX: int = 10
    DEFAULT_PIT_PROB: float = 0.15
    PITS_BAR_RATIO: float = 0.6
    PITS_BAR_PIT_PROB: float = 0.75
    PITS_RANDOM_PIT_PROB: float = 0.5
    BRIDGE_COUNT_MIN: int = 3
    BRIDGE_COUNT_MAX: int = 6
    PITS_TRENCH_ROW_OFFSET: int = 2
    PITS_EDGE_MARGIN: int = 2
    NON_MAZE_DENSITY: float = 0.5
    PITS_DENSITY: float = 0.6
    STAIRS_OBSTACLE_DENSITY: float = 0.35
    STAIRS_PLACEMENT_PROB: float = 0.75
    NON_MAZE_PILLAR_WEIGHT: float = 0.5


OBSTACLES = ObstacleConfig()
"""Singleton instance of obstacle configuration."""


# =============================================================================
# Helper Functions
# =============================================================================

def height_to_meters(height_value: int) -> float:
    """Convert discretized height value to meters."""
    return height_value * VERTICAL_SCALE


def meters_to_height(meters: float) -> int:
    """Convert meters to discretized height value."""
    return int(meters / VERTICAL_SCALE)


def is_valid_ground(height: int) -> bool:
    """Check if height value represents valid ground."""
    return THRESHOLDS.GROUND_MIN <= height <= THRESHOLDS.GROUND_MAX


def is_valid_platform(height: int) -> bool:
    """Check if height value represents valid platform."""
    return THRESHOLDS.PLATFORM_MIN <= height <= THRESHOLDS.PLATFORM_MAX


def is_valid_goal_position(height: int) -> bool:
    """Check if height value is valid for goal/spawn placement."""
    return is_valid_ground(height) or is_valid_platform(height)


def is_obstacle(height: int) -> bool:
    """Check if height value represents an obstacle (wall or pit)."""
    return height > THRESHOLDS.WALL_THRESHOLD or height < THRESHOLDS.PIT_THRESHOLD


def is_pit(height: int) -> bool:
    """Check if height value represents a pit."""
    return height < THRESHOLDS.PIT_THRESHOLD


def is_wall(height: int) -> bool:
    """Check if height value represents a wall."""
    return height > THRESHOLDS.WALL_THRESHOLD


def cell_to_pixels(cell_idx: int) -> Tuple[int, int]:
    """Convert cell index to pixel range."""
    start = cell_idx * CELL_PIXELS
    end = (cell_idx + 1) * CELL_PIXELS
    return start, end
