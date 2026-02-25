"""Custom command generators for terrain-based navigation.

ValidMaskPose2dCommand:
  Samples goals/spawns from terrain valid_mask.
  Outputs (num_envs, 4): [pos_x_body, pos_y_body, pos_z_body, heading_error]

ValidMaskPosition2dCommand / TerrainBasedPosition2dCommand:
  Thin wrappers that output only (num_envs, 2): [pos_x_body, pos_y_body]
  All sampling logic is inherited; only the `command` property is overridden.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommand
from isaaclab.envs.mdp.commands.commands_cfg import TerrainBasedPose2dCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from BlindBrigade_tasks.common.terrains.terrain_constants import VERTICAL_SCALE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# =============================================================================
# Position Sampler (adapted from SRU goal_commands.py)
# =============================================================================

class PositionSampler:
    """Samples positions uniformly from valid terrain cells.

    Uses pre-computed valid_mask from terrain generation (already has safety padding).
    For spawn positions, uses spawn_mask with larger padding to account for robot body.
    """

    def __init__(
        self,
        heights: torch.Tensor,
        valid_mask: torch.Tensor,
        platform_mask: torch.Tensor,
        terrain_size: float,
        horizontal_scale: float,
        device: torch.device,
        platform_repeat_count: int = 10,
        spawn_mask: torch.Tensor = None,
        border_width: float = 0.0,
    ):
        self.device = device
        self.terrain_size = terrain_size
        self.horizontal_scale = horizontal_scale
        self.heights = heights
        self.valid_mask = valid_mask
        self.platform_mask = platform_mask
        self.spawn_mask = spawn_mask if spawn_mask is not None else valid_mask

        self.cell_size = horizontal_scale
        self.border_pixels = int(border_width / horizontal_scale) + 1
        self.mesh_center = terrain_size / 2

        self._build_position_tables(platform_repeat_count)

    def _build_position_tables(self, platform_repeat_count: int):
        """Build pre-computed position tensors for efficient sampling."""
        num_terrains = self.valid_mask.shape[0]

        # Build GOAL position table (from valid_mask with platform repetition)
        valid_indices = self.valid_mask.nonzero(as_tuple=False)
        enhanced_indices = []

        for terrain_idx in range(num_terrains):
            terrain_valid = valid_indices[valid_indices[:, 0] == terrain_idx]

            if len(terrain_valid) == 0:
                enhanced_indices.append(terrain_valid)
                continue

            terrain_platform = self.platform_mask[terrain_idx]
            platform_positions = terrain_platform.nonzero(as_tuple=False)

            if len(platform_positions) > 0:
                valid_xy = terrain_valid[:, 1:]
                plat_xy = platform_positions
                matches = (valid_xy.unsqueeze(1) == plat_xy.unsqueeze(0)).all(dim=2)
                is_platform = matches.any(dim=1)
                platform_valid = terrain_valid[is_platform]
                if len(platform_valid) > 0:
                    repeated = platform_valid.repeat(platform_repeat_count, 1)
                    terrain_valid = torch.cat([terrain_valid, repeated], dim=0)

            enhanced_indices.append(terrain_valid)

        # Count positions per terrain for goals
        self.count_per_terrain = torch.zeros(num_terrains, dtype=torch.long, device=self.device)
        for terrain_idx in range(num_terrains):
            self.count_per_terrain[terrain_idx] = len(enhanced_indices[terrain_idx])

        max_count = max(1, self.count_per_terrain.max().item())
        self.positions = torch.full(
            (num_terrains, max_count, 3), -1, dtype=torch.long, device=self.device
        )
        for terrain_idx in range(num_terrains):
            terrain_positions = enhanced_indices[terrain_idx]
            num_pos = terrain_positions.shape[0]
            if num_pos > 0:
                self.positions[terrain_idx, :num_pos] = terrain_positions

        # Build SPAWN position table (from spawn_mask, no platform repetition)
        spawn_indices = self.spawn_mask.nonzero(as_tuple=False)
        self.spawn_count_per_terrain = torch.zeros(num_terrains, dtype=torch.long, device=self.device)
        spawn_positions_list = []

        for terrain_idx in range(num_terrains):
            terrain_spawn = spawn_indices[spawn_indices[:, 0] == terrain_idx]
            self.spawn_count_per_terrain[terrain_idx] = len(terrain_spawn)
            spawn_positions_list.append(terrain_spawn)

        max_spawn_count = max(1, self.spawn_count_per_terrain.max().item())
        self.spawn_positions = torch.full(
            (num_terrains, max_spawn_count, 3), -1, dtype=torch.long, device=self.device
        )
        for terrain_idx in range(num_terrains):
            terrain_positions = spawn_positions_list[terrain_idx]
            num_pos = terrain_positions.shape[0]
            if num_pos > 0:
                self.spawn_positions[terrain_idx, :num_pos] = terrain_positions

    def sample(self, terrain_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample GOAL positions for given terrain indices."""
        return self._sample_from_table(terrain_indices, self.positions, self.count_per_terrain)

    def sample_spawn(self, terrain_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample SPAWN positions for given terrain indices."""
        return self._sample_from_table(terrain_indices, self.spawn_positions, self.spawn_count_per_terrain)

    def _sample_from_table(
        self,
        terrain_indices: torch.Tensor,
        positions_table: torch.Tensor,
        count_per_terrain: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from a pre-computed position table."""
        num_samples = terrain_indices.shape[0]

        valid_counts = count_per_terrain[terrain_indices].float().clamp(min=1)
        random_indices = (torch.rand(num_samples, device=self.device) * valid_counts).long()

        selected = positions_table[terrain_indices, random_indices]
        is_valid = selected[:, 0] >= 0

        local_x = torch.zeros(num_samples, device=self.device)
        local_y = torch.zeros(num_samples, device=self.device)
        local_z = torch.zeros(num_samples, device=self.device)

        if is_valid.any():
            valid_selected = selected[is_valid]
            x_idx = valid_selected[:, 1]
            y_idx = valid_selected[:, 2]

            local_x[is_valid] = (x_idx.float() + self.border_pixels) * self.cell_size - self.mesh_center
            local_y[is_valid] = (y_idx.float() + self.border_pixels) * self.cell_size - self.mesh_center

            height_values = self.heights[valid_selected[:, 0], x_idx, y_idx]
            local_z[is_valid] = height_values.float() * VERTICAL_SCALE

        return local_x, local_y, local_z


# =============================================================================
# ValidMaskPose2dCommand
# =============================================================================

class ValidMaskPose2dCommand(CommandTerm):
    """Command generator that samples goal/spawn from terrain valid_mask.

    Outputs the same (num_envs, 4) format as TerrainBasedPose2dCommand:
      [pos_x_body, pos_y_body, pos_z_body, heading_error]

    On resample, it also updates ``env.scene.terrain.env_origins`` with the sampled
    spawn position, so that ``reset_root_state_uniform`` places the robot correctly.
    """

    cfg: ValidMaskPose2dCommandCfg

    def __init__(self, cfg: ValidMaskPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain

        terrain_gen_cfg = self.terrain.cfg.terrain_generator
        self.num_terrain_rows = terrain_gen_cfg.num_rows
        self.num_terrain_cols = terrain_gen_cfg.num_cols
        self.terrain_size = terrain_gen_cfg.size[0]

        # Command buffers: same shape as UniformPose2dCommand
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # Metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

        # Lazy-init sampler
        self._sampling_initialized = False
        self._position_sampler: PositionSampler | None = None

    def __str__(self) -> str:
        return f"ValidMaskPose2dCommand:\n\tCommand dimension: {tuple(self.command.shape[1:])}\n"

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    # --------------------------------------------------------------------- #
    # Sampling initialization
    # --------------------------------------------------------------------- #

    def _initialize_position_sampling(self):
        if self._sampling_initialized:
            return

        heights_raw = getattr(self.terrain, "_height_field_visual", None)
        valid_mask_raw = getattr(self.terrain, "_height_field_valid_mask", None)
        platform_mask_raw = getattr(self.terrain, "_height_field_platform_mask", None)
        spawn_mask_raw = getattr(self.terrain, "_height_field_spawn_mask", None)

        if heights_raw is None or valid_mask_raw is None:
            raise ValueError(
                "No height field data found on terrain. "
                "Ensure MazeTerrainImporter is used with add_goal=True in terrain configuration."
            )

        heights = heights_raw.to(self.device)
        valid_mask = valid_mask_raw.to(self.device)
        platform_mask = platform_mask_raw.to(self.device) if platform_mask_raw is not None else torch.zeros_like(valid_mask)
        spawn_mask = spawn_mask_raw.to(self.device) if spawn_mask_raw is not None else valid_mask

        horizontal_scale = self.terrain.cfg.terrain_generator.horizontal_scale
        sub_terrain_border_width = 0.0  # Default from HfTerrainBaseCfg

        self._position_sampler = PositionSampler(
            heights=heights,
            valid_mask=valid_mask,
            platform_mask=platform_mask,
            terrain_size=self.terrain_size,
            horizontal_scale=horizontal_scale,
            device=self.device,
            spawn_mask=spawn_mask,
            border_width=sub_terrain_border_width,
        )
        self._sampling_initialized = True

    def _get_terrain_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get terrain indices for given environment IDs."""
        levels = self.terrain.terrain_levels[env_ids]
        types = self.terrain.terrain_types[env_ids]

        terrain_cfg = self.terrain.cfg.terrain_generator
        if terrain_cfg.curriculum:
            return levels + types * self.num_terrain_rows
        else:
            return levels * self.num_terrain_cols + types

    # --------------------------------------------------------------------- #
    # CommandTerm interface
    # --------------------------------------------------------------------- #

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal and spawn positions from terrain masks."""
        self._initialize_position_sampling()

        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.clone().to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        terrain_indices = self._get_terrain_indices(env_ids_tensor)

        # Sample goal and spawn
        goal_x, goal_y, goal_z = self._position_sampler.sample(terrain_indices)
        spawn_x, spawn_y, spawn_z = self._position_sampler.sample_spawn(terrain_indices)

        # Convert to world coordinates
        levels = self.terrain.terrain_levels[env_ids]
        types = self.terrain.terrain_types[env_ids]
        terrain_origins = self.terrain.terrain_origins[levels, types]

        # Set goal position in world frame
        self.pos_command_w[env_ids, 0] = terrain_origins[:, 0] + goal_x
        self.pos_command_w[env_ids, 1] = terrain_origins[:, 1] + goal_y
        self.pos_command_w[env_ids, 2] = goal_z + self.robot.data.default_root_state[env_ids, 2]

        # Update env_origins with spawn position (consumed by reset_root_state_uniform)
        spawn_offset = 0.05
        self.terrain.env_origins[env_ids, 0] = terrain_origins[:, 0] + spawn_x
        self.terrain.env_origins[env_ids, 1] = terrain_origins[:, 1] + spawn_y
        self.terrain.env_origins[env_ids, 2] = spawn_z + spawn_offset

        # Sample heading
        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Transform goal to body frame — same as UniformPose2dCommand."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )

    # --------------------------------------------------------------------- #
    # Debug visualization
    # --------------------------------------------------------------------- #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_marker"):
                cfg = CUBOID_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/goal_position"
                cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.goal_marker = VisualizationMarkers(cfg)
            self.goal_marker.set_visibility(True)
        else:
            if hasattr(self, "goal_marker"):
                self.goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_marker.visualize(self.pos_command_w)


# =============================================================================
# Configuration
# =============================================================================

@configclass
class ValidMaskPose2dCommandCfg(CommandTermCfg):
    """Configuration for ValidMaskPose2dCommand."""

    class_type: type = ValidMaskPose2dCommand

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    simple_heading: bool = False
    """If True, heading points toward goal. If False, heading is randomly sampled."""

    @configclass
    class Ranges:
        heading: tuple[float, float] = (-3.14, 3.14)

    ranges: Ranges = Ranges()


# =============================================================================
# 2D Position-Only Commands  (drop z and heading — output shape: (N, 2))
# =============================================================================

class TerrainBasedPosition2dCommand(TerrainBasedPose2dCommand):
    """Terrain-based command that outputs only (x, y) in body frame.

    Inherits all terrain-patch sampling from TerrainBasedPose2dCommand.
    Optionally resamples the goal when the robot reaches it — set
    ``cfg.goal_reached_threshold > 0`` to enable.
    """

    cfg: TerrainBasedPosition2dCommandCfg

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D position in base frame. Shape is (num_envs, 2)."""
        return self.pos_command_b[:, :2]

    def _update_command(self):
        if self.cfg.goal_reached_threshold > 0.0:
            dist = torch.norm(self.pos_command_b[:, :2], dim=1)
            reached = (dist < self.cfg.goal_reached_threshold).nonzero(as_tuple=False).flatten()
            if len(reached) > 0:
                self.time_left[reached] = 0.0
        super()._update_command()


class ValidMaskPosition2dCommand(ValidMaskPose2dCommand):
    """ValidMask-based command that outputs only (x, y) in body frame.

    Inherits all mask-based terrain sampling from ValidMaskPose2dCommand.
    Optionally resamples the goal when the robot reaches it — set
    ``cfg.goal_reached_threshold > 0`` to enable.
    """

    cfg: ValidMaskPosition2dCommandCfg

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D position in base frame. Shape is (num_envs, 2)."""
        return self.pos_command_b[:, :2]

    def _update_command(self):
        # Check distance using pos_command_b from the previous step — i.e. what
        # the policy observed this step — before the parent re-transforms it.
        if self.cfg.goal_reached_threshold > 0.0:
            dist = torch.norm(self.pos_command_b[:, :2], dim=1)
            reached = (dist < self.cfg.goal_reached_threshold).nonzero(as_tuple=False).flatten()
            if len(reached) > 0:
                self.time_left[reached] = 0.0
        super()._update_command()


@configclass
class TerrainBasedPosition2dCommandCfg(TerrainBasedPose2dCommandCfg):
    """Configuration for TerrainBasedPosition2dCommand."""

    class_type: type = TerrainBasedPosition2dCommand


    goal_reached_threshold: float = 0.05
    """Distance (m) at which the goal is considered reached and resampled next step."""

    simple_heading: bool = False
    """Whether to use simple heading or not. Set to false for 2D position."""


@configclass
class ValidMaskPosition2dCommandCfg(ValidMaskPose2dCommandCfg):
    """Configuration for ValidMaskPosition2dCommand."""

    class_type: type = ValidMaskPosition2dCommand

    resampling_time_range: tuple[float, float] = (250.0, 250.0)
    """Resampling is disabled by default — goals only resample on goal reach."""

    goal_reached_threshold: float = 0.0
    """Distance (m) at which the goal is considered reached and resampled next step."""
