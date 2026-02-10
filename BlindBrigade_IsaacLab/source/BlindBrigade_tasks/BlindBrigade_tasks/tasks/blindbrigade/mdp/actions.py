# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for rigid body control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RigidBodyPDAction(ActionTerm):
    """Action term for rigid body control with PD position control for x/y and velocity control for yaw.

    This action term controls a RigidObject using:
    - PD control for x and y position (actions specify target positions)
    - Direct velocity control for yaw (action specifies angular velocity)

    The action space is 3D: [target_x, target_y, yaw_velocity]
    """

    cfg: RigidBodyPDActionCfg
    _asset: RigidObject

    def __init__(self, cfg: RigidBodyPDActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Get the rigid object from the scene
        self._asset = env.scene[cfg.asset_name]

        # Store gains
        self._kp = cfg.kp
        self._kd = cfg.kd

        # Initialize previous position error for derivative term
        self._prev_pos_error = torch.zeros(env.num_envs, 2, device=self._device)

        # Store processed actions
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self._device)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space: [target_x, target_y, yaw_vel]."""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions received from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions (same as raw for this term)."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions from policy.

        Args:
            actions: Raw actions tensor of shape (num_envs, 3).
                     Format: [target_x, target_y, yaw_velocity]
        """
        self._raw_actions = actions.clone()
        self._processed_actions = actions.clone()

    def apply_actions(self):
        """Apply PD control for position and direct velocity for yaw."""
        # Get current state
        current_pos = self._asset.data.root_pos_w[:, :2]  # (num_envs, 2) - x, y only
        current_lin_vel = self._asset.data.root_lin_vel_w[:, :2]  # (num_envs, 2)

        # Target positions from actions
        target_pos = self._processed_actions[:, :2]  # (num_envs, 2)

        # PD control for x, y position
        pos_error = target_pos - current_pos
        vel_error = -current_lin_vel  # target velocity is 0 for position control

        # Compute velocity commands using PD
        vel_xy = self._kp * pos_error + self._kd * vel_error

        # Update previous error for next iteration
        self._prev_pos_error = pos_error.clone()

        # Direct yaw velocity from actions
        yaw_vel = self._processed_actions[:, 2]

        # Build full velocity tensor: [vx, vy, vz, wx, wy, wz]
        root_vel = torch.zeros(self._num_envs, 6, device=self._device)
        root_vel[:, 0] = vel_xy[:, 0]  # vx
        root_vel[:, 1] = vel_xy[:, 1]  # vy
        root_vel[:, 5] = yaw_vel       # wz (yaw angular velocity)

        # Write velocities to simulation
        self._asset.write_root_velocity_to_sim(root_vel)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the action term for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Reset previous error for these environments
        self._prev_pos_error[env_ids] = 0.0


@configclass
class RigidBodyPDActionCfg(ActionTermCfg):
    """Configuration for rigid body PD action term.

    Attributes:
        asset_name: Name of the RigidObject in the scene.
        kp: Proportional gain for position control.
        kd: Derivative gain for velocity damping.
    """

    class_type: type = RigidBodyPDAction

    asset_name: str = "guide"
    """Name of the rigid object asset in the scene."""

    kp: float = 5.0
    """Proportional gain for position control (stiffness)."""

    kd: float = 1.0
    """Derivative gain for velocity control (damping)."""
