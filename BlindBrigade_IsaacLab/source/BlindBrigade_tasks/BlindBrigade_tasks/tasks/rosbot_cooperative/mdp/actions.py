"""Action terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


class SE2BaseMecanumDrive(ActionTerm):
    cfg: SE2BaseMecanumDriveCfg
    _asset: Articulation

    def __init__(self, cfg: SE2BaseMecanumDriveCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        # Find joint indices for the wheel joints
        self._wheel_joint_ids, _ = self._asset.find_joints(list(cfg.wheel_joints))
        self._prev_vel_cmd = torch.zeros(env.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()
        scaled_actions = actions * torch.tensor([self.cfg.max_vx, self.cfg.max_vy, self.cfg.max_wz], device=self.device)
        self._processed_actions = torch.clamp(
            scaled_actions,
            min=torch.tensor([-self.cfg.max_vx, -self.cfg.max_vy, -self.cfg.max_wz], device=self.device),
            max=torch.tensor([self.cfg.max_vx, self.cfg.max_vy, self.cfg.max_wz], device=self.device),
        )

    def apply_actions(self):
        # Extract body-frame commands
        vx = self._processed_actions[:, 0]
        vy = self._processed_actions[:, 1]
        wz = self._processed_actions[:, 2]

        if self.cfg.animate_wheels:
            # Compute wheel speeds from body velocities
            # Write to simulation
            # Formula for omnidirectional/mecanum drive:
            #   w_fl = (vx - vy - L*wz) / r
            #   w_fr = (vx + vy + L*wz) / r
            #   w_rl = (vx - vy + L*wz) / r
            #   w_rr = (vx + vy - L*wz) / r
            # where L = half_wheelbase, r = wheel_radius
            # Note: vy sign might need flipping depending on your coordinate system

            L = self.cfg.half_wheelbase
            r = self.cfg.wheel_radius

            if self.cfg.o_pattern:
                # O-pattern (mirrored): vy sign flips
                w_fl = (vx - vy - L * wz) / r
                w_fr = (vx + vy - L * wz) / r
                w_rl = (vx - vy + L * wz) / r
                w_rr = (vx + vy + L * wz) / r
            else:
                # X-pattern (standard)
                w_fl = (vx - vy + L * wz) / r
                w_fr = (vx + vy + L * wz) / r
                w_rl = (vx - vy - L * wz) / r
                w_rr = (vx + vy - L * wz) / r

            # Clamp wheel speeds
            w_fl = torch.clamp(w_fl, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
            w_fr = torch.clamp(w_fr, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
            w_rl = torch.clamp(w_rl, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
            w_rr = torch.clamp(w_rr, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
        
            # Set angular velocity targets for all 4 wheels
            wheel_vel_targets = torch.stack([w_fl, w_fr, w_rl, w_rr], dim=-1)
            self._asset.set_joint_velocity_target(wheel_vel_targets, joint_ids=self._wheel_joint_ids)

        # Only set linear velocity for base (wheels don't move)
        root_vel = self._asset.data.root_vel_w.clone()  # preserve current velocity (including gravity)

        # Transform body-frame linear and angular velocities to world-frame
        root_vel[:, :2] = quat_apply(
            self._asset.data.root_quat_w, 
            torch.stack((vx, vy, torch.zeros_like(vx)), dim=-1)
        )[:, :2] # only x, y

        root_vel[:, 5] = quat_apply(
            self._asset.data.root_quat_w, 
            torch.stack((torch.zeros_like(wz), torch.zeros_like(wz), wz), dim=-1)
        )[:, 2] # only yaw

        self._asset.write_root_velocity_to_sim(root_vel)
        self._prev_vel_cmd = self._processed_actions.clone()

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._prev_vel_cmd[env_ids] = 0.0


@configclass
class SE2BaseMecanumDriveCfg(ActionTermCfg):
    class_type: type = SE2BaseMecanumDrive  # set at bottom
    asset_name: str = "robot"

    # Wheel geometry for wheel animation (meters)
    wheel_radius: float = 0.05
    half_wheelbase: float = 0.12  # Lx
    half_track: float = 0.10      # Ly

    wheel_joints: tuple[str, str, str, str] = (
        "fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"
    )

    # Action limits (body frame)
    max_vx: float = 1.0
    max_vy: float = 1.0
    max_wz: float = 2.0

    # Clamp wheel speeds (rad/s)
    max_wheel_speed: float = 60.0

    # Flip vy contribution if your wheel mounting is mirrored (X vs O pattern)
    o_pattern: bool = False

    # If True: also spin wheels; if False: only move base
    animate_wheels: bool = True
