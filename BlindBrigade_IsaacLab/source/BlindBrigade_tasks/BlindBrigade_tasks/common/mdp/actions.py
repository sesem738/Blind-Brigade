"""Action terms for rigid body control."""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import MISSING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation

###############################################
#              Action Terms
###############################################

class SE2BaseMecanumDrive(ActionTerm):
    """Action term for SE2 planar control of a mecanum-wheeled base.

    Accepts body-frame velocity commands (vx, vy, wz), scales them by configured
    limits, and directly writes the resulting root velocity to simulation. Optionally
    computes and applies individual wheel speed targets for visual animation.
    """
    cfg: SE2BaseMecanumDriveCfg
    _asset: Articulation

    def __init__(self, cfg: SE2BaseMecanumDriveCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        # Find joint indices for the wheel joints
        self._wheel_joint_ids, _ = self._asset.find_joints(list(cfg.wheel_joints))
        self._prev_vel_cmd = torch.zeros(env.num_envs, 3, device=self.device)
        # Low-pass filter state
        self._lpf_state = torch.zeros(env.num_envs, 3, device=self.device)

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
        # Single-pole low-pass filter: y = α·x + (1-α)·y_prev
        alpha = self.cfg.action_lpf_alpha
        self._lpf_state = alpha * actions + (1.0 - alpha) * self._lpf_state
        scaled_actions = self._lpf_state * torch.tensor([self.cfg.max_vx, self.cfg.max_vy, self.cfg.max_wz], device=self.device)
        self._processed_actions = torch.clamp(
            scaled_actions,
            min=torch.tensor([-self.cfg.max_vx, -self.cfg.max_vy, -self.cfg.max_wz], device=self.device),
            max=torch.tensor([self.cfg.max_vx, self.cfg.max_vy, self.cfg.max_wz], device=self.device),
        )

    def apply_actions(self):
        # Extract body-frame commands
        vx = self._processed_actions[:, 0] * self.cfg.scale_factor_vx
        vy = self._processed_actions[:, 1] * self.cfg.scale_factor_vy
        wz = self._processed_actions[:, 2] * self.cfg.scale_factor_wz

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

        # Build root velocity from sim state, then overwrite commanded components
        cur_vel = self._asset.data.root_vel_w
        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        root_vel[:, 2] = cur_vel[:, 2]  # preserve vz for gravity
        # Damp roll/pitch rates to prevent runaway tilting while allowing slope response
        root_vel[:, 3:5] = cur_vel[:, 3:5] * self.cfg.rp_damping

        # Transform body-frame linear and angular velocities to world-frame
        root_vel[:, :2] = quat_apply(
            self._asset.data.root_quat_w,
            torch.stack((vx, vy, torch.zeros_like(vx)), dim=-1)
        )[:, :2]

        root_vel[:, 5] = quat_apply(
            self._asset.data.root_quat_w,
            torch.stack((torch.zeros_like(wz), torch.zeros_like(wz), wz), dim=-1)
        )[:, 2]

        self._asset.write_root_velocity_to_sim(root_vel)
        self._prev_vel_cmd = self._processed_actions.clone()

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._prev_vel_cmd[env_ids] = 0.0
        self._lpf_state[env_ids] = 0.0


class DifferentialDrive(ActionTerm):
    """Action term for differential-drive control.

    Accepts a 2D action (v, omega) — forward velocity and yaw rate — and converts
    them to left/right wheel velocity targets using differential-drive kinematics:

        w_left  = (v - omega * half_track) / wheel_radius
        w_right = (v + omega * half_track) / wheel_radius

    For a 4-wheel robot, the front and rear wheels on each side receive the same target.
    No root velocity override — the robot moves purely through wheel physics.
    """

    cfg: DifferentialDriveCfg
    _asset: Articulation

    def __init__(self, cfg: DifferentialDriveCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._wheel_joint_ids, _ = self._asset.find_joints(list(cfg.wheel_joints))
        self._raw_actions = torch.zeros(env.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 2, device=self.device)
        self._lpf_state = torch.zeros(env.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Low-pass filter
        alpha = self.cfg.action_lpf_alpha
        self._lpf_state = alpha * actions + (1.0 - alpha) * self._lpf_state
        # Scale to velocity limits
        limits = torch.tensor([self.cfg.max_v, self.cfg.max_omega], device=self.device)
        self._processed_actions = torch.clamp(self._lpf_state * limits, -limits, limits)

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        omega = self._processed_actions[:, 1]

        r = self.cfg.wheel_radius
        d = self.cfg.half_track  # half the distance between left and right wheels

        # Differential drive kinematics
        w_left = (v - omega * d) / r
        w_right = (v + omega * d) / r

        # Clamp wheel speeds
        w_left = torch.clamp(w_left, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
        w_right = torch.clamp(w_right, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)

        # Both front and rear on same side get the same target
        wheel_vel_targets = torch.stack([w_left, w_right, w_left, w_right], dim=-1)
        self._asset.set_joint_velocity_target(wheel_vel_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._lpf_state[env_ids] = 0.0


class AckermannDrive(ActionTerm):
    """
    Action term for Ackermann-steered vehicles (e.g. MuSHR, F1TENTH).
    Source: https://uwrobotlearning.github.io/WheeledLab/

    Accepts a 2D action (velocity, steering_angle) and converts to:
    - Joint *position* targets for steering joints (Ackermann geometry)
    - Joint *velocity* targets for throttle joints

    Unlike the mecanum/differential drives, this does NOT override root velocity.
    The robot moves purely through actuated joints.
    """

    cfg: AckermannDriveCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""
    _bounding_strategy: str | None
    """The strategy used to bound the actions. Can be 'clip' or 'tanh'. If None, no bounding is applied."""

    def __init__(self, cfg: AckermannDriveCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        wheel_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names)
        self._wheel_ids = wheel_ids
        self._wheel_names = wheel_names

        steering_ids, steering_names = self._asset.find_joints(cfg.steering_joint_names)
        self._steering_ids = steering_ids
        self._steering_names = steering_names

        # Action scaling and offset
        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
        self._bounding_strategy = cfg.bounding_strategy

        # Initialize tensors for actions
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)  # Placeholder for [velocity, steering_angle]

        self.base_length = torch.tensor(cfg.base_length, device=self.device)
        self.base_width = torch.tensor(cfg.base_width, device=self.device)
        self.wheel_rad = torch.tensor(cfg.wheel_radius, device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions

        if self._bounding_strategy == 'clip':
            self._processed_actions = torch.clip(actions, min=-1.0, max=1.0) * self._scale + self._offset

        elif self._bounding_strategy == 'tanh':
            self._processed_actions = torch.tanh(actions) * self._scale + self._offset

        else:
            self._processed_actions = actions * self._scale + self._offset

        if self.cfg.no_reverse:
            self._processed_actions[:, 0] = torch.clamp(self._processed_actions[:, 0], min=0.0)


    def apply_actions(self):

        left_rotator_angle, right_rotator_angle, wheel_speeds = self._calculate_ackermann_angles_and_velocities(
            target_velocity=self.processed_actions[:, 0], # Velocity for all cars
            target_steering_angle=self.processed_actions[:, 1] # Steering angle for all cars
        )
        front_wheel_angles = torch.stack([left_rotator_angle, right_rotator_angle], dim=1)

        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=self._wheel_ids)
        self._asset.set_joint_position_target(front_wheel_angles, joint_ids=self._steering_ids)


    def _calculate_ackermann_angles_and_velocities(self, target_steering_angle, target_velocity):
        """
        Calculates the steering angles for the left and right front wheels and the wheel velocities based on the
        Ackermann steering geometry.

        Parameters:
        ----------
        target_steering_angle : torch.Tensor
            Target steering angles in radians for each environment.
        target_velocity : torch.Tensor
            Target velocities for each environment in meters/second.

        Returns:
        -------
        delta_left : torch.Tensor
            Steering angles for the left front wheels.
        delta_right : torch.Tensor
            Steering angles for the right front wheels.
        wheel_speeds : torch.Tensor
            Speeds for each wheel.
        """
        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad

        # Ensure inputs are PyTorch tensors
        target_steering_angle = target_steering_angle.float()
        target_velocity = target_velocity.float()

        # Calculating the turn radius from the steering angle
        tan_steering = torch.tan(target_steering_angle)
        R = torch.where(tan_steering == 0, torch.full_like(tan_steering, 1e6), L / tan_steering)

        # Calculate the steering angles for the left and right front wheels in radians
        delta_left = torch.atan(L / (R - W / 2))
        delta_right = torch.atan(L / (R + W / 2))

        # Assuming the rear wheels follow the path's radius adjusted for their position
        R_rear_left = torch.sqrt((R - W/2)**2 + L**2)
        R_rear_right = torch.sqrt((R + W/2)**2 + L**2)

        # Velocity adjustment based on wheel's distance from the IC
        v_front_left = target_velocity * torch.abs(R_rear_left / (R*wheel_radius))
        v_front_right = target_velocity * torch.abs(R_rear_right / (R*wheel_radius))

        v_back_left = target_velocity * torch.abs((R - W/2) / (R*wheel_radius))
        v_back_right = target_velocity * torch.abs((R + W/2) / (R*wheel_radius))

        # Calculate target rotation for each wheel based on its velocity
        wheel_speeds = torch.stack([v_back_left, v_back_right, v_front_left, v_front_right], dim=1)

        return delta_left, delta_right, wheel_speeds


class AckermannRWDDrive(AckermannDrive):
    """
    MuSHR only uses tan steering and open diff throttle for drifting
    Source: https://uwrobotlearning.github.io/WheeledLab/
    TODO: simulated open diff throttle
    """

    def _calculate_ackermann_angles_and_velocities(self, target_velocity, target_steering_angle):
        """
        Calculates the steering angles for the left and right front wheels and the wheel velocities based on the
        Ackermann steering geometry.
        """
        tan_steering = torch.tan(target_steering_angle)
        target_ang_vel = target_velocity / self.wheel_rad

        # tan steering and fixed rear throttle
        delta_left = tan_steering
        delta_right = tan_steering
        v_back_left = target_ang_vel
        v_back_right = target_ang_vel

        throttle = torch.stack([v_back_left, v_back_right], dim=1)

        return delta_left, delta_right, throttle


class Ackermann4WDDrive(AckermannDrive):
    """4WD with tan steering and open diff throttle (simulated through ackermann adjusted throttle)
    Source: https://uwrobotlearning.github.io/WheeledLab/"""

    def _calculate_ackermann_angles_and_velocities(self, target_velocity, target_steering_angle):

        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad

        # Calculating the turn radius from the steering angle
        tan_steering = torch.tan(target_steering_angle)
        R = torch.where(tan_steering == 0, torch.full_like(tan_steering, 1e6), L / tan_steering)

        # Calculate the steering angles for the left and right front wheels in radians
        delta_left = tan_steering
        delta_right = tan_steering

        # Assuming the rear wheels follow the path's radius adjusted for their position
        R_rear_left = torch.sqrt((R - W/2)**2 + L**2)
        R_rear_right = torch.sqrt((R + W/2)**2 + L**2)

        # Velocity adjustment based on wheel's distance from the IC
        v_front_left = target_velocity * torch.abs(R_rear_left / (R*wheel_radius))
        v_front_right = target_velocity * torch.abs(R_rear_right / (R*wheel_radius))

        v_back_left = target_velocity * torch.abs((R - W/2) / (R*wheel_radius))
        v_back_right = target_velocity * torch.abs((R + W/2) / (R*wheel_radius))

        # Calculate target rotation for each wheel based on its velocity
        wheel_speeds = torch.stack([v_back_left, v_back_right, v_front_left, v_front_right], dim=1)

        return delta_left, delta_right, wheel_speeds

###############################################
#              Action Configs
###############################################

@configclass
class SE2BaseMecanumDriveCfg(ActionTermCfg):
    """Configuration for :class:`SE2BaseMecanumDrive`.

    Defines wheel geometry, body-frame velocity limits, wheel speed cap,
    mecanum roller pattern (X vs O), and whether to animate wheel joints.
    """
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

    # Acounting for model errors
    scale_factor_vx: float = 1.0
    scale_factor_vy: float = 1.0
    scale_factor_wz: float = 1.5

    # Clamp wheel speeds (rad/s)
    max_wheel_speed: float = 60.0

    # Flip vy contribution if your wheel mounting is mirrored (X vs O pattern)
    o_pattern: bool = False

    # If True: also spin wheels; if False: only move base
    animate_wheels: bool = True

    # Low-pass filter alpha (0→1): 1.0 = no filtering, lower = smoother
    action_lpf_alpha: float = 1.0

    # Per-step multiplier on roll/pitch angular velocity (0.0 = zero out, 1.0 = preserve fully)
    rp_damping: float = 0.0


@configclass
class DifferentialDriveCfg(ActionTermCfg):
    """Configuration for :class:`DifferentialDrive`."""

    class_type: type = DifferentialDrive
    asset_name: str = "robot"

    # Wheel geometry
    wheel_radius: float = 0.05
    half_track: float = 0.105  # half the distance between left and right wheels

    # Joint names: FL, FR, RL, RR
    wheel_joints: tuple[str, str, str, str] = (
        "fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"
    )

    # Action limits
    max_v: float = 1.0       # max forward/backward velocity (m/s)
    max_omega: float = 2.0   # max yaw rate (rad/s)

    # Clamp wheel speeds (rad/s)
    max_wheel_speed: float = 60.0

    # Low-pass filter alpha (0→1): 1.0 = no filtering, lower = smoother
    action_lpf_alpha: float = 1.0


@configclass
class AckermannDriveCfg(ActionTermCfg):
    """Configuration for the Ackermann steering action term.
    Source: https://uwrobotlearning.github.io/WheeledLab/

    This action term models a vehicle with Ackermann steering, where actions control
    the forward velocity and steering angle of the vehicle.

    Attributes:
        class_type: Reference to the AckermannAction class.
        velocity_joint_name: The name of the joint or mechanism controlling velocity.
        steering_joint_name: The name of the joint or mechanism controlling steering.
        scale: Scale factor for the action components (velocity, steering_angle).
        offset: Offset factor for the action components, useful for calibration.
    """
    class_type: type[ActionTerm] = AckermannDrive

    wheel_joint_names: list[str] = MISSING
    """The name of the joint or mechanism controlling the vehicle's velocity."""

    steering_joint_names: list[str] = MISSING
    """The name of the joint or mechanism controlling the vehicle's steering angle."""

    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factors for the action components: (velocity_scale, steering_angle_scale). Defaults to (1.0, 1.0)."""

    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factors for the action components: (velocity_offset, steering_angle_offset). Defaults to (0.0, 0.0)."""

    bounding_strategy: str | None = "tanh"
    """The strategy to bound the action values. Defaults to "tanh"."""

    base_length: float = 1.0
    """The length of the vehicle's base. Defaults to 1.0."""

    base_width: float = 1.0
    """The width of the vehicle's base. Defaults to 1.0."""

    wheel_radius: float = 1.0
    """The radius of the vehicle's wheels. Defaults to 1.0."""

    no_reverse: bool = False
    """Whether the vehicle can reverse; sets throttle min to 0 if True. Defaults to False."""


@configclass
class AckermannRWDDriveCfg(AckermannDriveCfg):
    """Rear-wheel-drive Ackermann config (2 throttle joints).
    Source: https://uwrobotlearning.github.io/WheeledLab/"""
    class_type: type[ActionTerm] = AckermannRWDDrive


@configclass
class Ackermann4WDDriveCfg(AckermannDriveCfg):
    """4WD Ackermann config (4 throttle joints).
    Source: https://uwrobotlearning.github.io/WheeledLab/"""
    class_type: type[ActionTerm] = Ackermann4WDDrive
