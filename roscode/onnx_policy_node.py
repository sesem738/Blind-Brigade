#!/usr/bin/env python3
"""
ROS 2 Humble node: deploy a trained ONNX policy on a physical ROSbot.

Observation vector layout (9-dim — matches ProprioceptiveCfg):
  [0:2]  pose_command   — goal in body frame (dx_body, dy_body)       no clip
  [2:5]  base_lin_vel   — body-frame linear velocity (vx, vy, vz)     clip ±1.0
  [5:6]  base_yaw_rate  — body-frame yaw rate wz                      clip ±2.0
  [6:9]  last_action    — previous raw policy output (vx, vy, wz)     clip ±1.0

Action vector (3-dim, raw ≈ [-1, 1]):
  [0] vx_raw  →  vx_body = vx_raw  * MAX_VX   (default 1.0 m/s)
  [1] vy_raw  →  vy_body = vy_raw  * MAX_VY   (default 1.0 m/s)
  [2] wz_raw  →  wz_body = wz_raw  * MAX_WZ   (default 2.0 rad/s)

Control frequency: 40 Hz (matching sim: dt=1/120 s, decimation=3).

Vicon interface:
  - Topic: motion_capture_tracking_interfaces/NamedPoseArray on "poses"
  - Velocities estimated via finite differences with an exponential LPF.

Safety:
  - Stops the robot if Vicon data is lost for > vicon_timeout_s seconds.
  - Clips all published velocities to [−max_vx, max_vx] / [−max_vy, max_vy] / [−max_wz, max_wz].

Multi-robot:
  - Launch with --ros-args -r __ns:=/robot1 to namespace cmd_vel automatically.
  - The Vicon topic is resolved relative to the namespace unless overridden as absolute.
"""

import math
import time

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from motion_capture_tracking_interfaces.msg import NamedPoseArray

# ---------------------------------------------------------------------------
# Helper: quaternion → yaw
# ---------------------------------------------------------------------------

def quat_to_yaw(q) -> float:
    """Extract yaw (rotation about Z) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_body(dx_w: float, dy_w: float, yaw: float):
    """Rotate a 2-D world-frame vector into body frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx_w + s * dy_w, -s * dx_w + c * dy_w


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class ONNXPolicyNode(Node):
    """Runs a trained ONNX policy on a physical ROSbot using Vicon localisation."""

    # ---- Observation / action constants matching training config ----
    OBS_DIM = 9
    ACT_DIM = 3

    def __init__(self):
        super().__init__("onnx_policy_node")

        # ---- ROS 2 parameters ----
        self.declare_parameter("onnx_path", "policy.onnx")
        self.declare_parameter("robot_name", "Bob")          # Vicon rigid-body name
        self.declare_parameter("goal_x", 2.0)               # World-frame goal X [m]
        self.declare_parameter("goal_y", 0.0)               # World-frame goal Y [m]
        self.declare_parameter("control_rate", 40.0)        # Hz
        self.declare_parameter("vicon_timeout_s", 0.5)      # s before emergency stop
        self.declare_parameter("pose_topic", "poses")       # Vicon NamedPoseArray topic
        self.declare_parameter("cmd_vel_topic", "cmd_vel")  # Relative to namespace
        # Action scaling — must match SE2BaseMecanumDriveCfg
        self.declare_parameter("max_vx", 1.0)   # m/s
        self.declare_parameter("max_vy", 1.0)   # m/s
        self.declare_parameter("max_wz", 2.0)   # rad/s
        # Hard velocity clip applied *after* policy scaling (safety margin)
        self.declare_parameter("clip_vx", 0.8)
        self.declare_parameter("clip_vy", 0.8)
        self.declare_parameter("clip_wz", 1.5)
        # Velocity estimator: exponential LPF alpha ∈ (0, 1]; 1.0 = no filter
        self.declare_parameter("vel_lpf_alpha", 0.4)
        # Low-pass filter on the action output (matches action_lpf_alpha in training)
        self.declare_parameter("action_lpf_alpha", 1.0)

        # ---- Read parameters ----
        onnx_path       = self.get_parameter("onnx_path").value
        self.robot_name = self.get_parameter("robot_name").value
        self.goal_w     = np.array([
            self.get_parameter("goal_x").value,
            self.get_parameter("goal_y").value,
        ], dtype=np.float32)
        control_rate        = self.get_parameter("control_rate").value
        self.vicon_timeout  = self.get_parameter("vicon_timeout_s").value
        self.max_vx         = self.get_parameter("max_vx").value
        self.max_vy         = self.get_parameter("max_vy").value
        self.max_wz         = self.get_parameter("max_wz").value
        self.clip_vx        = self.get_parameter("clip_vx").value
        self.clip_vy        = self.get_parameter("clip_vy").value
        self.clip_wz        = self.get_parameter("clip_wz").value
        self.vel_alpha      = self.get_parameter("vel_lpf_alpha").value
        self.act_alpha      = self.get_parameter("action_lpf_alpha").value

        # ---- Load ONNX policy ----
        self.get_logger().info(f"Loading ONNX policy from: {onnx_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.ort_input_name  = self.ort_session.get_inputs()[0].name
        self.ort_output_name = self.ort_session.get_outputs()[0].name
        self.get_logger().info(
            f"Policy loaded. Input: '{self.ort_input_name}' "
            f"Output: '{self.ort_output_name}'"
        )

        # ---- State ----
        # Pose
        self.x: float   = 0.0
        self.y: float   = 0.0
        self.yaw: float = 0.0
        # Velocity estimates (body frame, LPF smoothed)
        self.vx_b: float = 0.0   # body-frame forward velocity
        self.vy_b: float = 0.0   # body-frame lateral velocity
        self.vz_b: float = 0.0   # body-frame vertical (always ≈ 0 on flat floor)
        self.wz_b: float = 0.0   # body-frame yaw rate
        # Previous pose for finite-difference velocity
        self._prev_x:   float | None = None
        self._prev_y:   float | None = None
        self._prev_yaw: float | None = None
        self._prev_t:   float | None = None
        # Last action (raw policy output, used as observation)
        self._last_action = np.zeros(self.ACT_DIM, dtype=np.float32)
        # LPF state on action output
        self._act_lpf = np.zeros(self.ACT_DIM, dtype=np.float32)
        # Vicon health
        self._last_vicon_t: float = time.monotonic()
        self._vicon_valid: bool   = False

        # ---- Publishers / Subscribers ----
        pose_topic    = self.get_parameter("pose_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.sub_pose = self.create_subscription(
            NamedPoseArray,
            pose_topic,
            self._vicon_callback,
            10,
        )
        self.pub_cmd = self.create_publisher(Twist, cmd_vel_topic, 10)

        # ---- Control timer ----
        self._control_dt = 1.0 / control_rate
        self.timer = self.create_timer(self._control_dt, self._control_callback)

        self.get_logger().info(
            f"ONNXPolicyNode ready. Robot='{self.robot_name}', "
            f"goal=({self.goal_w[0]:.2f}, {self.goal_w[1]:.2f}), "
            f"rate={control_rate:.0f} Hz"
        )

    # ------------------------------------------------------------------
    # Vicon callback
    # ------------------------------------------------------------------

    def _vicon_callback(self, msg: NamedPoseArray):
        """Extract pose from the Vicon NamedPoseArray and estimate velocities."""
        now = time.monotonic()

        for named_pose in msg.poses:
            if named_pose.name != self.robot_name:
                continue

            x   = named_pose.pose.position.x
            y   = named_pose.pose.position.y
            yaw = quat_to_yaw(named_pose.pose.orientation)

            # ---- Finite-difference velocity estimation ----
            if self._prev_t is not None:
                dt = now - self._prev_t
                if dt > 1e-6:
                    # World-frame velocities
                    vx_w_raw = (x   - self._prev_x)   / dt
                    vy_w_raw = (y   - self._prev_y)   / dt
                    dyaw_raw = wrap_to_pi(yaw - self._prev_yaw) / dt

                    # Rotate to body frame
                    vx_b_raw, vy_b_raw = world_to_body(vx_w_raw, vy_w_raw, yaw)

                    # Exponential LPF
                    a = self.vel_alpha
                    self.vx_b = a * vx_b_raw  + (1.0 - a) * self.vx_b
                    self.vy_b = a * vy_b_raw  + (1.0 - a) * self.vy_b
                    self.wz_b = a * dyaw_raw  + (1.0 - a) * self.wz_b
                    # vz stays at 0 (flat floor assumption)

            # ---- Update state ----
            self.x   = x
            self.y   = y
            self.yaw = yaw
            self._prev_x   = x
            self._prev_y   = y
            self._prev_yaw = yaw
            self._prev_t   = now

            self._last_vicon_t = now
            self._vicon_valid  = True
            break   # found the robot — no need to iterate further

    # ------------------------------------------------------------------
    # Control callback — runs at fixed rate
    # ------------------------------------------------------------------

    def _control_callback(self):
        """Build observation → ONNX inference → publish cmd_vel."""
        now = time.monotonic()

        # ---- Safety: Vicon timeout ----
        if not self._vicon_valid or (now - self._last_vicon_t) > self.vicon_timeout:
            self._publish_stop("Vicon data lost — stopping robot.")
            return

        # ---- Build observation vector ----
        obs = self._build_observation()

        # ---- ONNX inference ----
        ort_inputs = {self.ort_input_name: obs.reshape(1, self.OBS_DIM)}
        raw_action = self.ort_session.run(
            [self.ort_output_name], ort_inputs
        )[0].flatten()   # shape (3,)

        # ---- Action low-pass filter (matches action_lpf_alpha in sim) ----
        a = self.act_alpha
        self._act_lpf = a * raw_action + (1.0 - a) * self._act_lpf

        # ---- Scale to physical velocities ----
        vx_cmd = float(self._act_lpf[0]) * self.max_vx
        vy_cmd = float(self._act_lpf[1]) * self.max_vy
        wz_cmd = float(self._act_lpf[2]) * self.max_wz

        # ---- Hard safety clip ----
        vx_cmd = max(-self.clip_vx, min(self.clip_vx, vx_cmd))
        vy_cmd = max(-self.clip_vy, min(self.clip_vy, vy_cmd))
        wz_cmd = max(-self.clip_wz, min(self.clip_wz, wz_cmd))

        # ---- Store last_action as raw policy output (matches training obs) ----
        self._last_action = np.clip(raw_action.astype(np.float32), -1.0, 1.0)

        # ---- Publish ----
        twist = Twist()
        twist.linear.x  = vx_cmd
        twist.linear.y  = vy_cmd
        twist.angular.z = wz_cmd
        self.pub_cmd.publish(twist)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """Assemble the 9-dim observation vector expected by the policy.

        Layout (matching ProprioceptiveCfg in mdp_cfg.py):
            [0:2] pose_command  — goal in body frame (dx, dy)
            [2:5] base_lin_vel  — body-frame velocity (vx, vy, vz), clip ±1
            [5:6] base_yaw_rate — yaw rate wz, clip ±2
            [6:9] last_action   — previous raw policy output, clip ±1
        """
        # ── pose_command: goal vector rotated to body frame ──────────────────
        dx_w = self.goal_w[0] - self.x
        dy_w = self.goal_w[1] - self.y
        dx_b, dy_b = world_to_body(dx_w, dy_w, self.yaw)

        # ── base_lin_vel (clipped) ────────────────────────────────────────────
        vx_b = float(np.clip(self.vx_b, -1.0, 1.0))
        vy_b = float(np.clip(self.vy_b, -1.0, 1.0))
        vz_b = 0.0   # flat floor

        # ── base_yaw_rate (clipped) ───────────────────────────────────────────
        wz_b = float(np.clip(self.wz_b, -2.0, 2.0))

        # ── last_action (already clipped in _control_callback) ───────────────
        obs = np.array(
            [
                dx_b, dy_b,          # 0:2  pose_command
                vx_b, vy_b, vz_b,    # 2:5  base_lin_vel
                wz_b,                # 5:6  base_yaw_rate
                self._last_action[0],# 6    last_action vx_raw
                self._last_action[1],# 7    last_action vy_raw
                self._last_action[2],# 8    last_action wz_raw
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_stop(self, reason: str | None = None):
        """Publish zero velocity and optionally log a warning."""
        if reason:
            self.get_logger().warn(reason, throttle_duration_sec=1.0)
        twist = Twist()
        self.pub_cmd.publish(twist)

    def stop_robot(self):
        """Send several stop commands to ensure the robot halts."""
        stop = Twist()
        for _ in range(10):
            self.pub_cmd.publish(stop)
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ONNXPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
