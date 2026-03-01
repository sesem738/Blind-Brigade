"""
ROS 2 launch file for the ONNX policy node.

Single robot (default namespace):
    ros2 launch roscode launch_policy.py onnx_path:=/path/to/policy.onnx robot_name:=Bob

Multi-robot (namespaced):
    ros2 launch roscode launch_policy.py onnx_path:=/path/to/policy.onnx \
        robot_name:=Bob namespace:=/robot1 goal_x:=2.0 goal_y:=0.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Declare launch arguments ──────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("onnx_path",   default_value="policy.onnx"),
        DeclareLaunchArgument("robot_name",  default_value="Bob"),
        DeclareLaunchArgument("goal_x",      default_value="2.0"),
        DeclareLaunchArgument("goal_y",      default_value="0.0"),
        DeclareLaunchArgument("namespace",   default_value=""),
        DeclareLaunchArgument("control_rate",       default_value="40.0"),
        DeclareLaunchArgument("vicon_timeout_s",    default_value="0.5"),
        DeclareLaunchArgument("max_vx",  default_value="1.0"),
        DeclareLaunchArgument("max_vy",  default_value="1.0"),
        DeclareLaunchArgument("max_wz",  default_value="2.0"),
        DeclareLaunchArgument("clip_vx", default_value="0.8"),
        DeclareLaunchArgument("clip_vy", default_value="0.8"),
        DeclareLaunchArgument("clip_wz", default_value="1.5"),
        DeclareLaunchArgument("vel_lpf_alpha",    default_value="0.4"),
        DeclareLaunchArgument("action_lpf_alpha", default_value="1.0"),
        DeclareLaunchArgument("pose_topic",    default_value="poses"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="cmd_vel"),
    ]

    # ── Policy node ───────────────────────────────────────────────────────────
    policy_node = Node(
        package="roscode",
        executable="onnx_policy_node",
        name="onnx_policy_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[{
            "onnx_path":         LaunchConfiguration("onnx_path"),
            "robot_name":        LaunchConfiguration("robot_name"),
            "goal_x":            LaunchConfiguration("goal_x"),
            "goal_y":            LaunchConfiguration("goal_y"),
            "control_rate":      LaunchConfiguration("control_rate"),
            "vicon_timeout_s":   LaunchConfiguration("vicon_timeout_s"),
            "max_vx":            LaunchConfiguration("max_vx"),
            "max_vy":            LaunchConfiguration("max_vy"),
            "max_wz":            LaunchConfiguration("max_wz"),
            "clip_vx":           LaunchConfiguration("clip_vx"),
            "clip_vy":           LaunchConfiguration("clip_vy"),
            "clip_wz":           LaunchConfiguration("clip_wz"),
            "vel_lpf_alpha":     LaunchConfiguration("vel_lpf_alpha"),
            "action_lpf_alpha":  LaunchConfiguration("action_lpf_alpha"),
            "pose_topic":        LaunchConfiguration("pose_topic"),
            "cmd_vel_topic":     LaunchConfiguration("cmd_vel_topic"),
        }],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(args + [policy_node])
