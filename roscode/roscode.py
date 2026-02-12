#!/usr/bin/env python3
"""
ROS2 Humble node: subscribes to Odometry and publishes to cmd_vel
to make a holonomic robot follow a circle trajectory.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math


class CircleTrajectoryNode(Node):
    """Makes a holonomic robot follow a circle using odometry and cmd_vel."""

    def __init__(self):
        super().__init__("circle_trajectory_node")

        # Parameters
        self.declare_parameter("radius", 1.0)  # circle radius [m]
        self.declare_parameter("linear_speed", 0.3)  # tangential speed [m/s]
        self.declare_parameter("odom_topic", "odometry/filtered")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("control_rate", 20.0)  # Hz

        self.radius = self.get_parameter("radius").value
        self.linear_speed = self.get_parameter("linear_speed").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        control_rate = self.get_parameter("control_rate").value

        # Circle center (set when we receive first odom)
        self.center_x: float | None = None
        self.center_y: float | None = None
        self.initialized = False

        # Current pose from odometry
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Subscriber and publisher
        self.sub_odom = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10,
        )
        self.pub_cmd = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Control timer
        self.timer = self.create_timer(1.0 / control_rate, self.control_callback)

        self.get_logger().info(
            f"Circle trajectory node started: radius={self.radius}m, "
            f"speed={self.linear_speed}m/s"
        )

    def quaternion_to_yaw(self, q):
        """Extract yaw from geometry_msgs/Quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        """Store current pose from odometry."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q)

        # Set circle center at first odom (relative to start pose)
        if not self.initialized:
            # Start circle to the left: center at (x, y + radius)
            self.center_x = self.x - self.radius * math.sin(self.yaw)
            self.center_y = self.y + self.radius * math.cos(self.yaw)
            self.initialized = True

    def control_callback(self):
        """Compute desired velocity tangent to circle and publish cmd_vel."""
        if not self.initialized:
            return

        # Vector from center to robot
        dx = self.x - self.center_x
        dy = self.y - self.center_y
        dist = math.sqrt(dx * dx + dy * dy)

        # Avoid division by zero; if we're at center, drive forward
        if dist < 1e-3:
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            self.pub_cmd.publish(twist)
            return

        # Tangent direction (perpendicular to radius, counter-clockwise):
        # tangent = (-dy/dist, dx/dist)
        tx = -dy / dist
        ty = dx / dist

        # Desired velocity in world frame (tangent to circle, magnitude = linear_speed)
        v_world_x = self.linear_speed * tx
        v_world_y = self.linear_speed * ty

        # Transform to body frame (robot's current yaw)
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        vx_body = c * v_world_x + s * v_world_y
        vy_body = -s * v_world_x + c * v_world_y

        twist = Twist()
        twist.linear.x = float(vx_body)
        twist.linear.y = float(vy_body)
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        self.pub_cmd.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = CircleTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot on shutdown
        stop = Twist()
        stop.linear.x = 0.0
        stop.linear.y = 0.0
        stop.angular.z = 0.0
        node.pub_cmd.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
