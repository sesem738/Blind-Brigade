"""ROSOrin robot articulation configurations for Isaac Lab.

This module provides articulation configurations for the ROSOrin robot platform
with three different drive configurations:
- Ackermann steering (4WD with front steering)
- Differential drive (2 wheel tank-style)
- Mecanum drive (4 wheel omnidirectional)
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
import isaaclab.sim as sim_utils

from . import BLINDBRIGADE_ASSETS_DATA_DIR

##
# Actuator Configurations
##

# 6DOF Omnidirectional Base Actuator Configuration
# Direct control of base x, y, yaw motion (holonomic constraint)
# This uses a 6DOF joint but only actuates 3 DOFs: translation X, Y and rotation Z (yaw)
ROSORIN_OMNI_6DOF_ACTUATOR_CFG = {
    "base_planar": ImplicitActuatorCfg(
        joint_names_expr=["base_joint.*"],  # The 6DOF joint connecting base to world/ground
        velocity_limit=2.0,  # Max linear velocity: 2 m/s, angular: 2 rad/s
        effort_limit=100.0,  # Force/torque limits
        stiffness=0.0,  # Pure velocity control
        damping=10.0,  # Light damping for stability
        friction=0.1,
    ),
}

# Ackermann Drive Actuator Configuration
# Similar to car steering with front steering wheels and all-wheel throttle
ROSORIN_ACKER_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["wheel_f[lr]"],  # Matches wheel_fl, wheel_fr (front left/right steering)
        velocity_limit=10.0,
        effort_limit=2.0,
        stiffness=100.0,
        damping=10.0,
        friction=0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=["wheel_[fb][lr]_throttle"],  # All wheel throttle joints
        saturation_effort=1.0,
        effort_limit=0.3,
        velocity_limit=400.0,
        stiffness=0.0,
        damping=1000.0,
        friction=0.0,
    ),
}

# Differential Drive Actuator Configuration
# Tank-style drive with independent left/right wheel control
ROSORIN_DIFF_ACTUATOR_CFG = {
    "drive_joints": DCMotorCfg(
        joint_names_expr=["wheel_[fb][lr]"],  # All wheels for differential (assuming 4 wheels with synced pairs)
        saturation_effort=1.0,
        effort_limit=0.4,  # Higher torque for differential drive
        velocity_limit=400.0,
        stiffness=0.0,
        damping=1000.0,
        friction=0.0,
    ),
}

# Mecanum Drive Actuator Configuration
# Omnidirectional drive with 4 independently controlled mecanum wheels
ROSORIN_MECANUM_ACTUATOR_CFG = {
    "mecanum_joints": DCMotorCfg(
        joint_names_expr=["wheel_[fb][lr]"],  # All 4 mecanum wheels independently controlled
        saturation_effort=1.0,
        effort_limit=0.35,
        velocity_limit=400.0,
        stiffness=0.0,
        damping=1000.0,
        friction=0.0,
    ),
}

##
# Initial State Configurations
##

# Ackermann initial state
_ACKER_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "wheel_fl": 0.0,  # Front left steering
        "wheel_fr": 0.0,  # Front right steering
        "wheel_fl_throttle": 0.0,
        "wheel_fr_throttle": 0.0,
        "wheel_bl_throttle": 0.0,
        "wheel_br_throttle": 0.0,
    },
)

# Differential drive initial state
_DIFF_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "wheel_fl": 0.0,
        "wheel_fr": 0.0,
        "wheel_bl": 0.0,
        "wheel_br": 0.0,
    },
)

# Mecanum drive initial state (same as differential)
_MECANUM_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "wheel_fl": 0.0,
        "wheel_fr": 0.0,
        "wheel_bl": 0.0,
        "wheel_br": 0.0,
    },
)

##
# Complete Articulation Configurations
##

# ROSOrin Ackermann Configuration
ROSORIN_ACKER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/ROSOrin/rosorin_acker.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,  # m/s
            max_angular_velocity=100000.0,  # deg/s
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_ACKER_INIT_STATE,
    actuators=ROSORIN_ACKER_ACTUATOR_CFG,
)

# ROSOrin Differential Drive Configuration
ROSORIN_DIFF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/ROSOrin/rosorin_diff.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,  # m/s
            max_angular_velocity=100000.0,  # deg/s
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_DIFF_INIT_STATE,
    actuators=ROSORIN_DIFF_ACTUATOR_CFG,
)

# ROSOrin Mecanum Drive Configuration
ROSORIN_MECANUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/ROSOrin/rosorin_mecanum.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,  # m/s
            max_angular_velocity=100000.0,  # deg/s
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_MECANUM_INIT_STATE,
    actuators=ROSORIN_MECANUM_ACTUATOR_CFG,
)
