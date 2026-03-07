"""
Source: https://uwrobotlearning.github.io/WheeledLab/
"""

from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from dataclasses import MISSING

from BlindBrigade_assets import BLINDBRIGADE_ASSETS_DATA_DIR
from .hound import (
    HOUND_2WD_ACTUATOR_CFG, 
    HOUND_SUS_2WD_ACTUATOR_CFG,
    HOUND_4WD_ACTUATOR_CFG, 
    HOUND_SUS_4WD_ACTUATOR_CFG,
) 

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        'back_left_wheel_throttle' : 0.0,
        'back_right_wheel_throttle' : 0.0,
        'front_left_wheel_steer' : 0.0,
        'front_right_wheel_steer' : 0.0,
        'front_left_wheel_throttle' : 0.0,
        'front_right_wheel_throttle' : 0.0,
    },
)

_ZERO_INIT_STATES_WITH_SUS = _ZERO_INIT_STATES.replace(
    joint_pos={
        **_ZERO_INIT_STATES.joint_pos,
        'front_left_wheel_suspension' : 0.0,
        'front_right_wheel_suspension' : 0.0,
        'back_left_wheel_suspension' : 0.0,
        'back_right_wheel_suspension' : 0.0,
    }
)

#------------------- Base Mushr Config ------------------- 

MUSHR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MISSING,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0, # m/s
            max_angular_velocity=100000.0, # deg/s
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
    init_state = MISSING,
    actuators = MISSING
)

#------------------- 4WD Configs ------------------- 

# W/O Suspension
MUSHR_4WD_CFG = MUSHR_CFG.replace(
    spawn=MUSHR_CFG.spawn.replace(
        usd_path=f"{BLINDBRIGADE_ASSETS_DATA_DIR}/UWPRL/mushr_nano.usd",
    ),
    init_state = _ZERO_INIT_STATES,
    actuators = HOUND_4WD_ACTUATOR_CFG,
)

# With Suspension
MUSHR_SUS_4WD_CFG = MUSHR_CFG.replace(
    spawn=MUSHR_CFG.spawn.replace(
        usd_path=f"{BLINDBRIGADE_ASSETS_DATA_DIR}/UWRLL/mushr_nano_v2.usd",
    ),
    init_state= _ZERO_INIT_STATES_WITH_SUS,
    actuators = HOUND_SUS_4WD_ACTUATOR_CFG,
)

#------------------- 2WD Configs ------------------- 

# W/O Suspension
MUSHR_2WD_CFG = MUSHR_CFG.replace(
    spawn=MUSHR_CFG.spawn.replace(
        usd_path=f"{BLINDBRIGADE_ASSETS_DATA_DIR}/UWPRL/mushr_nano.usd",
    ),
    init_state = _ZERO_INIT_STATES,
    actuators = HOUND_2WD_ACTUATOR_CFG,
)

# With Suspension
MUSHR_SUS_2WD_CFG = MUSHR_CFG.replace(
    spawn=MUSHR_CFG.spawn.replace(
        usd_path=f"{BLINDBRIGADE_ASSETS_DATA_DIR}/UWRLL/mushr_nano_v2.usd",
    ),
    init_state= _ZERO_INIT_STATES_WITH_SUS,
    actuators = HOUND_SUS_2WD_ACTUATOR_CFG,
)
