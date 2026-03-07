"""
Source: https://uwrobotlearning.github.io/WheeledLab/
"""

from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

## HOUND 4WD actuator config w/o suspension
HOUND_4WD_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr= ["front_left_wheel_steer", "front_right_wheel_steer"],
        velocity_limit=10.0,
        effort_limit=3.2,
        stiffness=100.0,
        damping=10.,
        friction=0.,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort=1.05,
        effort_limit=0.25,
        velocity_limit=450.,
        stiffness=0,
        damping=1000.,
        friction=0.0,
    ),
}


## HOUND 4WD actuator config with suspension
HOUND_SUS_4WD_ACTUATOR_CFG = {
    **HOUND_4WD_ACTUATOR_CFG,
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_suspension"],
        effort_limit=None, # Passive joint
        velocity_limit=None,
        stiffness=1e8,
        damping=0.,
        friction=.5,
    ),
}

## HOUND 2WD Configuration w/o suspension
HOUND_2WD_ACTUATOR_CFG = {
    "steering_joints": HOUND_4WD_ACTUATOR_CFG["steering_joints"],
    "throttle_joints": HOUND_4WD_ACTUATOR_CFG["throttle_joints"].replace(
        joint_names_expr=["back_.*throttle"],
        effort_limit=0.5, # More torque for two wheel drive
    ),
    "passive_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_.*throttle"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
}

## HOUND 2WD Configuration with suspension
HOUND_SUS_2WD_ACTUATOR_CFG = {
    **HOUND_2WD_ACTUATOR_CFG,
    "suspension": HOUND_SUS_4WD_ACTUATOR_CFG["suspension"],
}
