from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg
from .. import BLINDBRIGADE_ASSETS_DATA_DIR

ROSBOT_XL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path=f"{BLINDBRIGADE_ASSETS_DATA_DIR}/rosbot_xl_mecanum/rosbot_xl_mecanum.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=5.0,
            max_angular_velocity=10.0,
        ),
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"],
            velocity_limit=60.0,
            damping=625.0,
            stiffness=0.0
        )
    },
)
