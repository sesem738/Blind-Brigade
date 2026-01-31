# robot_cfg.py
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg


from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
import isaaclab.sim as sim_utils

from .. import BLINDBRIGADE_ASSETS_DATA_DIR

# scene_cfg.py
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, CollisionPropertiesCfg


ROSORIN_MECANNUM_SRB = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/rosorin_mecanum",
    spawn=UsdFileCfg(
        usd_path=f"{BLINDBRIGADE_ASSETS_DATA_DIR}/rosorin_meccanum_srb.usd",
        rigid_props=RigidBodyPropertiesCfg(
            kinematic_enabled=True,     # <-- enables direct pose/vel driving
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=5.0,
            max_angular_velocity=10.0,
        ),
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
)

# class RosorinRigidSceneCfg(InteractiveSceneCfg):
#     """Scene with a single rigid-body robot + a depth camera attached to camera_link0."""

#     # --- Robot as a single rigid body (kinematic) ---
#     robot: RigidObjectCfg = ROSORIN_MECANNUM_SRB

#     # --- Depth camera attached to an existing camera link frame on the robot ---
#     # Your screenshot shows camera_link0 under base_link.
#     depth_cam: CameraCfg = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/ROSORIN_SRB/base_link/camera_link0/depth_cam",
#         update_period=0.0,  # every sim step; set e.g. 0.1 for 10 Hz
#         width=640,
#         height=480,
#         data_types=[
#             "distance_to_image_plane",  # "depth" in meters along camera optical axis
#             # optional:
#             # "rgb",
#         ],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             horizontal_aperture=20.955,
#             clipping_range=(0.1, 1.0e5),
#         ),
#         # Offset relative to camera_link0 (keep small; set to (0,0,0, identity) if camera_link0 is already correct)
#         offset=CameraCfg.OffsetCfg(
#             pos=(0.0, 0.0, 0.0),
#             rot=(1.0, 0.0, 0.0, 0.0),   # quaternion (w,x,y,z)
#             convention="ros",
#         ),
#     )