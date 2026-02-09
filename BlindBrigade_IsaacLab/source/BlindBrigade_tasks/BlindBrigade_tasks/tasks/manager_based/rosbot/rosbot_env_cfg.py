import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.envs.mdp.terminations import illegal_contact
from isaaclab_tasks.manager_based.navigation.mdp import position_command_error_tanh, heading_command_error_abs
from isaaclab.utils import configclass
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    FlatPatchSamplingCfg
)
import torch

##
# Pre-defined configs
##
from BlindBrigade_assets.robot.rosbot_xl_mecanum import ROSBOT_XL_CFG
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg
from . import mdp


from isaaclab.envs import ManagerBasedEnv

##
# Scene definition
##

@configclass
class ROSBotSceneCfg(InteractiveSceneCfg):
    """Configuration for a rosbot scene.
    
    Note: Regarding robot_contact_sensor
    Ground and obstacles in generated terrain consist of a single mesh
    This makes detecting collision with just obstacles hard 
    Workaround: 
    - Added a cube mesh starting just above the base level of wheels
    - Check for contact forces w.r.t. the cube mesh (which remains above ground)
    
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        max_init_terrain_level=None,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            sub_terrains={
                "boxes":MeshRepeatedBoxesTerrainCfg(
                    platform_width=0,
                    platform_height=0,
                    object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=40, height=0.25, size=(0.05, 0.05), max_yx_angle=0.0, degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=60, height=0.45, size=(0.3, 0.3), max_yx_angle=60.0, degrees=True
                    ),
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
                        "target": FlatPatchSamplingCfg(
                            num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
                    },
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 0.0),
        ),
        debug_vis=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = ROSBOT_XL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROSBOT_XL_CFG.spawn.replace(activate_contact_sensors=True),
    )
    
    robot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=1,
    )

    ray_caster_cam = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",                                                                       
        update_period=0.0,
        data_types=["distance_to_image_plane"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.08),   # front of robot, slightly above base
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            width=int(64/2),      # low res for speed
            height=int(48/2),
        ),
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.filter_collisions = True
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.01))

        # 25 x 25 = 625 subterrains
        self.terrain.terrain_generator.num_rows = 25
        self.terrain.terrain_generator.num_cols = 25
        # self.terrain.terrain_generator.num_rows = 2 # TODO: Only use for debugging
        # self.terrain.terrain_generator.num_cols = 2

@configclass
class EventCfg:
    """Configuration for the events."""

    root_state = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

##
# MDP settings
##

def ray_caster_depth(env: ManagerBasedEnv) -> torch.Tensor:
    """Alternative:
    ObsTerm(
        func=mdp.image, clip=(0.0,1.0), 
        params={
            "sensor_cfg": SceneEntityCfg("ray_caster_cam"),
            "data_type":"distance_to_image_plane",
            "normalize":False
        }
    )
    * Needs to be flattened if using MLP instead of CNN
    """
    cam = env.scene["ray_caster_cam"]
    depth = cam.data.output["distance_to_image_plane"]
    depth = torch.nan_to_num(depth, nan=1.0) / 1.0
    return depth.reshape(env.num_envs, -1)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-1.0,1.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-2.0, 2.0))
        ray_caster   = ObsTerm(func=ray_caster_depth, clip=(0.0,1.0))
        last_action  = ObsTerm(func=mdp.last_action, clip=(-1., 1.))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Add mdp.undesired_contacts for collision reward, if required. Skipped here.

    terminating = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "goal_pose"},
    )
    position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "goal_pose"},
    )
    orientation_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "goal_pose"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_contact = DoneTerm(
        func=illegal_contact, 
        params={
            "threshold": 45,
            "sensor_cfg": SceneEntityCfg("robot_contact_sensor"),
            },
        )


@configclass
class CommandCfg:
    """Commands for the rosbot."""

    goal_pose = TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        ranges=TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(10.0, 10.0),  # resamples every 10 seconds
        simple_heading=True,
        debug_vis=True,
    )


@configclass
class RosbotActionCfg:
    """Actions for the rosbot."""

    base_twist: mdp.SE2BaseMecanumDriveCfg = mdp.SE2BaseMecanumDriveCfg(
        wheel_radius=0.05,     
        half_wheelbase=0.125,   
        half_track=0.105,       
        o_pattern=False,        
        animate_wheels=True,
    )


#####################################################################
#
#                   Environment Configuration
#
#####################################################################


@configclass
class RosbotNavBoxTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """
    This env config contains a rosbot and box obstacles
    """
    seed: int = 42

    # Scene settings
    scene: ROSBotSceneCfg = ROSBotSceneCfg(num_envs=4, env_spacing=0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: RosbotActionCfg = RosbotActionCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 3 # 40 Hz
        self.episode_length_s = 15
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 200000 # For large number for flat patches


@configclass
class RosbotNavFlatTerrainEnvCfg(RosbotNavBoxTerrainEnvCfg):
    """
    This is a test environment. Designed to test & train RosbotNavBoxTerrainEnvCfg on flat terrain.
    """
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": MeshPlaneTerrainCfg(
                size=(8.0,8.0),
                flat_patch_sampling={          
                "init_pos": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
                "target": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
                },
            ),
        }

        # Disables sensors not required for flat single robot env
        self.scene.ray_caster_cam = None
        self.observations.policy.ray_caster = None
        self.terminations.terrain_contact = None

    