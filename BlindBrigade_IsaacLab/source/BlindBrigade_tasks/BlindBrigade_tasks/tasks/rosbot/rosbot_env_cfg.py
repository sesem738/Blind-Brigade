import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurriculumTerm
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    RayCasterCfg,
    patterns
)
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    FlatPatchSamplingCfg
)
from isaaclab.envs.mdp.terminations import illegal_contact
from isaaclab_tasks.manager_based.navigation.mdp import position_command_error_tanh, heading_command_error_abs
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from BlindBrigade_assets.robot.rosbot_xl_mecanum import ROSBOT_XL_CFG
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg
from . import mdp

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
        max_init_terrain_level=0,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=1.0,
            border_height=0.1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            curriculum=True,
            use_cache=False,
            sub_terrains={
                "boxes":MeshRepeatedBoxesTerrainCfg(
                    platform_width=0,
                    platform_height=0,
                    object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=0, height=0.25, size=(0.05, 0.05), max_yx_angle=0.0, degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=40, height=0.45, size=(0.3, 0.3), max_yx_angle=60.0, degrees=True
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
        debug_vis=False,
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

    ray_caster_cam = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",                                                                       
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.08),   # front of robot, slightly above base
        ),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=10, vertical_fov_range=[-30, 10], horizontal_fov_range=[-55, 55], horizontal_res=2.0
        ),
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    
    ray_caster_lidar_blind_side = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",                                                                       
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.08),   # front of robot, slightly above base
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-5, 0.0),
            horizontal_fov_range=(-125.0, 125.0),
            horizontal_res=5.0, # ray every 5° → 18 rays
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


#####################################################################
#
#                           MDP settings
#
#####################################################################

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


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        pose_command   = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        base_lin_vel   = ObsTerm(func=mdp.base_lin_vel,       clip=(-1.0,1.0))
        base_yaw_rate  = ObsTerm(func=mdp.base_yaw_rate,      clip=(-2.0, 2.0))
        ray_caster_cam = ObsTerm(func=mdp.ray_caster_lidar,   clip=(0.0,1.0), params={"asset_cfg": SceneEntityCfg("ray_caster_cam")})
        last_action    = ObsTerm(func=mdp.last_action,        clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""
        pose_command   = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        base_lin_vel   = ObsTerm(func=mdp.base_lin_vel,       clip=(-1.0,1.0))
        base_yaw_rate  = ObsTerm(func=mdp.base_yaw_rate,      clip=(-2.0, 2.0))
        ray_caster_cam = ObsTerm(func=mdp.ray_caster_lidar,   clip=(0.0,1.0), params={"asset_cfg": SceneEntityCfg("ray_caster_cam")})
        lidar_left     = ObsTerm(func=mdp.ray_caster_lidar,   clip=(0.0,1.0), params={"asset_cfg": SceneEntityCfg("ray_caster_lidar_blind_side")})
        last_action    = ObsTerm(func=mdp.last_action,        clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Add mdp.undesired_contacts for collision reward, if required. Skipped here.

    # is_alive = RewTerm(func=mdp.is_alive, weight=-0.05)
    # goal_distance = RewTerm(func=mdp.goal_distance_penalty, weight=-0.2)
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)
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
    position_tracking_precision = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 0.05, "command_name": "goal_pose"},
    )
    orientation_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "goal_pose"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Make Agent Look Where It is Going
    blind_spot_vel = RewTerm(func=mdp.blind_spot_velocity_penalty, weight=-0.2)
    heading_vel_align = RewTerm(func=mdp.heading_velocity_alignment, weight=-0.3)



    # Non-contributing reward term. Used to track success
    track_success = RewTerm(
        func=mdp.track_goal_reached, 
        params={
            "promote_threshold": 0.05,
        },
        weight=1.0
    )


    

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
        resampling_time_range=(5.0, 5.0),
        simple_heading=False,
        debug_vis=True,
    )

@configclass
class CurriculumCfg:
    """Curriculum for the rosbot"""

    terrain_level = CurriculumTerm(func=mdp.terrain_levels_nav_success_based)

@configclass
class RosbotActionCfg:
    """Actions for the rosbot."""

    base_twist: mdp.SE2BaseMecanumDriveCfg = mdp.SE2BaseMecanumDriveCfg(
        wheel_radius=0.05,     
        half_wheelbase=0.125,   
        half_track=0.105,       
        o_pattern=False,        
        animate_wheels=False,
    )

@configclass
class RosbotDifferetialActionCfg:
    """Actions for the rosbot."""

    base_twist: mdp.DifferentialDriveCfg = mdp.DifferentialDriveCfg(                                                                              
        wheel_radius=0.05,                                                                                                                        
        half_track=0.105,                                                                                                                         
        animate_wheels=False,
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

    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 3 # 40 Hz
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 200000 # For large number for flat patches


@configclass
class RosbotNavBoxTerrainEnvPLAYCfg(RosbotNavBoxTerrainEnvCfg):
    """
    Play config for RosbotNavFlatTerrainEnvCfg
    """
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 3
        self.scene.ray_caster_cam.debug_vis = True
        self.scene.ray_caster_lidar_blind_side.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.actions.base_twist.animate_wheels = True


@configclass
class RosbotNavBoxTerrainDiffEnvCfg(RosbotNavBoxTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions : RosbotDifferetialActionCfg = RosbotDifferetialActionCfg()


@configclass
class RosbotNavBoxTerrainDiffEnvPLAYCfg(RosbotNavBoxTerrainDiffEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 3
        self.scene.ray_caster_cam.debug_vis = True
        self.scene.ray_caster_lidar_blind_side.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.actions.base_twist.animate_wheels = True

@configclass
class RosbotNavFlatTerrainEnvCfg(RosbotNavBoxTerrainEnvCfg):
    """
    This is a test environment. Designed to test & train RosbotNavBoxTerrainEnvCfg on flat terrain.
    """
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.curriculum = False
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
        self.scene.ray_caster_cam_front = None
        self.scene.ray_caster_cam_left = None
        self.scene.ray_caster_cam_right = None
        self.observations.policy.ray_caster = None
        self.terminations.terrain_contact = None
        self.curriculum = None
        self.rewards.lateral_vel = None

@configclass
class RosbotNavFlatTerrainEnvPLAYCfg(RosbotNavFlatTerrainEnvCfg):
    """
    This is a test environment. Designed to test & train RosbotNavBoxTerrainEnvCfg on flat terrain.
    """
    
    def __post_init__(self):
        super().__post_init__()
