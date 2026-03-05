"""
Task description

Task: Blind must reach its goal. Guide helps it get there safely.

Information design:

  ┌──────────────┬──────────────────────────────────────────────────┬────────────────────────────────────────────┐                                  
  │              │                  Guide (policy)                  │               Blind (policy)               │
  ├──────────────┼──────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Blind's goal │ relative to guide (so guide knows where to lead) │ not observed                               │
  ├──────────────┼──────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Partner      │ front ray caster                                 │ nothing                                    │
  ├──────────────┼──────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Obstacles    │ front ray caster                                 │ nothing                                    │
  ├──────────────┼──────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Self         │ lin_vel, yaw_rate                                │ nothing                                    │
  └──────────────┴──────────────────────────────────────────────────┴────────────────────────────────────────────┘

  The blind should not observe its own goal delta — otherwise it doesn't need the guide. The blind only sees the guide's
  relative position and learns to follow it. The guide sees everything: the blind's goal, the blind's position, and the 
  obstacles. The guide's job is to position itself such that the blind can follow it to the goal safely.

Reward: Both robots share the reward when the blind reaches the goal. Guide gets additional reward for staying near the 
blind (so it doesn't just run to the goal alone).

Leader-follower dependency:
  Goal → Guide (sees goal + obstacles + blind) → positions itself → Blind → follows
"""

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
    MultiMeshRayCasterCfg,
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Domain Randomization Considerations: 
# 1) ~Mass randomization~: Action term sets velocity directly, bypassing dynamics
# 2) ~Force~ : Ramdomly sets root velocity once, but apply_actions() overwrites root velocity on the very next step
# 3) Observation noise — add noise to sensor readings (most impactful for sim-to-real)
# 4) Action noise/delay — simulate motor latency or command imprecision
# 5) Sensor offset randomization — slight position/rotation offset on ray caster

##
# Pre-defined configs
##
from BlindBrigade_assets.robot.rosbot_xl import ROSBOT_XL_MECANUM_CFG
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

    guide: ArticulationCfg = ROSBOT_XL_MECANUM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Guide",
        spawn=ROSBOT_XL_MECANUM_CFG.spawn.replace(activate_contact_sensors=True),
    )

    blind: ArticulationCfg = ROSBOT_XL_MECANUM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Blind",
        spawn=ROSBOT_XL_MECANUM_CFG.spawn.replace(activate_contact_sensors=True),
    )
    
    guide_robot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Guide/base_link",
        update_period=0.0,
        history_length=1,
    )

    blind_robot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Blind/base_link",
        update_period=0.0,
        history_length=1,
    )

    guide_ray_caster_cam = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Guide/base_link",                                                                       
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.08),   # front of robot, slightly above base
            # rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=10, vertical_fov_range=[-30, 10], horizontal_fov_range=[-55, 55], horizontal_res=2.0
        ),
        max_distance=1.0,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Blind/base_link/collisions", track_mesh_transforms=True
            ),
        ],
        debug_vis=False,
    )

    guide_ray_caster_lidar_blind_side = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Guide/base_link",
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.08),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=5,
            vertical_fov_range=(-10, 0.0),
            horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=5.0,
        ),
        max_distance=1.0,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Blind/base_link/collisions", track_mesh_transforms=True
            ),
        ],
        debug_vis=False,
    )

    blind_ray_caster_lidar_blind_side = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Blind/base_link",
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.08),
        ),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-10, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        max_distance=1.0,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Guide/base_link/collisions", track_mesh_transforms=True
            ),
        ],
        debug_vis=False,
    )
    
    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.filter_collisions = True
        self.guide.init_state = self.guide.init_state.replace(pos=(0.0, 0.0, 0.01))
        self.blind.init_state = self.blind.init_state.replace(pos=(1.0, 0.0, 0.01))

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

    root_states = EventTerm(
        func=mdp.reset_two_robots,
        mode="reset",
        params={
            "robot1_cfg": SceneEntityCfg("guide"),
            "robot2_cfg": SceneEntityCfg("blind"),
            "min_distance": 0.5,
            "max_distance": 2.0
        },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group — everything from guide's perspective."""
        blind_goal     = ObsTerm(func=mdp.blind_goal_relative_to_guide,      params={"command_name": "goal_pose", "guide_cfg": SceneEntityCfg("guide")})
        guide_lin_vel  = ObsTerm(func=mdp.base_lin_vel,     clip=(-1.0,1.0), noise=Unoise(n_min=-0.05, n_max=0.05), params={"asset_cfg": SceneEntityCfg("guide")})
        guide_yaw_rate = ObsTerm(func=mdp.base_yaw_rate,    clip=(-2.0, 2.0),noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("guide")})
        guide_ray_cam  = ObsTerm(func=mdp.ray_caster_lidar, clip=(0.0,1.0),  noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("guide_ray_caster_cam")})
        last_action    = ObsTerm(func=mdp.last_action,      clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        pose_command    = ObsTerm(func=mdp.generated_commands,                params={"command_name": "goal_pose"})
        guide_lin_vel   = ObsTerm(func=mdp.base_lin_vel,     clip=(-1.0,1.0), params={"asset_cfg": SceneEntityCfg("guide")})
        guide_yaw_rate  = ObsTerm(func=mdp.base_yaw_rate,    clip=(-2.0, 2.0),params={"asset_cfg": SceneEntityCfg("guide")})
        blind_lin_vel   = ObsTerm(func=mdp.base_lin_vel,     clip=(-1.0,1.0), params={"asset_cfg": SceneEntityCfg("blind")})
        blind_yaw_rate  = ObsTerm(func=mdp.base_yaw_rate,    clip=(-2.0, 2.0),params={"asset_cfg": SceneEntityCfg("blind")})
        guide_ray_cam   = ObsTerm(func=mdp.ray_caster_lidar, clip=(0.0,1.0),  params={"asset_cfg": SceneEntityCfg("guide_ray_caster_cam")})
        guide_blindside = ObsTerm(func=mdp.ray_caster_lidar, clip=(0.0,1.0),  params={"asset_cfg": SceneEntityCfg("guide_ray_caster_lidar_blind_side")})
        blind_blindside = ObsTerm(func=mdp.ray_caster_lidar, clip=(0.0,1.0),  params={"asset_cfg": SceneEntityCfg("blind_ray_caster_lidar_blind_side")})
        last_action     = ObsTerm(func=mdp.last_action,      clip=(-1.0, 1.0))

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
    goal_distance = RewTerm(func=mdp.goal_distance_penalty, weight=-0.2)
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
    guide_blindside_vel = RewTerm(func=mdp.blind_spot_velocity_penalty, weight=-0.2,params={"asset_cfg": SceneEntityCfg("guide")},)

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
    guide_terrain_contact = DoneTerm(
        func=illegal_contact, 
        params={
            "threshold": 45,
            "sensor_cfg": SceneEntityCfg("guide_robot_contact_sensor"),
            },
        )
    blind_terrain_contact = DoneTerm(
        func=illegal_contact, 
        params={
            "threshold": 45,
            "sensor_cfg": SceneEntityCfg("blind_robot_contact_sensor"),
            },
        )


@configclass
class CommandCfg:
    """Commands for the rosbot."""

    goal_pose = TerrainBasedPose2dCommandCfg(
        asset_name="blind",
        ranges=TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(5.0, 5.0),
        simple_heading=False,
        debug_vis=False,
    )

@configclass
class CurriculumCfg:
    """Curriculum for the rosbot"""

    terrain_level = CurriculumTerm(func=mdp.terrain_levels_nav_success_based)

@configclass
class RosbotActionCfg:
    """Actions for the rosbot."""

    guide_base_twist: mdp.SE2BaseMecanumDriveCfg = mdp.SE2BaseMecanumDriveCfg(
        asset_name="guide",
        wheel_radius=0.05,     
        half_wheelbase=0.125,   
        half_track=0.105,       
        o_pattern=False,        
        animate_wheels=False,
    )
    blind_base_twist: mdp.SE2BaseMecanumDriveCfg = mdp.SE2BaseMecanumDriveCfg(
        asset_name="blind",
        wheel_radius=0.05,     
        half_wheelbase=0.125,   
        half_track=0.105,       
        o_pattern=False,        
        animate_wheels=False,
    )

#####################################################################
#
#                   Environment Configuration
#
#####################################################################


@configclass
class RosbotCoopNavBoxTerrainEnvCfg(ManagerBasedRLEnvCfg):
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
class RosbotCoopNavBoxTerrainEnvPLAYCfg(RosbotCoopNavBoxTerrainEnvCfg):
    """
    Play config for RosbotNavFlatTerrainEnvCfg
    """
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 3
        self.scene.guide_ray_caster_cam.debug_vis = True
        self.scene.guide_ray_caster_lidar_blind_side.debug_vis = True
        self.scene.blind_ray_caster_lidar_blind_side.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.actions.guide_base_twist.animate_wheels = True
        self.actions.blind_base_twist.animate_wheels = True
        self.commands.goal_pose.debug_vis = True
    

@configclass
class RosbotCoopNavFlatTerrainEnvCfg(RosbotCoopNavBoxTerrainEnvCfg):
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
        self.curriculum = None
        self.rewards.track_success = None

@configclass
class RosbotCoopNavFlatTerrainEnvPLAYCfg(RosbotCoopNavFlatTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.guide_ray_caster_cam.debug_vis = True
        self.scene.guide_ray_caster_lidar_blind_side.debug_vis = True
        self.scene.blind_ray_caster_lidar_blind_side.debug_vis = True
        self.terminations.guide_terrain_contact = None
        self.terminations.blind_terrain_contact = None
        self.curriculum = None
        self.rewards.track_success = None
        self.scene.terrain.debug_vis = True
        self.actions.guide_base_twist.animate_wheels = True
        self.actions.blind_base_twist.animate_wheels = True
        self.commands.goal_pose.debug_vis = True
