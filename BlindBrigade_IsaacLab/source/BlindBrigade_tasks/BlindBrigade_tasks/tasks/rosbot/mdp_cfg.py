from isaaclab.managers import (
    SceneEntityCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurriculumTerm,
)
from isaaclab.utils import configclass
from isaaclab.envs.mdp.terminations import illegal_contact
from isaaclab_tasks.manager_based.navigation.mdp import (
    position_command_error_tanh,
)
# from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg
from BlindBrigade_tasks.common import mdp


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

        pose_command  = ObsTerm(func=mdp.generated_commands,     params={"command_name": "goal_pose"})
        base_lin_vel  = ObsTerm(func=mdp.base_lin_vel,           clip=(-1.0, 1.0))
        base_yaw_rate = ObsTerm(func=mdp.base_yaw_rate,          clip=(-2.0, 2.0))
        height_scan   = ObsTerm(func=mdp.height_scan_normalized, clip=(-1.0, 1.0), params={"sensor_cfg": SceneEntityCfg("ray_caster_cam")})
        last_action   = ObsTerm(func=mdp.last_action,            clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        pose_command  = ObsTerm(func=mdp.generated_commands,     params={"command_name": "goal_pose"})
        base_lin_vel  = ObsTerm(func=mdp.base_lin_vel,           clip=(-1.0, 1.0))
        base_yaw_rate = ObsTerm(func=mdp.base_yaw_rate,          clip=(-2.0, 2.0))
        height_scan   = ObsTerm(func=mdp.height_scan_normalized, clip=(-1.0, 1.0), params={"sensor_cfg": SceneEntityCfg("ray_caster_cam")})
        last_action   = ObsTerm(func=mdp.last_action,            clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightScanCfg(ObsGroup):
        """Flat RayCaster state for MLP branch of CNN policy."""
        height_scan = ObsTerm(
            func=mdp.height_scan_normalized,
            clip=(-1.0, 1.0), 
            params={"sensor_cfg": SceneEntityCfg("ray_caster_cam")}
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioceptiveCfg(ObsGroup):
        """Flat state for MLP branch of CNN policy."""

        pose_command  = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        base_lin_vel  = ObsTerm(func=mdp.base_lin_vel,  clip=(-1.0, 1.0))
        base_yaw_rate = ObsTerm(func=mdp.base_yaw_rate, clip=(-2.0, 2.0))
        last_action   = ObsTerm(func=mdp.last_action,   clip=(-1.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # → (B, 9)

    @configclass
    class ExteroceptiveRayCasterCfg(ObsGroup):
        """Downward height map for CNN branch."""

        depth_map = ObsTerm(
            func=mdp.ray_caster_image,
            clip=(0.0, 2.0),
            params={"sensor_cfg": SceneEntityCfg("ray_caster_cam"), "grid_h": 16, "grid_w": 16},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # single 4D term → stays (B, 1, 64, 64)
    
    @configclass
    class ExteroceptiveCameraCfg(ObsGroup):
        """Downward height map for CNN branch."""

        depth_map = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("zed2_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # single 4D term → stays (B, 1, 64, 64)

    @configclass
    class ExteroceptiveCameraFlatCfg(ObsGroup):
        """Flattened ZED2 depth image for MLP encoder branch — outputs (B, H*W)."""

        depth_map = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("zed2_camera"), "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # single 1D term → stays (B, H*W)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    proprioceptive: ProprioceptiveCfg = ProprioceptiveCfg()
    # exteroceptive: ExteroceptiveCameraCfg = ExteroceptiveCameraCfg()
    # exteroceptive_flat: ExteroceptiveCameraFlatCfg = ExteroceptiveCameraFlatCfg()
    # heightscan: HeightScanCfg = HeightScanCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)

    # Linear distance penalty — provides a consistent gradient at all distances.
    position_linear = RewTerm(func=mdp.goal_distance_penalty, weight=-0.3)

    # Coarse position tracking (tanh kernel, wide std for medium-range signal)
    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "goal_pose"},
    )

    # Fine-grained position tracking near the goal
    position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 0.5, "command_name": "goal_pose"},
    )

    # Strong bonus that ramps up within 0.5 m of the goal
    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=5.0,
        params={"threshold": 0.5},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Penalize velocity directed toward detected obstacles.
    obstacle_approach = RewTerm(
        func=mdp.obstacle_approach_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("ray_caster_cam"),
            "danger_radius": 0.7,
            "safe_dist_normalized": 0.85,
        },
    )

    # Non-contributing reward term. Used to track success for curriculum.
    track_success = RewTerm(
        func=mdp.track_goal_reached,
        params={
            "goal_dist_threshold": 0.1,
        },
        weight=1.0,
    )


@configclass
class SRURewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)

    lateral_movement = RewTerm(func=mdp.lateral_movement, weight=-0.1)

    backward_movement_penalty = RewTerm(func=mdp.backward_movement_penalty, weight=-0.1)

    rot_movement = RewTerm(func=mdp.rot_movement, weight=-1e-5)

    # Goal rewards
    reach_goal_xy_soft = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=0.25,
        params={"command_name": "goal_pose", "sigmoid": 2.5, "T_r": 1.0, "probability": 0.01},
    )

    reach_goal_xy_tight = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=1.5,
        params={"command_name": "goal_pose", "sigmoid": 0.25, "T_r": 0.1, "probability": 0.01},
    )
    
    heading_velocity_alignment = RewTerm(
        func=mdp.heading_velocity_alignment, 
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_speed": 0.5},
    )

    action_rate = RewTerm(func=mdp.action_rate_l1, weight=-0.2)

    # Non-contributing reward term. Used to track success for curriculum.
    track_success = RewTerm(
        func=mdp.track_goal_reached,
        params={
            "goal_dist_threshold": 0.1,
        },
        weight=1.0,
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_contact = DoneTerm(
        func=illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("robot_contact_sensor"),
        },
    )


@configclass
class CommandCfg:
    """Commands for the rosbot."""

    goal_pose = mdp.TerrainBasedPosition2dCommandCfg(
        asset_name="robot",
        ranges=mdp.TerrainBasedPosition2dCommandCfg.Ranges(
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(1e9, 1e9),   # Large values to disable automatic resampling
        goal_reached_threshold=None,        # None to disable goal resampling on reached
        debug_vis=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum for the rosbot."""

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
