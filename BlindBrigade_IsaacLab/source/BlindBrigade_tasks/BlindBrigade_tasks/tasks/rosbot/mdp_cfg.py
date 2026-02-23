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
    heading_command_error_abs,
)
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg
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
        height_scan   = ObsTerm(func=mdp.height_scan_normalized, clip=(0.0, 2.0), params={"sensor_cfg": SceneEntityCfg("ray_caster_cam")})
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
        height_scan   = ObsTerm(func=mdp.height_scan_normalized, clip=(0.0, 2.0), params={"sensor_cfg": SceneEntityCfg("ray_caster_cam")})
        last_action   = ObsTerm(func=mdp.last_action,            clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightScanCfg(ObsGroup):
        """Flat RayCaster state for MLP branch of CNN policy."""
        height_scan = ObsTerm(
            func=mdp.height_scan_normalized,
            clip=(0.0, 2.0), 
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
            clip=(0.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("ray_caster_cam")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # single 4D term → stays (B, 1, 64, 64)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    proprioceptive: ProprioceptiveCfg = ProprioceptiveCfg()
    exteroceptive: ExteroceptiveRayCasterCfg = ExteroceptiveRayCasterCfg()
    heightscan: HeightScanCfg = HeightScanCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)
    position_linear = RewTerm(func=mdp.goal_distance_penalty, weight=-0.1)
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

    # Penalize velocity directed toward detected obstacles.
    # Moving parallel to obstacles (squeezing through a gap) is not penalized —
    # only actively closing the gap is. Scale with danger_radius and safe_dist_normalized
    # to tune how early and how aggressively the avoidance kicks in.
    obstacle_approach = RewTerm(
        func=mdp.obstacle_approach_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("ray_caster_cam"),
            "danger_radius": 0.7,
            "safe_dist_normalized": 0.85,
        },
    )

    # Make Agent Look Where it is Going
    heading_vel_align = RewTerm(func=mdp.heading_velocity_alignment, weight=-0.3)

    # Non-contributing reward term. Used to track success
    track_success = RewTerm(
        func=mdp.track_goal_reached,
        params={
            "promote_threshold": 0.05,
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
