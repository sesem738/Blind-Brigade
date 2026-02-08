# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.envs.mdp.terminations import illegal_contact
from isaaclab.utils import configclass
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshRepeatedBoxesTerrainCfg,
    FlatPatchSamplingCfg
)

from . import mdp

##
# Pre-defined configs
##
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg
from BlindBrigade_assets.robot.rosbot_xl_mecanum import ROSBOT_XL_CFG
from wheeledlab.envs.mdp.observations import root_euler_xyz
from .mdp.actions import SE2BaseMecanumDriveCfg

import torch
from isaaclab.envs import ManagerBasedEnv

##
# Scene definition
##

@configclass
class ROSBotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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
    
    # Ground and obstacles in generated terrain consist of a single mesh
    # This makes detecting collision with just obstacles hard 
    # Workaround: 
    # - Added a cube mesh starting just above the base level of wheels
    # - Check for contact forces w.r.t. the cube mesh (which remains above ground)
    robot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/ground"],
    )

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.filter_collisions = True
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.01)
        )

        # 25 x 25 = 625 subterrains
        # self.terrain.terrain_generator.num_rows = 25
        # self.terrain.terrain_generator.num_cols = 25
        self.terrain.terrain_generator.num_rows = 2 # TODO: Remove this, only for debugging
        self.terrain.terrain_generator.num_cols = 2

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

def goal_relative_xyz(env : ManagerBasedEnv):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]  # we only need the x, y coordinates
    rel_pos = goal_pos - pos[:, :2]
    return torch.nan_to_num(rel_pos, nan=0)

def goal_distance(env: ManagerBasedEnv) -> torch.Tensor:
    pos = env.scene["robot"].data.root_pos_w[:, :2]
    goal = env.command_manager.get_command("goal_pose")[:, :2]
    return -torch.norm(goal - pos, dim=1)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TeacherCfg(ObsTerm):
        """
        Observations for policy group.
        """
        goal_relative_xyz = ObsTerm(
            func=goal_relative_xyz,
        )
        world_euler_xyz = ObsTerm(
            func=root_euler_xyz,
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-10., 10.))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-10., 10.))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1., 1.)
        )
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        goal_relative_xyz = ObsTerm(
            func=goal_relative_xyz,
        )
        world_euler_xyz = ObsTerm(
            func=root_euler_xyz,
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-10., 10.))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-10., 10.))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1., 1.)
        )
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    distance = RewTerm(func=goal_distance,weight=1.0)


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

##
# Environment configuration
##

@configclass
class ElevationCommandCfg:
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

    base_twist: SE2BaseMecanumDriveCfg = SE2BaseMecanumDriveCfg(
        wheel_radius=0.05,     
        half_wheelbase=0.125,   
        half_track=0.105,       
        o_pattern=False,        
        animate_wheels=True,
    )

@configclass
class RosbotEnvCfg(ManagerBasedRLEnvCfg):

    seed: int = 42

    # Scene settings
    scene: ROSBotSceneCfg = ROSBotSceneCfg(num_envs=4, env_spacing=0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: RosbotActionCfg = RosbotActionCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = ElevationCommandCfg = ElevationCommandCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
