import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshRepeatedBoxesTerrainCfg,
    FlatPatchSamplingCfg
)
from isaaclab.sensors import RayCasterCfg, patterns

from . import mdp

##
# Pre-defined configs
##

from .mdp.events import reset_root_state
from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg
import isaaclab.utils.math as math_utils

import torch


@configclass
class GuideSceneCfg(InteractiveSceneCfg):
    """
    Scene Configuration

    This configuration sets up two mobile vehicles in an obstacle course.
    """

    # Ground Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            sub_terrains={
                # "flat": MeshPlaneTerrainCfg(proportion=0.2),
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
                        "root_blind_spawn": FlatPatchSamplingCfg(
                            num_patches=1, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
                        "root_priv_spawn": FlatPatchSamplingCfg(
                            num_patches=1, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
                        "target_spawn": FlatPatchSamplingCfg(
                            num_patches=1, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01),
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

    # Lights

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    guide: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/guide")
    blind: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/blind")

    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/guide/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 20.0)
        ),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(size=[100,100], resolution=0.1),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.guide.init_state = self.guide.init_state.replace(
            pos=(0.0, 0.0, 0.1)
        )
        self.blind.init_state = self.blind.init_state.replace(
            pos=(0.1, 0.1, 0.1)
        )
        
        self.terrain.terrain_generator.num_rows = int(self.num_envs / 2)
        self.terrain.terrain_generator.num_cols = int(self.num_envs / 2)
        self.terrain.terrain_generator.sub_terrains["boxes"].flat_patch_sampling["root_blind_spawn"].num_patches = self.num_envs
        self.terrain.terrain_generator.sub_terrains["boxes"].flat_patch_sampling["root_priv_spawn"].num_patches = self.num_envs
        self.terrain.terrain_generator.sub_terrains["boxes"].flat_patch_sampling["target_spawn"].num_patches = self.num_envs


##
# MDP settings
##

def root_euler_xyz(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root Euler XYZ angles in the environment frame."""
    xyz_tuple = math_utils.euler_xyz_from_quat(mdp.root_quat_w(env, asset_cfg=asset_cfg))
    return torch.stack(xyz_tuple, dim=-1)

def goal_relative_xyz(env : ManagerBasedRLEnv):
    pos = mdp.root_pos_w(env, asset_cfg=SceneEntityCfg("guide"))
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]  # we only need the x, y coordinates
    rel_pos = goal_pos - pos[:, :2]
    return torch.nan_to_num(rel_pos, nan=0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        goal_relative_xyz = ObsTerm(
            func=goal_relative_xyz,
        )
        world_euler_xyz = ObsTerm(
            func=root_euler_xyz,
            params={"asset_cfg": SceneEntityCfg("guide")},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("guide")}, clip=(-10., 10.))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("guide")}, clip=(-10., 10.))
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
class CommandCfg:
    """Configuration for the elevation commands."""

    goal_pose = UniformPose2dCommandCfg(
        asset_name="blind",
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-19.0, 19.0),
            pos_y=(-19.0, 19.0),
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(10.0, 10.0),
        simple_heading=True,
        debug_vis=True
    )

@configclass
class EventCfg:
    """Configuration for the events."""

    set_goal = EventTerm(
        func=reset_root_state,
        mode="reset",
        params={
            "blind_asset_cfg": SceneEntityCfg("blind"),
            "guide_asset_cfg": SceneEntityCfg("guide"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class GuideEnvCfg(ManagerBasedRLEnvCfg):
    """ Configuration for the Guide Obstacle Avoidance Environment"""

    seed: int = 42

    # Scene settings
    scene: GuideSceneCfg = GuideSceneCfg(num_envs=4, env_spacing=0)

    # MDP Settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()
    commands: CommandCfg = CommandCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # Fix asset_name for actions to match scene entity
        self.actions.throttle_steer.asset_name = "guide"

        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

