from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, FlatPatchSamplingCfg
from isaaclab.utils import configclass

from .scene_cfg import ROSBotSceneCfg
from .mdp_cfg import (
    ObservationsCfg,
    SRURewardsCfg,
    TerminationsCfg,
    EventCfg,
    CommandCfg,
    CurriculumCfg,
    RosbotActionCfg,
)


@configclass
class RosbotNavBoxTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """This env config contains a rosbot and box obstacles."""

    seed: int = 42

    # Scene settings
    scene: ROSBotSceneCfg = ROSBotSceneCfg(num_envs=4, env_spacing=0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: RosbotActionCfg = RosbotActionCfg()
    # MDP settings
    rewards: SRURewardsCfg = SRURewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 3  # 40 Hz
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 200000  # For large number of flat patches
        # Disable ZED2 camera during training — GPU rendering scales badly with num_envs.
        # Training uses ray_caster_cam (efficient) for depth observations.
        # self.scene.zed2_camera = None


@configclass
class RosbotNavBoxTerrainEnvPLAYCfg(RosbotNavBoxTerrainEnvCfg):
    """Play config for RosbotNavBoxTerrainEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        # self.episode_length_s = 100000
        self.scene.terrain.terrain_generator.difficulty_range = (0.5, 1.0)
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 6
        self.scene.ray_caster_cam.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.actions.base_twist.animate_wheels = True
        self.curriculum = None


@configclass
class RosbotNavFlatTerrainEnvCfg(RosbotNavBoxTerrainEnvCfg):
    """Test environment for training on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": MeshPlaneTerrainCfg(
                size=(8.0, 8.0),
                flat_patch_sampling={
                    "init_pos": FlatPatchSamplingCfg(
                        num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01
                    ),
                    "target": FlatPatchSamplingCfg(
                        num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01
                    ),
                },
            ),
        }
        self.terminations.terrain_contact = None
        self.curriculum = None


@configclass
class RosbotNavFlatTerrainEnvPLAYCfg(RosbotNavFlatTerrainEnvCfg):
    """Play config for RosbotNavFlatTerrainEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
