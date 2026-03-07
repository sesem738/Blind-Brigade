from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, FlatPatchSamplingCfg
from isaaclab.utils import configclass

from .scene_cfg import MushrSceneCfg
from .mdp_cfg import (
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    EventCfg,
    CommandCfg,
    CurriculumCfg,
    MushrActionCfg,
)


@configclass
class MushrNavBoxTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """MuSHR navigation with box obstacles."""

    seed: int = 42

    scene: MushrSceneCfg = MushrSceneCfg(num_envs=4, env_spacing=0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: MushrActionCfg = MushrActionCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4  # 50 Hz control (matching WheeledLab mushr tasks)
        self.episode_length_s = 10
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 200  # 200 Hz physics (matching WheeledLab)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 200000


@configclass
class MushrNavBoxTerrainEnvPLAYCfg(MushrNavBoxTerrainEnvCfg):
    """Play config for MuSHR box terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.difficulty_range = (0.5, 1.0)
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 6
        self.scene.ray_caster_cam.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.curriculum = None


@configclass
class MushrNavFlatTerrainEnvCfg(MushrNavBoxTerrainEnvCfg):
    """MuSHR navigation on flat terrain (no exteroception)."""

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
        self.observations.policy = None
        self.observations.critic = None
        self.observations.exteroceptive = None
        self.curriculum = None
        self.scene.robot_contact_sensor = None
        self.terminations.terrain_contact = None
        self.scene.ray_caster_cam = None

        # proprioceptive only
        self.observations.policy = None
        self.observations.critic = None
        self.observations.exteroceptive = None
        self.observations.heightscan = None


@configclass
class MushrNavFlatTerrainEnvPLAYCfg(MushrNavFlatTerrainEnvCfg):
    """Play config for MuSHR flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 6
        self.scene.terrain.debug_vis = True
