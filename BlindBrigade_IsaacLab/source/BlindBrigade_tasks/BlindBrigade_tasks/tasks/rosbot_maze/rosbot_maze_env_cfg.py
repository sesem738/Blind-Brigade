from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .scene_cfg import ROSBotMazeSceneCfg
from .mdp_cfg import (
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    EventCfg,
    CommandCfg,
    CurriculumCfg,
    RosbotActionCfg,
)


@configclass
class RosbotNavMazeEnvCfg(ManagerBasedRLEnvCfg):
    """Rosbot navigation in maze terrain."""

    seed: int = 42

    scene: ROSBotMazeSceneCfg = ROSBotMazeSceneCfg(num_envs=4, env_spacing=0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: RosbotActionCfg = RosbotActionCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 3  # 40 Hz
        self.episode_length_s = 20
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 200000


@configclass
class RosbotNavMazeEnvPLAYCfg(RosbotNavMazeEnvCfg):
    """Play config for RosbotNavMazeEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 100000
        self.scene.terrain.max_init_terrain_level = 3
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 3
        self.scene.ray_caster_cam.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.actions.base_twist.animate_wheels = True
