import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from BlindBrigade_assets.robot.rosbot_xl import ROSBOT_XL_MECANUM_CFG
from BlindBrigade_tasks.common.terrains import (
    MazeTerrainImporterCfg,
    MazeTerrainGeneratorCfg,
    HfMazeTerrainCfg,
)


@configclass
class ROSBotMazeSceneCfg(InteractiveSceneCfg):
    """Scene with maze terrain for rosbot navigation.

    Uses MazeTerrainImporter which collects valid_mask / spawn_mask from the
    terrain generation step. No FlatPatchSamplingCfg needed — position sampling
    is handled by ValidMaskPose2dCommand using the masks directly.
    """

    terrain = MazeTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        max_init_terrain_level=0,
        use_terrain_origins=True,
        terrain_generator=MazeTerrainGeneratorCfg(
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
                "maze": HfMazeTerrainCfg(
                    add_goal=True,
                    add_noise_to_flat=False,
                    randomize_wall=True,
                    random_wall_ratio=0.5,
                    non_maze_terrain=False,
                    grid_size=(4, 4),
                    cell_size=2.0,
                    wall_height=1.5,
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

    robot: ArticulationCfg = ROSBOT_XL_MECANUM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROSBOT_XL_MECANUM_CFG.spawn.replace(activate_contact_sensors=True),
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
            pos=(0.0, 0.0, 2.0),
        ),
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.2,
            size=(3.0, 3.0),
        ),
        max_distance=2.1,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.filter_collisions = True
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.01))

        self.terrain.terrain_generator.num_rows = 25
        self.terrain.terrain_generator.num_cols = 25
