import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, CameraCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshRepeatedBoxesTerrainCfg,
    FlatPatchSamplingCfg,
)
from isaaclab.utils import configclass

from BlindBrigade_assets.robot.rosbot_xl_mecanum import ROSBOT_XL_CFG


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
                "boxes": MeshRepeatedBoxesTerrainCfg(
                    platform_width=0,
                    platform_height=0,
                    object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=0, height=0.25, size=(0.05, 0.05), max_yx_angle=0.0, degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=5, height=1.0, size=(2.0, 2.0), max_yx_angle=60.0, degrees=True
                    ),
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01
                        ),
                        "target": FlatPatchSamplingCfg(
                            num_patches=50, patch_radius=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], max_height_diff=0.01
                        ),
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

    zed2_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/zed2_camera",
        update_period=1.0 / 30.0,  # 30 Hz, matching ZED 2 HD1080 mode
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,           # mm
            horizontal_aperture=6.05,    # mm  2 × 2.12 × tan(110/2°) ≈ 6.05 mm
            vertical_aperture=2.97,      # mm  2 × 2.12 × tan(70/2°) ≈ 6.05 mm
            clipping_range=(0.3, 20.0),  # metres — ZED 2 depth range
            f_stop=1.06,
        ),
        width=662,   # low res (662x376) @15/30/60/100fps binning 4x4 mode per camera
        height=376,
        offset=CameraCfg.OffsetCfg(
            pos=(0.17, 0.0, 0.15),      # 170 mm forward, centred, 150 mm up
            rot=(0.5, -0.5, 0.5, -0.5),  # rotate so camera faces +X (forward)
            convention="ros",
        ),
    )

    ray_caster_cam = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.15, 0.0, 20.0),  # front of robot, slightly above base
        ),
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05,
            size=(3.15, 3.15),  # arange includes both endpoints: (3.15/0.05)+1 = 64 pts/axis → 64×64
        ),
        max_distance=3.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.filter_collisions = True
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.01))

        # 25 x 25 = 625 subterrains
        self.terrain.terrain_generator.num_rows = 25
        self.terrain.terrain_generator.num_cols = 25
