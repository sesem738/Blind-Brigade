"""MazeTerrainGenerator and MazeTerrainImporter: clean subclasses of Isaac Lab's
terrain system that collect height-field masks produced by maze_terrain().

Instead of monkey-patching (as the SRU codebase does), we use the built-in
``class_type`` extension points on TerrainGeneratorCfg / TerrainImporterCfg.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml

if TYPE_CHECKING:
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

logger = logging.getLogger(__name__)

# Height field attribute names stored on the terrain cfg by maze_terrain()
_HEIGHT_FIELD_ATTRS = [
    "height_field_visual",
    "height_field_valid_mask",
    "height_field_platform_mask",
    "height_field_spawn_mask",
]


# =============================================================================
# MazeTerrainGenerator
# =============================================================================

class MazeTerrainGenerator(TerrainGenerator):
    """TerrainGenerator subclass that collects height-field masks from maze terrain configs.

    The parent's ``_get_terrain_mesh`` copies the cfg and calls ``cfg.function(difficulty, cfg)``.
    The generation function (``maze_terrain``) stores masks on that *copy*. However,
    ``_add_sub_terrain`` receives the *original* cfg (not the copy), so the masks are lost.

    We override ``_get_terrain_mesh`` to intercept the masks from the cfg copy before they
    go out of scope.
    """

    def __init__(self, cfg: MazeTerrainGeneratorCfg, device: str = "cpu"):
        # Pre-allocate collection lists *before* super().__init__ runs generation
        self._height_field_lists: dict[str, list[torch.Tensor]] = {
            attr: [] for attr in _HEIGHT_FIELD_ATTRS
        }
        # Run parent init — this calls _generate_curriculum_terrains / _generate_random_terrains
        # which calls our overridden _get_terrain_mesh for each tile
        super().__init__(cfg, device)

        # Consolidate collected height fields into tensors
        for attr in _HEIGHT_FIELD_ATTRS:
            data_list = self._height_field_lists[attr]
            if data_list:
                setattr(self, attr, torch.cat(data_list, dim=0))
            else:
                setattr(self, attr, None)
        del self._height_field_lists

    def _get_terrain_mesh(
        self, difficulty: float, cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Override to collect height-field masks from the cfg copy.

        Reproduces the parent's logic but intercepts the cfg copy after generation
        to extract masks stored by maze_terrain().
        """
        # Copy configuration (same as parent)
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed

        # Generate hash for caching
        # Clear non-serializable fields before hashing
        saved_masks = {}
        for attr in _HEIGHT_FIELD_ATTRS:
            if hasattr(cfg, attr):
                saved_masks[attr] = getattr(cfg, attr)
                setattr(cfg, attr, None)
        saved_rng = getattr(cfg, "rng", None)
        if hasattr(cfg, "rng"):
            cfg.rng = None

        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # Check cache
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            return mesh, origin

        # Set RNG for reproducible generation
        if hasattr(cfg, "rng") and self.cfg.seed is not None:
            cfg.rng = self.np_rng.spawn(1)[0]

        # Generate the terrain — maze_terrain() stores masks on cfg
        meshes, origin = cfg.function(difficulty, cfg)
        if not isinstance(meshes, list):
            meshes = [meshes]
        mesh = trimesh.util.concatenate(meshes)

        # Center the mesh
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]

        # *** KEY: Collect height field masks from the cfg copy ***
        for attr in _HEIGHT_FIELD_ATTRS:
            if hasattr(cfg, attr):
                data = getattr(cfg, attr)
                if data is not None:
                    self._height_field_lists[attr].append(data)
                    setattr(cfg, attr, None)

        # Cache if enabled
        if self.cfg.use_cache:
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)

        return mesh, origin


@configclass
class MazeTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration that routes to MazeTerrainGenerator via class_type."""

    class_type: type = MazeTerrainGenerator


# =============================================================================
# MazeTerrainImporter
# =============================================================================

# Module-level storage for passing data from generator to importer.
# Works because TerrainImporter creates TerrainGenerator synchronously in __init__.
_height_field_storage: dict[str, torch.Tensor | None] = {
    attr: None for attr in _HEIGHT_FIELD_ATTRS
}


class MazeTerrainImporter(TerrainImporter):
    """TerrainImporter subclass that captures height-field masks from MazeTerrainGenerator.

    The parent's ``__init__`` creates the generator via ``cfg.terrain_generator.class_type(...)``
    but does NOT store a reference to it. We use module-level shared storage to pass the
    masks from MazeTerrainGenerator to MazeTerrainImporter.
    """

    def __init__(self, cfg: MazeTerrainImporterCfg):
        # Clear shared storage
        for attr in _HEIGHT_FIELD_ATTRS:
            _height_field_storage[attr] = None

        # Call parent __init__ — this creates MazeTerrainGenerator (via class_type),
        # which populates _height_field_storage via the patched __init__
        super().__init__(cfg)

        # Capture height field data from shared storage
        for attr in _HEIGHT_FIELD_ATTRS:
            setattr(self, f"_{attr}", _height_field_storage.get(attr))

        # Log what we captured
        for attr in _HEIGHT_FIELD_ATTRS:
            val = getattr(self, f"_{attr}", None)
            if val is not None:
                logger.info(f"MazeTerrainImporter: captured {attr} with shape {val.shape}")
            else:
                logger.warning(f"MazeTerrainImporter: {attr} is None")


# Patch MazeTerrainGenerator.__init__ to push masks to shared storage
_original_maze_gen_init = MazeTerrainGenerator.__init__


def _maze_gen_init_with_storage(self, cfg, device="cpu"):
    _original_maze_gen_init(self, cfg, device)
    # After init completes, push masks to shared storage
    for attr in _HEIGHT_FIELD_ATTRS:
        _height_field_storage[attr] = getattr(self, attr, None)


MazeTerrainGenerator.__init__ = _maze_gen_init_with_storage


@configclass
class MazeTerrainImporterCfg(TerrainImporterCfg):
    """Configuration that routes to MazeTerrainImporter via class_type."""

    class_type: type = MazeTerrainImporter
