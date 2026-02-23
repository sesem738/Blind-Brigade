# BB-rosbot-maze

Rosbot navigation task using procedural maze terrains with explicit valid-position masks for goal/spawn sampling.

## Motivation

The original rosbot navigation task (`rosbot/`) uses box obstacles with `FlatPatchSamplingCfg` to find valid spawn/goal positions by scanning the terrain mesh for flat patches. This works but is limited to simple obstacle layouts.

The SRU navigation sim has a more sophisticated terrain system (mazes, pits, stairs, random obstacles) that generates explicit `valid_mask` / `spawn_mask` during terrain creation — no flat-patch scanning needed. This task brings those terrains to the rosbot while keeping its existing robot, actions, observations, and rewards.

## Architecture

### Terrain Pipeline

```
HfMazeTerrainCfg          # Terrain config (maze params, wall height, grid size, etc.)
    |
    v
maze_terrain()             # Generates height field + valid_mask + spawn_mask
    |                        (stores masks on cfg copy during generation)
    v
MazeTerrainGenerator       # Subclass of TerrainGenerator
    |                        Overrides _get_terrain_mesh to intercept masks
    |                        from the cfg copy before it goes out of scope
    v
MazeTerrainImporter        # Subclass of TerrainImporter
                             Captures masks via module-level shared storage
                             Exposes _height_field_visual, _height_field_valid_mask, etc.
```

### Command Pipeline

```
ValidMaskPose2dCommand     # Subclass of CommandTerm
    |
    +-- PositionSampler    # Pre-computes valid position tables per terrain
    |     +-- sample()           -> goal (x, y, z) from valid_mask
    |     +-- sample_spawn()     -> spawn (x, y, z) from spawn_mask
    |
    +-- _resample_command()
    |     1. Samples goal position from valid_mask
    |     2. Samples spawn position from spawn_mask
    |     3. Updates terrain.env_origins with spawn position
    |     4. Samples heading (random or toward goal)
    |
    +-- _update_command()
    |     Transforms goal to body frame -> [pos_x_b, pos_y_b, pos_z_b, heading_error]
    |     Same (num_envs, 4) format as TerrainBasedPose2dCommand
    |
    +-- command property -> (num_envs, 4) tensor
```

### Key Design Decision: No Monkey-Patching

The SRU codebase monkey-patches `TerrainGenerator.__init__` and `TerrainImporter.__init__` to inject mask collection. We instead use Isaac Lab's built-in `class_type` extension points:

- `MazeTerrainGeneratorCfg.class_type = MazeTerrainGenerator`
- `MazeTerrainImporterCfg.class_type = MazeTerrainImporter`

The one subtlety: the parent's `_get_terrain_mesh` creates a **copy** of the sub-terrain cfg, and `maze_terrain()` stores masks on that copy. But `_add_sub_terrain` receives the **original** cfg, not the copy. So we override `_get_terrain_mesh` (not `_add_sub_terrain`) to intercept the masks from the copy before it goes out of scope.

## Differences from rosbot/

| Aspect | `rosbot/` | `rosbot_maze/` |
|---|---|---|
| Terrain | `TerrainImporterCfg` + `MeshRepeatedBoxesTerrainCfg` | `MazeTerrainImporterCfg` + `HfMazeTerrainCfg` |
| Position sampling | `FlatPatchSamplingCfg` (scans mesh for flat patches) | `valid_mask` / `spawn_mask` (generated during terrain creation) |
| Commands | `TerrainBasedPose2dCommandCfg` | `ValidMaskPose2dCommandCfg` |
| Reset event | `reset_root_state_from_terrain` (samples from flat patches) | `reset_root_state_uniform` (env_origins set by command) |
| Observations | Identical | Identical |
| Rewards | Identical | Identical |
| Actions | Identical | Identical |
| Curriculum | Identical | Identical |

## Reset Ordering

Isaac Lab calls command reset **before** event reset. This matters because:

1. `ValidMaskPose2dCommand._resample_command()` updates `terrain.env_origins[env_ids]` with the sampled spawn position
2. `reset_root_state_uniform` then reads `env.scene.env_origins[env_ids]` to place the robot

## Vendored Code

The maze terrain generation code is vendored from the SRU navigation sim (`isaaclab_nav_task/terrains/`) into `common/terrains/`:

| File | Source |
|---|---|
| `terrain_constants.py` | Height values, padding configs, obstacle params |
| `hf_terrains_maze.py` | DFS maze generation, obstacle generators, mask creation |
| `hf_terrains_maze_cfg.py` | `HfMazeTerrainCfg` dataclass |

## Usage

```bash
# Train
python scripts/rsl_rl/train.py --task BB-rosbot-maze-v0 --num_envs 4096

# Play / teleop
python scripts/teleop_agent.py --task BB-rosbot-maze-PLAY-v0 --num_envs 1
```

## Terrain Configuration

The maze terrain is configured in `scene_cfg.py`. Key parameters:

```python
HfMazeTerrainCfg(
    add_goal=True,              # Enable mask generation (required)
    grid_size=(4, 4),           # 4x4 maze cells in 8x8m tile
    cell_size=2.0,              # Each cell is 2m x 2m
    wall_height=1.5,            # Wall height in meters
    randomize_wall=True,        # Use random obstacle shapes instead of full walls
    non_maze_terrain=False,     # True = scattered obstacles, False = DFS maze
    stairs=False,               # Add stair/platform structures
    dynamic_obstacles=False,    # Add pit/trough obstacles
)
```

Set `use_cache=False` in the generator config — cached terrains skip generation, so masks won't be created.
