"""Script to teleoperate a robot in an Isaac Lab environment using keyboard.

Automatically detects the drive type (mecanum, differential, ackermann) from
the environment's action configuration and maps keyboard inputs accordingly.

Usage:
    python scripts/agents/teleop_agent.py --task=BB-rosbot-box-PLAY-v0 --num_envs=1
    python scripts/agents/teleop_agent.py --task=BB-rosbot-diff-box-PLAY-v0 --num_envs=1

Controls (all drive types):
    Arrow Up / Numpad 8   : Forward  (+v)
    Arrow Down / Numpad 2 : Backward (-v)
    Q / Numpad 7          : Yaw left  (+wz / +steering)
    E / Numpad 9          : Yaw right (-wz / -steering)

Mecanum only:
    Arrow Left / Numpad 4 : Strafe left  (+vy)
    Arrow Right / Numpad 6: Strafe right (-vy)

    L                     : Reset commands to zero
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleop agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=0.5, help="Keyboard sensitivity (0-1).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import random

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

import BlindBrigade_tasks.tasks  # noqa: F401

# Drive type constants
DRIVE_MECANUM = "mecanum"
DRIVE_DIFFERENTIAL = "differential"
DRIVE_ACKERMANN = "ackermann"
DRIVE_UNKNOWN = "unknown"


def detect_drive_type(env) -> str:
    """Detect drive type from the environment's action manager.

    Inspects the action term class to determine the robot's drive type.
    Falls back to action dimension if the class is unrecognized.
    """
    from BlindBrigade_tasks.common.mdp.actions import SE2BaseMecanumDrive, DifferentialDrive, AckermannDrive
    from isaaclab.envs.mdp.actions.non_holonomic_actions import NonHolonomicAction

    action_manager = env.unwrapped.action_manager
    for term in action_manager._terms.values():
        if isinstance(term, SE2BaseMecanumDrive):
            return DRIVE_MECANUM
        elif isinstance(term, DifferentialDrive):
            return DRIVE_DIFFERENTIAL
        elif isinstance(term, AckermannDrive):
            return DRIVE_ACKERMANN
        elif isinstance(term, NonHolonomicAction):
            return DRIVE_DIFFERENTIAL

    # Fallback: guess from action dimension
    action_dim = env.action_space.shape[-1]
    if action_dim == 3:
        return DRIVE_MECANUM
    elif action_dim == 2:
        return DRIVE_DIFFERENTIAL
    return DRIVE_UNKNOWN


def teleop_to_action(teleop_cmd: torch.Tensor, drive_type: str, action_dim: int, device) -> torch.Tensor:
    """Map Se2Keyboard output (vx, vy, wz) to the correct action layout.

    Args:
        teleop_cmd: (3,) tensor from Se2Keyboard — (vx, vy, wz).
        drive_type: One of DRIVE_MECANUM, DRIVE_DIFFERENTIAL, DRIVE_ACKERMANN.
        action_dim: Total action dimension of the environment.
        device: Torch device.

    Returns:
        (action_dim,) tensor with the teleop command mapped to the right indices.
    """
    action = torch.zeros(action_dim, device=device)
    vx, vy, wz = teleop_cmd[0], teleop_cmd[1], teleop_cmd[2]

    if drive_type == DRIVE_MECANUM:
        # (vx, vy, wz)
        action[0] = vx
        action[1] = vy
        action[2] = wz
    elif drive_type == DRIVE_DIFFERENTIAL:
        # (v, omega) — no lateral
        action[0] = vx
        action[1] = wz
    elif drive_type == DRIVE_ACKERMANN:
        # (v, steering_angle) — wz maps to steering
        action[0] = vx
        action[1] = wz
    else:
        # Best-effort: fill what fits
        n = min(3, action_dim)
        action[:n] = teleop_cmd[:n]

    return action


def main():
    """Teleop agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # detect drive type
    drive_type = detect_drive_type(env)

    action_dim = env.action_space.shape[-1]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space      : {env.action_space}")
    print(f"[INFO]: Detected drive type   : {drive_type}")
    print(f"[INFO]: Action dim            : {action_dim}")

    if drive_type == DRIVE_MECANUM:
        print("[INFO]: Controls: Up/Down=forward/back, Left/Right=strafe, Z/X=yaw")
    elif drive_type == DRIVE_DIFFERENTIAL:
        print("[INFO]: Controls: Up/Down=forward/back, Z/X=yaw (no strafe)")
    elif drive_type == DRIVE_ACKERMANN:
        print("[INFO]: Controls: Up/Down=forward/back, Z/X=steering (no strafe)")

    # create keyboard controller
    sens = args_cli.sensitivity
    keyboard = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=sens,
        v_y_sensitivity=sens,
        omega_z_sensitivity=sens,
        sim_device=device,
    ))
    # Rebind yaw keys from Z/X to Q/E
    import numpy as np
    key_map = keyboard._INPUT_KEY_MAPPING
    key_map["Q"] = key_map.pop("Z")   # yaw left
    key_map["E"] = key_map.pop("X")   # yaw right

    # visualize flat patches if available
    flat_patches = env.unwrapped.scene.terrain.flat_patches
    if flat_patches:
        vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
        for name in flat_patches:
            vis_cfg.markers[name] = sim_utils.CylinderCfg(
                radius=0.2,
                height=0.1,
                visual_material=sim_utils.GlassMdlCfg(
                    glass_color=(random.random(), random.random(), random.random())
                ),
            )
        flat_patches_visualizer = VisualizationMarkers(vis_cfg)
        all_locations = []
        all_indices = []
        for i, locations in enumerate(flat_patches.values()):
            flat = locations.view(-1, 3)
            all_locations.append(flat)
            all_indices += [i] * flat.shape[0]
        flat_patches_visualizer.visualize(torch.cat(all_locations), marker_indices=all_indices)

    # reset
    env.reset()
    keyboard.reset()

    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Se2Keyboard outputs (vx, vy, wz) in [-sens, +sens]
            teleop_cmd = keyboard.advance()  # shape (3,)

            # map to the correct action layout for this drive type
            mapped = teleop_to_action(teleop_cmd, drive_type, action_dim, device)

            # broadcast to all envs
            actions = mapped.unsqueeze(0).expand(num_envs, -1)

            env.step(actions)

        # real-time pacing
        elapsed = time.time() - start_time
        if args_cli.real_time and elapsed < dt:
            time.sleep(dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
