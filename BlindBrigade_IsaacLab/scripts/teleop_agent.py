"""Script to teleoperate a robot in an Isaac Lab environment using keyboard.

Usage:
    python scripts/teleop_agent.py --task=BB-rosbot-coop-flat-v0 --num_envs=1
    python scripts/teleop_agent.py --task=BB-rosbot-box-PLAY-v0 --num_envs=1

Controls:
    Arrow Up / Numpad 8   : Forward  (+vx)
    Arrow Down / Numpad 2 : Backward (-vx)
    Arrow Left / Numpad 4 : Strafe right (+vy)
    Arrow Right / Numpad 6: Strafe left  (-vy)
    Z / Numpad 7          : Yaw left  (+wz)
    X / Numpad 9          : Yaw right (-wz)
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

import BlindBrigade_tasks.tasks  # noqa: F401


def main():
    """Teleop agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space      : {env.action_space}")

    action_dim = env.action_space.shape[-1]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # create keyboard controller
    # Se2Keyboard outputs (vx, vy, wz) in [-sensitivity, +sensitivity]
    sens = args_cli.sensitivity
    keyboard = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=sens,
        v_y_sensitivity=sens,
        omega_z_sensitivity=sens,
        sim_device=device,
    ))
    print(keyboard)
    print(f"\n[INFO]: Action dim = {action_dim}. Keyboard controls first 3 dims (guide robot).")
    print("[INFO]: Remaining action dims (blind robot) are zeroed out.\n")

    # reset
    env.reset()
    keyboard.reset()

    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # get keyboard SE(2) command: (vx, vy, wz) in [-sens, +sens]
            teleop_cmd = keyboard.advance()  # shape (3,)

            # build full action tensor
            actions = torch.zeros(num_envs, action_dim, device=device)
            # fill first 3 dims (guide robot) for all envs
            actions[:, :3] = teleop_cmd.unsqueeze(0)

            env.step(actions)

        # real-time pacing
        elapsed = time.time() - start_time
        if args_cli.real_time and elapsed < dt:
            time.sleep(dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
