# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test lateral movement of mecanum wheel robot."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test lateral movement for mecanum robot.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BB-rosbot-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import BlindBrigade_tasks.tasks  # noqa: F401


def main():
    """Test lateral movement with constant velocity commands."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Action space shape: {env.action_space.shape}")
    print("[INFO]: Actions are [vx, vy, wz] - testing pure lateral (vy) motion")

    env.reset()

    step_count = 0
    phase_duration = 200  # steps per phase

    while simulation_app.is_running():
        with torch.inference_mode():
            # Create action tensor: [vx, vy, wz]
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

            # Cycle through different movement patterns
            phase = (step_count // phase_duration) % 6

            if phase == 0:
                # Pure lateral RIGHT (positive vy)
                actions[:, 1] = 0.5
                if step_count % phase_duration == 0:
                    print("[TEST] Phase 0: Moving RIGHT (vy = +0.5)")
            elif phase == 1:
                # Pure lateral LEFT (negative vy)
                actions[:, 1] = -0.5
                if step_count % phase_duration == 0:
                    print("[TEST] Phase 1: Moving LEFT (vy = -0.5)")
            elif phase == 2:
                # Pure forward (positive vx)
                actions[:, 0] = 0.5
                if step_count % phase_duration == 0:
                    print("[TEST] Phase 2: Moving FORWARD (vx = +0.5)")
            elif phase == 3:
                # Pure backward (negative vx)
                actions[:, 0] = -0.5
                if step_count % phase_duration == 0:
                    print("[TEST] Phase 3: Moving BACKWARD (vx = -0.5)")
            elif phase == 4:
                # Pure rotation CW
                actions[:, 2] = 5
                if step_count % phase_duration == 0:
                    print("[TEST] Phase 4: Rotating CW (wz = +5)")
            elif phase == 5:
                # Pure rotation CCW
                actions[:, 2] = -5
                if step_count % phase_duration == 0:
                    print("[TEST] Phase 5: Rotating CCW (wz = -5)")

            env.step(actions)
            step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
