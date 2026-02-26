# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--max_frames", type=int, default=200, help="Number of frames to save.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import BlindBrigade_tasks.tasks  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm

# Grid parameters (must match scene_cfg ray_caster_cam)
GRID_N = 16
GRID_SIZE_M = 3.0  # metres
UPSCALE = 256  # output image size

def render_occupancy_frame(distances: torch.Tensor, goal_body: torch.Tensor, max_distance: float) -> Image.Image:
    """Render a single occupancy grid frame.

    Args:
        distances: Raw distances for env 0, shape (N,) where N = GRID_N*GRID_N.
        goal_body: Goal position in body frame [x, y], shape (2,).
        max_distance: Sensor max distance for normalization.

    Returns:
        PIL Image (256x256 RGB).
    """
    # Reshape to 2D grid — GridPatternCfg ordering='xy': row=y, col=x
    grid = distances.reshape(GRID_N, GRID_N).cpu().numpy()

    # Normalize to [0, 255]: large distance = ground/free (white), small = obstacle (dark)
    gray = np.clip(grid / (max_distance + 0.01) * 255, 0, 255).astype(np.uint8)

    # Create grayscale image and upscale with nearest-neighbor
    img = Image.fromarray(gray, mode="L")
    img = img.resize((UPSCALE, UPSCALE), Image.NEAREST)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    # Robot marker at grid center (red dot)
    cx = UPSCALE / 2
    cy = UPSCALE / 2
    r = 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")

    # Goal marker (green dot) — convert body-frame metres to pixel coords
    # Grid spans [-1.5, 1.5] in both x (col) and y (row)
    gx, gy = goal_body[0].item(), goal_body[1].item()
    px = (gx + GRID_SIZE_M / 2) / GRID_SIZE_M * UPSCALE  # x → column → PIL x
    py = (gy + GRID_SIZE_M / 2) / GRID_SIZE_M * UPSCALE  # y → row    → PIL y

    # Clamp to image bounds
    px = max(0, min(UPSCALE - 1, px))
    py = max(0, min(UPSCALE - 1, py))

    r_goal = 6
    draw.ellipse([px - r_goal, py - r_goal, px + r_goal, py + r_goal], fill="green")

    return img

def render_binary_occupancy_frame(
    distances: torch.Tensor, goal_body: torch.Tensor, max_distance: float, height_margin: float = -0.05
) -> Image.Image:
    """Render a binary occupancy grid frame (black = obstacle, white = free).

    Drop-in replacement for render_occupancy_frame. The threshold is adaptive:
    anything 2cm (height_margin) above the ground level at the robot is occupied.

    Args:
        distances: Raw distances for env 0, shape (N,) where N = GRID_N*GRID_N.
        goal_body: Goal position in body frame [x, y], shape (2,).
        max_distance: Sensor max distance for normalization.
        height_margin: Height above robot ground level (metres) to count as
            occupied. Default 0.02 (2 cm).

    Returns:
        PIL Image (256x256 RGB).
    """
    grid = distances.reshape(GRID_N, GRID_N).cpu().numpy()

    # Ground level = average distance at the 4 center cells (robot position)
    # No ray lands exactly at (0,0); closest are indices 7,8 in each axis.
    ground_dist = grid[GRID_N//2 - 1:GRID_N//2 + 1, GRID_N//2 - 1:GRID_N//2 + 1].mean()

    # Shorter distance = taller surface. Occupied if >= height_margin above ground.
    binary = np.where(grid < ground_dist - height_margin, 255, 0).astype(np.uint8)

    img = Image.fromarray(binary, mode="L")
    img = img.resize((UPSCALE, UPSCALE), Image.NEAREST)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    # Robot marker at grid center (red dot)
    cx = UPSCALE / 2
    cy = UPSCALE / 2
    r = 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")

    # Goal marker (green dot)
    gx, gy = goal_body[0].item(), goal_body[1].item()
    px = (gx + GRID_SIZE_M / 2) / GRID_SIZE_M * UPSCALE
    py = (gy + GRID_SIZE_M / 2) / GRID_SIZE_M * UPSCALE
    px = max(0, min(UPSCALE - 1, px))
    py = max(0, min(UPSCALE - 1, py))
    r_goal = 6
    draw.ellipse([px - r_goal, py - r_goal, px + r_goal, py + r_goal], fill="green")

    return img

def render_heatmap_occupancy_frame(
    distances: torch.Tensor,
    goal_body: torch.Tensor,
    max_distance: float,
    height_margin: float = 0.02,
) -> Image.Image:
    """Render a heatmap occupancy frame.

    Color meaning (default turbo colormap):
        blue   = lower than ground (free space)
        green  = near ground
        red    = higher than ground (likely obstacle)

    Args:
        distances: Raw distances for env 0, shape (N,)
        goal_body: Goal position in body frame [x, y], shape (2,)
        max_distance: Sensor max distance for normalization
        height_margin: Height offset above ground (metres)

    Returns:
        PIL Image (256x256 RGB)
    """
    grid = distances.reshape(GRID_N, GRID_N).cpu().numpy()

    # Estimate local ground level at robot position
    ground_dist = grid[
        GRID_N // 2 - 1 : GRID_N // 2 + 1,
        GRID_N // 2 - 1 : GRID_N // 2 + 1,
    ].mean()

    # Relative height estimate
    # shorter distance => taller obstacle
    rel_height = (ground_dist - grid)

    # Optional shift so "free" stays cooler colors
    rel_height = rel_height - height_margin

    # Normalize into [0,1]
    # clip range can be tuned depending on sensor scale
    min_h = -0.1
    max_h = 0.3
    norm = np.clip((rel_height - min_h) / (max_h - min_h), 0.0, 1.0)

    # Apply colormap (returns RGBA in [0,1])
    cmap = cm.get_cmap("turbo")  # good perceptual contrast
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    img = Image.fromarray(colored, mode="RGB")
    img = img.resize((UPSCALE, UPSCALE), Image.NEAREST)

    draw = ImageDraw.Draw(img)

    # Robot marker (red)
    cx = UPSCALE / 2
    cy = UPSCALE / 2
    r = 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")

    # Goal marker (green)
    gx, gy = goal_body[0].item(), goal_body[1].item()
    px = (gx + GRID_SIZE_M / 2) / GRID_SIZE_M * UPSCALE
    py = (gy + GRID_SIZE_M / 2) / GRID_SIZE_M * UPSCALE
    px = max(0, min(UPSCALE - 1, px))
    py = max(0, min(UPSCALE - 1, py))
    r_goal = 6
    draw.ellipse([px - r_goal, py - r_goal, px + r_goal, py + r_goal], fill="green")

    return img

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # create output directory
    output_dir = os.path.abspath("occupancy_frames")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving frames to: {output_dir}", flush=True)

    # get the raw sensor for distance computation
    sensor = env.unwrapped.scene.sensors["ray_caster_cam"]
    max_dist = sensor.cfg.max_distance
    print(f"[INFO] Sensor max_distance={max_dist}, rays={sensor.data.ray_hits_w.shape}", flush=True)

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    print("[INFO] Got initial observations, starting loop...", flush=True)
    timestep = 0
    step_count = 0
    num_saved = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # --- capture frame from env 0 ---
            distances = obs['heightscan']
            # distances = torch.nan_to_num(distances, nan=max_dist)
            # distances = torch.clamp(input=distances,min=-max_dist,max=max_dist) 
            # Get goal in body frame from command manager
            goal_body = env.unwrapped.command_manager.get_command("goal_pose")[0, :2]

             # Render and save
            img = render_heatmap_occupancy_frame(distances, goal_body, max_dist)
            img.save(os.path.join(output_dir, f"frame_{step_count:05d}.png"))
            num_saved += 1

            if step_count % 50 == 0:
                print(f"[INFO] Step {step_count}/{args_cli.max_frames}", flush=True)
            step_count += 1

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[INFO] Saved {num_saved} frames to {output_dir}", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
