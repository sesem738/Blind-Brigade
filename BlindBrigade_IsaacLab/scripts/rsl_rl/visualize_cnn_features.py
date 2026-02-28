"""Visualize CNN feature maps from a trained StudentTeacherCNN distillation policy.

Hooks into the Conv2d layers of the student CNN encoder and saves feature map
grids alongside the input depth image at each captured step.

Usage (from scripts/rsl_rl/):
    python scripts/rsl_rl/visualize_cnn_features.py \
        --task BB-rosbot-box-PLAY-v0 \
        --agent rsl_rl_distil_cnn_cfg_entry_point \
        --load_run 2026-02-27_09-07-42 \
        --checkpoint model_1999.pt \
        --num_envs 1 \
        --enable_cameras \
        --headless \
        --capture_every 30 \
        --out_dir logs/feature_maps
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Visualize CNN feature maps from a distillation policy.")
parser.add_argument("--num_envs",       type=int,  default=1)
parser.add_argument("--task",           type=str,  default=None)
parser.add_argument("--agent",          type=str,  default="rsl_rl_distil_cnn_cfg_entry_point")
parser.add_argument("--capture_every",  type=int,  default=30,   help="Save feature maps every N steps.")
parser.add_argument("--max_steps",      type=int,  default=300,  help="Stop after this many steps.")
parser.add_argument("--env_idx",        type=int,  default=0,    help="Which environment to visualise.")
parser.add_argument("--out_dir",        type=str,  default="logs/feature_maps")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything after sim launch ───────────────────────────────────────────────

import os
import time
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")          # no GUI needed — saves to disk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import BlindBrigade_tasks.tasks  # noqa: F401


# ── Feature-map capture via forward hooks ─────────────────────────────────────

class ConvFeatureCapture:
    """Registers forward hooks on every Conv2d inside a module and stores outputs."""

    def __init__(self, cnn: nn.Module) -> None:
        self.maps: dict[str, torch.Tensor] = {}
        self._handles = []
        for name, module in cnn.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.maps[name] = output.detach().cpu()
        return hook

    def clear(self) -> None:
        self.maps.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _to_numpy_img(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) or (H, W) depth tensor → normalised uint8 (H, W)."""
    img = img_tensor.squeeze().numpy().astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img


def save_feature_maps(
    depth_img: torch.Tensor,
    feature_maps: dict[str, torch.Tensor],
    env_idx: int,
    step: int,
    out_dir: str,
) -> None:
    """Save the depth image and all feature map grids to PNG files."""
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Input depth image ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(_to_numpy_img(depth_img[env_idx]), cmap="viridis", aspect="auto")
    ax.set_title(f"Input depth image  (step {step})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step{step:04d}_input.png"), dpi=100)
    plt.close(fig)

    # ── 2. Feature maps per conv layer ───────────────────────────────────────
    for layer_name, fmap in feature_maps.items():
        # fmap shape: (B, C, H, W)
        channels = fmap[env_idx]  # (C, H, W)
        C = channels.shape[0]

        cols = min(8, C)
        rows = (C + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 2, rows * 2 + 0.6))
        fig.suptitle(f"Conv layer '{layer_name}'  —  {C} channels  (step {step})", fontsize=10)
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.05)

        for ch in range(C):
            ax = fig.add_subplot(gs[ch // cols, ch % cols])
            ax.imshow(_to_numpy_img(channels[ch]), cmap="RdBu_r", aspect="auto")
            ax.set_title(f"ch{ch}", fontsize=6, pad=1)
            ax.axis("off")

        # hide unused cells
        for idx in range(C, rows * cols):
            fig.add_subplot(gs[idx // cols, idx % cols]).axis("off")

        safe_name = layer_name.replace(".", "_")
        fig.savefig(
            os.path.join(out_dir, f"step{step:04d}_layer{safe_name}.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"[visualize] Saved feature maps for step {step} → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path   = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy    = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    # ── Locate the CNN encoder ────────────────────────────────────────────────
    if not hasattr(policy_nn, "student_cnns") or policy_nn.student_cnns is None:
        raise RuntimeError(
            "policy_nn has no student_cnns. Is this a CNN distillation checkpoint?"
        )
    # Use the first (and typically only) CNN encoder
    cnn_group = list(policy_nn.student_cnns.keys())[0]
    cnn       = policy_nn.student_cnns[cnn_group]
    print(f"[visualize] Hooking into CNN for obs group '{cnn_group}'")
    print(f"[visualize] CNN architecture:\n{cnn}")

    capture = ConvFeatureCapture(cnn)

    # ── Run and capture ───────────────────────────────────────────────────────
    obs      = env.get_observations()
    out_dir  = args_cli.out_dir
    step     = 0

    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if step % args_cli.capture_every == 0:
            # Grab the raw depth image for the selected env
            depth_img = env.unwrapped.obs_buf.get("exteroceptive", None)
            if depth_img is None:
                # fall back to the obs TensorDict if available
                depth_img = obs.get("exteroceptive", None)

            save_feature_maps(
                depth_img   = depth_img.cpu() if depth_img is not None else torch.zeros(args_cli.num_envs, 1, 1),
                feature_maps= capture.maps,
                env_idx     = args_cli.env_idx,
                step        = step,
                out_dir     = out_dir,
            )
            capture.clear()

        step += 1

    capture.remove()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
