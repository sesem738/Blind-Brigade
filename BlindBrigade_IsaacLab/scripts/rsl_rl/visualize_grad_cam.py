"""Grad-CAM visualisation for a trained StudentTeacherCNN distillation policy.

For each captured step, produces an overlay of the depth image with a heatmap
showing which pixels drove each action output (linear velocity, angular velocity).

Grad-CAM reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.

Usage (from scripts/rsl_rl/):
    python scripts/rsl_rl/visualize_grad_cam.py \
        --task BB-rosbot-box-PLAY-v0 \
        --agent rsl_rl_distil_cnn_cfg_entry_point \
        --load_run 2026-02-27_09-07-42 \
        --checkpoint model_1999.pt \
        --num_envs 1 \
        --enable_cameras \
        --headless \
        --capture_every 30 \
        --max_steps 300 \
        --out_dir logs/grad_cam
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Grad-CAM visualisation for distillation policy.")
parser.add_argument("--num_envs",      type=int, default=1)
parser.add_argument("--task",          type=str, default=None)
parser.add_argument("--agent",         type=str, default="rsl_rl_distil_cnn_cfg_entry_point")
parser.add_argument("--capture_every", type=int, default=30)
parser.add_argument("--max_steps",     type=int, default=300)
parser.add_argument("--env_idx",       type=int, default=0,  help="Which environment to visualise.")
parser.add_argument("--out_dir",       type=str, default="logs/grad_cam")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything after sim launch ───────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import BlindBrigade_tasks.tasks  # noqa: F401


# ── Grad-CAM implementation ───────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM on the last Conv2d of a CNN module.

    Computes a per-action saliency map by backpropagating from each action
    output through the full policy network to the last convolutional layer.
    """

    def __init__(self, cnn: nn.Module) -> None:
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        # Find the last Conv2d in the CNN
        last_conv = None
        for module in cnn.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise RuntimeError("No Conv2d found in the provided CNN module.")

        last_conv.register_forward_hook(self._fwd_hook)
        last_conv.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out  # (B, C, H, W)  — keep in graph for backward

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()  # (B, C, H, W)

    def compute(
        self,
        policy_nn,
        obs: dict,
        action_idx: int,
        env_idx: int,
        input_hw: tuple[int, int],
    ) -> np.ndarray:
        """Return a Grad-CAM heatmap (H_in, W_in) in [0, 1] for one action dimension.

        Args:
            policy_nn:   The StudentTeacherCNN module.
            obs:         Full observation TensorDict from the environment.
            action_idx:  Which action output to compute CAM for.
            env_idx:     Which batch element to visualise.
            input_hw:    (H, W) of the original depth image for upsampling.
        """
        policy_nn.zero_grad()

        # Forward with gradient tracking — run only the student branch
        encoded = policy_nn._encode_student_obs(obs)   # CNN + 1D obs → (B, F)
        actions = policy_nn.student(encoded)            # (B, num_actions)

        # Scalar target: action[env_idx, action_idx]
        target = actions[env_idx, action_idx]
        target.backward(retain_graph=True)

        # Grad-CAM: weight each channel by its mean gradient
        # gradients / activations: (B, C, H, W)
        grads = self._gradients[env_idx]          # (C, H, W)
        acts  = self._activations[env_idx]        # (C, H, W)

        weights = grads.mean(dim=(1, 2))           # (C,)  — global avg pool of gradients
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (H, W)
        cam = F.relu(cam)                          # keep only positive contributions

        # Upsample to original image resolution
        cam = cam.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W)
        cam = F.interpolate(cam, size=input_hw, mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        policy_nn.zero_grad()
        return cam


# ── Plotting ──────────────────────────────────────────────────────────────────

ACTION_LABELS = ["linear vel  (forward +)", "angular vel  (left +)"]


def save_grad_cam(
    depth_img: torch.Tensor,
    grad_cams: list[np.ndarray],
    actions:   torch.Tensor,
    env_idx:   int,
    step:      int,
    out_dir:   str,
) -> None:
    """Save Grad-CAM overlays for all actions."""
    os.makedirs(out_dir, exist_ok=True)

    num_actions = len(grad_cams)
    fig, axes = plt.subplots(1, num_actions + 1, figsize=(5 * (num_actions + 1), 4),
                             constrained_layout=True)

    # ── Raw depth image ───────────────────────────────────────────────────────
    img_np = depth_img[env_idx].squeeze().cpu().numpy().astype(np.float32)
    lo, hi = img_np.min(), img_np.max()
    if hi > lo:
        img_np = (img_np - lo) / (hi - lo)

    axes[0].imshow(img_np, cmap="viridis", aspect="auto")
    axes[0].set_title("Depth image", fontsize=10)
    axes[0].axis("off")

    # ── Grad-CAM overlays ─────────────────────────────────────────────────────
    for i, (cam, label) in enumerate(zip(grad_cams, ACTION_LABELS)):
        # Convert greyscale depth to RGB so we can overlay the heatmap
        depth_rgb = cm.viridis(img_np)[:, :, :3]           # (H, W, 3)
        heatmap   = cm.jet(cam)[:, :, :3]                   # (H, W, 3)  red = high importance
        overlay   = 0.55 * depth_rgb + 0.45 * heatmap       # blend

        axes[i + 1].imshow(overlay, aspect="auto")
        action_val = actions[env_idx, i].item()
        axes[i + 1].set_title(
            f"Grad-CAM: {label}\naction = {action_val:+.3f}", fontsize=9
        )
        axes[i + 1].axis("off")

    fig.suptitle(f"Grad-CAM  —  step {step}", fontsize=11)
    path = os.path.join(out_dir, f"step{step:04d}_gradcam.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grad-cam] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed            = agent_cfg.seed
    env_cfg.sim.device      = args_cli.device or env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path   = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = (DistillationRunner if agent_cfg.class_name == "DistillationRunner"
              else OnPolicyRunner)(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy_nn = runner.alg.policy
    policy_nn.eval()

    if not hasattr(policy_nn, "student_cnns") or policy_nn.student_cnns is None:
        raise RuntimeError("No student_cnns found — load a CNN distillation checkpoint.")

    cnn_group = list(policy_nn.student_cnns.keys())[0]
    cnn       = policy_nn.student_cnns[cnn_group]
    print(f"[grad-cam] Attaching to CNN for obs group '{cnn_group}'")

    grad_cam  = GradCAM(cnn)
    num_actions = env.num_actions

    # Get input image shape for upsampling
    obs = env.get_observations()
    img_shape = obs[cnn_group].shape[-2:]   # (H, W)
    print(f"[grad-cam] Image resolution: {img_shape}")

    step = 0
    while simulation_app.is_running() and step < args_cli.max_steps:

        if step % args_cli.capture_every == 0:
            # Snapshot obs (detach from any existing graph)
            obs_snap = {k: v.clone() for k, v in obs.items()}

            # Grad-CAM requires gradients — run outside inference_mode
            with torch.enable_grad():
                # Re-attach image to graph for backward
                obs_snap[cnn_group] = obs_snap[cnn_group].requires_grad_(False)

                cams = []
                for action_idx in range(num_actions):
                    cam = grad_cam.compute(
                        policy_nn  = policy_nn,
                        obs        = obs_snap,
                        action_idx = action_idx,
                        env_idx    = args_cli.env_idx,
                        input_hw   = tuple(img_shape),
                    )
                    cams.append(cam)

                # Action values at this step (inference, no grad needed)
                with torch.no_grad():
                    encoded = policy_nn._encode_student_obs(obs_snap)
                    actions = policy_nn.student(encoded)

            save_grad_cam(
                depth_img  = obs[cnn_group],
                grad_cams  = cams,
                actions    = actions,
                env_idx    = args_cli.env_idx,
                step       = step,
                out_dir    = args_cli.out_dir,
            )

        # Step environment in inference mode
        with torch.inference_mode():
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            actions_env = policy(obs)
            obs, _, dones, _ = env.step(actions_env)
            policy_nn.reset(dones)

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()