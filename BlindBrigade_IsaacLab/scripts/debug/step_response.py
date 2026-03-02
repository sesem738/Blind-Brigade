"""Step-response test: measure achievable velocities of the rosbot in simulation.

Sweeps multiple action set-points on each axis (vx, vy, wz), records the
steady-state achieved velocity, and plots:
  1. Time-domain step responses for each set point (3 subplots)
  2. Achieved vs commanded velocity with 1:1 reference (linearity plot)

Usage:
    python scripts/debug/step_response.py --task BB-rosbot-flat-PLAY-v0 --num_envs 1 --headless
    python scripts/debug/step_response.py --set_points 0.25 0.5 0.75 1.0 --num_steps 150 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-response velocity test for mecanum robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BB-rosbot-flat-PLAY-v0", help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=150, help="Steps per trial (~25ms each at 40Hz).")
parser.add_argument(
    "--set_points", type=float, nargs="+",
    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0],
    help="Action magnitudes to test (in [-1, 1] normalized space).",
)
parser.add_argument("--settle_frac", type=float, default=0.5, help="Fraction of steps to skip before averaging (settling time).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import BlindBrigade_tasks.tasks  # noqa: F401

# Test definitions: (label, action_index, vel_index, is_angular)
TESTS = [
    ("Forward (vx)", 0, 0, False),
    ("Lateral (vy)", 1, 1, False),
    ("Yaw (wz)", 2, 2, True),
]


def run_trial(env, action_term, act_idx, vel_idx, is_angular, set_point, num_steps):
    """Run a single constant-action trial. Returns (time, cmd, achieved) lists."""
    uw = env.unwrapped
    step_dt = uw.step_dt
    robot = uw.scene["robot"]

    # Reset env + LPF state
    env.reset()
    action_term.reset(torch.arange(uw.num_envs, device=uw.device))

    time_log, cmd_log, ach_log = [], [], []

    for step_i in range(num_steps):
        if not simulation_app.is_running():
            break

        actions = torch.zeros(env.action_space.shape, device=uw.device)
        actions[:, act_idx] = set_point
        env.step(actions)

        cmd_vel_val = action_term.processed_actions[0, vel_idx].item()
        if is_angular:
            ach_vel_val = robot.data.root_ang_vel_b[0, vel_idx].item()
        else:
            ach_vel_val = robot.data.root_lin_vel_b[0, vel_idx].item()

        time_log.append(step_i * step_dt)
        cmd_log.append(cmd_vel_val)
        ach_log.append(ach_vel_val)

    return time_log, cmd_log, ach_log


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    uw = env.unwrapped

    step_dt = uw.step_dt
    action_term = uw.action_manager.get_term("base_twist")
    max_limits = [action_term.cfg.max_vx, action_term.cfg.max_vy, action_term.cfg.max_wz]
    set_points = args_cli.set_points
    n_steps = args_cli.num_steps
    settle_start = int(n_steps * args_cli.settle_frac)

    print(f"[INFO] step_dt={step_dt:.4f}s  max_limits(vx,vy,wz)={max_limits}")
    print(f"[INFO] set_points={set_points}")
    print(f"[INFO] {n_steps} steps/trial, averaging after step {settle_start}")
    total_trials = len(TESTS) * len(set_points)
    print(f"[INFO] Total trials: {total_trials}")

    # results[axis_label][set_point] = {time, cmd, ach, ss_cmd, ss_ach}
    results = {}

    trial_num = 0
    for label, act_idx, vel_idx, is_angular in TESTS:
        print(f"\n=== {label} (max={max_limits[vel_idx]}) ===")
        results[label] = {}

        for sp in set_points:
            trial_num += 1
            time_log, cmd_log, ach_log = run_trial(
                env, action_term, act_idx, vel_idx, is_angular, sp, n_steps
            )

            # Steady-state average (past settling time)
            ss_cmd = sum(cmd_log[settle_start:]) / max(1, len(cmd_log[settle_start:]))
            ss_ach = sum(ach_log[settle_start:]) / max(1, len(ach_log[settle_start:]))
            ratio = ss_ach / ss_cmd if abs(ss_cmd) > 1e-6 else float("nan")

            results[label][sp] = {
                "time": time_log, "cmd": cmd_log, "ach": ach_log,
                "ss_cmd": ss_cmd, "ss_ach": ss_ach,
            }
            print(f"  [{trial_num:2d}/{total_trials}] action={sp:+.2f}  cmd={ss_cmd:+.4f}  ach={ss_ach:+.4f}  ratio={ratio:.3f}")

    env.close()

    # --- Plotting ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = ["m/s", "m/s", "rad/s"]
    colors = plt.cm.viridis([i / (len(set_points) - 1) for i in range(len(set_points))])

    # --- Figure 1: Time-domain step responses ---
    fig1, axes1 = plt.subplots(len(TESTS), 1, figsize=(10, 3 * len(TESTS)), sharex=True)
    if len(TESTS) == 1:
        axes1 = [axes1]

    for ax, (label, _, _, _), unit in zip(axes1, TESTS, units):
        axis_data = results[label]
        for sp, color in zip(set_points, colors):
            d = axis_data[sp]
            ax.plot(d["time"], d["ach"], "-", color=color, linewidth=1.5, label=f"ach {sp:.1f}")
            ax.plot(d["time"], d["cmd"], "--", color=color, linewidth=1, alpha=0.5)
        # Mark settle boundary
        ax.axvline(x=settle_start * step_dt, color="red", linestyle=":", alpha=0.5, label="settle")
        ax.set_ylabel(unit)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=4, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes1[-1].set_xlabel("Time (s)")
    fig1.suptitle("Step Response: Commanded (dashed) vs Achieved (solid)", fontsize=13)
    fig1.tight_layout()
    fig1.savefig("scripts/debug/step_response.png", dpi=150)
    print(f"\n[INFO] Step-response plot saved to scripts/debug/step_response.png")
    plt.close(fig1)

    # --- Figure 2: Linearity — achieved vs commanded steady-state ---
    fig2, axes2 = plt.subplots(1, len(TESTS), figsize=(5 * len(TESTS), 5))
    if len(TESTS) == 1:
        axes2 = [axes2]

    for ax, (label, _, _, _), unit in zip(axes2, TESTS, units):
        axis_data = results[label]
        cmds = [axis_data[sp]["ss_cmd"] for sp in set_points]
        achs = [axis_data[sp]["ss_ach"] for sp in set_points]

        # 1:1 reference
        all_vals = cmds + achs
        if any(abs(v) > 1e-6 for v in all_vals):
            lo = min(0, min(all_vals) * 1.1)
            hi = max(all_vals) * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="1:1")

        ax.scatter(cmds, achs, c=range(len(set_points)), cmap="viridis", s=60, zorder=5)
        ax.plot(cmds, achs, "-", color="tab:blue", alpha=0.6)

        # Annotate ratios
        for sp, cmd, ach in zip(set_points, cmds, achs):
            ratio = ach / cmd if abs(cmd) > 1e-6 else float("nan")
            ax.annotate(f"{ratio:.2f}", (cmd, ach), textcoords="offset points",
                        xytext=(6, -4), fontsize=7, color="gray")

        ax.set_xlabel(f"Commanded ({unit})")
        ax.set_ylabel(f"Achieved ({unit})")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    fig2.suptitle("Linearity: Achieved vs Commanded (steady-state)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig("scripts/debug/step_response_linearity.png", dpi=150)
    print(f"[INFO] Linearity plot saved to scripts/debug/step_response_linearity.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
    simulation_app.close()
