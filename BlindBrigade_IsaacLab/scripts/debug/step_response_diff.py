"""Step-response test for the differential-drive rosbot.

Sweeps multiple action set-points on each axis (v, omega), records the
steady-state achieved velocity, and plots:
  1. Time-domain step responses for each set point (2 subplots: forward vel, yaw rate)
  2. Achieved vs commanded velocity with 1:1 reference (linearity plot)
  3. Wheel speed plot — commanded left/right wheel speeds vs achieved

Also logs individual wheel joint velocities to verify the differential
kinematics are correct (left wheels match, right wheels match).

Usage:
    python scripts/debug/step_response_diff.py --task BB-rosbot-diff-flat-PLAY-v0 --num_envs 1 --headless
    python scripts/debug/step_response_diff.py --set_points 0.25 0.5 0.75 1.0 --num_steps 150 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-response velocity test for differential-drive robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BB-rosbot-diff-flat-PLAY-v0", help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=150, help="Steps per trial.")
parser.add_argument(
    "--set_points", type=float, nargs="+",
    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    help="Action magnitudes to test (in normalized [-1, 1] space).",
)
parser.add_argument("--settle_frac", type=float, default=0.5,
                    help="Fraction of steps to skip before averaging (settling time).")
parser.add_argument("--output_dir", type=str, default="scripts/debug",
                    help="Directory to save output plots.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import BlindBrigade_tasks.tasks  # noqa: F401

# Test definitions: (label, action_index, unit)
#   action_index: which dim of the 2D action (v, omega) to excite
TESTS = [
    ("Forward velocity (v)", 0, "m/s"),
    ("Yaw rate (omega)",     1, "rad/s"),
]


def run_trial(env, action_term, act_idx, set_point, num_steps):
    """Run a single constant-action trial.

    Returns dict with time-series logs for body velocity, wheel velocities, etc.
    """
    uw = env.unwrapped
    step_dt = uw.step_dt
    robot = uw.scene["robot"]

    # Reset env + action term LPF state
    env.reset()
    action_term.reset(torch.arange(uw.num_envs, device=uw.device))

    logs = {
        "time": [], "cmd_v": [], "cmd_omega": [],
        "ach_vx": [], "ach_vy": [], "ach_omega": [],
        "wheel_vel_fl": [], "wheel_vel_fr": [], "wheel_vel_rl": [], "wheel_vel_rr": [],
    }

    wheel_ids = action_term._wheel_joint_ids

    for step_i in range(num_steps):
        if not simulation_app.is_running():
            break

        actions = torch.zeros(env.action_space.shape, device=uw.device)
        actions[:, act_idx] = set_point
        env.step(actions)

        # Commanded velocities from action term
        proc = action_term.processed_actions[0]
        logs["cmd_v"].append(proc[0].item())
        logs["cmd_omega"].append(proc[1].item())

        # Achieved body velocities
        logs["ach_vx"].append(robot.data.root_lin_vel_b[0, 0].item())
        logs["ach_vy"].append(robot.data.root_lin_vel_b[0, 1].item())
        logs["ach_omega"].append(robot.data.root_ang_vel_b[0, 2].item())

        # Individual wheel joint velocities
        jvel = robot.data.joint_vel[0]
        logs["wheel_vel_fl"].append(jvel[wheel_ids[0]].item())
        logs["wheel_vel_fr"].append(jvel[wheel_ids[1]].item())
        logs["wheel_vel_rl"].append(jvel[wheel_ids[2]].item())
        logs["wheel_vel_rr"].append(jvel[wheel_ids[3]].item())

        logs["time"].append(step_i * step_dt)

    return logs


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    uw = env.unwrapped

    step_dt = uw.step_dt
    action_term = uw.action_manager.get_term("base_twist")
    cfg = action_term.cfg

    set_points = args_cli.set_points
    n_steps = args_cli.num_steps
    settle_start = int(n_steps * args_cli.settle_frac)

    print(f"[INFO] Drive type     : DifferentialDrive")
    print(f"[INFO] wheel_radius   : {cfg.wheel_radius}")
    print(f"[INFO] half_track     : {cfg.half_track}")
    print(f"[INFO] max_v          : {cfg.max_v}  max_omega: {cfg.max_omega}")
    print(f"[INFO] step_dt        : {step_dt:.4f}s")
    print(f"[INFO] set_points     : {set_points}")
    print(f"[INFO] {n_steps} steps/trial, averaging after step {settle_start}")

    # results[test_label][set_point] = logs dict
    results = {}

    total_trials = len(TESTS) * len(set_points)
    trial_num = 0

    for label, act_idx, unit in TESTS:
        print(f"\n=== {label} ===")
        results[label] = {}

        for sp in set_points:
            trial_num += 1
            logs = run_trial(env, action_term, act_idx, sp, n_steps)

            # Steady-state averages
            if act_idx == 0:
                cmd_key, ach_key = "cmd_v", "ach_vx"
            else:
                cmd_key, ach_key = "cmd_omega", "ach_omega"

            ss_cmd = sum(logs[cmd_key][settle_start:]) / max(1, len(logs[cmd_key][settle_start:]))
            ss_ach = sum(logs[ach_key][settle_start:]) / max(1, len(logs[ach_key][settle_start:]))
            ss_vy  = sum(logs["ach_vy"][settle_start:]) / max(1, len(logs["ach_vy"][settle_start:]))
            ratio = ss_ach / ss_cmd if abs(ss_cmd) > 1e-6 else float("nan")

            results[label][sp] = {**logs, "ss_cmd": ss_cmd, "ss_ach": ss_ach, "ss_vy": ss_vy}
            print(f"  [{trial_num:2d}/{total_trials}] action={sp:+.2f}  "
                  f"cmd={ss_cmd:+.4f}  ach={ss_ach:+.4f}  lat_vy={ss_vy:+.4f}  ratio={ratio:.3f}")

    env.close()

    # ---- Plotting ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    out_dir = args_cli.output_dir
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.cm.viridis([i / max(1, len(set_points) - 1) for i in range(len(set_points))])

    # --- Figure 1: Time-domain step responses ---
    fig1, axes1 = plt.subplots(len(TESTS), 1, figsize=(10, 4 * len(TESTS)), sharex=True)
    if len(TESTS) == 1:
        axes1 = [axes1]

    for ax, (label, act_idx, unit) in zip(axes1, TESTS):
        axis_data = results[label]
        ach_key = "ach_vx" if act_idx == 0 else "ach_omega"
        cmd_key = "cmd_v" if act_idx == 0 else "cmd_omega"

        for sp, color in zip(set_points, colors):
            d = axis_data[sp]
            ax.plot(d["time"], d[ach_key], "-", color=color, linewidth=1.5, label=f"ach {sp:.1f}")
            ax.plot(d["time"], d[cmd_key], "--", color=color, linewidth=1, alpha=0.5)

        ax.axvline(x=settle_start * step_dt, color="red", linestyle=":", alpha=0.5, label="settle")
        ax.set_ylabel(unit)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=4, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes1[-1].set_xlabel("Time (s)")
    fig1.suptitle("Differential Drive — Step Response: Commanded (dashed) vs Achieved (solid)", fontsize=13)
    fig1.tight_layout()
    path1 = os.path.join(out_dir, "step_response_diff.png")
    fig1.savefig(path1, dpi=150)
    print(f"\n[INFO] Step-response plot saved to {path1}")
    plt.close(fig1)

    # --- Figure 2: Linearity — achieved vs commanded ---
    fig2, axes2 = plt.subplots(1, len(TESTS), figsize=(5 * len(TESTS), 5))
    if len(TESTS) == 1:
        axes2 = [axes2]

    for ax, (label, _, unit) in zip(axes2, TESTS):
        axis_data = results[label]
        cmds = [axis_data[sp]["ss_cmd"] for sp in set_points]
        achs = [axis_data[sp]["ss_ach"] for sp in set_points]

        all_vals = cmds + achs
        if any(abs(v) > 1e-6 for v in all_vals):
            lo = min(0, min(all_vals) * 1.1)
            hi = max(all_vals) * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="1:1")

        ax.scatter(cmds, achs, c=range(len(set_points)), cmap="viridis", s=60, zorder=5)
        ax.plot(cmds, achs, "-", color="tab:blue", alpha=0.6)

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

    fig2.suptitle("Differential Drive — Linearity: Achieved vs Commanded (steady-state)", fontsize=13)
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "step_response_diff_linearity.png")
    fig2.savefig(path2, dpi=150)
    print(f"[INFO] Linearity plot saved to {path2}")
    plt.close(fig2)

    # --- Figure 3: Wheel velocities ---
    # Show wheel speeds for each test at the highest set point
    fig3, axes3 = plt.subplots(len(TESTS), 1, figsize=(10, 4 * len(TESTS)), sharex=True)
    if len(TESTS) == 1:
        axes3 = [axes3]

    max_sp = set_points[-1]
    for ax, (label, act_idx, unit) in zip(axes3, TESTS):
        d = results[label][max_sp]
        t = d["time"]

        ax.plot(t, d["wheel_vel_fl"], label="FL (left)", color="tab:blue", linewidth=1.5)
        ax.plot(t, d["wheel_vel_fr"], label="FR (right)", color="tab:orange", linewidth=1.5)
        ax.plot(t, d["wheel_vel_rl"], label="RL (left)", color="tab:blue", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.plot(t, d["wheel_vel_rr"], label="RR (right)", color="tab:orange", linewidth=1.5, linestyle="--", alpha=0.7)

        ax.axvline(x=settle_start * step_dt, color="red", linestyle=":", alpha=0.5)
        ax.set_ylabel("Wheel vel (rad/s)")
        ax.set_title(f"{label} — wheel speeds at action={max_sp:.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle("Differential Drive — Wheel Velocities (FL/RL=left, FR/RR=right)", fontsize=13)
    fig3.tight_layout()
    path3 = os.path.join(out_dir, "step_response_diff_wheels.png")
    fig3.savefig(path3, dpi=150)
    print(f"[INFO] Wheel velocity plot saved to {path3}")
    plt.close(fig3)

    # --- Figure 4: Lateral drift check ---
    # For forward velocity tests, plot vy to check for unwanted lateral motion
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fwd_label = TESTS[0][0]
    for sp, color in zip(set_points, colors):
        d = results[fwd_label][sp]
        ax4.plot(d["time"], d["ach_vy"], "-", color=color, linewidth=1.2, label=f"v={sp:.1f}")

    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax4.axvline(x=settle_start * step_dt, color="red", linestyle=":", alpha=0.5, label="settle")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Lateral velocity vy (m/s)")
    ax4.set_title("Lateral Drift During Forward Velocity Commands")
    ax4.legend(fontsize=7, ncol=4, loc="upper right")
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    path4 = os.path.join(out_dir, "step_response_diff_lateral_drift.png")
    fig4.savefig(path4, dpi=150)
    print(f"[INFO] Lateral drift plot saved to {path4}")
    plt.close(fig4)


if __name__ == "__main__":
    main()
    simulation_app.close()
