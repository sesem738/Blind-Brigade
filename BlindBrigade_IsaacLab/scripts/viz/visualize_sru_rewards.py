"""Visualize all SRU reward terms with 2D and 3D plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── Shared config ──────────────────────────────────────────────────
WEIGHTS = {
    "terminating": -50.0,
    "rot_movement": -1e-5,
    "reach_goal_xy_soft": 0.25,
    "reach_goal_xy_tight": 1.5,
    "heading_velocity_alignment": -0.2,
    "action_rate": -0.1,
}


def fig_terminating():
    """is_terminated: binary penalty on collision/termination."""
    fig, ax = plt.subplots(figsize=(5, 3))
    states = ["Alive", "Terminated"]
    raw = [0.0, 1.0]
    weighted = [0.0, WEIGHTS["terminating"]]
    x = np.arange(len(states))
    ax.bar(x - 0.15, raw, 0.3, label="Raw", color="steelblue")
    ax.bar(x + 0.15, weighted, 0.3, label="Weighted", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel("Reward")
    ax.set_title("terminating  (w = -50)")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    return fig


def fig_rot_movement():
    """rot_movement: ||ω_b|| penalized with L2 norm."""
    w = np.linspace(0, 5, 200)  # rad/s
    raw = w  # ||ω||
    weighted = raw * WEIGHTS["rot_movement"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(w, raw, "steelblue", lw=2)
    axes[0].set_xlabel("Angular velocity ||ω|| (rad/s)")
    axes[0].set_ylabel("Raw reward")
    axes[0].set_title("rot_movement — raw")

    axes[1].plot(w, weighted, "salmon", lw=2)
    axes[1].set_xlabel("Angular velocity ||ω|| (rad/s)")
    axes[1].set_ylabel("Weighted reward")
    axes[1].set_title(f"rot_movement — weighted (w = {WEIGHTS['rot_movement']:.0e})")

    fig.tight_layout()
    return fig


def fig_reach_goal():
    """reach_goal_xyz: 1/(1+(d/σ)²)/T_r — soft vs tight, 2D + 3D."""
    # ── Parameters ──
    soft_sigma, soft_Tr, soft_w = 2.5, 1.0, WEIGHTS["reach_goal_xy_soft"]
    tight_sigma, tight_Tr, tight_w = 0.25, 0.1, WEIGHTS["reach_goal_xy_tight"]

    def reward(d, sigma, T_r):
        return 1.0 / (1.0 + (d / sigma) ** 2) / T_r

    # ── 2D: distance vs reward ──
    d = np.linspace(0, 8, 400)
    r_soft = reward(d, soft_sigma, soft_Tr)
    r_tight = reward(d, tight_sigma, tight_Tr)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(d, r_soft, "steelblue", lw=2, label=f"soft (σ={soft_sigma}, T_r={soft_Tr})")
    ax1.plot(d, r_tight, "coral", lw=2, label=f"tight (σ={tight_sigma}, T_r={tight_Tr})")
    ax1.set_xlabel("Distance to goal (m)")
    ax1.set_ylabel("Raw reward")
    ax1.set_title("reach_goal — raw")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(d, r_soft * soft_w, "steelblue", lw=2, label=f"soft × {soft_w}")
    ax2.plot(d, r_tight * tight_w, "coral", lw=2, label=f"tight × {tight_w}")
    ax2.plot(d, r_soft * soft_w + r_tight * tight_w, "k--", lw=1.5, label="combined")
    ax2.set_xlabel("Distance to goal (m)")
    ax2.set_ylabel("Weighted reward")
    ax2.set_title("reach_goal — weighted + combined")
    ax2.legend()

    # ── 3D surface: x_err, y_err → reward ──
    g = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(g, g)
    D = np.sqrt(X**2 + Y**2)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    Z_soft = reward(D, soft_sigma, soft_Tr) * soft_w
    ax3.plot_surface(X, Y, Z_soft, cmap=cm.viridis, alpha=0.85, edgecolor="none")
    ax3.set_xlabel("x error (m)")
    ax3.set_ylabel("y error (m)")
    ax3.set_zlabel("Weighted reward")
    ax3.set_title(f"soft (σ={soft_sigma}, w={soft_w})")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    Z_tight = reward(D, tight_sigma, tight_Tr) * tight_w
    ax4.plot_surface(X, Y, Z_tight, cmap=cm.magma, alpha=0.85, edgecolor="none")
    ax4.set_xlabel("x error (m)")
    ax4.set_ylabel("y error (m)")
    ax4.set_zlabel("Weighted reward")
    ax4.set_title(f"tight (σ={tight_sigma}, w={tight_w})")

    fig.tight_layout()
    return fig


def fig_heading_velocity_alignment():
    """heading_velocity_alignment: speed_weight × |atan2(vy,vx)| / π — 2D + 3D."""
    max_speed = 0.5
    w = WEIGHTS["heading_velocity_alignment"]

    # ── 2D: angle vs penalty at various speeds ──
    angle = np.linspace(-np.pi, np.pi, 400)
    speeds = [0.1, 0.25, 0.5, 0.75, 1.0]

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    for s in speeds:
        sw = np.clip(s / max_speed, 0, 1)
        penalty = sw * np.abs(angle) / np.pi
        ax1.plot(np.degrees(angle), penalty * w, lw=1.8, label=f"speed={s} m/s")
    ax1.set_xlabel("Heading error (deg)")
    ax1.set_ylabel("Weighted reward")
    ax1.set_title(f"heading_velocity_alignment (w = {w})")
    ax1.legend(fontsize=8)
    ax1.axhline(0, color="k", lw=0.5)

    # ── 3D surface: vx, vy → penalty ──
    v = np.linspace(-1.0, 1.0, 200)
    VX, VY = np.meshgrid(v, v)
    speed = np.sqrt(VX**2 + VY**2)
    angle_err = np.abs(np.arctan2(VY, VX))
    sw = np.clip(speed / max_speed, 0, 1)
    Z = sw * angle_err / np.pi * w

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(VX, VY, Z, cmap=cm.coolwarm, alpha=0.85, edgecolor="none")
    # Zero reference lines along vx and vy axes
    zline = np.zeros_like(v)
    ax2.plot(v, zline, zline, color="k", lw=1.5, label="vx axis (vy=0)")
    ax2.plot(zline, v, zline, color="k", lw=1.5, ls="--", label="vy axis (vx=0)")
    ax2.set_xlabel("vx (m/s)")
    ax2.set_ylabel("vy (m/s)")
    ax2.set_zlabel("Weighted reward")
    ax2.set_title("heading_velocity_alignment — body frame")
    ax2.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    return fig


def fig_action_rate():
    """action_rate_l1: Σ|a_t - a_{t-1}| penalized."""
    # For a 3-dim action space (vx, vy, wz), show single-axis delta
    delta = np.linspace(0, 2, 200)
    n_dims = 3
    # Worst case: all dims change equally
    raw_single = delta
    raw_all = delta * n_dims
    w = WEIGHTS["action_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(delta, raw_single, "steelblue", lw=2, label="1 axis changes")
    axes[0].plot(delta, raw_all, "coral", lw=2, label="All 3 axes change equally")
    axes[0].set_xlabel("|Δa| per axis")
    axes[0].set_ylabel("Raw reward (Σ|Δa|)")
    axes[0].set_title("action_rate_l1 — raw")
    axes[0].legend()

    axes[1].plot(delta, raw_single * w, "steelblue", lw=2, label="1 axis changes")
    axes[1].plot(delta, raw_all * w, "coral", lw=2, label="All 3 axes change equally")
    axes[1].set_xlabel("|Δa| per axis")
    axes[1].set_ylabel("Weighted reward")
    axes[1].set_title(f"action_rate_l1 — weighted (w = {w})")
    axes[1].legend()
    axes[1].axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    return fig


def fig_combined_goal():
    """Combined reward landscape: all terms as a function of distance + heading."""
    d = np.linspace(0, 6, 300)

    # Goal rewards (always active for this viz)
    soft_r = 1.0 / (1.0 + (d / 2.5) ** 2) / 1.0 * WEIGHTS["reach_goal_xy_soft"]
    tight_r = 1.0 / (1.0 + (d / 0.25) ** 2) / 0.1 * WEIGHTS["reach_goal_xy_tight"]
    total_goal = soft_r + tight_r

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d, soft_r, "--", color="steelblue", lw=1.5, label="reach_soft")
    ax.plot(d, tight_r, "--", color="coral", lw=1.5, label="reach_tight")
    ax.plot(d, total_goal, "k", lw=2.5, label="combined goal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Distance to goal (m)")
    ax.set_ylabel("Weighted reward per step")
    ax.set_title("Combined goal reward landscape")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    figs = {
        "1_terminating": fig_terminating(),
        "2_rot_movement": fig_rot_movement(),
        "3_reach_goal": fig_reach_goal(),
        "4_heading_velocity_alignment": fig_heading_velocity_alignment(),
        "5_action_rate": fig_action_rate(),
        "6_combined_goal": fig_combined_goal(),
    }

    for name, fig in figs.items():
        fig.savefig(f"sru_reward_{name}.png", dpi=150, bbox_inches="tight")
        print(f"Saved sru_reward_{name}.png")

    plt.show()
