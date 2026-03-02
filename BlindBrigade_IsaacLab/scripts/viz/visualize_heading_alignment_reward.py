"""Visualize the heading_velocity_alignment reward function.

Shows how the penalty varies across velocity space for different
max_speed and weight values to help tune the reward term.

Usage:
    python scripts/viz/visualize_heading_alignment_reward.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Reward function (mirrors rewards.py exactly) ────────────────────────────

def heading_velocity_alignment(vx, vy, max_speed=0.5):
    speed = np.sqrt(vx**2 + vy**2)
    angle_error = np.abs(np.arctan2(vy, vx))          # 0 = forward, pi = backward
    speed_weight = np.clip(speed / max_speed, 0.0, 1.0)
    return speed_weight * (angle_error / np.pi)        # raw penalty in [0, 1]


# ── Grid ─────────────────────────────────────────────────────────────────────

V = 2.0          # velocity axis range (m/s) — rosbot max ~0.5 m/s linear
N = 300
vx = np.linspace(-V, V, N)
vy = np.linspace(-V, V, N)
VX, VY = np.meshgrid(vx, vy)

MAX_SPEEDS  = [0.4, 1.0, 1.2]
WEIGHTS     = [-0.05, -0.1, -0.2, -0.5]

# ── Figure 1: velocity-space heatmaps for each max_speed ─────────────────────

fig1, axes = plt.subplots(1, len(MAX_SPEEDS), figsize=(14, 4.5), constrained_layout=True)
fig1.suptitle("heading_velocity_alignment  —  raw penalty in [0, 1]  (weight not applied)\n"
              "Axes = body-frame velocity (vx forward, vy lateral)", fontsize=11)

for ax, ms in zip(axes, MAX_SPEEDS):
    Z = heading_velocity_alignment(VX, VY, max_speed=ms)
    im = ax.contourf(VX, VY, Z, levels=50, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.contour(VX, VY, Z, levels=[0.25, 0.5, 0.75], colors="white", linewidths=0.8, linestyles="--")
    plt.colorbar(im, ax=ax, label="penalty")

    # annotate key directions
    ax.axhline(0, color="white", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="white", linewidth=0.5, linestyle=":")
    ax.set_xlabel("vx  (m/s)  →  forward")
    ax.set_ylabel("vy  (m/s)  →  left")
    ax.set_title(f"max_speed = {ms} m/s")
    ax.set_aspect("equal")

    # mark the saturation circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ms * np.cos(theta), ms * np.sin(theta), "w-", linewidth=1.5, label=f"saturation @ {ms} m/s")
    ax.legend(fontsize=7, loc="upper left")

# ── Figure 2: 1D slices — penalty vs angle at different speeds ───────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
fig2.suptitle("Penalty vs heading angle at different speeds  |  Effect of weight on training signal",
              fontsize=11)

# Left: angle slice for several speeds, for the chosen max_speed
chosen_ms = 0.5
angles = np.linspace(0, np.pi, 300)  # 0 = forward, pi = backward
speeds_to_plot = [0.1, 0.2, 0.5, 0.8, 1.2]

ax = axes2[0]
for sp in speeds_to_plot:
    # vx = sp*cos(a), vy = sp*sin(a) — robot moving at angle `a` from forward
    pen = np.clip(sp / chosen_ms, 0, 1) * (angles / np.pi)
    ax.plot(np.degrees(angles), pen, label=f"speed = {sp} m/s")

ax.set_xlabel("heading error (°)   0° = pure forward,  180° = pure backward")
ax.set_ylabel("raw penalty")
ax.set_title(f"max_speed = {chosen_ms} m/s")
ax.legend(fontsize=8)
ax.set_xlim(0, 180)
ax.set_ylim(0, 1.05)
ax.axvline(90, color="gray", linestyle="--", linewidth=0.8, label="pure lateral")
ax.grid(True, alpha=0.3)
ax.set_xticks([0, 45, 90, 135, 180])

# Right: weighted penalty per step at max misalignment (moving backward)
#        as a fraction of the total reward budget
ax2 = axes2[1]
worst_raw = 1.0   # speed >= max_speed and moving backward
best_reach_rew = 1.5  # reach_goal_xy_tight weight — a useful reference

for ms in MAX_SPEEDS:
    # penalty at a range of speeds, 90 deg misalignment (pure lateral = common bad case)
    sp_arr = np.linspace(0, 1.5, 300)
    lateral_pen = np.clip(sp_arr / ms, 0, 1) * (90 / 180)  # 90 deg heading error
    ax2.plot(sp_arr, lateral_pen, label=f"max_speed={ms}, 90° error", linestyle="--")

for ms in MAX_SPEEDS:
    sp_arr = np.linspace(0, 1.5, 300)
    back_pen = np.clip(sp_arr / ms, 0, 1) * 1.0             # 180 deg heading error
    ax2.plot(sp_arr, back_pen, label=f"max_speed={ms}, 180° error")

ax2.set_xlabel("robot speed  (m/s)")
ax2.set_ylabel("raw penalty  (multiply by |weight| for actual signal)")
ax2.set_title("Raw penalty vs speed  —  for lateral (90°) and backward (180°) travel")
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="rosbot max speed")

# ── Figure 3: weight sensitivity ─────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
fig3.suptitle("Weighted penalty per step at full backward speed\n"
              "Compare against termination (-50), reach rewards (+0.25, +1.5)", fontsize=10)

ms_chosen = 0.5
angles_deg = np.array([45, 90, 135, 180])
x = np.arange(len(WEIGHTS))
width = 0.18

for i, ang in enumerate(angles_deg):
    raw = np.clip(1.0 / ms_chosen, 0, 1) * (ang / 180)  # at max speed
    weighted = [abs(w) * raw for w in WEIGHTS]
    ax3.bar(x + i * width, weighted, width, label=f"{ang}° error")

ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels([str(w) for w in WEIGHTS])
ax3.set_xlabel("|weight|")
ax3.set_ylabel("penalty magnitude per step")
ax3.legend(title="heading error", fontsize=8)
ax3.axhline(0.1, color="green", linestyle="--", linewidth=1, label="action_rate ref (-0.1)")
ax3.grid(True, alpha=0.3, axis="y")

# reference lines
ax3.text(3.8, 0.105, "action_rate scale", color="green", fontsize=7, va="bottom")

plt.show()
