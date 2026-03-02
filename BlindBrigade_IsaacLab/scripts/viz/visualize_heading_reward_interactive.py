"""Interactive visualization of heading_velocity_alignment reward.

Sliders:
  - max_speed: speed at which penalty reaches full strength
  - weight: reward weight multiplier
  - speed_exponent: shape of speed weighting curve (1=linear, 2=quadratic, 0.5=sqrt)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib import cm

# ── Initial parameters ─────────────────────────────────────────────
INIT_MAX_SPEED = 0.5
INIT_WEIGHT = -0.2
INIT_EXPONENT = 1.0
V_RANGE = 1.0
GRID_RES = 200

# ── Precompute grid ────────────────────────────────────────────────
v = np.linspace(-V_RANGE, V_RANGE, GRID_RES)
VX, VY = np.meshgrid(v, v)
speed = np.sqrt(VX**2 + VY**2)
angle_err = np.abs(np.arctan2(VY, VX))  # [0, π]

angle_1d = np.linspace(-np.pi, np.pi, 400)
speed_samples = [0.1, 0.25, 0.5, 0.75, 1.0]


def compute(max_speed, weight, exponent):
    sw = np.clip(speed / max_speed, 0, 1) ** exponent
    Z = sw * angle_err / np.pi * weight
    return Z


def compute_2d(max_speed, weight, exponent):
    lines = []
    for s in speed_samples:
        sw = np.clip(s / max_speed, 0, 1) ** exponent
        penalty = sw * np.abs(angle_1d) / np.pi * weight
        lines.append(penalty)
    return lines


# ── Figure layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(bottom=0.25)

ax_2d = fig.add_subplot(1, 2, 1)
ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

# Initial plots
lines_2d_data = compute_2d(INIT_MAX_SPEED, INIT_WEIGHT, INIT_EXPONENT)
line_objs = []
for i, s in enumerate(speed_samples):
    (ln,) = ax_2d.plot(np.degrees(angle_1d), lines_2d_data[i], lw=1.8, label=f"speed={s}")
    line_objs.append(ln)
ax_2d.axhline(0, color="k", lw=0.5)
ax_2d.set_xlabel("Heading error (deg)")
ax_2d.set_ylabel("Weighted reward")
ax_2d.legend(fontsize=8)
title_2d = ax_2d.set_title(f"max_speed={INIT_MAX_SPEED}, w={INIT_WEIGHT}, exp={INIT_EXPONENT}")

Z0 = compute(INIT_MAX_SPEED, INIT_WEIGHT, INIT_EXPONENT)
surf = [ax_3d.plot_surface(VX, VY, Z0, cmap=cm.coolwarm, alpha=0.85, edgecolor="none")]
zline = np.zeros_like(v)
ax_3d.plot(v, zline, zline, color="k", lw=1.5)        # vx axis
ax_3d.plot(zline, v, zline, color="k", lw=1.5, ls="--")  # vy axis
z_range = np.linspace(Z0.min(), 0, len(v))
z_axis_line = [ax_3d.plot(zline, zline, z_range, color="k", lw=1.5, ls=":")[0]]  # Z axis
ax_3d.set_xlabel("vx (m/s)")
ax_3d.set_ylabel("vy (m/s)")
ax_3d.set_zlabel("Weighted reward")
ax_3d.set_title("Body frame velocity → penalty")

# ── View radio buttons ─────────────────────────────────────────────
# (elev, azim) for each named view
VIEWS = {
    "3D":     (30, -60),
    "XY top": (90, -90),
    "XZ":     (0, -90),
    "YZ":     (0, 0),
}
ax_radio = fig.add_axes([0.91, 0.55, 0.08, 0.20])
radio = RadioButtons(ax_radio, list(VIEWS.keys()), active=0)


def set_view(label):
    elev, azim = VIEWS[label]
    ax_3d.view_init(elev=elev, azim=azim)
    fig.canvas.draw_idle()


radio.on_clicked(set_view)

# ── Sliders ────────────────────────────────────────────────────────
ax_ms = fig.add_axes([0.15, 0.13, 0.65, 0.03])
ax_wt = fig.add_axes([0.15, 0.08, 0.65, 0.03])
ax_ex = fig.add_axes([0.15, 0.03, 0.65, 0.03])

s_ms = Slider(ax_ms, "max_speed", 0.05, 2.0, valinit=INIT_MAX_SPEED, valstep=0.05)
s_wt = Slider(ax_wt, "weight", -1.0, 0.0, valinit=INIT_WEIGHT, valstep=0.01)
s_ex = Slider(ax_ex, "exponent", 0.1, 3.0, valinit=INIT_EXPONENT, valstep=0.1)


def update(val):
    ms = s_ms.val
    wt = s_wt.val
    ex = s_ex.val

    # Update 2D lines
    new_lines = compute_2d(ms, wt, ex)
    for ln, data in zip(line_objs, new_lines):
        ln.set_ydata(data)
    y_min = min(d.min() for d in new_lines)
    ax_2d.set_ylim(y_min * 1.1, max(0.01, -y_min * 0.1))
    title_2d.set_text(f"max_speed={ms:.2f}, w={wt:.2f}, exp={ex:.1f}")

    # Update 3D surface and Z-axis line (remove old, draw new)
    surf[0].remove()
    z_axis_line[0].remove()
    Z = compute(ms, wt, ex)
    surf[0] = ax_3d.plot_surface(VX, VY, Z, cmap=cm.coolwarm, alpha=0.85, edgecolor="none")
    new_z_range = np.linspace(Z.min(), 0, len(v))
    z_axis_line[0] = ax_3d.plot(zline, zline, new_z_range, color="k", lw=1.5, ls=":")[0]
    ax_3d.set_zlim(Z.min() * 1.1, max(0.01, -Z.min() * 0.1))

    fig.canvas.draw_idle()


s_ms.on_changed(update)
s_wt.on_changed(update)
s_ex.on_changed(update)

plt.show()
