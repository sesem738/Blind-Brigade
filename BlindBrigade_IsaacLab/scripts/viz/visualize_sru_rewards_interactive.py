"""Interactive visualization of all SRU reward terms.

Left plot:  individual reward terms vs distance to goal
Right plot: combined total reward vs distance + per-step breakdown bar at cursor

Sliders grouped into:
  - Weights (per term)
  - Function parameters (sigma, T_r, max_speed)
  - Assumed robot state (angular vel, heading angle, action delta)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyBboxPatch

# ── Distance axis ──────────────────────────────────────────────────
D = np.linspace(0, 8, 500)

# ── Defaults from SRURewardsCfg ────────────────────────────────────
DEFAULTS = dict(
    # weights
    w_soft=0.25, w_tight=1.5, w_rot=-1e-5, w_heading=-0.2, w_action=-0.2,
    # reach_goal params
    sigma_soft=2.5, sigma_tight=0.25, Tr_soft=1.0, Tr_tight=0.1,
    # heading params
    max_speed=0.2,
    # assumed robot state
    ang_vel=1.0, heading_angle=30.0, speed=0.3, action_delta=0.3,
)


# ── Reward formulas ────────────────────────────────────────────────
def reach_goal(d, sigma, T_r):
    return 1.0 / (1.0 + (d / sigma) ** 2) / T_r


def rot_penalty(ang_vel):
    """||omega_b|| — raw value, constant w.r.t. distance."""
    return np.abs(ang_vel)


def heading_penalty(speed, heading_deg, max_speed):
    """clamp(speed/max_speed) * |heading| / pi."""
    sw = np.clip(speed / max_speed, 0, 1)
    return sw * np.abs(np.radians(heading_deg)) / np.pi


def action_rate_penalty(delta):
    """Sum |Δa| across 3 axes — assume same delta on each."""
    return np.abs(delta) * 3


# ── Compute all terms ──────────────────────────────────────────────
def compute_all(p):
    soft  = reach_goal(D, p["sigma_soft"], p["Tr_soft"]) * p["w_soft"]
    tight = reach_goal(D, p["sigma_tight"], p["Tr_tight"]) * p["w_tight"]
    rot   = np.full_like(D, rot_penalty(p["ang_vel"]) * p["w_rot"])
    head  = np.full_like(D, heading_penalty(p["speed"], p["heading_angle"], p["max_speed"]) * p["w_heading"])
    act   = np.full_like(D, action_rate_penalty(p["action_delta"]) * p["w_action"])
    total = soft + tight + rot + head + act
    return dict(soft=soft, tight=tight, rot=rot, heading=head, action=act, total=total)


# ── Colors ─────────────────────────────────────────────────────────
COLORS = dict(
    soft="#2196F3", tight="#FF5722", rot="#9C27B0",
    heading="#FF9800", action="#4CAF50", total="k",
)
LABELS = dict(
    soft="reach_soft", tight="reach_tight", rot="rot_movement",
    heading="heading_align", action="action_rate", total="combined",
)

# ── Figure layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.subplots_adjust(left=0.06, right=0.72, bottom=0.06, top=0.95, wspace=0.25)

ax_ind = fig.add_subplot(2, 1, 1)   # individual terms
ax_com = fig.add_subplot(2, 1, 2)   # combined + breakdown

# ── Initial plot ───────────────────────────────────────────────────
data0 = compute_all(DEFAULTS)

lines_ind = {}
for key in ["soft", "tight", "rot", "heading", "action"]:
    lw = 2.0
    ls = "-" if key in ("soft", "tight") else "--"
    lines_ind[key], = ax_ind.plot(D, data0[key], color=COLORS[key], lw=lw, ls=ls, label=LABELS[key])

ax_ind.axhline(0, color="gray", lw=0.5)
ax_ind.set_xlabel("Distance to goal (m)")
ax_ind.set_ylabel("Weighted reward per step")
ax_ind.set_title("Individual SRU reward terms")
ax_ind.legend(loc="lower right", fontsize=9)

line_total, = ax_com.plot(D, data0["total"], color="k", lw=2.5, label="total")
# shade positive/negative
fill_pos = ax_com.fill_between(D, 0, data0["total"], where=data0["total"] >= 0, alpha=0.15, color="green")
fill_neg = ax_com.fill_between(D, 0, data0["total"], where=data0["total"] < 0, alpha=0.15, color="red")
fills = [fill_pos, fill_neg]

ax_com.axhline(0, color="gray", lw=0.5)
ax_com.set_xlabel("Distance to goal (m)")
ax_com.set_ylabel("Weighted reward per step")
ax_com.set_title("Combined reward")
ax_com.legend(loc="lower right", fontsize=9)

# ── Sliders (right panel) ─────────────────────────────────────────
slider_x = 0.76
slider_w = 0.20
slider_h = 0.018
gap = 0.028

sliders = {}


def make_slider(y, label, vmin, vmax, vinit, vstep=None, color="lightsteelblue"):
    ax = fig.add_axes([slider_x, y, slider_w, slider_h])
    kw = dict(valinit=vinit)
    if vstep is not None:
        kw["valstep"] = vstep
    s = Slider(ax, label, vmin, vmax, **kw, color=color)
    return s


def add_label(y, text):
    fig.text(slider_x + slider_w / 2, y + 0.008, text,
             ha="center", va="bottom", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="#E0E0E0", ec="none"))


# ── Group: Weights ──
y = 0.92
add_label(y, "Weights")
y -= gap
sliders["w_soft"]    = make_slider(y, "w_soft",     0.0,  5.0, DEFAULTS["w_soft"],    0.05, COLORS["soft"]);    y -= gap
sliders["w_tight"]   = make_slider(y, "w_tight",   0.0, 10.0, DEFAULTS["w_tight"],   0.1,  COLORS["tight"]);   y -= gap
sliders["w_rot"]     = make_slider(y, "w_rot",     -1e-3, 0.0, DEFAULTS["w_rot"],     1e-5, COLORS["rot"]);     y -= gap
sliders["w_heading"] = make_slider(y, "w_head",    -2.0,  0.0, DEFAULTS["w_heading"], 0.05, COLORS["heading"]); y -= gap
sliders["w_action"]  = make_slider(y, "w_action",  -2.0,  0.0, DEFAULTS["w_action"],  0.05, COLORS["action"]);  y -= gap

# ── Group: Function Parameters ──
y -= 0.01
add_label(y, "Function Params")
y -= gap
sliders["sigma_soft"]  = make_slider(y, "σ_soft",  0.1, 5.0, DEFAULTS["sigma_soft"],  0.1);  y -= gap
sliders["sigma_tight"] = make_slider(y, "σ_tight", 0.05, 2.0, DEFAULTS["sigma_tight"], 0.05); y -= gap
sliders["Tr_soft"]     = make_slider(y, "Tr_soft", 0.1, 5.0, DEFAULTS["Tr_soft"],     0.1);  y -= gap
sliders["Tr_tight"]    = make_slider(y, "Tr_tight", 0.01, 2.0, DEFAULTS["Tr_tight"],  0.01); y -= gap
sliders["max_speed"]   = make_slider(y, "max_spd", 0.05, 2.0, DEFAULTS["max_speed"],  0.05); y -= gap

# ── Group: Assumed Robot State ──
y -= 0.01
add_label(y, "Assumed State")
y -= gap
sliders["ang_vel"]       = make_slider(y, "ω (r/s)",  0.0, 5.0,   DEFAULTS["ang_vel"],       0.1);  y -= gap
sliders["heading_angle"] = make_slider(y, "θ_err (°)", 0.0, 180.0, DEFAULTS["heading_angle"], 5.0);  y -= gap
sliders["speed"]         = make_slider(y, "speed",    0.0, 1.5,   DEFAULTS["speed"],         0.05); y -= gap
sliders["action_delta"]  = make_slider(y, "|Δa|",    0.0, 2.0,   DEFAULTS["action_delta"],  0.05); y -= gap


# ── Update callback ───────────────────────────────────────────────
def update(val):
    p = {k: s.val for k, s in sliders.items()}
    data = compute_all(p)

    # Update individual lines
    for key in ["soft", "tight", "rot", "heading", "action"]:
        lines_ind[key].set_ydata(data[key])

    # Auto-scale individual plot
    all_vals = np.concatenate([data[k] for k in ["soft", "tight", "rot", "heading", "action"]])
    ymin, ymax = all_vals.min(), all_vals.max()
    margin = max(0.05, (ymax - ymin) * 0.1)
    ax_ind.set_ylim(ymin - margin, ymax + margin)

    # Update combined
    line_total.set_ydata(data["total"])

    # Remove old fills, draw new
    for f in fills:
        f.remove()
    fills[0] = ax_com.fill_between(D, 0, data["total"], where=data["total"] >= 0, alpha=0.15, color="green")
    fills[1] = ax_com.fill_between(D, 0, data["total"], where=data["total"] < 0, alpha=0.15, color="red")

    ymin_t, ymax_t = data["total"].min(), data["total"].max()
    margin_t = max(0.05, (ymax_t - ymin_t) * 0.1)
    ax_com.set_ylim(ymin_t - margin_t, ymax_t + margin_t)

    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(update)

plt.show()
