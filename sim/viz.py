"""
Visualization module for Intelli Recover Simulation:
1. Animated GIF generator for the 9 runs (S1/S2/S3 x 3 controllers).
2. Five high-resolution diagnostic and comparative plots:
   - Lateral error vs time per scenario (3 controllers overlaid)
   - S2 dynamic thresholding & ambient lux tracking
   - S3 confidence timeline & blind distance recovery
   - Table 5 measured metrics comparison bar charts
   - Dynamic fuzzy-PID gain adaptation profiles
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.metrics import RunMetrics, SimulationStepLog
from sim.track import Obstacle, Track

# Consistent professional color scheme
COLOR_STATIC = "#d9534f"      # Coral Red for baseline StaticPID
COLOR_ADAPTIVE = "#f0ad4e"    # Amber/Gold for AdaptivePID
COLOR_INTELLI = "#198754"     # Forest/Emerald Green for IntelliRecover
CONTROLLER_COLORS = {
    "StaticPID": COLOR_STATIC,
    "AdaptivePID": COLOR_ADAPTIVE,
    "IntelliRecover": COLOR_INTELLI,
}


def render_run_animation(
    track: Track,
    logs: List[SimulationStepLog],
    scenario_name: str,
    controller_name: str,
    output_gif_path: str,
    gap_s_range: Optional[tuple] = None,
    obstacle: Optional[Obstacle] = None,
    subsample: int = 12,
    fps: int = 20,
) -> None:
    """
    Renders top-down animated GIF showing robot, track, 8 IR sensors, trail, and HUD.
    Subsamples frames to ensure fast generation and compact GIF file sizes (~1-2 MB).
    """
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    frames_log = logs[::subsample]
    if not frames_log:
        return

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    # Draw track centerline
    ax.plot(track.waypoints[:, 0], track.waypoints[:, 1], color="#495057", lw=4.0, zorder=2, label="Line Track (18mm)")

    # Draw gap if present
    if gap_s_range is not None:
        idx_gap = (track.s >= gap_s_range[0]) & (track.s <= gap_s_range[1])
        if np.any(idx_gap):
            ax.plot(track.waypoints[idx_gap, 0], track.waypoints[idx_gap, 1],
                    color="#ffffff", lw=5.0, zorder=3)
            ax.plot(track.waypoints[idx_gap, 0], track.waypoints[idx_gap, 1],
                    color="#dc3545", lw=2.5, ls="--", zorder=4, label="Discontinuity Gap (20cm)")

    # Draw obstacle if present
    if obstacle is not None:
        obs_circle = patches.Circle((obstacle.x, obstacle.y), obstacle.radius + 0.02,
                                    facecolor="#fd7e14", edgecolor="#d9534f", lw=2, zorder=5, label="Static Obstacle")
        ax.add_patch(obs_circle)
        ax.text(obstacle.x + 0.06, obstacle.y, "Obstacle", color="#d9534f", fontsize=9, fontweight="bold", zorder=6)

    # Path trail artist
    trail_line, = ax.plot([], [], color=CONTROLLER_COLORS.get(controller_name, "#007bff"),
                          lw=1.8, alpha=0.65, ls="-", zorder=3)

    # Robot chassis rectangle (oriented)
    robot_chassis = patches.Polygon([[0, 0], [0, 0], [0, 0], [0, 0]],
                                    closed=True, facecolor="#212529", edgecolor="#343a40", lw=1.5, zorder=7)
    ax.add_patch(robot_chassis)

    # Heading indicator
    heading_line, = ax.plot([], [], color="#0dcaf0", lw=2.5, zorder=8)

    # 8 IR sensor spots
    sensor_scat = ax.scatter([], [], s=25, c=[], cmap="Greens", vmin=0, vmax=1,
                             edgecolor="#212529", lw=0.5, zorder=9)

    # Dynamic HUD text box
    hud_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", ha="left",
                       fontsize=9.5, fontfamily="monospace",
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#ced4da", alpha=0.92),
                       zorder=10)

    # View limits with margin
    x_min, x_max = float(np.min(track.waypoints[:, 0])), float(np.max(track.waypoints[:, 0]))
    y_min, y_max = float(np.min(track.waypoints[:, 1])), float(np.max(track.waypoints[:, 1]))
    margin = 0.35
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    ax.set_title(f"{scenario_name} — {controller_name}", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # Pre-calculate robot geometry
    l_box, w_box = 0.12, 0.09  # chassis dimensions (m)
    box_corners = np.array([
        [-l_box / 2, -w_box / 2],
        [l_box / 2, -w_box / 2],
        [l_box / 2, w_box / 2],
        [-l_box / 2, w_box / 2]
    ])

    trail_xs, trail_ys = [], []

    def update(frame_idx):
        log = frames_log[frame_idx]
        trail_xs.append(log.true_x)
        trail_ys.append(log.true_y)
        trail_line.set_data(trail_xs, trail_ys)

        # Rotate and translate chassis
        c_th = np.cos(log.true_theta)
        s_th = np.sin(log.true_theta)
        rot_mat = np.array([[c_th, -s_th], [s_th, c_th]])
        trans_box = (rot_mat @ box_corners.T).T + np.array([log.true_x, log.true_y])
        robot_chassis.set_xy(trans_box)

        # Heading arrow
        arrow_len = 0.10
        hx = [log.true_x, log.true_x + arrow_len * c_th]
        hy = [log.true_y, log.true_y + arrow_len * s_th]
        heading_line.set_data(hx, hy)

        # IR sensor positions
        idx = np.arange(8) - 3.5
        offsets = idx * 0.009
        cx = log.true_x + 0.08 * c_th
        cy = log.true_y + 0.08 * s_th
        nx, ny = -s_th, c_th
        sx = cx + offsets * nx
        sy = cy + offsets * ny
        sensor_scat.set_offsets(np.column_stack((sx, sy)))

        # Color sensors: active (green) vs inactive (dim gray)
        active_color = "#20c997" if log.confidence > 0.20 else "#adb5bd"
        sensor_scat.set_color(active_color)

        # Update HUD text
        status_color = "#198754" if log.status in ("TRACKING", "RE-ACQUIRED") else ("#fd7e14" if log.status == "RECOVERING" else "#dc3545")
        hud_str = (
            f"Controller : {controller_name:<14}\n"
            f"Time       : {log.time_s:5.2f} s | Lux: {log.ambient_lux:4.0f} lx\n"
            f"Confidence : {log.confidence:4.2f}   | Status: {log.status}\n"
            f"Threshold  : {log.threshold:5.1f} | LatErr: {log.true_lateral_error_m*1000:+5.1f} mm\n"
            f"Gains      : Kp={log.kp:.2f} Ki={log.ki:.2f} Kd={log.kd:.2f}"
        )
        hud_text.set_text(hud_str)

        return trail_line, robot_chassis, heading_line, sensor_scat, hud_text

    anim = animation.FuncAnimation(fig, update, frames=len(frames_log), blit=True)
    anim.save(output_gif_path, writer="pillow", fps=fps)
    plt.close(fig)


def plot_lateral_error_vs_time(
    all_logs: Dict[str, Dict[str, List[SimulationStepLog]]],
    output_png_path: str
) -> None:
    """
    Plot 1: Lateral error vs time per scenario (3 controllers overlaid).
    Subplots for S1 (Normal), S2 (Lighting), S3 (Gap).
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False, dpi=300)
    scenarios = [("S1", "Scenario 1: Normal Track (Constant 500 Lux)"),
                 ("S2", "Scenario 2: Dynamic Lighting (500 -> 80 -> 900 Lux)"),
                 ("S3", "Scenario 3: Discontinuity (20 cm Gap) & Obstacle Avoidance")]

    for ax, (sc_id, sc_title) in zip(axes, scenarios):
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle=":", alpha=0.5)

        for ctrl in ["StaticPID", "AdaptivePID", "IntelliRecover"]:
            if sc_id in all_logs and ctrl in all_logs[sc_id]:
                logs = all_logs[sc_id][ctrl]
                times = [l.time_s for l in logs]
                errors_mm = [l.true_lateral_error_m * 1000.0 for l in logs]
                ax.plot(times, errors_mm, label=ctrl, color=CONTROLLER_COLORS[ctrl],
                        lw=1.8 if ctrl != "IntelliRecover" else 2.4,
                        alpha=0.85 if ctrl != "IntelliRecover" else 1.0)

        # 5 mm specification band
        ax.axhline(5.0, color="#6c757d", ls="--", lw=1.0, alpha=0.7, label="Spec Limit (+-5mm)" if sc_id == "S1" else "")
        ax.axhline(-5.0, color="#6c757d", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel("Lateral Error (mm)", fontsize=10, fontweight="bold")
        ax.set_title(sc_title, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylim(-28, 28)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    axes[-1].set_xlabel("Simulation Time (s)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)


def plot_lighting_adaptation_s2(
    logs_static: List[SimulationStepLog],
    logs_adaptive: List[SimulationStepLog],
    logs_intelli: List[SimulationStepLog],
    output_png_path: str
) -> None:
    """
    Plot 2: S2 threshold vs time and lux vs time (demonstrating ADTSF).
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

    # Subplot 1: Ambient Lux Profile
    times = [l.time_s for l in logs_intelli]
    lux_vals = [l.ambient_lux for l in logs_intelli]
    ax1.plot(times, lux_vals, color="#495057", lw=2.2, label="Ambient Illumination (Lux)")
    ax1.axvspan(3.0, 6.0, color="#ced4da", alpha=0.3, label="Dim Light Step (80 lx)")
    ax1.axvspan(6.0, 9.0, color="#ffe8cc", alpha=0.35, label="Glare Ramp (up to 900 lx)")
    ax1.set_ylabel("Ambient Lux", fontsize=10, fontweight="bold")
    ax1.set_title("Scenario S2: Dynamic Environmental Illumination Profile", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Subplot 2: Threshold Adaptations
    thresh_static = [l.threshold for l in logs_static]
    thresh_adaptive = [l.threshold for l in logs_adaptive]
    thresh_intelli = [l.threshold for l in logs_intelli]

    ax2.plot(times, thresh_static, color=COLOR_STATIC, lw=1.8, ls="--", label="StaticPID Fixed Threshold (500)")
    ax2.plot(times, thresh_adaptive, color=COLOR_ADAPTIVE, lw=1.8, label="AdaptivePID (ADTSF Dynamic)")
    ax2.plot(times, thresh_intelli, color=COLOR_INTELLI, lw=2.2, label="IntelliRecover (ADTSF Fast Recalibration)")
    ax2.set_xlabel("Simulation Time (s)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("IR Threshold (ADC counts)", fontsize=10, fontweight="bold")
    ax2.set_title("ADTSF Threshold Tracking vs Fixed Baseline Threshold", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)


def plot_gap_recovery_s3(
    logs_intelli: List[SimulationStepLog],
    output_png_path: str
) -> None:
    """
    Plot 3: S3 confidence, recovery status timeline, and blind distance vs time.
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

    times = [l.time_s for l in logs_intelli]
    conf = [l.confidence for l in logs_intelli]
    lat_err = [l.true_lateral_error_m * 1000.0 for l in logs_intelli]

    # Subplot 1: Confidence & Status
    ax1.plot(times, conf, color="#0d6efd", lw=2.0, label="ADTSF Confidence")
    ax1.axhline(0.25, color="#dc3545", ls="--", lw=1.2, label="Discontinuity Threshold (0.25)")
    ax1.axhline(0.50, color="#198754", ls=":", lw=1.2, label="Re-acquisition Threshold (0.50)")

    # Color highlight recovery intervals
    recov_times = [l.time_s for l in logs_intelli if l.status == "RECOVERING"]
    if recov_times:
        ax1.axvspan(min(recov_times), max(recov_times), color="#fff3cd", alpha=0.55, label="PPDR Active Recovery Window")

    ax1.set_ylabel("Confidence Ratio", fontsize=10, fontweight="bold")
    ax1.set_title("Scenario S3: Predictive Path Discontinuity Recovery (PPDR) Timeline", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Lateral Error & Re-entry Damping
    ax2.plot(times, lat_err, color=COLOR_INTELLI, lw=2.0, label="IntelliRecover Lateral Deviation")
    ax2.axhline(5.0, color="#6c757d", ls="--", lw=1.0)
    ax2.axhline(-5.0, color="#6c757d", ls="--", lw=1.0, label="Target Bound (+-5mm)")
    ax2.set_xlabel("Simulation Time (s)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Lateral Deviation (mm)", fontsize=10, fontweight="bold")
    ax2.set_title("Ground-Truth Lateral Deviation Across 20 cm Gap and Re-entry", fontsize=11, fontweight="bold")
    ax2.set_ylim(-20, 20)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)


def plot_gain_adaptation(
    logs_adaptive: List[SimulationStepLog],
    logs_intelli: List[SimulationStepLog],
    output_png_path: str
) -> None:
    """
    Plot 5: Gain adaptation (Kp, Ki, Kd vs time) for HFPID.
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)

    times = [l.time_s for l in logs_intelli]
    kp_vals = [l.kp for l in logs_intelli]
    ki_vals = [l.ki for l in logs_intelli]
    kd_vals = [l.kd for l in logs_intelli]

    ax1.plot(times, kp_vals, color="#e63946", lw=2.0, label="Kp (Proportional Gain)")
    ax1.axhline(0.8, color="#6c757d", ls="--", lw=1.0, label="Nominal Kp0 (0.8)")
    ax1.set_ylabel("Kp Gain", fontsize=9.5, fontweight="bold")
    ax1.set_title("HFPID Takagi-Sugeno Adaptive Gain Tuning During Trajectory", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.plot(times, ki_vals, color="#457b9d", lw=2.0, label="Ki (Integral Gain)")
    ax2.axhline(0.05, color="#6c757d", ls="--", lw=1.0, label="Nominal Ki0 (0.05)")
    ax2.set_ylabel("Ki Gain", fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    ax3.plot(times, kd_vals, color="#2a9d8f", lw=2.0, label="Kd (Derivative Gain)")
    ax3.axhline(0.2, color="#6c757d", ls="--", lw=1.0, label="Nominal Kd0 (0.2)")
    ax3.set_xlabel("Simulation Time (s)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Kd Gain", fontsize=9.5, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)


def plot_table5_metrics_bar_chart(
    df_metrics: pd.DataFrame,
    output_png_path: str
) -> None:
    """
    Plot 4: Bar chart comparing each Table 5 metric across the 3 controllers,
    with target threshold lines drawn (equivalent to Figure 4 in report with measured values).
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    def parse_mean(val_str: str) -> float:
        val_str = str(val_str).strip()
        if "Failed" in val_str or "DNF" in val_str:
            return 0.0
        if "+/-" in val_str:
            try:
                return float(val_str.split("+/-")[0].strip())
            except Exception:
                return 0.0
        if "-" in val_str:
            try:
                parts = val_str.split("-")
                return float(parts[1].strip())
            except Exception:
                return 0.0
        try:
            return float(val_str)
        except Exception:
            return 0.0

    controllers = ["StaticPID", "AdaptivePID", "IntelliRecover"]
    colors = [COLOR_STATIC, COLOR_ADAPTIVE, COLOR_INTELLI]

    charts = [
        ("Line-Tracking Accuracy (%)", axes[0], 90.0, ">=", "%"),
        ("Lateral Deviation (mm)", axes[1], 5.0, "<=", "mm"),
        ("Static Obstacle Recovery Time (ms)", axes[2], 900.0, "<=", "ms"),
        ("Line Discontinuity Recovery (<=20 cm gap) (ms)", axes[3], 1300.0, "<=", "ms"),
        ("Settling Time (90-degree turn) (ms)", axes[4], 600.0, "<=", "ms"),
        ("Continuous Operation Battery Life (min)", axes[5], 90.0, ">=", "min"),
    ]

    for title, ax, target_val, direction, unit in charts:
        row = df_metrics[df_metrics["Metric"] == title]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [parse_mean(str(row[c])) for c in controllers]

        bars = ax.bar(controllers, vals, color=colors, width=0.55, edgecolor="#212529", lw=0.8)
        ax.axhline(target_val, color="#dc3545", ls="--", lw=1.8, label=f"Target ({direction}{target_val}{unit})")

        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        ax.set_ylabel(unit, fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.legend(loc="upper right", fontsize=8)

        # Label values on bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02 * target_val,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2.0, 0.05 * target_val,
                        "DNF", ha="center", va="bottom", fontsize=8.5, color="#dc3545", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)
