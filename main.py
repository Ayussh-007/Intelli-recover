"""
Main simulation execution script for Intelli Recover.

Runs the complete experiment matrix:
- 3 Scenarios: S1 (Normal), S2 (Lighting Change), S3 (Line Gap & Static Obstacle)
- 3 Controllers: StaticPID, AdaptivePID, IntelliRecover
- N = 10 random seeds per configuration (90 total simulation runs)
- Sub-runs for line gap sensitivity (10 cm, 15 cm, 20 cm)
- Generates 9 animated GIFs (FuncAnimation + Pillow)
- Generates 5 publication-ready diagnostic plots
- Outputs outputs/results.csv and outputs/results_table.md
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add local path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.controllers import (
    AdaptivePIDController,
    BaseController,
    IntelliRecoverController,
    StaticPIDController,
)
from sim.metrics import (
    RunMetrics,
    SimulationStepLog,
    aggregate_metrics_table,
    compute_run_metrics,
    export_markdown_table,
)
from sim.robot import Robot, RobotConfig
from sim.track import (
    LineGap,
    Obstacle,
    ScenarioConfig,
    Track,
    build_scenario_configs,
    get_ambient_lux,
)
from sim.viz import (
    plot_gain_adaptation,
    plot_gap_recovery_s3,
    plot_lateral_error_vs_time,
    plot_lighting_adaptation_s2,
    plot_table5_metrics_bar_chart,
    render_run_animation,
)


def run_simulation(
    track: Track,
    scenario: ScenarioConfig,
    controller_cls: type,
    seed: int,
    dt_physics: float = 0.005,  # 200 Hz physics
    dt_ctrl: float = 0.010,     # 100 Hz control loop
    max_time_s: float = 16.0,
) -> Tuple[List[SimulationStepLog], RunMetrics]:
    """
    Simulates one end-to-end run of a robot along the track.
    Returns:
        logs: list of SimulationStepLog
        metrics: RunMetrics instance
    """
    robot = Robot(config=RobotConfig(), seed=seed)
    controller: BaseController = controller_cls(base_speed=0.30)
    controller.reset()

    # Initial pose at start of track (x=0, y=0, heading=0)
    start_pt = track.waypoints[0]
    start_theta = track.tangent_angles[0]
    robot.set_pose(start_pt[0], start_pt[1], start_theta)

    logs: List[SimulationStepLog] = []
    ctrl_steps_ratio = int(round(dt_ctrl / dt_physics))
    step_count = 0
    sim_time = 0.0

    last_ctrl_out = None
    hint_idx = 0

    gap_range = (scenario.gap.s_start, scenario.gap.s_end) if scenario.gap else None
    obs_s = scenario.obstacle.s_track if scenario.obstacle else None

    # Track progress
    while sim_time < max_time_s:
        # Check current ground truth pose on track
        px, py, s_val, theta_t, e_lat, hint_idx = track.get_closest_point(robot.x, robot.y, hint_idx=hint_idx)
        heading_err = (robot.theta - theta_t + np.pi) % (2 * np.pi) - np.pi
        line_exists = track.is_line_present(s_val, scenario.gap)

        # Ambient lux for current step
        lux = get_ambient_lux(scenario, sim_time, robot.rng)

        # Control loop step (100 Hz)
        if step_count % ctrl_steps_ratio == 0:
            ctrl_out = controller.update(
                robot=robot,
                track=track,
                ambient_lux=lux,
                dt_ctrl=dt_ctrl,
                obstacle=scenario.obstacle,
                closest_hint_idx=hint_idx,
            )
            last_ctrl_out = ctrl_out

        # Step physics (200 Hz)
        target_v_l = last_ctrl_out.target_v_l if last_ctrl_out else 0.0
        target_v_r = last_ctrl_out.target_v_r if last_ctrl_out else 0.0
        delta_dist = robot.step_physics(target_v_l, target_v_r, dt_physics)
        robot.v_linear_step = delta_dist

        # Log telemetry at control rate
        if step_count % ctrl_steps_ratio == 0 and last_ctrl_out is not None:
            cycle_lat = robot.latencies_ms[-1] if robot.latencies_ms else 15.0
            batt_life = robot.estimate_battery_runtime_minutes()
            is_avoiding = (last_ctrl_out.status == "OBSTACLE_AVOIDING")

            step_log = SimulationStepLog(
                time_s=sim_time,
                true_x=robot.x,
                true_y=robot.y,
                true_theta=robot.theta,
                dr_x=robot.x_dr,
                dr_y=robot.y_dr,
                dr_theta=robot.theta_dr,
                s_track=s_val,
                true_lateral_error_m=e_lat,
                true_heading_error_rad=heading_err,
                ambient_lux=lux,
                threshold=last_ctrl_out.threshold,
                confidence=last_ctrl_out.confidence,
                status=last_ctrl_out.status,
                kp=last_ctrl_out.kp,
                ki=last_ctrl_out.ki,
                kd=last_ctrl_out.kd,
                line_exists=line_exists,
                obstacle_avoiding=is_avoiding,
                battery_runtime_min=batt_life,
                cycle_latency_ms=cycle_lat,
            )
            logs.append(step_log)

        sim_time += dt_physics
        step_count += 1

        # Check completion: reached end of closed circuit (s near total_length)
        if s_val > (track.total_length - 0.20):
            break

        # Early termination if robot has completely halted or been lost for > 2.5s
        if last_ctrl_out and last_ctrl_out.status == "LOST" and sim_time > 4.5:
            # Check if stationary
            if abs(robot.v_l) < 0.01 and abs(robot.v_r) < 0.01:
                # Fill remaining logs up to a short buffer and break
                break

    # Compute metrics from ground-truth logs
    metrics = compute_run_metrics(
        logs=logs,
        scenario_id=scenario.scenario_id,
        controller_name=controller.name,
        seed=seed,
        turn_90_s_start=scenario.turn_90_s_start,
        turn_90_s_end=scenario.turn_90_s_end,
        gap_s_range=gap_range,
        obstacle_s=obs_s,
    )

    return logs, metrics


def run_all_experiments(num_seeds: int = 10, output_dir: str = "outputs") -> None:
    """Executes the full experiment matrix and generates all deliverables."""
    print("=" * 76)
    print("  INTELLI RECOVER SIMULATION: AUTONOMOUS VALIDATION SUITE")
    print("=" * 76)
    start_total_time = time.time()

    anim_dir = os.path.join(output_dir, "animations")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(anim_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    track = Track()
    print(f"[Track] Initialized parametric track: Total length = {track.total_length:.2f} m")
    print(f"        Sharp 90-degree corner: s in [{track.turn_90_s_start:.2f}, {track.turn_90_s_end:.2f}] m")

    s1, s2, s3 = build_scenario_configs(track, gap_length=0.20)
    scenarios = [s1, s2, s3]

    controller_map = {
        "StaticPID": StaticPIDController,
        "AdaptivePID": AdaptivePIDController,
        "IntelliRecover": IntelliRecoverController,
    }

    all_metrics: List[RunMetrics] = []
    representative_logs: Dict[str, Dict[str, List[SimulationStepLog]]] = {
        "S1": {}, "S2": {}, "S3": {}
    }

    print(f"\n[Matrix] Executing 3 Scenarios x 3 Controllers x {num_seeds} Seeds = {3 * 3 * num_seeds} runs...")

    for sc in scenarios:
        print(f"\n--- Scenario {sc.scenario_id}: {sc.name} ---")
        for ctrl_name, ctrl_cls in controller_map.items():
            print(f"  -> Running {ctrl_name:<14} (Seeds 1..{num_seeds})...", end="", flush=True)
            ctrl_start = time.time()

            for seed in range(1, num_seeds + 1):
                logs, metrics = run_simulation(
                    track=track,
                    scenario=sc,
                    controller_cls=ctrl_cls,
                    seed=seed,
                )
                all_metrics.append(metrics)

                # Store seed 42 (or seed 1) as representative for animations and plots
                if seed == 1:
                    representative_logs[sc.scenario_id][ctrl_name] = logs

            ctrl_dur = time.time() - ctrl_start
            print(f" Done ({ctrl_dur:.1f}s)")

    # -----------------------------------------------------------------------
    # Gap Sensitivity Sub-Runs (10 cm, 15 cm, 20 cm)
    # -----------------------------------------------------------------------
    print("\n[Gap Sensitivity Analysis] Evaluating PPDR across 10 cm, 15 cm, 20 cm discontinuities...")
    gap_lengths = [0.10, 0.15, 0.20]
    gap_results = {}
    for g_len in gap_lengths:
        success_count = 0
        _, _, sc_gap = build_scenario_configs(track, gap_length=g_len)
        for seed in range(1, num_seeds + 1):
            _, m = run_simulation(track, sc_gap, IntelliRecoverController, seed=seed)
            if m.gap_recovered:
                success_count += 1
        rate = (success_count / num_seeds) * 100.0
        gap_results[f"{int(g_len * 100)}cm"] = rate
        print(f"  Gap {int(g_len * 100)} cm: Success Rate = {rate:.1f}% ({success_count}/{num_seeds})")

    # -----------------------------------------------------------------------
    # Export Results CSV & Markdown Table
    # -----------------------------------------------------------------------
    print("\n[Exporting Metrics] Computing Table 5 aggregates and exporting files...")
    df_runs = pd.DataFrame([m.__dict__ for m in all_metrics])
    csv_path = os.path.join(output_dir, "results.csv")
    df_runs.to_csv(csv_path, index=False)
    print(f"  Saved raw metrics to {csv_path}")

    df_table5 = aggregate_metrics_table(all_metrics)
    md_table_path = os.path.join(output_dir, "results_table.md")
    export_markdown_table(df_table5, md_table_path)
    print(f"  Saved Table 5 markdown to {md_table_path}")

    # Print markdown table to stdout
    print("\n" + "=" * 76)
    print("  MEASURED TABLE 5 RESULTS SUMMARY")
    print("=" * 76)
    print(df_table5.to_string(index=False))
    print("=" * 76)

    # -----------------------------------------------------------------------
    # Generate Visualizations (5 Plots)
    # -----------------------------------------------------------------------
    print("\n[Visualizations] Generating 5 publication-ready diagnostic plots...")
    p1 = os.path.join(plots_dir, "lateral_error_vs_time.png")
    plot_lateral_error_vs_time(representative_logs, p1)
    print(f"  [1/5] Plot saved: {p1}")

    p2 = os.path.join(plots_dir, "lighting_adaptation_s2.png")
    plot_lighting_adaptation_s2(
        representative_logs["S2"]["StaticPID"],
        representative_logs["S2"]["AdaptivePID"],
        representative_logs["S2"]["IntelliRecover"],
        p2
    )
    print(f"  [2/5] Plot saved: {p2}")

    p3 = os.path.join(plots_dir, "gap_recovery_s3.png")
    plot_gap_recovery_s3(representative_logs["S3"]["IntelliRecover"], p3)
    print(f"  [3/5] Plot saved: {p3}")

    p4 = os.path.join(plots_dir, "table5_metrics_comparison.png")
    plot_table5_metrics_bar_chart(df_table5, p4)
    print(f"  [4/5] Plot saved: {p4}")

    p5 = os.path.join(plots_dir, "gain_adaptation.png")
    plot_gain_adaptation(
        representative_logs["S1"]["AdaptivePID"],
        representative_logs["S1"]["IntelliRecover"],
        p5
    )
    print(f"  [5/5] Plot saved: {p5}")

    # -----------------------------------------------------------------------
    # Generate 9 Animated GIFs
    # -----------------------------------------------------------------------
    print("\n[Animations] Rendering 9 animated GIFs (subsampled Pillow render)...")
    anim_idx = 1
    for sc in scenarios:
        gap_range = (sc.gap.s_start, sc.gap.s_end) if sc.gap else None
        for ctrl_name in ["StaticPID", "AdaptivePID", "IntelliRecover"]:
            gif_filename = f"{sc.scenario_id}_{ctrl_name}.gif"
            gif_path = os.path.join(anim_dir, gif_filename)
            t_gif_start = time.time()
            render_run_animation(
                track=track,
                logs=representative_logs[sc.scenario_id][ctrl_name],
                scenario_name=f"{sc.scenario_id}: {sc.name}",
                controller_name=ctrl_name,
                output_gif_path=gif_path,
                gap_s_range=gap_range,
                obstacle=sc.obstacle,
                subsample=10,
                fps=20,
            )
            gif_sz_kb = os.path.getsize(gif_path) / 1024.0
            dur = time.time() - t_gif_start
            print(f"  [{anim_idx}/9] Saved {gif_filename:<24} ({gif_sz_kb:5.1f} KB, {dur:4.1f}s)")
            anim_idx += 1

    total_elapsed = time.time() - start_total_time
    print(f"\n[Complete] All 90 simulation runs, 9 GIFs, 5 plots, CSV, and markdown table generated in {total_elapsed:.1f} s.")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    run_all_experiments(num_seeds=10, output_dir=out_dir)
