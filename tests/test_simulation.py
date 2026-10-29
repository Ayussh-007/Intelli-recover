"""
Sanity tests for Intelli Recover simulation suite.

Tests:
1. Differential-drive kinematics & straight line tracking.
2. ADTSF dynamic threshold recalibration under ambient light shifts.
3. HFPID fuzzy gain tuning output within bounded multipliers.
4. PPDR activation when confidence drops below 0.25 and heading hold behavior.
5. Metrics calculation correctness on known synthetic trajectories.
"""

import sys
import os
import numpy as np
import pytest

# Ensure sim package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.algorithms import (
    ADTSFState,
    HFPIDState,
    PPDRState,
    adtsf_step,
    hfpid_fuzzy_rule_base,
    hfpid_step,
    ppdr_step,
)
from sim.controllers import AdaptivePIDController, IntelliRecoverController, StaticPIDController
from sim.metrics import SimulationStepLog, compute_run_metrics
from sim.robot import Robot, RobotConfig
from sim.track import Track, build_scenario_configs


def test_robot_kinematics():
    """Verify differential-drive forward kinematics and speed bounds."""
    robot = Robot(config=RobotConfig(), seed=42)
    robot.set_pose(0.0, 0.0, 0.0)

    # Commanded forward movement: 0.3 m/s for 1 second (200 physics steps at dt=0.005)
    dt = 0.005
    for _ in range(200):
        robot.step_physics(target_v_l=0.30, target_v_r=0.30, dt=dt)

    # Robot should have moved forward ~0.3 m along +X with heading ~0
    assert 0.25 <= robot.x <= 0.32
    assert abs(robot.y) < 0.02
    assert abs(robot.theta) < 0.05


def test_adtsf_dynamic_recalibration():
    """Verify ADTSF recalibrates threshold when ambient light experiences large delta (> 100 lux)."""
    state = ADTSFState(prev_threshold=500.0, last_lux=500.0)

    # Synthetic IR array at 500 lux: line on sensor 3 & 4
    ir_500 = np.array([210, 205, 215, 820, 830, 205, 195, 200], dtype=float)
    line_pos, conf, thresh_init, is_disc = adtsf_step(ir_500, ambient_lux=500.0, state=state)
    assert conf >= 0.25
    assert not is_disc

    # Sudden step down to 80 lux: floor drops to ~20, line drops to ~380
    ir_80 = np.array([25, 20, 30, 385, 390, 25, 20, 15], dtype=float)
    line_pos_80, conf_80, thresh_80, is_disc_80 = adtsf_step(ir_80, ambient_lux=80.0, state=state)

    # Under 80 lux, threshold should immediately recalibrate below 300 (midpoint of 20 and 390 is ~205)
    assert thresh_80 < 350.0
    # And line should still be detected!
    assert conf_80 >= 0.25


def test_hfpid_gain_bounds():
    """Verify HFPID gains stay strictly within [0.5x, 1.5x] bounds."""
    state = HFPIDState(kp0=0.8, ki0=0.05, kd0=0.2)

    # Large rising error test
    left_v, right_v, kp, ki, kd, corr = hfpid_step(
        line_position_m=0.030, state=state, base_speed=0.30, dt=0.010
    )

    assert 0.40 <= kp <= 1.20001  # 0.8 * [0.5, 1.5] = [0.4, 1.2]
    assert 0.025 <= ki <= 0.07501
    assert 0.10 <= kd <= 0.30001


def test_ppdr_trigger_and_recovery():
    """Verify PPDR activates when confidence < 0.25 and transitions on line recovery."""
    state = PPDRState(max_blind_distance_m=0.20, max_recovery_time_s=1.2)

    # Step 1: Normal confidence
    cmd, status, gain_scale = ppdr_step(
        confidence=0.50, current_time_s=1.0, delta_distance_m=0.003,
        current_heading_rad=0.0, state=state, base_speed=0.30
    )
    assert status == "TRACKING"
    assert not state.recovery_active
    assert cmd is None

    # Step 2: Line lost (confidence drops to 0.0) -> triggers PPDR
    cmd, status, gain_scale = ppdr_step(
        confidence=0.0, current_time_s=1.01, delta_distance_m=0.003,
        current_heading_rad=0.0, state=state, base_speed=0.30
    )
    assert status == "RECOVERING"
    assert state.recovery_active
    assert cmd is not None  # Heading hold command provided

    # Step 3: Advance 0.15 m under recovery
    cmd, status, gain_scale = ppdr_step(
        confidence=0.0, current_time_s=1.50, delta_distance_m=0.15,
        current_heading_rad=0.0, state=state, base_speed=0.30
    )
    assert status == "RECOVERING"

    # Step 4: Line re-detected (confidence >= 0.50) -> re-acquisition
    cmd, status, gain_scale = ppdr_step(
        confidence=0.60, current_time_s=1.60, delta_distance_m=0.01,
        current_heading_rad=0.0, state=state, base_speed=0.30
    )
    assert status == "RE-ACQUIRED"
    assert not state.recovery_active
    assert gain_scale < 1.0  # Damped re-entry factor active


def test_metrics_calculation_synthetic():
    """Verify metric calculations on a known synthetic trajectory."""
    logs = []
    # 100 steps: all perfectly on track with 2 mm lateral error
    for t in np.arange(0, 1.0, 0.01):
        logs.append(SimulationStepLog(
            time_s=float(t),
            true_x=float(t * 0.3),
            true_y=0.0,
            true_theta=0.0,
            dr_x=float(t * 0.3),
            dr_y=0.0,
            dr_theta=0.0,
            s_track=float(t * 0.3),
            true_lateral_error_m=0.002,  # 2 mm
            true_heading_error_rad=0.01,
            ambient_lux=500.0,
            threshold=500.0,
            confidence=0.8,
            status="TRACKING",
            kp=0.8, ki=0.05, kd=0.2,
            line_exists=True,
            obstacle_avoiding=False,
            battery_runtime_min=98.0,
            cycle_latency_ms=15.2,
        ))

    m = compute_run_metrics(
        logs=logs,
        scenario_id="S1",
        controller_name="IntelliRecover",
        seed=1,
        turn_90_s_start=1.5,
        turn_90_s_end=1.8,
    )

    assert m.tracking_accuracy_pct == 100.0
    assert abs(m.mean_lateral_dev_mm - 2.0) < 0.01
    assert m.mean_latency_ms == pytest.approx(15.2, rel=1e-2)
    assert m.battery_life_min == pytest.approx(98.0, rel=1e-2)
