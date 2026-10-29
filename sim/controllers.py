"""
Controller implementations for the Intelli Recover simulation:
1. StaticPID: Baseline with fixed threshold and fixed gains, no recovery.
2. AdaptivePID: ADTSF dynamic thresholding + HFPID fuzzy gain tuning, no PPDR recovery.
3. IntelliRecover: Full system with ADTSF + HFPID + PPDR + damped re-entry + obstacle avoidance + power management.
"""

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple
import numpy as np

from sim.algorithms import (
    ADTSFState,
    HFPIDState,
    PPDRState,
    RecoveryEvent,
    adtsf_step,
    hfpid_step,
    ppdr_step,
)
from sim.robot import Robot
from sim.track import Obstacle, Track


@dataclass
class ControllerOutput:
    """Standardized output structure for robot controllers."""
    target_v_l: float
    target_v_r: float
    threshold: float
    line_position: float
    confidence: float
    kp: float
    ki: float
    kd: float
    status: str  # "TRACKING", "RECOVERING", "RE-ACQUIRED", "OBSTACLE_AVOIDING", "LOST"
    duty_cycle_active: bool = False
    re_entry_overshoot_deg: Optional[float] = None


class BaseController:
    """Abstract base class for line following controllers."""

    def __init__(self, name: str, base_speed: float = 0.30):
        self.name = name
        self.base_speed = base_speed
        self.logged_events: List[RecoveryEvent] = []

    def reset(self) -> None:
        self.logged_events.clear()

    def update(
        self,
        robot: Robot,
        track: Track,
        ambient_lux: float,
        dt_ctrl: float,
        obstacle: Optional[Obstacle] = None,
        closest_hint_idx: Optional[int] = None,
    ) -> ControllerOutput:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. StaticPID Controller (Baseline)
# ---------------------------------------------------------------------------

class StaticPIDController(BaseController):
    """
    Baseline controller:
    - Fixed threshold = 500.0 (fails when ambient lux alters reflectance).
    - Fixed gains: Kp0 = 0.8, Ki0 = 0.05, Kd0 = 0.2.
    - No PPDR: on line loss (confidence < 0.25), maintains last command or stops.
    - Continuous 100% sensor polling (higher power consumption).
    """

    def __init__(self, base_speed: float = 0.30, fixed_threshold: float = 500.0):
        super().__init__("StaticPID", base_speed)
        self.fixed_threshold = fixed_threshold
        self.kp = 0.80
        self.ki = 0.05
        self.kd = 0.20
        self.prev_error: float = 0.0
        self.integral: float = 0.0
        self.integral_clamp: float = 0.25
        self.status = "TRACKING"

    def reset(self) -> None:
        super().reset()
        self.prev_error = 0.0
        self.integral = 0.0
        self.status = "TRACKING"

    def update(
        self,
        robot: Robot,
        track: Track,
        ambient_lux: float,
        dt_ctrl: float,
        obstacle: Optional[Obstacle] = None,
        closest_hint_idx: Optional[int] = None,
    ) -> ControllerOutput:
        t0 = time.perf_counter()

        # Check obstacle collision
        obs_dist = robot.detect_obstacle(obstacle)
        if obs_dist < 0.12:
            # Baseline robot has no autonomous avoidance: stops / stuck
            self.status = "LOST"
            compute_time = time.perf_counter() - t0
            robot.log_cycle_latency(compute_time)
            robot.duty_cycle_active = False
            return ControllerOutput(
                target_v_l=0.0, target_v_r=0.0,
                threshold=self.fixed_threshold, line_position=0.0,
                confidence=0.0, kp=self.kp, ki=self.ki, kd=self.kd,
                status="LOST"
            )

        # Fixed threshold binarization
        ir_readings = robot.read_ir_sensors(track, ambient_lux, closest_hint_idx=closest_hint_idx)
        binary = (ir_readings > self.fixed_threshold).astype(np.float64)
        active_count = float(np.sum(binary))
        confidence = active_count / 8.0

        if active_count > 0:
            line_pos = float(np.sum(robot.sensor_offsets_m * binary) / active_count)
            self.status = "TRACKING"
        else:
            # Line lost: StaticPID has no recovery logic, retains last line position or zero
            line_pos = self.prev_error * 0.0315
            self.status = "LOST"

        # Fixed PID
        error_norm = float(np.clip(line_pos / 0.0315, -1.5, 1.5))
        self.integral = float(np.clip(self.integral + error_norm * dt_ctrl,
                                      -self.integral_clamp, self.integral_clamp))
        derivative = (error_norm - self.prev_error) / max(dt_ctrl, 1e-4)

        # If line completely lost, open-loop coast or stop
        if confidence < 0.10:
            # Drifts off or stops
            correction = (self.kp * error_norm) * self.base_speed * 1.5
        else:
            correction = (self.kp * error_norm + self.ki * self.integral + self.kd * derivative * 0.01) * self.base_speed * 1.5

        left_v = self.base_speed - correction
        right_v = self.base_speed + correction

        self.prev_error = error_norm
        compute_time = time.perf_counter() - t0
        robot.log_cycle_latency(compute_time)
        robot.duty_cycle_active = False

        return ControllerOutput(
            target_v_l=left_v, target_v_r=right_v,
            threshold=self.fixed_threshold, line_position=line_pos,
            confidence=confidence, kp=self.kp, ki=self.ki, kd=self.kd,
            status=self.status
        )


# ---------------------------------------------------------------------------
# 2. AdaptivePID Controller
# ---------------------------------------------------------------------------

class AdaptivePIDController(BaseController):
    """
    Adaptive PID controller:
    - ADTSF (Algorithm 1): dynamic thresholding adapts to changing ambient light.
    - HFPID (Algorithm 2): fuzzy tuning for Kp, Ki, Kd gains.
    - NO PPDR: when confidence < 0.25 (line gap), lacks heading hold / recovery state machine.
    """

    def __init__(self, base_speed: float = 0.30):
        super().__init__("AdaptivePID", base_speed)
        self.adtsf_state = ADTSFState()
        self.hfpid_state = HFPIDState()
        self.status = "TRACKING"

    def reset(self) -> None:
        super().reset()
        self.adtsf_state = ADTSFState()
        self.hfpid_state = HFPIDState()
        self.status = "TRACKING"

    def update(
        self,
        robot: Robot,
        track: Track,
        ambient_lux: float,
        dt_ctrl: float,
        obstacle: Optional[Obstacle] = None,
        closest_hint_idx: Optional[int] = None,
    ) -> ControllerOutput:
        t0 = time.perf_counter()

        # Check obstacle collision
        obs_dist = robot.detect_obstacle(obstacle)
        if obs_dist < 0.12:
            self.status = "LOST"
            compute_time = time.perf_counter() - t0
            robot.log_cycle_latency(compute_time)
            robot.duty_cycle_active = False
            return ControllerOutput(
                target_v_l=0.0, target_v_r=0.0,
                threshold=self.adtsf_state.prev_threshold, line_position=0.0,
                confidence=0.0, kp=self.hfpid_state.kp0, ki=self.hfpid_state.ki0, kd=self.hfpid_state.kd0,
                status="LOST"
            )

        # Algorithm 1: ADTSF
        ir_readings = robot.read_ir_sensors(track, ambient_lux, closest_hint_idx=closest_hint_idx)
        line_pos, confidence, thresh, possible_gap = adtsf_step(
            ir_readings, ambient_lux, self.adtsf_state, robot.sensor_offsets_m
        )

        if confidence < 0.25:
            # Without PPDR, AdaptivePID continues to run PID on last/noisy input
            self.status = "LOST"
        else:
            self.status = "TRACKING"

        # Algorithm 2: HFPID
        left_v, right_v, kp, ki, kd, _ = hfpid_step(
            line_pos, self.hfpid_state, self.base_speed, dt_ctrl, gain_scale_factor=1.0
        )

        compute_time = time.perf_counter() - t0
        robot.log_cycle_latency(compute_time)
        robot.duty_cycle_active = False

        return ControllerOutput(
            target_v_l=left_v, target_v_r=right_v,
            threshold=thresh, line_position=line_pos,
            confidence=confidence, kp=kp, ki=ki, kd=kd,
            status=self.status
        )


# ---------------------------------------------------------------------------
# 3. IntelliRecover Controller (Full Proposed Architecture)
# ---------------------------------------------------------------------------

class IntelliRecoverController(BaseController):
    """
    IntelliRecover Full Proposed Architecture:
    - Layer 2: ADTSF (Algorithm 1) for dynamic thresholding and sensor fusion.
    - Layer 3: HFPID (Algorithm 2) for fuzzy adaptive gain tuning.
    - Layer 4: PPDR (Algorithm 3) for predictive path discontinuity recovery.
    - Damped re-entry correction limiting overshoot.
    - Autonomous obstacle circumventing maneuver with smooth re-entry.
    - Layer 5: Duty-cycled sensor power management (saves battery runtime).
    """

    def __init__(self, base_speed: float = 0.30):
        super().__init__("IntelliRecover", base_speed)
        self.adtsf_state = ADTSFState()
        self.hfpid_state = HFPIDState()
        self.ppdr_state = PPDRState()

        # Obstacle avoidance state machine
        self.obstacle_state: str = "CLEAR"  # "CLEAR", "CIRCUMVENT_RIGHT", "BYPASS_FORWARD", "REJOIN_LEFT", "RE_ENTRY_LOCK"
        self.obstacle_timer_s: float = 0.0
        self.obstacle_encounter_time: float = 0.0
        self.obstacle_reacquired_time: float = 0.0
        self.peak_reentry_overshoot_deg: float = 0.0
        self.obstacle_passed: bool = False

    def reset(self) -> None:
        super().reset()
        self.adtsf_state = ADTSFState()
        self.hfpid_state = HFPIDState()
        self.ppdr_state = PPDRState()
        self.obstacle_state = "CLEAR"
        self.obstacle_timer_s = 0.0
        self.obstacle_encounter_time = 0.0
        self.obstacle_reacquired_time = 0.0
        self.peak_reentry_overshoot_deg = 0.0
        self.obstacle_passed = False

    def update(
        self,
        robot: Robot,
        track: Track,
        ambient_lux: float,
        dt_ctrl: float,
        obstacle: Optional[Obstacle] = None,
        closest_hint_idx: Optional[int] = None,
    ) -> ControllerOutput:
        t0 = time.perf_counter()
        current_time = robot.simulated_time_s

        # -------------------------------------------------------------------
        # Autonomous Obstacle Circumventing State Machine
        # -------------------------------------------------------------------
        obs_dist = robot.detect_obstacle(obstacle) if not self.obstacle_passed else 999.0

        if obs_dist < 0.16 and self.obstacle_state == "CLEAR":
            self.obstacle_state = "CIRCUMVENT_RIGHT"
            self.obstacle_encounter_time = current_time
            self.obstacle_timer_s = 0.0

        if self.obstacle_state != "CLEAR":
            self.obstacle_timer_s += dt_ctrl
            v_avoid = self.base_speed * 0.75

            if self.obstacle_state == "CIRCUMVENT_RIGHT":
                # Steer right to clear obstacle radius
                left_v = v_avoid * 1.30
                right_v = v_avoid * 0.65
                if self.obstacle_timer_s > 0.16:
                    self.obstacle_state = "BYPASS_FORWARD"
                    self.obstacle_timer_s = 0.0

            elif self.obstacle_state == "BYPASS_FORWARD":
                # Pass alongside obstacle
                left_v = v_avoid
                right_v = v_avoid
                if self.obstacle_timer_s > 0.20:
                    self.obstacle_state = "REJOIN_LEFT"
                    self.obstacle_timer_s = 0.0

            elif self.obstacle_state == "REJOIN_LEFT":
                # Steer left back toward the track line
                left_v = v_avoid * 0.65
                right_v = v_avoid * 1.30

                # Check if sensor array re-detects the line
                ir_readings = robot.read_ir_sensors(track, ambient_lux, closest_hint_idx=closest_hint_idx)
                line_pos, conf, thresh, _ = adtsf_step(ir_readings, ambient_lux, self.adtsf_state, robot.sensor_offsets_m)

                if conf >= 0.25 and self.obstacle_timer_s > 0.10:
                    self.obstacle_state = "RE_ENTRY_LOCK"
                    self.obstacle_timer_s = 0.0
                    self.obstacle_reacquired_time = current_time
                    self.obstacle_passed = True
                elif self.obstacle_timer_s > 0.35:
                    self.obstacle_state = "RE_ENTRY_LOCK"
                    self.obstacle_timer_s = 0.0
                    self.obstacle_passed = True

            elif self.obstacle_state == "RE_ENTRY_LOCK":
                # Damped re-entry mode: reduced gains to suppress overshoot
                ir_readings = robot.read_ir_sensors(track, ambient_lux, closest_hint_idx=closest_hint_idx)
                line_pos, conf, thresh, _ = adtsf_step(ir_readings, ambient_lux, self.adtsf_state, robot.sensor_offsets_m)

                # Measure heading overshoot relative to track tangent
                _, _, _, theta_t, _, _ = track.get_closest_point(robot.x, robot.y, hint_idx=closest_hint_idx)
                h_err_deg = abs(np.degrees((robot.theta - theta_t + np.pi) % (2 * np.pi) - np.pi))
                self.peak_reentry_overshoot_deg = max(self.peak_reentry_overshoot_deg, h_err_deg)

                # Damped HFPID step (0.50x gain factor)
                left_v, right_v, kp, ki, kd, _ = hfpid_step(
                    line_pos, self.hfpid_state, self.base_speed, dt_ctrl, gain_scale_factor=0.50
                )

                if self.obstacle_timer_s > 0.20:
                    self.obstacle_state = "CLEAR"

            compute_time = time.perf_counter() - t0
            robot.log_cycle_latency(compute_time)
            robot.duty_cycle_active = False

            return ControllerOutput(
                target_v_l=left_v, target_v_r=right_v,
                threshold=self.adtsf_state.prev_threshold, line_position=0.0,
                confidence=0.8, kp=self.hfpid_state.kp0, ki=self.hfpid_state.ki0, kd=self.hfpid_state.kd0,
                status="OBSTACLE_AVOIDING",
                re_entry_overshoot_deg=self.peak_reentry_overshoot_deg
            )

        # -------------------------------------------------------------------
        # Standard Line Tracking with ADTSF + HFPID + PPDR
        # -------------------------------------------------------------------

        # Algorithm 1: ADTSF
        ir_readings = robot.read_ir_sensors(track, ambient_lux, closest_hint_idx=closest_hint_idx)
        line_pos, confidence, thresh, possible_discontinuity = adtsf_step(
            ir_readings, ambient_lux, self.adtsf_state, robot.sensor_offsets_m
        )

        # Algorithm 3: PPDR
        delta_dist_encoder = robot.v_linear_step if hasattr(robot, 'v_linear_step') else (robot.cfg.base_speed * dt_ctrl)
        recovery_cmd, ppdr_status, gain_scale = ppdr_step(
            confidence=confidence,
            current_time_s=current_time,
            delta_distance_m=delta_dist_encoder,
            current_heading_rad=robot.theta_dr,
            state=self.ppdr_state,
            base_speed=self.base_speed,
            logged_events=self.logged_events,
        )

        if recovery_cmd is not None:
            # Under blind predictive heading-hold
            left_v, right_v = recovery_cmd
            kp = self.hfpid_state.kp0
            ki = self.hfpid_state.ki0
            kd = self.hfpid_state.kd0
        else:
            # Algorithm 2: HFPID with optional damped gain_scale
            left_v, right_v, kp, ki, kd, _ = hfpid_step(
                line_pos, self.hfpid_state, self.base_speed, dt_ctrl, gain_scale_factor=gain_scale
            )

        # Layer 5: Duty-cycled sensor power layer
        # When confidence is high and track error is low (robot is centered on straight),
        # duty-cycle sensor polling to save power
        duty_cycle_active = bool(confidence >= 0.70 and abs(line_pos) < 0.006)
        robot.duty_cycle_active = duty_cycle_active

        compute_time = time.perf_counter() - t0
        robot.log_cycle_latency(compute_time)

        return ControllerOutput(
            target_v_l=left_v, target_v_r=right_v,
            threshold=thresh, line_position=line_pos,
            confidence=confidence, kp=kp, ki=ki, kd=kd,
            status=ppdr_status,
            duty_cycle_active=duty_cycle_active,
            re_entry_overshoot_deg=self.peak_reentry_overshoot_deg if self.obstacle_passed else None
        )
