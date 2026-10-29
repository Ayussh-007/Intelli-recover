"""
Core algorithms for the Intelli Recover system:
- Algorithm 1: ADTSF (Adaptive Dynamic Thresholding & Sensor Fusion)
- Algorithm 2: HFPID (Hybrid Fuzzy-PID Adaptive Gain Tuning)
- Algorithm 3: PPDR (Predictive Path Discontinuity Recovery)

Implemented strictly according to the microproject specification.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Algorithm 1: ADTSF (Adaptive Dynamic Thresholding & Sensor Fusion)
# ---------------------------------------------------------------------------

@dataclass
class ADTSFState:
    """State memory for the ADTSF algorithm."""
    prev_threshold: float = 500.0
    last_lux: float = 500.0
    history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=20))
    alpha: float = 0.3
    lux_delta_threshold: float = 100.0


def adtsf_step(
    ir_readings: np.ndarray,
    ambient_lux: float,
    state: ADTSFState,
    sensor_positions_m: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, bool]:
    """
    Algorithm 1: Adaptive Dynamic Thresholding & Sensor Fusion (ADTSF).

    Inputs:
        ir_readings: Array of 8 analog IR values.
        ambient_lux: Current ambient lux reading.
        state: ADTSFState holding rolling window and threshold history.
        sensor_positions_m: Physical lateral offset of each sensor (meters).
            Defaults to 8 sensors spaced 9 mm apart centered at 0.

    Outputs:
        line_position: Centered line lateral position (m or normalized [-1, 1]).
        confidence: Active sensor ratio (0.0 to 1.0).
        updated_threshold: Computed threshold for binarization.
        possible_discontinuity: True if confidence < 0.25.
    """
    if sensor_positions_m is None:
        # Default 8 sensors, 9 mm spacing, centered at 0
        sensor_positions_m = np.linspace(-0.0315, 0.0315, 8)

    # Maintain 20-sample rolling history
    state.history.append(ir_readings.copy())

    # Step 1-3: Threshold computation
    lux_delta = abs(ambient_lux - state.last_lux)
    if lux_delta > state.lux_delta_threshold:
        # Fast recalibration: immediate midpoint between current min and max
        min_val = float(np.min(ir_readings))
        max_val = float(np.max(ir_readings))
        updated_threshold = (min_val + max_val) / 2.0
    else:
        # Rolling baseline calculation over window
        stacked = np.array(state.history)
        rolling_min = float(np.min(stacked))
        rolling_max = float(np.max(stacked))
        baseline = (rolling_min + rolling_max) / 2.0
        updated_threshold = state.alpha * baseline + (1.0 - state.alpha) * state.prev_threshold

    # Step 4: Binary classification
    binary = (ir_readings > updated_threshold).astype(np.float64)
    active_count = float(np.sum(binary))

    # Step 5: Weighted centroid
    if active_count > 0:
        line_position = float(np.sum(sensor_positions_m * binary) / active_count)
    else:
        line_position = 0.0

    # Step 6-7: Confidence & discontinuity detection
    confidence = active_count / 8.0
    possible_discontinuity = bool(confidence < 0.25)

    # Step 8: Update state
    state.prev_threshold = updated_threshold
    state.last_lux = ambient_lux

    return line_position, confidence, updated_threshold, possible_discontinuity


# ---------------------------------------------------------------------------
# Algorithm 2: HFPID (Hybrid Fuzzy-PID Adaptive Gain Tuning)
# ---------------------------------------------------------------------------

@dataclass
class HFPIDState:
    """State for HFPID adaptive controller."""
    kp0: float = 0.8
    ki0: float = 0.05
    kd0: float = 0.2
    prev_error: float = 0.0
    integral: float = 0.0
    integral_clamp: float = 0.25
    gain_bound_low: float = 0.5
    gain_bound_high: float = 1.5


def _fuzzify_error(error_abs: float) -> Tuple[float, float, float]:
    """
    Fuzzifies absolute normalized error [0, 1] into {Small, Medium, Large}.
    Triangular / trapezoidal membership functions.
    """
    # Small: 1 at 0, 0 at 0.35
    if error_abs <= 0.15:
        mu_s = 1.0
    elif error_abs < 0.35:
        mu_s = (0.35 - error_abs) / 0.20
    else:
        mu_s = 0.0

    # Medium: 0 at 0.15, 1 at 0.40, 0 at 0.70
    if error_abs <= 0.15 or error_abs >= 0.70:
        mu_m = 0.0
    elif error_abs < 0.40:
        mu_m = (error_abs - 0.15) / 0.25
    else:
        mu_m = (0.70 - error_abs) / 0.30

    # Large: 0 at 0.45, 1 above 0.75
    if error_abs <= 0.45:
        mu_l = 0.0
    elif error_abs < 0.75:
        mu_l = (error_abs - 0.45) / 0.30
    else:
        mu_l = 1.0

    total = mu_s + mu_m + mu_l
    if total > 1e-6:
        return mu_s / total, mu_m / total, mu_l / total
    return 1.0, 0.0, 0.0


def _fuzzify_error_rate(error_rate: float) -> Tuple[float, float, float]:
    """
    Fuzzifies error rate into {Falling, Steady, Rising}.
    Steady is centered near 0.
    """
    # Falling (error magnitude decreasing or negative rate)
    if error_rate <= -0.05:
        mu_f = 1.0
    elif error_rate < 0.0:
        mu_f = -error_rate / 0.05
    else:
        mu_f = 0.0

    # Steady (rate near 0)
    rate_abs = abs(error_rate)
    if rate_abs <= 0.02:
        mu_st = 1.0
    elif rate_abs < 0.08:
        mu_st = (0.08 - rate_abs) / 0.06
    else:
        mu_st = 0.0

    # Rising (error increasing)
    if error_rate <= 0.0:
        mu_r = 0.0
    elif error_rate < 0.05:
        mu_r = error_rate / 0.05
    else:
        mu_r = 1.0

    total = mu_f + mu_st + mu_r
    if total > 1e-6:
        return mu_f / total, mu_st / total, mu_r / total
    return 0.0, 1.0, 0.0


def hfpid_fuzzy_rule_base(error_norm: float, error_rate_norm: float) -> Tuple[float, float, float]:
    """
    Takagi-Sugeno 3x3 Rule Table for (Kp_adj, Ki_adj, Kd_adj).

    Rule Table Structure:
    Error \\ Rate | Falling (F)         | Steady (ST)        | Rising (R)
    --------------+---------------------+--------------------+-------------------
    Small (S)     | (0.9, 1.0, 0.9)     | (1.0, 1.0, 1.0)    | (1.1, 0.9, 1.2)
    Medium (M)    | (1.0, 0.9, 1.2)     | (1.2, 1.0, 1.1)    | (1.3, 0.7, 1.35)
    Large (L)     | (1.1, 0.6, 1.4)     | (1.4, 0.8, 1.2)    | (1.5, 0.5, 1.5)

    Rationale:
    - On large rising error (sharp turns): maximum Kp and Kd for fast damping,
      lower Ki to prevent integrator wind-up.
    - On large falling error (returning to center): moderate Kp, high Kd to prevent overshoot.
    - On small steady error (straight track): baseline nominal gains (1.0, 1.0, 1.0).
    """
    mu_s, mu_m, mu_l = _fuzzify_error(abs(error_norm))
    mu_f, mu_st, mu_r = _fuzzify_error_rate(error_rate_norm)

    # 3x3 rule consequences (Kp_adj, Ki_adj, Kd_adj)
    rules = {
        ('S', 'F'):  (0.90, 1.00, 0.90),
        ('S', 'ST'): (1.00, 1.00, 1.00),
        ('S', 'R'):  (1.10, 0.90, 1.20),
        ('M', 'F'):  (1.00, 0.90, 1.20),
        ('M', 'ST'): (1.20, 1.00, 1.10),
        ('M', 'R'):  (1.35, 0.70, 1.35),
        ('L', 'F'):  (1.15, 0.60, 1.40),
        ('L', 'ST'): (1.40, 0.80, 1.20),
        ('L', 'R'):  (1.50, 0.50, 1.50),
    }

    err_weights = [('S', mu_s), ('M', mu_m), ('L', mu_l)]
    rate_weights = [('F', mu_f), ('ST', mu_st), ('R', mu_r)]

    kp_adj, ki_adj, kd_adj = 0.0, 0.0, 0.0
    for e_name, e_w in err_weights:
        for r_name, r_w in rate_weights:
            w = e_w * r_w
            kp_r, ki_r, kd_r = rules[(e_name, r_name)]
            kp_adj += w * kp_r
            ki_adj += w * ki_r
            kd_adj += w * kd_r

    return kp_adj, ki_adj, kd_adj


def hfpid_step(
    line_position_m: float,
    state: HFPIDState,
    base_speed: float,
    dt: float,
    gain_scale_factor: float = 1.0,
    max_sensor_span_m: float = 0.0315,
) -> Tuple[float, float, float, float, float, float]:
    """
    Algorithm 2: Hybrid Fuzzy-PID Adaptive Gain Tuning (HFPID).

    Inputs:
        line_position_m: Lateral line offset from sensor center (m).
        state: HFPIDState.
        base_speed: Nominal forward speed (m/s).
        dt: Time step (s).
        gain_scale_factor: Multiplier for damped re-entry or recovery mode.
        max_sensor_span_m: Maximum lateral range for normalization.

    Outputs:
        left_v: Target left wheel speed (m/s).
        right_v: Target right wheel speed (m/s).
        kp: Active proportional gain.
        ki: Active integral gain.
        kd: Active derivative gain.
        correction: Net steering correction.
    """
    # 1. Error = Target_Center - Line_Position
    # Line_Position > 0 means line is to the left of robot centerline.
    # To steer towards the line (counter-clockwise / left), right wheel must speed up
    # relative to left wheel (v_R > v_L), so Correction must be positive.
    # Normalizing line position to [-1, 1] range for fuzzy inference
    error_norm = float(np.clip(line_position_m / max_sensor_span_m, -1.5, 1.5))
    error_raw = line_position_m

    # 2. Error_Rate
    error_rate_norm = (error_norm - state.prev_error) / max(dt, 1e-4) * 0.01

    # 3-4. Fuzzy Rule Base
    kp_adj, ki_adj, kd_adj = hfpid_fuzzy_rule_base(error_norm, error_rate_norm)

    # 5. Clamped Gains
    kp = float(np.clip(state.kp0 * kp_adj * gain_scale_factor,
                       state.kp0 * state.gain_bound_low, state.kp0 * state.gain_bound_high))
    ki = float(np.clip(state.ki0 * ki_adj * gain_scale_factor,
                       state.ki0 * state.gain_bound_low, state.ki0 * state.gain_bound_high))
    kd = float(np.clip(state.kd0 * kd_adj * gain_scale_factor,
                       state.kd0 * state.gain_bound_low, state.kd0 * state.gain_bound_high))

    # 6. Anti-windup Integral
    state.integral = float(np.clip(state.integral + error_norm * dt,
                                   -state.integral_clamp, state.integral_clamp))

    # 7. Correction calculation
    # Normalized error used for steering correction
    correction = (kp * error_norm + ki * state.integral + kd * (error_norm - state.prev_error)) * base_speed * 1.5

    # 8. Left & Right wheel target linear speeds
    left_v = base_speed - correction
    right_v = base_speed + correction

    state.prev_error = error_norm
    return left_v, right_v, kp, ki, kd, correction


# ---------------------------------------------------------------------------
# Algorithm 3: PPDR (Predictive Path Discontinuity Recovery)
# ---------------------------------------------------------------------------

@dataclass
class PPDRState:
    """State memory for the PPDR recovery algorithm."""
    recovery_active: bool = False
    start_time_s: float = 0.0
    blind_distance_m: float = 0.0
    last_known_heading_rad: float = 0.0
    status: str = "TRACKING"  # "TRACKING", "RECOVERING", "RE-ACQUIRED", "LOST"
    re_entry_timer_s: float = 0.0
    max_blind_distance_m: float = 0.20  # 20 cm
    max_recovery_time_s: float = 1.20   # 1200 ms
    damped_re_entry_duration_s: float = 0.35  # Damping window after re-acquisition


@dataclass
class RecoveryEvent:
    """Log record of a discontinuity recovery event."""
    start_time: float
    end_time: float
    outcome: str  # "Re-acquired" or "Lost"
    blind_distance_m: float
    re_entry_overshoot_deg: float


def ppdr_step(
    confidence: float,
    current_time_s: float,
    delta_distance_m: float,
    current_heading_rad: float,
    state: PPDRState,
    base_speed: float,
    logged_events: Optional[List[RecoveryEvent]] = None,
) -> Tuple[Optional[Tuple[float, float]], str, float]:
    """
    Algorithm 3: Predictive Path Discontinuity Recovery (PPDR).

    Inputs:
        confidence: Line confidence from ADTSF (0.0 to 1.0).
        current_time_s: Current simulation timestamp.
        delta_distance_m: Wheel distance traveled in current step (from encoders).
        current_heading_rad: Current heading estimate from dead-reckoning.
        state: PPDRState.
        base_speed: Configured base speed.
        logged_events: Optional list to store completed recovery event records.

    Outputs:
        command_override: (left_v, right_v) if recovery active, else None.
        status: "TRACKING", "RECOVERING", "RE-ACQUIRED", or "LOST".
        gain_scale_factor: 1.0 normal, 0.6 during damped re-entry.
    """
    gain_scale_factor = 1.0

    # Step 1: Trigger recovery if confidence drops below 0.25
    if confidence < 0.25 and not state.recovery_active:
        state.recovery_active = True
        state.start_time_s = current_time_s
        state.blind_distance_m = 0.0
        state.last_known_heading_rad = current_heading_rad
        state.status = "RECOVERING"

    # Step 2: Active recovery handling
    if state.recovery_active:
        state.blind_distance_m += delta_distance_m
        elapsed = current_time_s - state.start_time_s

        # Check line re-acquisition: confidence >= 0.5 (or >= 0.25 confirmed line)
        if confidence >= 0.50:
            state.recovery_active = False
            state.status = "RE-ACQUIRED"
            state.re_entry_timer_s = state.damped_re_entry_duration_s

            if logged_events is not None:
                logged_events.append(RecoveryEvent(
                    start_time=state.start_time_s,
                    end_time=current_time_s,
                    outcome="Re-acquired",
                    blind_distance_m=state.blind_distance_m,
                    re_entry_overshoot_deg=0.0  # Measured during re-entry phase
                ))
            return None, state.status, 0.60

        # Check timeout / distance limits: blind distance > 20 cm or elapsed > 1200 ms
        elif state.blind_distance_m > state.max_blind_distance_m or elapsed > state.max_recovery_time_s:
            state.recovery_active = False
            state.status = "LOST"

            if logged_events is not None:
                logged_events.append(RecoveryEvent(
                    start_time=state.start_time_s,
                    end_time=current_time_s,
                    outcome="Lost",
                    blind_distance_m=state.blind_distance_m,
                    re_entry_overshoot_deg=0.0
                ))
            # Safe stop
            return (0.0, 0.0), state.status, 1.0

        else:
            # Command = Hold_Heading(Last_Known_Heading, reduced_speed=True)
            v_blind = base_speed * 0.70
            # Heading error relative to locked heading
            heading_err = state.last_known_heading_rad - current_heading_rad
            # Normalize heading error to [-pi, pi]
            heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
            steer = float(np.clip(2.5 * heading_err, -0.4 * v_blind, 0.4 * v_blind))
            return (v_blind - steer, v_blind + steer), "RECOVERING", 1.0

    # Damped re-entry window handling
    if state.re_entry_timer_s > 0:
        state.re_entry_timer_s -= 0.01  # decrement by control step
        gain_scale_factor = 0.60
        if state.re_entry_timer_s <= 0:
            state.status = "TRACKING"

    return None, state.status, gain_scale_factor
