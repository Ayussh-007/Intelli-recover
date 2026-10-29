"""
Robot and Physics Model for Differential-Drive Robot.

Features:
- Differential-drive kinematics at dt = 0.005 s (200 Hz).
- 8-channel IR sensor array (mounted 0.08 m forward, 9 mm spacing).
- Ambient light sensor with realistic phototransistor optical scaling.
- Quadrature encoders with slip and dead-reckoning drift.
- Motor dynamics with first-order lag and PWM speed saturation.
- Sensing-to-actuation latency tracking (hardware delay + perf_counter).
- Battery energy consumption model with duty-cycled power saving.
"""

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple
import numpy as np

from sim.track import Obstacle, Track


@dataclass
class RobotConfig:
    """Robot physical specifications."""
    wheel_base: float = 0.12        # 12 cm axle width
    wheel_radius: float = 0.03      # 3 cm wheel radius
    sensor_forward_dist: float = 0.08  # 8 cm forward from axle
    sensor_spacing: float = 0.009   # 9 mm between sensor centers
    num_sensors: int = 8
    max_wheel_speed: float = 0.50   # 0.5 m/s maximum speed
    base_speed: float = 0.30        # 0.3 m/s nominal forward speed
    motor_time_constant: float = 0.02  # First-order motor lag (s)
    ir_noise_std: float = 12.0      # ADC noise std dev
    encoder_noise_std: float = 0.0002  # Encoder measurement noise (m)
    wheel_slip_rate: float = 0.015  # Wheel slip factor causing dead-reckoning drift
    hardware_latency_ms: float = 14.5  # Fixed sensor/driver hardware latency
    battery_capacity_wh: float = 14.8  # 2000 mAh @ 7.4V Li-ion (14.8 Wh)


class Robot:
    """Differential-drive mobile robot with sensors and power tracking."""

    def __init__(self, config: Optional[RobotConfig] = None, seed: int = 42):
        self.cfg = config or RobotConfig()
        self.rng = np.random.default_rng(seed)

        # Pose: [x (m), y (m), theta (rad)]
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0

        # Wheel actual speeds (m/s)
        self.v_l: float = 0.0
        self.v_r: float = 0.0

        # Dead-reckoning pose estimate (from noisy encoders)
        self.x_dr: float = 0.0
        self.y_dr: float = 0.0
        self.theta_dr: float = 0.0

        # Total distance from encoders
        self.distance_traveled_dr: float = 0.0

        # Sensor positions along array bar (meters)
        # Centered around 0: [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5] * 0.009
        idx = np.arange(self.cfg.num_sensors) - (self.cfg.num_sensors - 1) / 2.0
        self.sensor_offsets_m = idx * self.cfg.sensor_spacing

        # Battery energy tracking (Joules)
        self.total_energy_consumed_j: float = 0.0
        self.simulated_time_s: float = 0.0
        self.duty_cycle_active: bool = False

        # Latency log (ms)
        self.latencies_ms: List[float] = []

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Initializes robot true and dead-reckoning pose."""
        self.x = x
        self.y = y
        self.theta = theta
        self.x_dr = x
        self.y_dr = y
        self.theta_dr = theta
        self.v_l = 0.0
        self.v_r = 0.0
        self.distance_traveled_dr = 0.0
        self.total_energy_consumed_j = 0.0
        self.simulated_time_s = 0.0
        self.latencies_ms.clear()

    def get_ir_sensor_positions(self) -> np.ndarray:
        """
        Returns (8, 2) array of global (x, y) coordinates for all 8 IR sensors.
        Sensor bar is mounted sensor_forward_dist ahead of axle.
        """
        c_th = np.cos(self.theta)
        s_th = np.sin(self.theta)

        # Center of sensor bar
        cx = self.x + self.cfg.sensor_forward_dist * c_th
        cy = self.y + self.cfg.sensor_forward_dist * s_th

        # Normal unit vector perpendicular to robot heading: (-sin(theta), cos(theta))
        nx = -s_th
        ny = c_th

        pts = np.zeros((self.cfg.num_sensors, 2))
        for i, off in enumerate(self.sensor_offsets_m):
            pts[i, 0] = cx + off * nx
            pts[i, 1] = cy + off * ny
        return pts

    def read_ir_sensors(
        self,
        track: Track,
        ambient_lux: float,
        gap_s_range: Optional[Tuple[float, float]] = None,
        closest_hint_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulates 8-channel analog IR reflectance readings.

        Physical optical model:
        - Reflectance has high contrast on line (800 count) vs floor (200 count) at 500 lux.
        - Ambient light changes both DC floor level and contrast:
            Floor_ADC = 200 + 0.8 * (Lux - 500)
            Line_ADC = Floor_ADC + 600 * (Lux / 500)^0.25
        - At 80 lux: Floor ~ 15, Line ~ 415 -> Fixed 500 threshold detects 0 (lost).
        - At 900 lux: Floor ~ 520, Line ~ 950 -> Fixed 500 threshold detects ALL 1s (saturated).
        - Gaussian sensor noise added to each photodiode.
        """
        sensor_pts = self.get_ir_sensor_positions()
        readings = np.zeros(self.cfg.num_sensors)

        # Ambient baseline calculations
        lux_ratio = float(np.clip(ambient_lux / 500.0, 0.05, 3.0))
        dc_floor = 200.0 + 0.80 * (ambient_lux - 500.0)
        contrast = 600.0 * (lux_ratio ** 0.25)

        for i in range(self.cfg.num_sensors):
            sx, sy = sensor_pts[i]
            px, py, s_val, theta_t, e_lat, _ = track.get_closest_point(sx, sy, hint_idx=closest_hint_idx)

            # Check if line exists at this track position (gap check)
            line_exists = True
            if gap_s_range is not None:
                if gap_s_range[0] <= s_val <= gap_s_range[1]:
                    line_exists = False

            # Distance from sensor to line centerline
            dist_to_centerline = abs(e_lat)

            # Line overlap factor (line width 18 mm, half width 9 mm)
            half_w = track.line_width / 2.0
            if line_exists and dist_to_centerline <= half_w:
                # Fully on line
                overlap = 1.0
            elif line_exists and dist_to_centerline <= (half_w + 0.005):
                # Partial footprint overlap
                overlap = (half_w + 0.005 - dist_to_centerline) / 0.005
            else:
                overlap = 0.0

            # Raw signal + noise
            noise = self.rng.normal(0.0, self.cfg.ir_noise_std)
            raw_val = dc_floor + overlap * contrast + noise
            readings[i] = float(np.clip(raw_val, 0.0, 1023.0))

        return readings

    def read_ambient_sensor(self, ambient_lux: float) -> float:
        """Reads ambient light sensor with small measurement noise."""
        noise = self.rng.normal(0.0, 2.5)
        return float(np.clip(ambient_lux + noise, 10.0, 1500.0))

    def detect_obstacle(self, obstacle: Optional[Obstacle]) -> float:
        """
        Simulates forward range sensor detecting obstacle distance ahead of front bumper.
        Returns distance (m), or 999.0 if no obstacle ahead.
        """
        if obstacle is None:
            return 999.0

        # Vector from robot to obstacle
        dx = obstacle.x - self.x
        dy = obstacle.y - self.y
        dist = float(np.hypot(dx, dy))

        # Check if obstacle is roughly in front (angle within +- 45 deg)
        angle_to_obs = np.arctan2(dy, dx)
        angle_diff = (angle_to_obs - self.theta + np.pi) % (2 * np.pi) - np.pi

        if abs(angle_diff) < np.pi / 4.0 and dist < 0.60:
            bumper_dist = max(0.0, dist - obstacle.radius - 0.06)
            return float(bumper_dist)
        return 999.0

    def step_physics(
        self,
        target_v_l: float,
        target_v_r: float,
        dt: float,
    ) -> float:
        """
        Integrates differential-drive physics at 200 Hz (dt = 0.005 s).
        Includes first-order motor lag, wheel velocity saturation,
        encoder measurement with slip, and dead-reckoning accumulation.
        Returns incremental distance traveled in this step (m).
        """
        # Apply speed saturation [-max_speed, max_speed]
        v_l_sat = float(np.clip(target_v_l, -0.20, self.cfg.max_wheel_speed))
        v_r_sat = float(np.clip(target_v_r, -0.20, self.cfg.max_wheel_speed))

        # Motor first-order lag
        alpha_m = dt / (self.cfg.motor_time_constant + dt)
        self.v_l += alpha_m * (v_l_sat - self.v_l)
        self.v_r += alpha_m * (v_r_sat - self.v_r)

        # Differential kinematics
        v_linear = (self.v_l + self.v_r) / 2.0
        omega = (self.v_r - self.v_l) / self.cfg.wheel_base

        # True pose update
        delta_dist_true = v_linear * dt
        self.theta += omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.x += delta_dist_true * np.cos(self.theta)
        self.y += delta_dist_true * np.sin(self.theta)

        # Quadrature encoder simulation with slip and noise
        slip_l = self.rng.normal(self.cfg.wheel_slip_rate, 0.005)
        slip_r = self.rng.normal(self.cfg.wheel_slip_rate, 0.005)
        noise_l = self.rng.normal(0.0, self.cfg.encoder_noise_std)
        noise_r = self.rng.normal(0.0, self.cfg.encoder_noise_std)

        d_wheel_l = (self.v_l * dt) * (1.0 - slip_l) + noise_l
        d_wheel_r = (self.v_r * dt) * (1.0 - slip_r) + noise_r

        # Dead-reckoning pose update
        d_center_dr = (d_wheel_l + d_wheel_r) / 2.0
        d_theta_dr = (d_wheel_r - d_wheel_l) / self.cfg.wheel_base

        self.theta_dr += d_theta_dr
        self.theta_dr = (self.theta_dr + np.pi) % (2 * np.pi) - np.pi
        self.x_dr += d_center_dr * np.cos(self.theta_dr)
        self.y_dr += d_center_dr * np.sin(self.theta_dr)
        self.distance_traveled_dr += abs(d_center_dr)

        # Battery energy consumption update
        # P = P_idle + P_motor + P_sensors
        # Nominal system: idle ~ 1.2 W, motor ~ 4.5 W * (speed/0.3), sensors ~ 1.5 W (or 0.35 W duty-cycled)
        p_idle = 1.20
        motor_factor = (abs(self.v_l) + abs(self.v_r)) / (2.0 * self.cfg.base_speed)
        p_motor = 4.80 * motor_factor
        p_sensors = 0.38 if self.duty_cycle_active else 1.52

        p_total = p_idle + p_motor + p_sensors
        self.total_energy_consumed_j += p_total * dt
        self.simulated_time_s += dt

        return abs(delta_dist_true)

    def log_cycle_latency(self, compute_duration_s: float) -> float:
        """
        Logs sensing-to-actuation latency for this control cycle.
        Latency = Hardware pipeline delay (ADC + bus + driver) + CPU algorithm execution.
        """
        compute_ms = compute_duration_s * 1000.0
        total_latency_ms = self.cfg.hardware_latency_ms + compute_ms
        self.latencies_ms.append(total_latency_ms)
        return total_latency_ms

    def estimate_battery_runtime_minutes(self) -> float:
        """
        Computes continuous operation battery life in minutes based on average power.
        Battery capacity = 14.8 Wh = 53,280 J.
        """
        if self.simulated_time_s <= 0.0 or self.total_energy_consumed_j <= 0.0:
            return 95.0
        avg_power_w = self.total_energy_consumed_j / self.simulated_time_s
        capacity_joules = self.cfg.battery_capacity_wh * 3600.0
        runtime_sec = capacity_joules / max(avg_power_w, 0.5)
        return float(runtime_sec / 60.0)
