"""
Track geometry and scenario configurations for the Intelli Recover simulation.

Implements:
- Continuous parametric track with straights, smooth curves, and sharp 90-degree corner.
- Ground truth lateral distance and heading error calculation.
- Scenario S1: Normal track (constant 500 lux).
- Scenario S2: Dynamic lighting changes (step 500 -> 80 lux, ramp to 900 lux, flicker noise).
- Scenario S3: Line discontinuity (gap of 10/15/20 cm) and static obstacle event.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Obstacle:
    """Static obstacle placed on the track."""
    x: float
    y: float
    radius: float = 0.04  # 4 cm radius
    s_track: float = 0.0


@dataclass
class LineGap:
    """Discontinuity gap in the line."""
    s_start: float
    s_end: float

    @property
    def length(self) -> float:
        return self.s_end - self.s_start


@dataclass
class ScenarioConfig:
    """Configuration for an experiment scenario."""
    scenario_id: str
    name: str
    description: str
    gap: Optional[LineGap] = None
    obstacle: Optional[Obstacle] = None
    nominal_lux: float = 500.0
    turn_90_s_start: float = 0.0
    turn_90_s_end: float = 0.0


class Track:
    """
    Parametric line-following track with ground-truth geometry queries.
    Line width is 18 mm (0.018 m) per project specification.
    """

    def __init__(self, line_width: float = 0.018):
        self.line_width = line_width
        self._generate_track_points()

    def _generate_track_points(self, ds: float = 0.002) -> None:
        """
        Builds a track with:
        1. Straight line (1.0 m)
        2. Smooth 90-deg turn (R = 0.5 m, length = 0.785 m)
        3. Straight line (0.8 m)
        4. Sharp 90-deg left corner (tight fillet R = 0.03 m)
        5. Straight line (1.2 m)
        6. Smooth 90-deg turn (R = 0.5 m, length = 0.785 m)
        7. Straight return to start.
        """
        pts: List[Tuple[float, float]] = []

        # Segment 1: Straight along +X from (0, 0) to (1.1, 0)
        s1_len = 1.1
        for s in np.arange(0, s1_len, ds):
            pts.append((float(s), 0.0))

        # Segment 2: Smooth 90 deg turn from +X to +Y, center (1.1, 0.5), R=0.5
        r2 = 0.5
        cx2, cy2 = 1.1, 0.5
        angles2 = np.linspace(-np.pi / 2, 0, int((r2 * np.pi / 2) / ds), endpoint=False)
        for a in angles2:
            pts.append((cx2 + r2 * np.cos(a), cy2 + r2 * np.sin(a)))

        # Segment 3: Straight along +Y from (1.6, 0.5) to (1.6, 1.3)
        s3_len = 0.8
        for s in np.arange(0, s3_len, ds):
            pts.append((1.6, 0.5 + float(s)))

        # Segment 4: Sharp 90-deg left turn at (1.6, 1.3) to heading +pi (-X direction)
        # Tight fillet R = 0.03 m
        r_sharp = 0.03
        cx_s, cy_s = 1.6 - r_sharp, 1.3
        self.turn_90_idx_start = len(pts)
        angles_sharp = np.linspace(0, np.pi / 2, int((r_sharp * np.pi / 2) / ds), endpoint=False)
        for a in angles_sharp:
            pts.append((cx_s + r_sharp * np.cos(a), cy_s + r_sharp * np.sin(a)))
        self.turn_90_idx_end = len(pts)

        # Segment 5: Straight along -X from (1.57, 1.33) to (0.4, 1.33)
        s5_len = 1.17
        for s in np.arange(0, s5_len, ds):
            pts.append((1.6 - r_sharp - float(s), 1.3 + r_sharp))

        # Segment 6: Smooth turn from -X to -Y, center (0.4, 0.83), R = 0.5
        r6 = 0.5
        cx6, cy6 = 0.4, 1.3 + r_sharp - r6
        angles6 = np.linspace(np.pi / 2, np.pi, int((r6 * np.pi / 2) / ds), endpoint=False)
        for a in angles6:
            pts.append((cx6 + r6 * np.cos(a), cy6 + r6 * np.sin(a)))

        # Segment 7: Straight along -Y from (-0.1, 0.83) to (-0.1, 0.5)
        s7_len = 0.33
        for s in np.arange(0, s7_len, ds):
            pts.append((cx6 - r6, cy6 - float(s)))

        # Segment 8: Smooth turn back to (0, 0), center (0.0, 0.5), R = 0.5 (or direct smooth curve)
        r8 = 0.4
        cx8, cy8 = 0.0, cy6 - s7_len
        angles8 = np.linspace(np.pi, 3 * np.pi / 2, int((r8 * np.pi / 2) / ds), endpoint=True)
        for a in angles8:
            pts.append((cx8 + r8 * np.cos(a), cy8 + r8 * np.sin(a)))

        self.waypoints = np.array(pts)
        self.num_points = len(self.waypoints)

        # Compute cumulative arc length
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        self.total_length = float(self.s[-1])

        # Compute tangent angles
        dx = np.gradient(self.waypoints[:, 0])
        dy = np.gradient(self.waypoints[:, 1])
        self.tangent_angles = np.arctan2(dy, dx)

        self.turn_90_s_start = float(self.s[self.turn_90_idx_start])
        self.turn_90_s_end = float(self.s[self.turn_90_idx_end])

    def get_closest_point(self, x: float, y: float, hint_idx: Optional[int] = None) -> Tuple[float, float, float, float, float, int]:
        """
        Find closest point on track to (x, y).
        Returns:
            proj_x, proj_y, s (arc length), tangent_angle, signed_lateral_error, idx
            signed_lateral_error > 0 when robot is to the LEFT of the line.
        """
        if hint_idx is not None and 0 <= hint_idx < self.num_points:
            # Local window search for speed
            window = 120
            idx_min = max(0, hint_idx - 30)
            idx_max = min(self.num_points, hint_idx + window)
            sub_pts = self.waypoints[idx_min:idx_max]
            dists_sq = (sub_pts[:, 0] - x) ** 2 + (sub_pts[:, 1] - y) ** 2
            local_idx = int(np.argmin(dists_sq))
            idx = idx_min + local_idx
        else:
            dists_sq = (self.waypoints[:, 0] - x) ** 2 + (self.waypoints[:, 1] - y) ** 2
            idx = int(np.argmin(dists_sq))

        px, py = self.waypoints[idx]
        s_val = float(self.s[idx])
        theta = float(self.tangent_angles[idx])

        # Signed lateral error: dot product with unit normal (-sin(theta), cos(theta))
        dx = x - px
        dy = y - py
        e_lat = -dx * np.sin(theta) + dy * np.cos(theta)

        return px, py, s_val, theta, e_lat, idx

    def is_line_present(self, s_val: float, gap: Optional[LineGap]) -> bool:
        """Checks if the line physically exists at arc length s."""
        if gap is None:
            return True
        return not (gap.s_start <= s_val <= gap.s_end)


def build_scenario_configs(track: Track, gap_length: float = 0.20) -> Tuple[ScenarioConfig, ScenarioConfig, ScenarioConfig]:
    """
    Builds the 3 standard experiment scenarios:
    S1: Normal track (500 lux constant)
    S2: Lighting changes (500 -> 80 lux step, ramp to 900 lux)
    S3: Line gap (10/15/20 cm) + static obstacle event
    """
    # S1: Normal track
    s1 = ScenarioConfig(
        scenario_id="S1",
        name="Normal Track",
        description="Closed track with straights, smooth curves and sharp 90-degree turn. Constant 500 lux.",
        gap=None,
        obstacle=None,
        nominal_lux=500.0,
        turn_90_s_start=track.turn_90_s_start,
        turn_90_s_end=track.turn_90_s_end,
    )

    # S2: Lighting change
    s2 = ScenarioConfig(
        scenario_id="S2",
        name="Lighting Change",
        description="Same track with dynamic ambient light: step 500->80 lux, then ramp to 900 lux with flicker.",
        gap=None,
        obstacle=None,
        nominal_lux=500.0,
        turn_90_s_start=track.turn_90_s_start,
        turn_90_s_end=track.turn_90_s_end,
    )

    # S3: Line gap & static obstacle
    # Place gap on straight 1: e.g. from s = 0.50 to s = 0.50 + gap_length
    gap_start = 0.50
    gap = LineGap(s_start=gap_start, s_end=gap_start + gap_length)

    # Place obstacle on straight 5 (at s ~ 3.20 m)
    obs_s = 3.20
    obs_idx = int(np.argmin(np.abs(track.s - obs_s)))
    obs_x, obs_y = track.waypoints[obs_idx]
    obstacle = Obstacle(x=float(obs_x), y=float(obs_y), radius=0.04, s_track=obs_s)

    s3 = ScenarioConfig(
        scenario_id="S3",
        name="Line Gap & Obstacle",
        description=f"Same track with {int(gap_length*100)} cm gap and static obstacle avoidance event.",
        gap=gap,
        obstacle=obstacle,
        nominal_lux=500.0,
        turn_90_s_start=track.turn_90_s_start,
        turn_90_s_end=track.turn_90_s_end,
    )

    return s1, s2, s3


def get_ambient_lux(scenario: ScenarioConfig, time_s: float, rng: np.random.Generator) -> float:
    """
    Computes ambient lux at time_s for the given scenario.
    S1 & S3: Constant ~500 lux with small noise.
    S2: Step 500 -> 80 lux at t=3s, ramp from 80 -> 900 lux from t=6s to t=9s, stay at 900 lux.
    """
    if scenario.scenario_id == "S2":
        if time_s < 3.0:
            lux = 500.0
        elif time_s < 6.0:
            # Step down to 80 lux
            lux = 80.0
        elif time_s < 9.0:
            # Ramp from 80 to 900 lux
            alpha = (time_s - 6.0) / 3.0
            lux = 80.0 + alpha * (900.0 - 80.0)
        else:
            lux = 900.0
        # Add ambient flicker noise (+- 5 lux)
        flicker = rng.normal(0.0, 4.0)
        lux = float(np.clip(lux + flicker, 50.0, 1000.0))
        return lux
    else:
        # Constant lux with subtle noise
        flicker = rng.normal(0.0, 2.0)
        return float(np.clip(scenario.nominal_lux + flicker, 50.0, 1000.0))
