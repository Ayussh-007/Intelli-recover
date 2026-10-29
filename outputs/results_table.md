# Table 5: Measured Simulation Performance vs. Report Specifications

Comparison of baseline **StaticPID**, intermediate **AdaptivePID**, and full **IntelliRecover** architecture across 10 random seeds (mean +/- std).

| Metric | Target | StaticPID | AdaptivePID | IntelliRecover | Target met? (IntelliRecover) |
|---|---|---|---|---|:---:|
| Line-Tracking Accuracy (%) | >= 90 | 62.2 +/- 36.57 | 98.5 +/- 0.03 | 98.5 +/- 0.03 | **PASS** |
| Lateral Deviation (mm) | <= 5 | 2.4 +/- 0.01 | 2.5 +/- 0.01 | 2.5 +/- 0.01 | **PASS** |
| Static Obstacle Recovery Time (ms) | <= 900 | Failed / DNF | Failed / DNF | 884.0 +/- 6.63 | **PASS** |
| Line Discontinuity Recovery (<=20 cm gap) (ms) | <= 1300 | 670.0 +/- 0.00 | 670.0 +/- 0.00 | 670.0 +/- 0.00 | **PASS** |
| Sensing-to-Actuation Latency (ms) | <= 60 | 14.6 +/- 0.00 | 14.6 +/- 0.01 | 14.6 +/- 0.01 | **PASS** |
| Settling Time (90-degree turn) (ms) | <= 600 | 352.0 +/- 6.00 | 348.0 +/- 6.00 | 348.0 +/- 6.00 | **PASS** |
| Illumination Tolerance Range (lux) | 50-1000 | 400 - 620 | 50 - 1000 | 50 - 1000 | **PASS** |
| Re-entry Overshoot after Obstacle (deg) | <= 5 | 18.5 +/- 0.00 | 15.4 +/- 0.00 | 4.9 +/- 0.12 | **PASS** |
| Control-Loop Update Rate (Hz) | 100-200 | 100.0 +/- 0.00 | 100.0 +/- 0.00 | 100.0 +/- 0.00 | **PASS** |
| Continuous Operation Battery Life (min) | >= 90 | 118.5 +/- 0.29 | 118.4 +/- 0.25 | 119.1 +/- 0.76 | **PASS** |
