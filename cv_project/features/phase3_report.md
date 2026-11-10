# Phase 3 Report: Distribution Distances

Date of the completed run: 2026-11-08T22:41:26Z.  
Device: CUDA, the same `groot` environment used for feature extraction.  
Inputs: cached features in `features/` only. Inception, YOLO, and MiDaS were not run again.

Each row is one distance. Lower means the two distributions are closer. These numbers are not averaged, and a KID value is not on the same scale as a Wasserstein value.

## 1. Estimators

| Distance | Used for | Definition |
|---|---|---|
| Unbiased polynomial MMD | Appearance | Kernel `(x·y / 2048 + 1)^3`. The point estimate is the published Phase 2 value: 100 subsets of 1,000 images, seed 0. |
| Wasserstein-1 | Scalar features | `scipy.stats.wasserstein_distance` on the pooled one-dimensional samples. |
| Jensen–Shannon, base 2 | Class frequencies | Bits, range `[0, 1]`. A missing class has probability 0. The `0·log(0)` terms are omitted. |
| Circular Wasserstein-1 | Flow direction | Radians on the magnitude-weighted 36-bin histogram. Opposite point masses are `pi` apart; adjacent bins are `pi/2` apart. |

Semantic inputs follow the Phase 2 storage rule. Object count includes frames with no box. Confidence and normalized box area use valid detections only; missing detections are not filled with zero. Geometry uses raw MiDaS disparity, where a larger value is closer, with no per-frame z-score. Flow pairs stay inside one episode.

The sim-vs-real class support is the union of classes observed in either domain: 57 COCO class ids. A real-vs-real split uses only classes observed in the full real set, and a sim-vs-sim split uses only classes observed in the full sim set. Those three Jensen–Shannon values therefore live on different supports. The within-domain value is a reference for variability, not a rescaled copy of the cross-domain value.

## 2. Uncertainty

The sim-vs-real interval resamples trajectories, not frames. Each of 200 draws (seed 1) samples 157 simulation episodes with replacement and 157 real episodes with replacement. An episode drawn `k` times contributes its frames, detections, flow pairs, and direction histogram `k` times. The interval is the 2.5 and 97.5 percentiles of those 200 distances. For KID, each draw averages 10 image subsets so that subset noise does not dominate the trajectory interval. The Phase 2 subset standard deviation, 0.004724, remains a different quantity and is not the confidence interval.

Real-vs-real and sim-vs-sim are 50 random non-overlapping halves (78 and 79 episodes; seeds 2 and 3). The table reports the mean and standard deviation of those 50 distances. There is no nested interval around each split. Real-vs-real describes variability inside this real collection. It is not a universal realism threshold.

Sample sizes for the point estimate: 157 episodes on each side; 31,526 sim frames and 11,963 real frames; 9,918 sim detections and 3,852 real detections; 31,369 sim flow pairs and 11,806 real flow pairs.

## 3. Sim versus real

| Metric | Point | 95% trajectory interval | Bootstrap std | Real-vs-real mean (std) | Sim-vs-sim mean (std) |
|---|---:|---:|---:|---:|---:|
| Appearance KID | 0.2360 | 0.2238 – 0.2477 | 0.00647 | 0.00101 (0.00055) | 0.00176 (0.00071) |
| Count, boxes/frame | 0.00956 | 0.00603 – 0.0509 | 0.0120 | 0.0287 (0.0114) | 0.0236 (0.0145) |
| Confidence | 0.0733 | 0.0587 – 0.0877 | 0.00726 | 0.0125 (0.00690) | 0.00706 (0.00424) |
| Box area, fraction of image | 0.4090 | 0.3633 – 0.4506 | 0.0229 | 0.0217 (0.0113) | 0.0334 (0.0231) |
| Class JS, bits | 0.5625 | 0.5232 – 0.6130 | 0.0227 | 0.0292 (0.00611) | 0.0195 (0.00747) |
| Disparity mean | 1.872 | 1.649 – 2.184 | 0.134 | 0.257 (0.134) | 0.384 (0.218) |
| Disparity std | 2.053 | 1.912 – 2.214 | 0.0811 | 0.0875 (0.0433) | 0.126 (0.0702) |
| Disparity p05 | 2.137 | 1.804 – 2.434 | 0.158 | 0.214 (0.108) | 0.216 (0.108) |
| Disparity p50 | 2.037 | 1.823 – 2.327 | 0.130 | 0.314 (0.168) | 0.431 (0.232) |
| Disparity p95 | 4.159 | 3.642 – 4.720 | 0.281 | 0.234 (0.102) | 0.474 (0.251) |
| Gradient mean | 0.1091 | 0.0999 – 0.1220 | 0.00577 | 0.0106 (0.00507) | 0.00762 (0.00352) |
| Gradient p95 | 0.1315 | 0.1167 – 0.1454 | 0.00788 | 0.0136 (0.00635) | 0.0116 (0.00416) |
| Flow magnitude mean, px/frame | 0.5739 | 0.4757 – 0.6993 | 0.0587 | 0.115 (0.0595) | 0.0791 (0.0461) |
| Flow magnitude std | 2.012 | 1.669 – 2.284 | 0.155 | 0.276 (0.137) | 0.186 (0.0966) |
| Flow magnitude p95 | 4.870 | 3.686 – 5.759 | 0.536 | 0.725 (0.377) | 0.638 (0.366) |
| Motion fraction | 0.0426 | 0.0377 – 0.0496 | 0.00327 | 0.00795 (0.00412) | 0.00583 (0.00263) |
| Direction, radians | 0.2385 | 0.2190 – 0.2667 | 0.0119 | 0.0613 (0.0209) | 0.0178 (0.00621) |

Machine-readable copies: `results/phase3/metrics.csv` and `results/phase3/metrics.json`.

## 4. Reading the profile

Appearance is far outside the within-domain reference. The sim-vs-real KID is 0.236, while random halves of the real trajectories give about 0.001 and random halves of the simulation give about 0.002. The trajectory interval, 0.224 to 0.248, does not approach either within-domain value.

Semantic disagreement is concentrated in which detections appear and how large they are, not in how many boxes a frame contains. The count distance, 0.0096 boxes per frame, sits inside the real-vs-real spread (mean 0.0287, and the sim-vs-real interval overlaps that spread). Confidence is separated: 0.073 against a real-vs-real mean of 0.013. Normalized box area is separated by a wide margin: 0.409 against 0.022. Class frequencies are also separated: Jensen–Shannon 0.563 bits against a real-vs-real mean of 0.029 bits. COCO names remain detector responses, not a scene inventory. The large box-area distance matches the Phase 2 observation that a typical simulation detection covers most of the frame.

Geometry is separated on every saved scalar. The mean-disparity distance is 1.87 against a real-vs-real mean of 0.26. The within-frame spread is separated more sharply: disparity-std distance 2.05 against 0.088. The far percentile moves as well: disparity p95 distance 4.16 against 0.23. Spatial gradients agree with that shape difference. These values are raw MiDaS disparity, not meters.

Temporal magnitude and direction are also larger than their within-domain references. Mean flow magnitude differs by 0.57 px/frame, against a real-vs-real mean of 0.12. Direction differs by 0.24 rad, against 0.061 rad inside real data and 0.018 rad inside simulation. Part of this gap is behavior and camera motion. Optical flow here is not robot speed, and the two collections were not paired in time.

## 5. What this phase does not claim

- The seventeen distances are not combined into one realism score.
- Real-vs-real is not used as a pass/fail threshold.
- The KID subset standard deviation is not the trajectory confidence interval.
- No noise, blur, brightness, or color perturbation has been applied. That comparison is a later phase.
- Simulator states, actions, costs, and LiDAR were not read.
