# Phase 2 Report: Visual Feature Extraction

Date of the completed run: 2026-11-08T22:06:05Z.  
Device: NVIDIA GeForce RTX 5080 Laptop GPU, CUDA, PyTorch 2.9.0+cu128.  
Wall time of the full extraction: 1.97 hours.  
This report covers feature extraction only. Distribution distances other than KID, Real-vs-Real references, and trajectory-level confidence intervals are not part of this phase.

## 1. Inputs

Both domains are unpaired RGB trajectories. Robot states, actions, costs, LiDAR, and simulator ground truth were not read.

| | Simulation | Real |
|---|---:|---:|
| Directory | `data/sim_obses_15fps/obses_15fps` | `data/real_obses_15fps/obses_15fps` |
| Episodes | 157 (`episode_0` … `episode_156`) | 157 (`episode_0` … `episode_156`) |
| Frames | 31,526 | 11,963 |
| Tensor | `uint8`, shape `(T, 480, 640, 3)`, RGB, range 0–255 | same |
| Declared rate | 15 fps | 15 fps |
| Duration at 15 fps | 2,101.73 s | 797.53 s |
| Per-frame timestamps inside the RGB tensor | none | none |

Frame order inside an episode is the tensor index. Consecutive-frame operators never cross an episode boundary. The same code path and the same hyperparameters are used after the file is loaded. There is no simulator-specific branch in the extractors.

Episode length differs. Simulation episodes contain 44–503 frames (mean 200.8). Real episodes contain 35–355 frames (mean 76.2). Features are stored per episode, so later metrics can resample by trajectory rather than by frame.

## 2. Interface and storage

Entry point: `phase2.extract.extract_features`.

```text
features = extract_features(config, domains, modules, limit_episodes)
```

Modules, in order, are `appearance`, `semantic`, `geometry`, and `temporal`. Each finished episode is written before the next one starts. An existing `episode_XXXX.npz` for a module is skipped, so a rerun does not repeat that inference.

```text
features/
  sim/{appearance,semantic,geometry,temporal}/episode_0000.npz … episode_0156.npz
  real/{appearance,semantic,geometry,temporal}/episode_0000.npz … episode_0156.npz
  yolo_classes.tsv
  run_meta.json
  kid.json
```

All four modules are present for all 157 episodes on both sides. `run_meta.json` records the device, package version, module list, and the resolved configuration.

## 3. Appearance

**Model.** `torchvision.models.inception_v3`, weights `Inception_V3_Weights.DEFAULT` (checkpoint `inception_v3_google-0cc3c7bd.pth`). `torchvision` 0.24.0+cu128. The final fully connected layer is replaced by an identity, so the saved vector is the 2048-dimensional pool feature. `transform_input` is disabled because normalization is applied explicitly. Inference uses `eval` mode and `torch.inference_mode`. Batch size 32.

**Preprocessing, identical for both domains.**

1. Convert `uint8` RGB to float and divide by 255.
2. Bilinear resize from 480×640 to 299×299 (`align_corners=False`). The full frame is kept. There is no center crop.
3. ImageNet normalization: mean `(0.485, 0.456, 0.406)`, standard deviation `(0.229, 0.224, 0.225)`.

**Saved per frame.** `embedding` `(T, 2048)` float32, `rgb_mean` `(T, 3)`, `rgb_std` `(T, 3)`, `frame_index` `(T,)`.

**KID.** Computed from the saved embeddings, not from a second forward pass. The estimator is the unbiased polynomial MMD of Binkowski et al. For embeddings \(x, y \in \mathbb{R}^{d}\) with \(d = 2048\),

\[
k(x,y) = \left(\frac{x^{\top} y}{d} + 1\right)^{3}.
\]

For a pair of subsets \(X, Y\) of size \(n = m = 1000\),

\[
\mathrm{KID}(X,Y)
= \frac{\sum_{i \neq j} k(x_i,x_j)}{n(n-1)}
+ \frac{\sum_{i \neq j} k(y_i,y_j)}{m(m-1)}
- \frac{2}{nm}\sum_{i,j} k(x_i,y_j).
\]

One hundred subsets were drawn without replacement, using NumPy seed 0. The same subset size was used for both domains. The reported uncertainty is the sample standard deviation across those 100 subset scores (`ddof=1`). It is the sampling uncertainty of this KID estimator. It is not a trajectory-level bootstrap.

| Quantity | Value |
|---|---:|
| Sim embeddings | 31,526 |
| Real embeddings | 11,963 |
| Subset size | 1,000 |
| Repeats | 100 |
| Seed | 0 |
| KID mean | 0.236023 |
| KID std | 0.004724 |

Lower KID means the Inception feature distributions are closer. This number is not on the same scale as a Wasserstein distance and should not be averaged with the other three gaps.

**RGB statistics, frame means then averaged over frames.**

| Channel | Sim mean | Sim std of frame means | Real mean | Real std of frame means |
|---|---:|---:|---:|---:|
| R | 131.683 | 26.066 | 107.859 | 13.775 |
| G | 124.337 | 26.062 | 112.520 | 9.341 |
| B | 121.379 | 25.535 | 114.200 | 10.177 |

Mean within-frame channel standard deviation: simulation `(41.97, 42.85, 42.51)`, real `(28.74, 20.07, 23.03)`. Simulation frames are brighter, more variable across the trajectory, and have higher within-frame contrast. Real frames are slightly blue-weighted. These statistics are descriptive. They are not the KID.

## 4. Semantic

**Model.** Ultralytics YOLOv8n, package 8.4.208, weights `cv_project/yolov8n.pt`. Confidence threshold 0.25, the same threshold as the existing course notebook. Batch size 16. Class names are the COCO names written to `yolo_classes.tsv`.

**Channel order.** Stored frames are RGB. Ultralytics treats a NumPy array as BGR, so each frame is channel-reversed before `predict`. The network therefore receives RGB, matching the notebook path that reads files with OpenCV.

**Empty frames.** If a frame has no box at or above 0.25, its object count is 0. No confidence, class, or box row is written for that frame. Confidence is not filled with zero.

**Saved quantities.** Per frame: `count`. Per detection: source frame index, confidence, class id, `xyxy` in pixels, width and height divided by 640 and 480, normalized area (product of those two), and normalized box center.

| | Simulation | Real |
|---|---:|---:|
| Frames | 31,526 | 11,963 |
| Frames with at least one box | 28.30% | 28.20% |
| Frames with zero boxes | 71.70% | 71.80% |
| Mean boxes per frame | 0.315 | 0.322 |
| Std of boxes per frame | 0.531 | 0.553 |
| Median / 95th percentile of count | 0 / 1 | 0 / 1 |
| Detections | 9,918 | 3,852 |
| Distinct COCO classes | 40 | 45 |
| Confidence mean / std / median | 0.388 / 0.122 / 0.351 | 0.461 / 0.170 / 0.417 |
| Normalized area mean / std / median | 0.641 / 0.408 / 0.958 | 0.232 / 0.192 / 0.198 |
| Mean normalized width, height | 0.710, 0.779 | 0.358, 0.535 |

The two domains have almost the same fraction of frames with a detection and almost the same mean count. The conditional distributions differ. Real detections have higher confidence and smaller boxes. Simulation boxes are often large: the median normalized area is 0.958, so a typical simulation box covers most of the frame.

Most frequent classes, as a fraction of detections rather than of frames:

| Rank | Simulation | Share | Real | Share |
|---|---|---:|---|---:|
| 1 | bed | 33.88% | vase | 38.81% |
| 2 | umbrella | 19.83% | person | 11.94% |
| 3 | person | 10.68% | suitcase | 8.72% |
| 4 | cake | 6.07% | kite | 7.76% |
| 5 | pizza | 4.87% | cup | 7.17% |
| 6 | vase | 4.46% | airplane | 4.60% |
| 7 | cat | 3.31% | chair | 2.62% |
| 8 | cup | 2.81% | cell phone | 2.36% |

These labels are COCO names emitted by a detector that was not trained on this bin-and-floor scene. A “bed” or “vase” label is not evidence that the scene contains that object. The semantic module measures how the fixed detector responds, including its domain-specific errors. It does not measure geometric correctness of the simulator.

## 5. Geometry

**Model.** Intel MiDaS `DPT_Large`, loaded with `torch.hub` from the cached `intel-isl/MiDaS` repository and checkpoint `dpt_large_384.pt`. `timm` 1.0.31. Batch size 8.

**Preprocessing.** The hub `dpt_transform` is applied to the RGB `uint8` frame, the same transform used by the course notebook. Because the tensor is already RGB, it is not converted from BGR. The network output is bilinearly upsampled with `bicubic` interpolation and `align_corners=False` back to 480×640.

**Convention.** The raw DPT output is a disparity-like relative value: a larger number means closer, not a larger metric distance. This was checked on the real episode-0 frame. The nearby floor at the bottom of the downward-looking view has a higher mean (26.89) than the more distant bin region at the top (12.72). No per-frame shift or scale is applied. Mean and variance remain comparable across frames and across domains. Full disparity maps are not stored.

**Saved per frame.** `disparity_mean`, `disparity_std`, percentiles 5, 25, 50, 75, and 95, plus `grad_mean` and `grad_p95`. The gradient is the magnitude of a 3×3 Sobel filter on the raw disparity map.

| Feature | Sim mean | Sim std | Sim median | Real mean | Real std | Real median |
|---|---:|---:|---:|---:|---:|---:|
| disparity mean | 19.406 | 5.031 | 20.657 | 18.438 | 3.602 | 18.363 |
| disparity std | 8.898 | 1.703 | 9.134 | 6.845 | 1.567 | 6.876 |
| disparity p05 | 5.979 | 2.739 | 6.361 | 8.116 | 2.674 | 8.200 |
| disparity p50 | 19.350 | 5.618 | 20.943 | 18.257 | 3.964 | 18.548 |
| disparity p95 | 33.612 | 6.455 | 34.506 | 29.553 | 5.139 | 29.059 |
| grad mean | 0.534 | 0.096 | 0.536 | 0.425 | 0.111 | 0.420 |
| grad p95 | 0.668 | 0.189 | 0.649 | 0.536 | 0.155 | 0.510 |

Simulation disparity maps are slightly higher on average, more spread out within a frame, and have stronger spatial gradients. These are statistics of the monocular predictor, not metric depth errors.

## 6. Temporal

**Algorithm.** Farneback dense optical flow, OpenCV 4.11.0. Parameters are copied from the course notebook: `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`. Input is grayscale converted from RGB. Flow is computed only from frame \(t\) to \(t+1\) inside the same episode. The first frame of every episode has no pair. There are 31,369 simulation pairs and 11,806 real pairs.

**Saved per pair.**

- `mag_mean`, `mag_std`, `mag_p95`: mean, standard deviation, and 95th percentile of the flow magnitude, in pixels per frame.
- `motion_frac`: fraction of pixels whose magnitude exceeds 1.0 pixel.
- `dir_cos`, `dir_sin`: magnitude-weighted mean of \(\cos\theta\) and \(\sin\theta\), with \(\theta = \operatorname{atan2}(f_y, f_x)\). This keeps \(-\pi\) and \(\pi\) on the same circle.
- `dir_hist`: 36 bins on \([-\pi, \pi)\), weighted by flow magnitude and not normalized. Histograms can be summed across pairs before a circular comparison.

| Feature | Sim mean | Sim std | Sim median | Real mean | Real std | Real median |
|---|---:|---:|---:|---:|---:|---:|
| mag mean (px/frame) | 1.615 | 1.345 | 1.263 | 1.068 | 1.469 | 0.441 |
| mag std | 5.642 | 3.831 | 5.159 | 3.655 | 4.153 | 2.437 |
| mag p95 | 10.935 | 10.509 | 7.403 | 6.135 | 9.647 | 0.673 |
| motion fraction | 0.114 | 0.070 | 0.103 | 0.075 | 0.088 | 0.042 |

Simulation pairs move more on average. The real median magnitude is 0.441 px/frame, against a simulation median of 1.263. Part of this difference can come from different amounts of camera ego-motion and from different episode lengths, not only from rendering. The module measures the distribution of image motion. It does not measure physical velocity error.

## 7. What this phase does not claim

- The four feature families are not combined into one realism score.
- KID and the scalar feature summaries are not on a common scale.
- COCO class names are detector outputs, not scene labels.
- MiDaS values are not meters.
- Optical flow is not a measurement of robot speed.
- The KID standard deviation is subset noise. A trajectory bootstrap, a Real-vs-Real split, and Wasserstein or Jensen–Shannon distances on the saved tables are the next phase.
- No image was blurred, noised, recolored, or otherwise perturbed.

## 8. Reproducing the run

Environment: `groot` (`torch 2.9.0+cu128`, `torchvision 0.24.0+cu128`, `opencv 4.11.0`, `numpy 1.26.4`). `ultralytics 8.4.208`, `timm 1.0.31`, and `polars` were installed into `cv_project/.vendor` and placed on `PYTHONPATH`. `env_isaaclab` was not used.

```text
python -m phase2.extract
```

Configuration defaults live in `phase2/config.py` and are copied into `features/run_meta.json`.
