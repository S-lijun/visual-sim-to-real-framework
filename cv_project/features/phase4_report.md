# Phase 4 Report: Baseline Sim-to-Real Evaluation

Date of the figure run: 2026-11-08.  
Question: with no added image perturbation, how large is the visual gap between the original simulation RGB and the original real RGB?

The distances are the Phase 3 estimates. Inception, YOLO, and MiDaS were not run again, and no noise, blur, brightness, or color change was applied. This report adds the distribution figures and the comparison of each distance with the spread inside the real collection.

## 1. Setup

Both domains are the original 157 trajectories. The real images are the reference and were not modified. Preprocessing and features are those already stored under `features/`.

For each distance \(D_k\),

\[
G_k^{\mathrm{base}} = D_k(P_{\mathrm{sim}}^k, P_{\mathrm{real}}^k).
\]

The four families stay separate. A KID value is not on the scale of a Wasserstein value, and the seventeen rows are not averaged.

The point estimate pools every frame, detection, or flow pair. Longer episodes therefore contribute more samples. Simulation episodes average 200.8 frames and real episodes average 76.2, so the pooled sim distribution gives long episodes more weight. The 95% interval still resamples whole trajectories: 200 draws, seed 1, 157 episodes with replacement on each side. A drawn episode contributes all of its frames. Real-vs-real is the mean over 50 non-overlapping halves of the real trajectories, seed 2. It describes variability in this collection. It is the reference used below, and it is not treated as a universal cutoff.

Sample sizes are unchanged: 31,526 sim frames and 11,963 real frames; 9,918 sim detections and 3,852 real detections; 31,369 sim flow pairs and 11,806 real flow pairs.

## 2. Baseline distances

The ratio in the last column is \(G_k^{\mathrm{base}}\) divided by the real-vs-real mean of the same distance. Compare ratios only within a row's own unit. The full intervals and the sim-vs-sim diagnostics are in `features/phase3_report.md` and `results/phase3/metrics.csv`.

| Distance | Baseline | Real-vs-real mean | Ratio to real-vs-real |
|---|---:|---:|---:|
| Appearance KID | 0.2360 | 0.00101 | 234.7 |
| Box count (boxes/frame) | 0.00956 | 0.0287 | 0.334 |
| Confidence | 0.0733 | 0.0125 | 5.85 |
| Box area (fraction of image) | 0.4090 | 0.0217 | 18.8 |
| Class JS (bits) | 0.5625 | 0.0292 | 19.3 |
| Disparity mean | 1.872 | 0.257 | 7.28 |
| Disparity std | 2.053 | 0.0875 | 23.5 |
| Disparity p05 | 2.137 | 0.214 | 10.0 |
| Disparity p50 | 2.037 | 0.314 | 6.49 |
| Disparity p95 | 4.159 | 0.234 | 17.8 |
| Gradient mean | 0.1091 | 0.0106 | 10.3 |
| Gradient p95 | 0.1315 | 0.0136 | 9.66 |
| Flow magnitude mean (px/frame) | 0.574 | 0.115 | 4.97 |
| Flow magnitude std | 2.012 | 0.276 | 7.29 |
| Flow magnitude p95 | 4.870 | 0.725 | 6.72 |
| Motion fraction | 0.0426 | 0.00795 | 5.35 |
| Direction (radians) | 0.2385 | 0.0613 | 3.89 |

`results/phase4/figures/gap_vs_real_variability.png` shows these ratios on a logarithmic axis. The vertical line is a ratio of 1, where the sim-to-real distance equals the average distance between random halves of the real data.

Class JS uses a wider support for sim-vs-real (the 57-class union) than for real-vs-real (classes seen in the real set). Part of that ratio comes from labels that appear on only one side. The other rows use the same one-dimensional sample on both comparisons.

## 3. Distribution figures

Figures are written to `results/phase4/figures/`. Orange is simulation and blue is real. Histogram panels are density-normalized, so the larger sim frame count does not raise the curve by itself.

| File | What is drawn |
|---|---|
| `semantic_distributions.png` | Box count as a fraction of frames; confidence and normalized box area as densities over valid detections. |
| `class_frequencies.png` | The ten labels with the largest share in either domain, as a fraction of detections. |
| `geometry_distributions.png` | Disparity mean, std, median, and p95, plus gradient mean and p95. |
| `temporal_distributions.png` | Flow magnitude mean and p95, motion fraction, and the magnitude-weighted direction histogram. |
| `appearance_pca.png` | First two principal components of 4,000 embeddings from each domain. |
| `gap_vs_real_variability.png` | Each baseline distance divided by its real-vs-real mean. |

Count is grouped as 0, 1, 2, and 3 or more. Confidence is plotted from the operating threshold 0.25, because scores below that threshold were not stored. Box area is plotted on \([0, 1]\). Geometry and flow histograms show the window from the 1st to the 99th percentile of the two domains combined, so a heavy tail that still enters the Wasserstein distance can fall outside the panel. The direction curve is cut at the bin where the combined histogram is smallest, and the two edges of that panel are the same angle.

The PCA figure uses NumPy seed 4. Each domain contributes 4,000 uniformly sampled frames. The components are fit on that balanced subsample after subtracting its mean. PC1 accounts for 18.4% of the subsample variance and PC2 for 6.7%, together 25.1%. The scatter is an illustration. The appearance distance is the KID.

## 4. Where the baseline exceeds the real-data spread

Relative to its own real-vs-real mean, the largest ratios are:

1. Appearance KID, 235 times the within-real distance. The trajectory interval, 0.224 to 0.248, stays far above the real-vs-real mean of 0.0010 and the sim-vs-sim mean of 0.0018.
2. Within-frame disparity standard deviation, 23.5 times. Disparity p95 is 17.8 times and disparity p05 is 10.0 times. The disparity mean is lower in this ranking, at 7.3 times.
3. Detector class frequencies, 19.3 times, and normalized box area, 18.8 times. Confidence is separated more modestly, at 5.8 times.
4. Spatial gradient of disparity, about 10 times for both the mean and the p95.
5. Optical-flow magnitude, 5.0 to 7.3 times depending on the summary, and direction, 3.9 times.

Box count is the exception. Its baseline distance, 0.0096 boxes per frame, is about one third of the real-vs-real mean. The two domains fire the detector on a similar fraction of frames. About 72% of frames in each domain have count 0, and almost all of the rest have one box.

## 5. Interpretation

**Appearance.** The two-dimensional projection already separates the domains: simulation embeddings sit to the left of the real cloud, with limited overlap. Most of the 2048-dimensional variance is outside these two axes, so the plot cannot replace KID. The scalar color statistics point in the same direction without being the metric. Simulation frames are brighter, with mean RGB \((131.7, 124.3, 121.4)\) against real \((107.9, 112.5, 114.2)\), and they have higher within-frame channel standard deviation. A plausible source is rendered shading, texture, and color balance. The baseline does not isolate which of those factors moves KID. That isolation is the later perturbation experiment.

**Scene content and detector behavior.** The count match says the detector is not simply silent on one domain. What it returns is different. Real detections have higher confidence (mean 0.461 against 0.388) and much smaller boxes (mean normalized area 0.232 against 0.641; the sim median area is 0.958). The area panel shows the sim mass piled against a box that covers essentially the whole frame. The leading labels are also different responses to this bin-and-floor scene: simulation detections are led by bed (33.9%) and umbrella (19.8%), real detections by vase (38.8%), person (11.9%), and suitcase (8.7%). Person is common in both, at about 11–12%. These COCO names are outputs of a detector that was not trained on this scene. They record a difference in detector behavior, including domain-specific mistakes. They do not inventory the objects that are present.

**Inferred depth structure.** MiDaS disparity is a relative, larger-is-closer output, checked on the downward-looking real view. The sim maps stretch farther across near and far values: the low percentile is smaller (mean p05 5.98 against 8.12) and the high percentile is larger (mean p95 33.6 against 29.6), so the within-frame standard deviation moves more than the mean. Gradients are stronger in simulation as well. This can come from scene layout and camera pitch, and also from textures or edges that change the monocular predictor. It is not a metric depth error in meters.

**Camera motion.** Real flow magnitude is concentrated near zero. The real median of `mag_mean` is 0.441 px/frame, against a simulation median of 1.263, while the real mean of 1.068 is pulled up by a smaller set of fast pairs. Simulation direction concentrates on two opposite lobes, near \(0.11\pi\) and \(-0.89\pi\), which is one axis of image motion seen in both directions. Real direction is broader. Episode length, gait, and head motion can produce this pattern even if the renderer were identical. The flow distance therefore mixes image appearance with how the camera moved. It is not a measurement of robot speed, and the trajectories are not paired in time.

**Frame sampling.** Because the point estimate pools frames, a long simulation episode influences \(P_{\mathrm{sim}}\) more than a short one. The trajectory interval gives every episode the same chance to be drawn, then keeps the unequal lengths. The baseline numbers should be read with that weighting. They are not an average of 157 episode-level distances.

## 6. What this phase does not claim

- The ratios are not a single realism score, and a large KID ratio is not interchangeable with a large Wasserstein ratio.
- Real-vs-real is the internal spread of this real collection under random halves.
- The PCA plane explains 25.1% of the variance in the plotted subsample. Separation there supports the KID result and does not replace it.
- COCO labels are detector outputs. MiDaS values are not meters. Flow is not robot velocity.
- No simulation image was perturbed. Changes under noise, blur, brightness, and color are outside this baseline.
