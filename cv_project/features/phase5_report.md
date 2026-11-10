# Phase 5 Report: Single-Factor Visual Ablation

Extraction finished: 2026-11-09T08:35:22Z. Wall time for the 54-episode extraction: 7.86 hours.  
Question: when one controlled change is applied to the original simulation RGB, how do the four sim-to-real distances move?

Each condition changes one factor. Noise, blur, brightness, and color are never applied together. The real images were not modified and were not passed through the transforms. Original simulation files were only read.

## 1. Sample

The perturbations use simulation episodes 0 through 53: 54 trajectories and 11,054 frames. The real reference remains the full set of 157 trajectories and 11,963 frames. The baseline is the original, unperturbed simulation on those same 54 episodes, compared with the same real set. It is not the Phase 4 number from all 157 simulation episodes. On KID the two baselines are close: 0.2344 here, 0.2360 on the full set. Box count is the quantity that moves the most between the two samples: 0.0410 on these 54 episodes, 0.0096 on all 157.

Distances use the Phase 3 definitions. The point estimate pools frames. The interval resamples the 54 simulation trajectories and all 157 real trajectories, 200 draws, seed 1. KID inside each draw averages 10 subsets. The reported KID point estimate uses 100 subsets of 1,000, NumPy seed 0, and the same polynomial kernel as Phase 2. Class Jensen–Shannon uses the fixed 57-class support from the full-data comparison, so every condition is scored on the same set of labels.

\[
\Delta G_k^{(a)} = G_k^{(a)} - G_k^{\mathrm{base}}
\]

A negative change means that distance became smaller. It means the transformed simulation distribution moved closer to the real distribution under that one metric. It does not by itself mean the image became more realistic.

## 2. Transforms

Pixel values stay in 0–255 and are clipped after the shift. Blur, brightness, and color use one fixed operator on every frame, so they do not add flicker. Noise is independent at each frame. Its random seed is the pair `(5, episode id)`, so the same episode is perturbed the same way on a rerun.

| Factor | Levels | Operation |
|---|---|---|
| Gaussian noise | σ = 5, 15, 30 | Add Gaussian noise, round, clip. |
| Gaussian blur | σ = 0.6, 1.2, 2.4 | Kernel width `2 * ceil(3σ) + 1`, so 5, 9, and 17. |
| Brightness | −30, −15, +15, +30 | Add the same integer to R, G, and B, then clip. |
| Color temperature | −32, −16, −8, +8, +16, +32 | Add the value to R and subtract it from B. G is unchanged. Positive is warmer. |

A contact sheet of simulation episode 0, frame 20, is saved at `results/phase5/preview_episode0.png`.

## 3. Baseline on these 54 episodes

| Distance | Baseline |
|---|---:|
| Appearance KID | 0.2344 |
| Box count (boxes/frame) | 0.0410 |
| Confidence | 0.0683 |
| Box area (fraction of image) | 0.3779 |
| Class JS (bits) | 0.5402 |
| Disparity mean | 2.029 |
| Disparity std | 2.231 |
| Disparity p95 | 4.723 |
| Gradient mean | 0.1183 |
| Flow magnitude mean (px/frame) | 0.529 |
| Motion fraction | 0.0398 |
| Direction (radians) | 0.2392 |

The full table, including the other geometry and flow summaries, is `results/phase5/metrics.csv`.

## 4. What each factor moves

Figures are in `results/phase5/figures/`. Each factor has one panel per distance, and the panel title names the metric and its family: Appearance, Semantic, Geometry, or Temporal. The black point at intensity 0 is the baseline above. Error bars are the 95% trajectory interval of the perturbed distance. `delta_heatmap.png` shows the sign of each change after scaling every metric by its own largest absolute change. The heatmap is only a map of direction. The numbers below are the distances.

**Noise.** Medium and high noise lower KID. At σ = 15, KID is 0.1595 (interval 0.150–0.175). At σ = 30 it is 0.1371 (0.131–0.149). Both intervals sit below the baseline 0.2344. σ = 5 is 0.2226 (0.210–0.240) and still covers the baseline. The same noise increases the temporal gap sharply, because a new noise field is drawn on every frame. Mean flow magnitude goes from 0.529 px/frame to 1.082, 10.25, and 17.56. Motion fraction goes to 0.111, 0.839, and 0.906. Box count rises from 0.041 to about 0.19–0.20 at every noise level, and the intervals stay above the baseline. Box area falls from 0.378 to 0.112, 0.130, and 0.071. Class JS falls at σ = 5 (0.293, interval 0.242–0.379) and rises again at σ = 30 (0.623, interval 0.589–0.672). Disparity distances fall only once the noise is strong: disparity std goes from 2.231 to 1.284 at σ = 15 and 0.199 at σ = 30.

**Blur.** KID barely moves. The three values are 0.2332, 0.2313, and 0.2299, and every interval covers 0.2344. Flow magnitude dips at σ = 1.2 (0.364, interval 0.303–0.438) and returns near the baseline at σ = 2.4. Direction falls at σ = 2.4, from 0.239 to 0.142 (interval 0.119–0.177). Strong blur also raises the box-count distance to 0.138 (0.094–0.187) and lowers class JS to 0.476 (0.429–0.554). Disparity mean rises across the three levels, to 2.183, 2.397, and 2.725.

**Brightness.** Shifts of ±15 and ±30 do not push KID outside its trajectory interval. Darkening by 15 leaves KID at 0.2342. Darkening by 30 leaves it at 0.2317 (0.218–0.253). Brightening by 30 raises it to 0.2412 (0.225–0.266). The simulation mean is about 14 levels brighter than the real mean, so −15 is the step that matches that average, and KID does not follow it. Flow, direction, box area, and disparity stay near the baseline. The largest brightness changes are in class JS at the extremes: 0.604 at −30 and 0.586 at +30, against 0.540. Those intervals still reach the baseline.

**Color temperature.** KID stays inside the baseline interval at every tested shift. The largest point estimate is 0.2484 at −16 (interval 0.230–0.273). Flow and direction are flat, which is the expected result for a transform that is the same on every frame. Semantics and geometry are not flat. A cooler shift shrinks box area and disparity spread: at −32, box area is 0.106 (0.085–0.133) and disparity std is 1.012 (0.760–1.239). A warmer shift moves both the other way: at +32, box area is 0.566 (0.524–0.610), class JS is 0.707 (0.660–0.763), and disparity std is 2.732 (2.482–2.942).

## 5. Reading the sensitivity

Appearance, as measured by KID, responds to added noise and does not respond clearly to blur, brightness, or color temperature inside the ranges above. Closing the mean brightness gap, or shifting color temperature, leaves KID within the trajectory interval of the original simulation.

The detector metrics respond to noise and to color. Noise changes how many boxes appear and how large they are. Color temperature moves box area and, in the warm direction, class JS, while leaving the optical flow essentially unchanged. Blur has to reach σ = 2.4 before count, class JS, and flow direction leave the baseline interval. Brightness is the weakest of the four factors for this set of distances.

Geometry responds to strong noise and to color temperature. Strong noise and a cool shift reduce the disparity distances. A warm shift increases them. Those reductions sit next to large changes in other metrics, especially flow under noise, so they are a statement about the MiDaS statistic under that transform.

The temporal module is the check on whether a transform flickers. Independent frame noise drives mean flow from 0.53 px/frame to 17.6. Constant brightness and constant color leave flow near 0.53. Blur, which is also constant across time, changes direction only at the strongest setting.

## 6. What this phase does not claim

- A smaller distance is not a claim that the perturbed image is perceptually more realistic.
- The factors were not combined. A result for noise does not predict noise plus blur.
- KID, Wasserstein, and Jensen–Shannon remain different units. The heatmap scales each row by itself so the signs can be seen together.
- These 54 simulation episodes are the sample for every condition and for the baseline in this report. The Phase 4 table remains the full-data baseline.
- COCO labels are detector outputs. MiDaS values are not meters. Flow is not robot velocity.
