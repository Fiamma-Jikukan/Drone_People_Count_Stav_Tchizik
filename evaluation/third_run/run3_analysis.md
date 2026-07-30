# Run 3 Analysis — Confidence sweep + operating-point evaluation

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Run 3 answers the question runs 1–2 could not: **where should the per-modality
confidence threshold sit, and what does the best thermal operating point actually
buy us?** It has two parts:

1. **A low-floor confidence sweep** (`sweep_run/sweep/`) — one capture at a confidence floor of
   **0.05**, then the metrics re-computed at a range of thresholds. This maps the whole
   precision/recall/F1/MAE curve for each modality without re-running the model.
2. **An operating-point run** (`after_sweep_run/`) — a normal, baseline-faithful run at
   the thresholds the sweep recommends (**RGB 0.25 / thermal 0.075**, CLAHE off), scored
   with per-image metrics, plots, and FP/FN example overlays.

Everything else is held at the run-2 configuration: YOLO11x, SAHI on, CLAHE off,
NMS IoU 0.5.

---

## Part A — The confidence sweep (floor 0.05)

**Why one run is enough.** The confidence threshold is applied *after* detection, so
re-thresholding the saved detections at any value T is **near-exact** — close to running
the whole pipeline at T. (Our NMS and the score-ordering make the suppression part exact;
it is not *bit*-exact because SAHI's pinned GREEDYNMM *merges* overlapping tile boxes, so
a lower capture floor feeds the merge marginally more candidates — a small effect, see
Caveats.) One low-floor capture therefore covers the sweep. The floor was set to **0.05**
(`--base-conf 0.05 --rgb-conf 0.05 --thermal-conf 0.05`) so the saved JSON keeps every
detection down to 0.05 — including the weak thermal ones we want to study.

### Sweep results

| thr | RGB P | RGB R | RGB F1 | RGB MAE | · | Thermal P | Thermal R | Thermal F1 | Thermal MAE |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0.05  | 0.691 | 0.920 | 0.790 | 21.75 | · | 0.697 | **0.662** | **0.679** | 10.00 |
| 0.075 | 0.748 | 0.913 | 0.822 | 14.50 | · | 0.719 | 0.606 | 0.658 | **7.75** |
| 0.10  | 0.793 | 0.905 | 0.845 | 9.25  | · | 0.732 | 0.512 | 0.603 | 12.00 |
| 0.15  | 0.843 | 0.901 | 0.871 | 4.50  | · | 0.776 | 0.412 | 0.539 | 18.75 |
| 0.20  | 0.876 | 0.859 | 0.868 | **2.75** | · | 0.836 | 0.319 | 0.462 | 24.75 |
| 0.25  | 0.917 | 0.840 | **0.877** | 5.50 | · | 0.860 | 0.269 | 0.410 | 27.50 |
| 0.30  | 0.934 | 0.810 | 0.868 | 8.75  | · | 0.927 | 0.237 | 0.378 | 29.75 |
| 0.35  | 0.932 | 0.776 | 0.846 | 11.00 | · | 0.941 | 0.200 | 0.330 | 31.50 |

![Recall vs threshold](sweep_run/sweep/recall_vs_threshold.png)
![Precision vs threshold](sweep_run/sweep/precision_vs_threshold.png)
![F1 vs threshold](sweep_run/sweep/f1_vs_threshold.png)
![MAE vs threshold](sweep_run/sweep/mae_vs_threshold.png)

### What the sweep shows

- **Thermal is under-confident, not blind.** At a low threshold the model actually fires
  on **66 % of thermal people** (recall 0.662 at 0.05) — they were simply scoring below
  the 0.20 default and being discarded. The problem was never "the model can't see
  thermal people"; it was "their scores sit near the noise floor."
- **Thermal's sweet spot is ~0.05–0.075.** F1 peaks at 0.05 (0.679) and counting error
  (MAE) bottoms out at 0.075 (7.75) — versus 24.75 at the 0.20 default. That is the
  single biggest thermal-counting lever we found.
- **Precision is the ceiling.** Recovering that recall costs false positives
  (precision drops to 0.697 at 0.05), so thermal F1 caps around **0.68** wherever you sit
  — you cannot get high recall *and* high precision by threshold alone. Only a fine-tuned
  thermal model lifts the whole curve (docs 6–8).
- **RGB wants the opposite end.** It is recall-saturated at low thresholds (0.92 at 0.05)
  but with many false positives (MAE 21.75), so it wants a *higher* threshold: F1 peaks at
  ~0.25 (0.877) and counting (MAE 2.75) is best at ~0.20. This confirms the **per-modality
  threshold design** — the two modalities' optima are on opposite sides of the range.

---

## Part B — Operating-point run (RGB 0.25 / thermal 0.075)

The sweep's optima are RGB ≈ 0.25 and thermal ≈ 0.075 (thermal's MAE-min, and near its
F1-peak); we evaluate a real run at **RGB 0.25 / thermal 0.075**. This is a normal run at
those thresholds (`--rgb-conf 0.25 --thermal-conf 0.075 --base-conf 0.075`), so the
annotated images and metrics are real and inspectable — not a re-threshold.

### Headline metrics

| Modality | Precision | Recall | F1 | MAE |
|---|---:|---:|---:|---:|
| **RGB** (0.25) | 0.925 | 0.848 | **0.885** | **5.50** |
| **Thermal** (0.075) | 0.730 | 0.625 | 0.673 | 7.25 |
| Overall | 0.854 | 0.764 | 0.806 | 6.375 |

**RGB is byte-identical to runs 1 & 2** (pred 241, TP 223, FP 18, FN 40) — confirming
this run uses the same detector path as the baseline, so the comparison below is clean.

### Effect of dropping thermal 0.20 → 0.075 (both CLAHE off)

| Thermal | Run 2 (thr 0.20) | Run 3 op (thr 0.075) |
|---|---:|---:|
| True positives | 51 | **100** |
| False positives | 10 | 37 |
| False negatives | 109 | **60** |
| Recall | 0.319 | **0.625** |
| F1 | 0.462 | **0.673** |
| MAE | 24.75 | **7.25** |

Lowering the thermal threshold found **49 more real people** (TP 51 → 100) and **cut the
counting error to under a third** (MAE 24.75 → 7.25), at the cost of 27 extra false
positives (precision 0.836 → 0.730). For a *counting* task that is a clear win — the
undercount was the dominant error, and this closes most of it.

![Counts per image](after_sweep_run/results/plots/counts_per_image.png)
![Detection scores by modality](after_sweep_run/results/plots/detection_scores.png)
![MAE by modality](after_sweep_run/results/plots/mae.png)

### Per-image thermal breakdown — the failure is image-specific

| Thermal image | GT | pred | TP | FP | FN | Recall |
|---|---:|---:|---:|---:|---:|---:|
| `0004_T` | 66 | 65 | 50 | 15 | 16 | 0.758 |
| `0006_T` (190910) | 59 | 62 | 43 | 19 | 16 | 0.729 |
| `0006_T` (190803) | 10 | 5 | 3 | 2 | 7 | 0.300 |
| `0005_T` | 25 | 5 | 4 | 1 | 21 | **0.160** |

Thermal is **bimodal**: on two frames the model recovers ~73–76 % of people, but on
**`0005_T` it finds only 4 of 25** even at the 0.075 threshold. That single image drags the
aggregate thermal recall down. It is the clearest illustration of domain mismatch — some
thermal scenes render people in a way the COCO-pretrained model simply does not fire on,
and no threshold recovers what was never detected.

![0005_T — thermal failure case](after_sweep_run/results/examples/DJI_20260621190900_0005_T.png)
![0004_T — thermal partial success](after_sweep_run/results/examples/DJI_20260621190747_0004_T.png)
![0006_V — RGB example](after_sweep_run/results/examples/DJI_20260621190803_0006_V.png)

*(Overlay key: green box = true positive, red box = false positive, blue point = matched
ground-truth person, yellow point = missed person.)*

---

## Caveats

- **The sweep closely matches the real run.** With the tile-merge **pinned** (GREEDYNMM,
  forced at every confidence floor — `src/detection.py`), the sweep's re-thresholded
  thermal @ 0.075 reads MAE 7.75 / F1 0.658 vs the actual run's **7.25 / 0.673**, and
  RGB @ 0.25 reads 5.50 / 0.877 vs **5.50 / 0.885**; the sweep's thermal @ 0.20 even
  reproduces run 2 exactly (24.75 / 0.462). The tiny residual is that GREEDYNMM *merges*
  overlapping boxes (not pure suppression), so a lower capture floor feeds the merge
  marginally more candidates — making re-thresholding **near-exact**, not bit-exact.
  *(Before the postprocess was pinned, SAHI's confidence-dependent auto-switch to NMS/IOU
  at the 0.05 floor made the sweep visibly optimistic; pinning removed that.)*
- **Small, single-condition sample** (8 images / 4 thermal, one dusk session). The
  recommended thresholds are indicative and would shift on new data.
- **Precision is the real thermal ceiling**, not the threshold — the durable fix is a
  fine-tuned thermal detector (docs 6–8).

---

## Conclusion & recommended operating point

- **RGB: threshold ≈ 0.25** — F1 0.885, MAE 5.5; robust, insensitive across 0.20–0.30.
- **Thermal: threshold ≈ 0.075** — cuts counting error to under a third of the 0.20
  default (MAE 24.75 → 7.25, F1 0.462 → 0.673) by recovering under-confident detections.
- The two modalities want **opposite ends** of the threshold range, which validates the
  pipeline's separate per-modality thresholds.
- Thermal remains capped by **precision / domain mismatch**; threshold tuning takes it
  from "mostly misses" to "usable," but a fine-tuned model is needed to go further.

Artefacts: `sweep_run/sweep/` (sweep charts + `sweep.csv`), `after_sweep_run/results/`
(`summary.json`, `metrics_per_image.csv`, `plots/`, `examples/`).
