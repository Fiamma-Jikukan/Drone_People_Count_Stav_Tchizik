# Run 2 Analysis — CLAHE off vs baseline, plus the confidence sweep

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Run 2 is identical to the baseline (run 1) in every setting **except thermal
preprocessing**: CLAHE is turned **off** (`--no-clahe`). Everything else is held fixed
(YOLO11x, SAHI on, confidence RGB 0.25 / thermal 0.20, NMS IoU 0.5), so any difference
is attributable to CLAHE alone. This document compares run 2 to run 1 and then reads the
confidence sweep run on run 2's saved detections.

Sources: `first_run/results/summary.json`, `second_run/results/summary.json`,
`second_run/sweep/sweep.csv`.

---

## 1. Run 1 → Run 2, per modality

| | Metric | Run 1 (CLAHE **on**) | Run 2 (CLAHE **off**) | Δ |
|---|---|---:|---:|---:|
| **RGB** | Precision | 0.925 | 0.925 | — |
| | Recall | 0.848 | 0.848 | — |
| | F1 | 0.885 | 0.885 | — |
| | MAE | 5.50 | 5.50 | — |
| **Thermal** | Precision | 0.833 | 0.836 | +0.003 |
| | Recall | 0.250 | **0.319** | **+0.069** |
| | F1 | 0.385 | **0.462** | **+0.077** |
| | MAE | 28.00 | **24.75** | **−3.25** |

Overall (both modalities): F1 0.739 → **0.756**, recall 0.622 → **0.648**,
MAE 16.75 → **15.125**.

---

## 2. RGB is the control — and it is byte-identical

RGB is a pass-through in preprocessing (CLAHE only ever touches thermal), so RGB **must**
be unchanged if the experiment is clean — and it is: same predicted total (241), same
TP/FP/FN (223 / 18 / 40), same precision, recall, F1, and MAE to the digit. This is the
sanity check that makes the thermal comparison trustworthy: the only thing that moved
between the two runs is thermal, so the thermal change is caused by CLAHE and nothing
else.

---

## 3. Thermal: turning CLAHE off helped, modestly

| Thermal detections | Run 1 (CLAHE on) | Run 2 (CLAHE off) |
|---|---:|---:|
| True positives (TP) | 40 | **51** |
| False positives (FP) | 8 | 10 |
| False negatives (FN) | 120 | **109** |
| Predicted total | 48 | 61 |

CLAHE off recovered **11 more real people** (TP 40 → 51) at the cost of only **2 extra
false positives** (FP 8 → 10). Because the new detections were mostly correct,
**precision barely moved** (0.833 → 0.836) while **recall rose** (0.25 → 0.319) — so F1
went up and counting error dropped (MAE 28 → 24.75).

**Why.** CLAHE was added to normalise the thermal camera's per-frame auto-gain, but
piling local contrast enhancement onto the already contrast-processed `WhiteHot` JPEGs
distorts the warm blobs *away* from what a COCO-pretrained detector expects. Removing it
leaves the blobs closer to the model's learned "person" appearance, so a few more clear
the detection floor.

**But the gain is small and the ceiling is unchanged.** Even at its better setting,
thermal (F1 0.462, MAE 24.75) is still far behind RGB (F1 0.885, MAE 5.50), and 109 of
160 thermal people are still missed. Contrast normalisation was never the bottleneck —
**domain mismatch** is (a COCO model on thermal), and only a thermal-appropriate
(fine-tuned) model can close that (docs 6–8). **CLAHE off is the recommended thermal
setting** from here on.

---

## 4. Confidence sweep on run 2

Charts and table: `second_run/sweep/`. The sweep re-applies a range of confidence
thresholds to run 2's *saved* detections (exact — no re-run — because SAHI's merge and
our NMS are greedy by score).

| thr | RGB F1 | RGB MAE | Thermal F1 | Thermal MAE | Thermal recall |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.885 | 5.50 | **0.462** | **24.75** | **0.319** |
| 0.25 | 0.885 | 5.50 | 0.410 | 27.50 | 0.269 |
| 0.30 | 0.876 | 8.75 | 0.378 | 29.75 | 0.237 |
| 0.35 | 0.855 | 11.00 | 0.330 | 31.50 | 0.200 |
| 0.40 | 0.822 | 15.25 | 0.261 | 34.00 | 0.150 |

- **Raising the threshold only makes things worse.** Both modalities were already at (or
  below) their optimum at run 2's saved floor, so every step up trades recall away: F1
  and MAE degrade monotonically. RGB is best at ≤ 0.25 (F1 0.885); thermal is best at the
  lowest available point, 0.20.
- **RGB @ 0.20 reads identical to @ 0.25** — because run 2 only *saved* detections down
  to each modality's confidence threshold (RGB 0.25, thermal 0.20). There are no
  detections below the floor to add back, so any threshold under the floor is flat and
  uninformative.

### Limitation — what this sweep cannot answer

The interesting thermal question is *"how much recall do we recover by going **lower**
than 0.20?"* Run 2 cannot answer it: `main.py` applied the per-modality confidence
threshold **before** writing the JSON, so nothing below 0.20 (thermal) / 0.25 (RGB) was
saved — even though the model's detection floor (`base_conf`) was 0.10. Probing below the
floor requires a **re-run** that saves detections down low (e.g. `--thermal-conf 0.10`).
That low-floor sweep is out of scope for the run 1 vs run 2 ablation, which isolates
CLAHE.

---

## 5. Conclusion

- **CLAHE off is a small, free win on thermal** (F1 +0.077, MAE −3.25) with RGB provably
  unaffected — so adopt CLAHE off.
- **RGB is strong and robust**; its optimum sits at ≤ 0.25 and it is insensitive to the
  exact threshold.
- **The thermal ceiling is the model, not the settings.** Preprocessing and threshold
  tuning give only marginal thermal gains; the durable fix is a fine-tuned thermal
  detector (docs 6–8).
