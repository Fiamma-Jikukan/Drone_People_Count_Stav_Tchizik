# Consolidated Results Comparison

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

All scored runs and both modalities in **one place** — the results-package summary /
comparison table. Every number is computed against the same 8-image point ground truth
(4 RGB / 4 thermal; RGB 263 people, thermal 160) with the same metrics
([`metrics.py`](../metrics.py)). Per-run detail: [`ablation_results.md`](ablation_results.md),
[`second_run/results/run2_analysis.md`](../second_run/results/run2_analysis.md),
[`third_run/run3_analysis.md`](../third_run/run3_analysis.md).

## The runs

| Run | Config (vs baseline) | Output |
|-----|----------------------|--------|
| **1 — baseline** | CLAHE **on**, RGB 0.25 / thermal **0.20** | `first_run/predictions/`, `first_run/results/` |
| **2 — CLAHE off** | CLAHE **off**, RGB 0.25 / thermal 0.20 | `second_run/results/` |
| **3 — operating point** | CLAHE **off**, RGB 0.25 / thermal **0.10** | `third_run/after_sweep_run/results/` |

All runs: YOLO11x, SAHI on, NMS IoU 0.5. Only the **bold** factor changes at each step.

---

## 1. Per-modality metrics across runs

| Run | Modality | MAE | Precision | Recall | F1 |
|-----|----------|----:|----------:|-------:|---:|
| **1 — baseline** | RGB | 5.50 | 0.925 | 0.848 | **0.885** |
|  | Thermal | 28.00 | 0.833 | 0.250 | 0.385 |
| **2 — CLAHE off** | RGB | 5.50 | 0.925 | 0.848 | **0.885** |
|  | Thermal | 24.75 | 0.836 | 0.319 | 0.462 |
| **3 — op. point** | RGB | 5.50 | 0.925 | 0.848 | **0.885** |
|  | Thermal | **11.75** | 0.735 | 0.519 | **0.608** |

**RGB is identical in all three runs** — CLAHE never touches it and its threshold (0.25)
never changed, so it is the clean control. **All the movement is thermal**, and it only
improves: **MAE 28.00 → 24.75 → 11.75** (roughly halved overall), **F1 0.385 → 0.462 →
0.608**.

## 2. Thermal detection outcomes (where the recovery comes from)

Thermal ground truth = **160** people across 4 images.

| Run | Thermal config | Predicted | TP | FP | FN | Recall |
|-----|----------------|----------:|---:|---:|---:|-------:|
| **1 — baseline** | CLAHE on, 0.20 | 48 | 40 | 8 | 120 | 0.250 |
| **2 — CLAHE off** | CLAHE off, 0.20 | 61 | 51 | 10 | 109 | 0.319 |
| **3 — op. point** | CLAHE off, 0.10 | 113 | **83** | 30 | **77** | 0.519 |

Turning CLAHE off and lowering the threshold recovers real people (**TP 40 → 83**, missed
**FN 120 → 77**). The cost is more false positives (**FP 8 → 30**) — the precision trade-off
that becomes thermal's ceiling.

## 3. Overall (both modalities combined)

| Run | MAE | Precision | Recall | F1 |
|-----|----:|----------:|-------:|---:|
| **1 — baseline** | 16.75 | 0.910 | 0.622 | 0.739 |
| **2 — CLAHE off** | 15.13 | 0.907 | 0.648 | 0.756 |
| **3 — op. point** | **8.63** | 0.864 | 0.723 | **0.788** |

---

## Reading the table

- **RGB is strong and stable** (F1 0.885, MAE 5.5) — no tuning needed; it is the control
  that proves the comparison is clean.
- **Thermal improves monotonically** as we turn CLAHE off and lower its threshold, but even
  at its best (F1 0.608) it trails RGB by a wide margin.
- **The gap is recall / domain mismatch, not preprocessing or thresholds** — a CLAHE
  clip × tile grid found **no setting beats "off"** (see
  [`ablation_results.md`](ablation_results.md) §3), and a confidence sweep shows thermal is
  **under-confident, not blind** but capped by precision (see
  [`third_run/run3_analysis.md`](../third_run/run3_analysis.md)). The durable fix is a
  thermal-appropriate (fine-tuned) model
  ([doc 8](../../answers_documents/8_finetuning_future_work.md)).

## Notes

- The shipped defaults are the **run-1 baseline** (CLAHE on, thermal 0.20), kept
  deliberately conservative; runs 2–3 **document** the better operating point rather than
  changing the default.
- Run 3's numbers are the **operating-point run** (a real run at RGB 0.25 / thermal 0.10),
  which is baseline-faithful — its RGB reproduces runs 1–2 byte-for-byte. The confidence
  *sweep* (`third_run/sweep_run/sweep/`) is read for **shape**, not absolute values (it re-thresholds
  a low-floor capture where SAHI switches its tile-merge).
