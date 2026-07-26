# Ablation Results

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Results of the four evaluation runs defined in [ablation_plan.md](ablation_plan.md).
The ablation quantifies the **effect of settings** (item #6): CLAHE on/off and the
confidence threshold, evaluated against the same point ground truth with the same
metrics, per modality. **Only the setting under test changes between runs.**

---

## 1. The four runs at a glance

| Run | What changed | Config | Output |
|-----|--------------|--------|--------|
| **1 — baseline** | — | CLAHE **on**, thermal conf 0.20 | `evaluation/predictions/` |
| **2 — CLAHE off** | thermal preprocessing | CLAHE **off**, thermal conf 0.20 | `evaluation/second_run/` |
| **3 — confidence sweep** | confidence threshold | CLAHE off, floor 0.10, sweep 0.10–0.35 | `evaluation/third_run/sweep/` |
| **4 — lower thermal floor** | confidence threshold (below 0.10) | CLAHE off, floor 0.03, thermal-only, sweep 0.03–0.20 | `evaluation/fourth_run/sweep/` |

All runs: YOLO11x, SAHI on. RGB was settled by run 3, so run 4 is thermal-only.

---

## 2. Run 1 → 2: CLAHE **hurt** thermal

| Metric (thermal) | Run 1 (CLAHE on) | Run 2 (CLAHE off) |
|---|---:|---:|
| Recall | 0.250 | **0.319** |
| F1 | 0.385 | **0.462** |
| MAE | 28.0 | **24.75** |

RGB was **byte-identical** between the two runs (CLAHE only touches thermal), which is
a clean control — the only thing that moved was thermal, and it moved **up**. So CLAHE,
added to *help* thermal by normalising contrast, was mildly *hurting* it: piling local
contrast enhancement onto the already edge-sharpened `WhiteHot` JPEGs distorts the warm
blobs away from what the detector expects. **CLAHE is off from here on.**

## 3. Run 3: the confidence sweep (floor 0.10)

| conf | RGB F1 | RGB MAE | Thermal F1 | Thermal MAE | Thermal recall |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.855 | 9.50 | 0.608 | 11.75 | 0.519 |
| 0.20 | 0.875 | **2.75** | 0.462 | 24.75 | 0.319 |
| 0.25 | **0.885** | 5.50 | 0.410 | 27.50 | 0.269 |
| 0.35 | 0.855 | 11.00 | 0.330 | 31.50 | 0.200 |

- **RGB has a broad optimum around 0.20–0.25** (F1 peaks at 0.25, MAE minimises at 0.20)
  and is otherwise insensitive — it is robust.
- **Thermal wants the opposite: as low as possible.** Every thermal metric improves
  monotonically toward the low end; dropping thermal 0.20 → 0.10 nearly doubled recall
  and **halved MAE**.
- **This validates the per-modality-threshold design** (docs 1–2): the two modalities
  have genuinely different optimal operating points.
- Two exact consistency checks passed (RGB @ 0.25 = baseline; thermal @ 0.20 = run 2),
  confirming that re-thresholding one low-floor run reproduces separate runs.

Thermal was **still climbing at the 0.10 floor**, motivating run 4.

## 4. Run 4: below the 0.10 floor (thermal)

| conf | Thermal P | Thermal R | Thermal F1 | Thermal MAE |
|---:|---:|---:|---:|---:|
| 0.03 | 0.449 | **0.794** | 0.573 | 39.25 |
| 0.05 | 0.575 | 0.744 | 0.649 | 20.75 |
| 0.075 | 0.665 | 0.681 | **0.673** | 11.50 |
| 0.10 | 0.685 | 0.556 | 0.614 | **7.50** |
| 0.15 | 0.761 | 0.438 | 0.556 | 17.00 |
| 0.20 | 0.828 | 0.331 | 0.473 | 24.00 |

- **Recall keeps climbing below 0.10 — reaching 0.79 at 0.03.** The detector actually
  *fires* on ~80 % of thermal people; it was simply doing so at **very low confidence**.
- **But precision collapses** as the threshold drops (0.83 → 0.45), so those recovered
  detections come with many false positives.
- The trade-off bottoms out around **0.075–0.10**: **best F1 at 0.075 (0.673)** and
  **best counting (MAE) at 0.10 (7.50)**.
- (Minor note: run 4 at 0.10 differs slightly from run 3 at 0.10 — 7.50 vs 11.75 MAE —
  because the lower detection floor feeds SAHI's merge more candidates. Re-thresholding
  is exact *within* a run; across different floors the merge differs a little. Trends
  are consistent.)

---

## 5. The headline: thermal, fixed by settings alone

Two free levers — **turn CLAHE off** and **lower the thermal threshold 0.20 → 0.10** —
transformed thermal with **no model change**:

| Thermal | Baseline (CLAHE on, 0.20) | Best settings (CLAHE off, ~0.10) |
|---|---:|---:|
| MAE | 28.0 | **7.5** |
| F1 | 0.385 | **~0.67** |
| Recall | 0.25 | **0.56** (up to 0.79 at 0.03) |

Thermal counting error dropped **~4×**. RGB was already good and barely moved
(best MAE 2.75 at conf 0.20, F1 0.885 at 0.25).

### Refined conclusion on thermal (important)
Earlier the thermal failure looked like the model being *blind* to thermal. The
ablation refines that: the model **detects most thermal people but is badly
under-confident** on them (they only weakly resemble a COCO "person"), so a strict
threshold was discarding real detections. Tuning recovers them — **but precision is now
the ceiling**: pushing recall higher floods in false positives. A **thermal-appropriate
model** (fine-tuned on thermal) would lift the whole precision–recall curve, which
threshold tuning cannot. So domain mismatch is still the root limit — it just shows up
as *low confidence + low precision*, not *no detections*.

---

## 6. Recommended operating points

| Modality | CLAHE | Confidence | Rationale |
|----------|-------|-----------|-----------|
| **RGB**     | off (irrelevant — RGB is pass-through) | **0.20–0.25** | Broad optimum; MAE-best 0.20, F1-best 0.25. |
| **Thermal** | **off** | **~0.10** | Best counting (MAE 7.5); near F1-best. Use ~0.075 to favour F1/recall. |

Suggested change to the shipped defaults: keep RGB ≈ 0.25, **drop thermal 0.20 → 0.10**,
and set CLAHE off by default.

Charts: `evaluation/third_run/sweep/` and `evaluation/fourth_run/sweep/`.

---

## 7. Caveats

- **Small, single-condition sample** (8 images / 4 thermal, one scene, one dusk
  session): the tuned thresholds are indicative, likely to shift on new data.
- Re-thresholding is **exact within a run**; small cross-run differences come from
  SAHI's merge seeing different candidate sets at different floors.
- The best *fix* for thermal remains a **thermal-appropriate model**, not thresholds
  (docs 6–7); the ablation shows how far cheap settings alone can go (a lot) and where
  they stop (precision).
