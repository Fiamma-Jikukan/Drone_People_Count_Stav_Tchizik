# 6. Result Analysis

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

This document analyses the evaluation results (item #6). The metrics, figures, and
false-positive / false-negative examples it refers to are in
[doc 5](5_evaluation_and_analysis.md); the deeper modality comparison is in
[doc 7](7_rgb_vs_thermal_comparison.md).

---

## Where the model did well

RGB is strong: **F1 0.885, MAE 5.5** people across frames holding 25–111 people, with
**precision 0.93**. Counting error is small even on the densest frame (111 → 100). The
detector reliably finds upright and seated people on grass and promenade.

## Where the count was wrong

Thermal is poor: **recall 0.25, MAE 28**, and one frame (`0005_T`, 25 people) produced
**zero detections**. All modality-level error is **under-counting** (predicted totals
below ground truth: RGB 241/263, thermal 48/160).

## Likely causes of false positives and false negatives

- **False negatives dominate**, and are the whole story in thermal. The COCO-pretrained
  YOLO11x has never seen `WhiteHot`, edge-sharpened thermal imagery, so a warm human
  blob simply does not match its learned "person" appearance — exactly the
  domain-mismatch risk flagged in docs 1–2. Per-frame auto-gain variation (doc 1) makes
  it worse: `0005_T` normalised such that people barely stood out, and the detector
  found nothing.
- In **RGB**, the residual FNs are **dense-cluster** members (overlapping bodies merged
  or suppressed) and the **smallest / most-occluded** people.
- **False positives are rare** in both — the model is conservative at these thresholds,
  so precision is high and the accuracy ceiling is set by recall.

## Effect of confidence / input resolution / IoU / NMS settings

*(Qualitative; a quantitative ablation is the planned next step.)*

- **Confidence:** precision is high and recall low, so the operating point is
  conservative. **Lowering** the confidence threshold (especially for thermal) would
  recover missed people and raise recall, at some precision cost — the main knob to
  explore next. Raising it would do the opposite.
- **SAHI / input resolution:** tiled inference is doing heavy lifting — without it a
  12 MP frame downscaled to 640 px would lose almost all of these small people (doc 1).
  It is essential to the RGB result.
- **NMS IoU (0.5):** a moderate value; too low would merge adjacent people in dense
  clusters (more FN), too high would split individuals (more FP). Its effect is
  concentrated in the dense RGB clusters.

## Main difference between RGB and thermal performance

RGB vastly outperforms thermal (**F1 0.885 vs 0.385**, **MAE 5.5 vs 28**), and the gap
is almost entirely **recall** (0.85 vs 0.25), not precision (0.93 vs 0.83) — i.e. the
thermal model *misses* people rather than inventing them. The full comparison, and why,
is in [doc 7](7_rgb_vs_thermal_comparison.md).

## Recommended next steps (analysis)

- **Thermal needs a thermal-appropriate model.** The single highest-impact step is
  fine-tuning on a **thermal pedestrian dataset (e.g. FLIR ADAS)** or using a
  thermal-native detector; the pretrained COCO model is the wrong tool for `WhiteHot`
  imagery. Lowering the thermal confidence threshold is a cheap partial mitigation.
- **Quantitative ablation** of confidence / NMS / SAHI is pending, to put numbers on
  the "effect of settings" above.