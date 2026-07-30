# 8. Fine-tuning: Decision and Future-Work Plan

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

This addresses the **optional** fine-tuning extension. The assignment states that a
*"technically sound decision not to fine-tune is acceptable when supported by
evidence… In that case, explain how fine-tuning should be performed in future work."*
This document does exactly that: the reasoned **decision not to fine-tune now**, and a
concrete **plan** for doing it later.

---

## 8.1 Decision: no fine-tuning in this prototype

We deliberately did **not** fine-tune. Fine-tuning updates the model's weights by
training on labelled data (learning rate, epochs, batch size, …) and produces a new
model — distinct from the inference/preprocessing **ablation** we ran (CLAHE on/off),
which never changed a single weight. The decision rests on evidence:

- **No training data.** The 8 hand-annotated images are an **evaluation** set, and the
  ground truth is **points** (for counting), not training boxes. Training on 8 images —
  one location, one dusk session — would **overfit** immediately and tell us nothing
  generalisable (doc 4).
- **RGB is already strong** without training (F1 0.885, MAE 5.5) — no need.
- **Exhaust the free settings first.** The ablation showed that the free settings — CLAHE
  **off** plus lowering the thermal threshold to **0.075** — cut thermal counting error to
  roughly a quarter (MAE 28 → 7.25) and lifted F1 0.385 → 0.673, with **no training at all**
  (ablation_results.md, run3_analysis.md). The correct engineering order is: take the free
  settings wins, *then* train — and we did.
- **But the ablation also located the ceiling that only training can lift.** Even at its
  best setting thermal stays clearly behind RGB, because the COCO-pretrained model has never
  seen `WhiteHot` thermal and only weakly recognises the warm blobs as people (domain
  mismatch — docs 1, 6). No preprocessing or threshold knob can move that curve;
  **fine-tuning is precisely the tool that can.** So fine-tuning is the right *future*
  step, targeted specifically at **thermal**.

The choice of detector supports this: Ultralytics YOLO11 has a first-class fine-tuning
API, so the plan below is low-friction when data exists (doc 2).

---

## 8.2 When fine-tuning becomes justified (triggers)

- A **labelled in-domain thermal training set** exists (aerial, `WhiteHot`, ~45 m).
- Thermal performance must exceed the **current best settings** (CLAHE off, threshold
  0.075: thermal F1 ≈ 0.67, MAE ≈ 7.25) — i.e. higher precision *and* recall together,
  which no preprocessing or threshold knob can give.
- RGB does **not** need it; fine-tuning effort is thermal-only.

---

## 8.3 Future-work fine-tuning plan (thermal detector)

Documented against the fields the assignment lists.

- **Target & initial weights.** Fine-tune **YOLO11x**, starting from the **COCO-pretrained
  `yolo11x.pt`** (transfer learning — reuse the strong RGB/object features). Optionally
  warm-start instead from a **thermal-pretrained checkpoint** (e.g. FLIR ADAS-trained) if
  one is available, to shrink the domain gap further.
- **Dataset.** The gap is *aerial* `WhiteHot` thermal. Options, best-first:
  1. **Collect & annotate** a few hundred–few thousand aerial thermal drone frames from
     this platform (boxes around people). Highest value, matches the domain.
  2. **Public thermal pedestrian data** (FLIR ADAS) as a bridge — but it is *ground-level
     automotive*, so it has its own domain gap; use for pre-conditioning, not as the sole
     source.
- **Dataset split.** ~**70 / 15 / 15** train / val / test, **split by scene/flight** (not
  by frame) to prevent leakage of near-duplicate frames across splits; stratify by
  density (sparse/medium/dense). Keep the current 8-image point set as an **independent
  held-out check** scored with the existing `evaluation/` pipeline.
- **Input size.** Train on **640-px tiles** to match the SAHI inference used in production
  (so train/serve resolution agree); thermal frames are 1280×1024, so tiling also
  multiplies effective training samples.
- **Batch size.** **16** (adjust to GPU memory; use gradient accumulation if smaller).
- **Learning rate.** Fine-tuning LR, lower than from-scratch: **`lr0` ≈ 1e-3** with warmup
  and cosine decay; **freeze the early backbone** (`freeze≈10`) for the first epochs, then
  unfreeze for a lower-LR full pass.
- **Epochs.** **50–100** with **early stopping** (`patience` ≈ 15) on validation fitness.
- **Augmentations (thermal-aware — this matters).** Keep geometric augs (mosaic, scale,
  translate, `fliplr`); **disable colour/hue augmentation** (`hsv_h`, `hsv_s` → 0) because
  the 3-channel thermal is a *replicated single intensity channel*, so hue jitter is
  physically meaningless and harmful; **keep mild brightness/`hsv_v`** to simulate the
  camera's per-frame auto-gain variation (doc 1). Consider adding randomised CLAHE/contrast
  as a domain-specific aug.
- **Checkpoint selection.** Ultralytics saves **`best.pt`** by validation fitness; because
  the goal is **counting**, select/monitor by **recall and mAP@0.5** on the val set (and
  confirm on the held-out point set), not training loss.
- **Overfitting controls.** The dominant risk given limited data: **heavy augmentation**,
  **backbone freezing**, **early stopping**, weight decay, and a small LR; watch the
  **train/val gap** and keep the held-out test untouched until the end.
- **Evaluation.** Re-run the **same metrics pipeline** (`evaluation/metrics.py`) on the
  held-out thermal test and compare to the settings-tuned baseline. **Success = the
  precision–recall curve shifts up** (higher precision at equal recall), i.e. beating the
  ceiling thresholds cannot move.

---

## 8.4 Cheaper experiment to try first

Before a full fine-tune, evaluate an **off-the-shelf thermal-pretrained detector**
(e.g. FLIR-trained) on the existing 8-image set with the current pipeline. It's a
one-evening experiment that either gives a quick thermal win or confirms that
*aerial* thermal is different enough to require in-domain data — informing whether to
invest in data collection.
