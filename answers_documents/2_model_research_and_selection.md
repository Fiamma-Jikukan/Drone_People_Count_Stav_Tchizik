# 2. Model Research and Selection

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

This document reviews relevant models and technical approaches, compares them
against the criteria the assignment lists, and selects one primary approach to
implement. The choice is driven directly by the conditions established in
[doc 1](1_data_analysis.md): **very small people**, **dense seated clusters**,
**two unaligned sensors** (high-res RGB with tiny targets; low-res thermal with
salient but unstable blobs), a **small single-condition dataset**, and a
**~30-hour prototype** scope.

---

## 2.1 Framing: two decisions, not one

Selecting an "approach" here means answering two separate questions:

1. **Which detection paradigm?** — object detection (boxes) vs. density-map
   crowd counting (a learned "people per region" heatmap).
2. **Which detector + inference strategy?** — the model family (YOLO / DETR /
   two-stage) and *how* it is run on a 12 MP image where people are ~30 px
   (single pass vs. tiled/sliced inference).

### Paradigm: detection vs. density estimation

**Density-map crowd counting** (CSRNet, P2PNet, and similar) regresses a count
without per-object boxes. It excels in *very* dense crowds (stadiums, marches)
where individual detection breaks down.

I **reject it as the primary approach** because:

- The assignment explicitly requires **detection-level outputs** (bounding boxes
  or points) for validation, duplicate reduction, and per-detection metrics
  (precision/recall). Density maps give a number, not verifiable detections.
- Our scenes are **moderately** dense (small seated groups), not the extreme
  crowds density models are built for — detection is both feasible and more
  informative here.
- Pretrained density models are trained almost entirely on **RGB, ground-level,
  head-annotated** data (ShanghaiTech, etc.) — a poor match for **aerial full-body
  thermal**.

So the project uses **object detection**, and the rest of this document compares
detectors and inference strategies. (Density estimation is noted as a possible
future direction for genuinely dense scenes.)

---

## 2.2 Candidate approaches

| # | Approach | One-line description |
|---|----------|----------------------|
| A | **Ultralytics YOLO (YOLO11), single 640-px pass** | The obvious COCO-pretrained baseline; fast, simple, `person` class built in. |
| B | **Ultralytics YOLO + SAHI tiled inference** | Same detector, but the image is sliced into tiles so small people stay large enough to detect. |
| C | **RT-DETR / RF-DETR (transformer)** | End-to-end, NMS-free transformer detector; Apache-2.0; strong accuracy, heavier. |
| D | **Aerial-domain-pretrained YOLO (VisDrone weights)** | YOLO weights trained on drone imagery rather than generic COCO. |

All four are variations on modern detectors; the meaningful axes are **how small
objects are handled**, **licensing**, and **cost**. They are compared below
against the assignment's required criteria.

---

## 2.3 Comparison against the required criteria

### (a) Suitability for small-object detection in aerial imagery

- **A (YOLO single-pass):** Weak. Resizing 4032×3024 → 640 shrinks a 30 px person
  to ~5 px; recall on small people collapses. This is exactly the failure doc 1
  predicts.
- **B (YOLO + SAHI):** **Strong.** Slicing the image into (e.g.) 640-px tiles with
  overlap means each person is detected at near-native scale, then detections are
  merged back with NMS. SAHI was designed for precisely this problem and reports
  large recall gains on small objects. Best fit for our dominant challenge.
- **C (RT-DETR/RF-DETR):** Good, and better than single-pass YOLO on small objects
  by architecture; can *also* be combined with SAHI. But single-pass at a
  tractable resolution still under-resolves 30-px people.
- **D (VisDrone-pretrained):** Good priors for aerial *scale and viewpoint*
  (objects are small by design in VisDrone), but VisDrone is RGB-only and its
  label set/quality varies; still benefits from tiling.

### (b) Suitability for RGB and thermal input

- No COCO/VisDrone detector is trained on **thermal**. However, the provided
  thermal is a `WhiteHot` 8-bit image that **preserves human shape**, so a
  shape-driven detector can still fire on it when the single channel is
  **replicated to 3 channels** and **per-frame CLAHE** stabilizes contrast (doc 1).
- This works for **A–D equally** — thermal handling is a preprocessing decision,
  not a model-family decision. The realistic expectation is **lower thermal
  recall** than RGB from any COCO-pretrained model, to be quantified in evaluation
  and, if needed, closed later by **fine-tuning on a thermal set** (see 2.5).
- A genuinely thermal-native option exists (models fine-tuned on **FLIR ADAS** /
  thermal pedestrian datasets); noted as a fine-tuning path rather than the
  baseline, to keep one unified pipeline for both modalities.

### (c) Availability and relevance of pretrained weights

- **A/B:** Excellent — Ultralytics ships COCO-pretrained weights with a first-class
  `person` class, downloaded automatically; the largest ecosystem and docs.
- **C:** Good — official RT-DETR and Roboflow RF-DETR COCO weights are available
  (Apache-2.0), and SAHI supports them.
- **D:** Available (community VisDrone-trained YOLO checkpoints) but less
  standardized; `person`/`pedestrian` class definitions differ from COCO and need
  remapping.

### (d) Inference speed and hardware requirements

- **A:** Fastest (one small pass).
- **B:** **N× slower**, where N = number of tiles (a 12 MP image at 640-px tiles is
  ~30–50 tiles). This is the main cost of the accuracy it buys — acceptable for an
  offline prototype counting a folder of images, not for high-FPS video.
- **C:** Transformers are heavier per image and more memory-hungry; real-time
  variants exist but need a decent GPU.
- **D:** Same cost profile as A/B (it *is* a YOLO).
- All run on a single modern GPU; YOLO (A/B/D) also runs acceptably on CPU for a
  small image set, which suits a prototype.

### (e) Support for fine-tuning

- **A/B/D:** **Best-in-class.** Ultralytics has a one-command training/fine-tuning
  API, easy dataset formatting (YOLO txt), and abundant tutorials — the smoothest
  path if we later fine-tune on thermal or aerial data.
- **C:** Fine-tunable, but the workflow is more involved and slower to converge
  (transformers are data-hungry).

### (f) Model and code licensing

- **A/B/D — Ultralytics YOLO11: AGPL-3.0** (or paid Enterprise license). AGPL is
  fine for this **evaluation prototype and internal research**, but it is
  **copyleft**: shipping it in a closed-source product requires open-sourcing the
  connected code or buying the Enterprise license. This is the one real
  disadvantage of the YOLO route and is flagged for productization.
- **B — SAHI: MIT** (permissive; no constraint added by the tiling layer).
- **C — RT-DETR / RF-DETR: Apache-2.0** (permissive) — the licensing-clean choice
  if closed-source deployment is a hard requirement.

### (g) Potential future use in cloud or edge environments

- **A/B/D:** Strong. YOLO exports to ONNX / TensorRT / CoreML / TFLite and runs on
  Jetson-class edge devices; SAHI tiling can be toggled off (or coarsened) for a
  fast single-pass edge mode, giving a natural accuracy/latency dial.
- **C:** Deployable and increasingly edge-friendly, but transformer export/latency
  tuning is more work than YOLO's mature toolchain.

### Summary matrix

| Criterion | A: YOLO 1-pass | **B: YOLO + SAHI** | C: RT-DETR/RF-DETR | D: VisDrone-YOLO |
|-----------|:--:|:--:|:--:|:--:|
| Small aerial objects | ✗ | **✓✓** | ✓ | ✓ |
| RGB + thermal (via preproc) | ✓ | ✓ | ✓ | ✓ |
| Pretrained weights | ✓✓ | ✓✓ | ✓ | ~ |
| Speed / HW cost | ✓✓ | ~ (N× tiles) | ~ | ✓ |
| Fine-tuning ease | ✓✓ | ✓✓ | ✓ | ✓✓ |
| Licensing | AGPL | AGPL+MIT | **Apache** | AGPL |
| Cloud / edge | ✓✓ | ✓ | ✓ | ✓✓ |

---

## 2.4 Selected approach

**Primary implementation: Ultralytics YOLO11, COCO-pretrained, run with SAHI
sliced/tiled inference, applied to both modalities** (thermal replicated to
3 channels + CLAHE), filtered to the `person` class, with confidence and NMS/IoU
tuned **per modality** on the annotated evaluation sample.

### Why this approach

- **It directly attacks the dominant challenge.** Tiny people are the #1 problem
  (doc 1); SAHI tiling is the single most effective, lowest-effort lever against it
  and turns a weak baseline (A) into a usable one.
- **Fastest path to a reliable baseline in ~30 h.** Mature weights, one `person`
  class, huge ecosystem, trivial setup — maximizing time spent on validation and
  analysis (60 %+ of the grade) rather than on model plumbing.
- **One pipeline for both modalities.** The same detector handles RGB and thermal;
  only preprocessing differs — clean, testable, and honest about each modality's
  performance.
- **Best fine-tuning on-ramp.** If thermal recall proves inadequate, Ultralytics
  gives the smoothest route to fine-tune on a thermal/aerial set (doc on optional
  work).
- **Deployment optionality.** Export + a SAHI on/off latency dial cover both cloud
  batch and edge use.

### Advantages
- Highest recall on small people for the least engineering effort (tiling).
- Excellent tooling, docs, reproducibility, and community support.
- Trivial to demonstrate live (swap thresholds, run on a new image) — which the
  technical review explicitly asks for.

### Disadvantages
- **AGPL-3.0** license: acceptable for this prototype/evaluation, but a blocker for
  closed-source productization without an Enterprise license. *Mitigation:* the
  pipeline is written **detector-agnostic** so the model can be swapped for
  Apache-licensed **RT-DETR/RF-DETR** (SAHI supports both) if licensing becomes a
  constraint.
- **SAHI multiplies inference time** (N tiles per image). Acceptable offline;
  documented as a cost, with single-pass mode available for speed.
- **Not thermal-native**: expect lower thermal recall from COCO weights.

### Expected limitations (to be confirmed in evaluation)
- Under-counting in **dense seated clusters** (NMS merges) and for **very small /
  occluded** people.
- **False positives** from person-like objects (cones, chairs, statues, lamp posts)
  in RGB and warm blobs (heat lamps, sun-warmed surfaces) in thermal.
- **Tile-boundary effects**: a person split across two tiles may be double-counted
  or missed; mitigated by tile **overlap** + merge NMS, but not eliminated.
- Results are **indicative only** — one location, one evening, 13 pairs; no claim of
  generalization.

---

## 2.5 Fine-tuning position (brief)

Fine-tuning is **not** part of the mandatory baseline and is **not** justified up
front: the dataset is tiny and single-condition, with no train/val split large
enough to avoid overfitting, and COCO-pretrained YOLO gives a reasonable person
detector out of the box. The sound engineering decision is to **establish and
validate the pretrained baseline first**, then decide from evidence. If thermal
recall is the main gap, the recommended future step is fine-tuning on a public
**thermal pedestrian dataset (e.g. FLIR ADAS)** and/or an **aerial dataset
(VisDrone)**, using Ultralytics' training API — documented fully in the optional-work
write-up rather than performed here.

---

## Sources

- [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11) and
  [Ultralytics License](https://www.ultralytics.com/license) — YOLO11/YOLO26, AGPL-3.0 vs Enterprise.
- [SAHI (obss/sahi) GitHub](https://github.com/obss/sahi) and
  [Ultralytics SAHI tiled-inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference) — MIT license, framework-agnostic slicing, supports YOLO/RT-DETR/RF-DETR.
- [SAHI paper (arXiv:2202.06934)](https://arxiv.org/abs/2202.06934) — slicing-aided inference for small-object detection.
- [RT-DETR (lyuwenyu/RT-DETR)](https://github.com/lyuwenyu/RT-DETR) and
  [RF-DETR (Roboflow)](https://blog.roboflow.com/rf-detr/) — transformer detectors, Apache-2.0.
- [UAV-DETR (PMC12349633)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349633/) — DETR variant for small-object UAV imagery.
