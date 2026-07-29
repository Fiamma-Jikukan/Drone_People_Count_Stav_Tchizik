# 2. Model Research and Selection

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

This document reviews two relevant models, compares them against the criteria the
assignment lists, and selects one primary approach to implement. The choice is
driven directly by the conditions established in [doc 1](1_data_analysis.md):
**very small people**, **dense seated clusters**, **two unaligned sensors**
(high-res RGB with tiny targets; low-res thermal with salient but unstable blobs),
a **small single-condition dataset**, and a **~30-hour prototype** scope.

---

## 2.1 The two candidate models

**Why these two.** Modern object detection is dominated by two model families, and
these candidates are the strongest, best-supported representative of each. Rather
than compare several near-identical CNNs, a more useful comparison pits the two
*paradigms* against each other: a **one-stage CNN** (YOLO — the industry default
for fast, practical detection) versus a **detection transformer** (RT-DETR/RF-DETR
— the newer architecture increasingly challenging YOLO on accuracy). Both have
COCO-pretrained `person` weights, both are actively maintained, and both work with
SAHI tiling — so they are a fair, real-world fork in the road for a person-counting
prototype, differing on exactly the axes we care about (small-object handling,
speed, fine-tuning, and licensing).

**How each works (in short):**

- **YOLO11 (one-stage CNN).** A convolutional network looks at the whole image
  **once** and, in a single forward pass, directly predicts a grid of candidate
  boxes with a class and confidence for each. Overlapping duplicates are then
  removed by **Non-Max Suppression (NMS)**. It is fast and mature but, on a single
  pass, struggles with tiny objects — which is why we wrap it in **SAHI**: SAHI
  slices the large image into overlapping tiles, runs YOLO on each tile (so a small
  person is large *within* its tile), then stitches and de-duplicates the tile
  detections back into full-image coordinates.

- **RT-DETR / RF-DETR (detection transformer).** A CNN backbone extracts image
  features, then a **transformer** uses attention to reason about all objects
  jointly and outputs a **fixed set of predictions directly** — no grid of anchors
  and **no NMS** (the model itself learns not to emit duplicates). Attention gives
  it strong context and competitive small-object accuracy, at the cost of a heavier,
  more GPU-hungry, more data-hungry model.

| | **Model 1 — Ultralytics YOLO11 (+ SAHI tiling)** | **Model 2 — RT-DETR / RF-DETR** |
|---|---|---|
| Family | One-stage CNN detector | End-to-end detection transformer (DETR) |
| Small-object strategy | COCO-pretrained detector run with **SAHI** sliced/tiled inference so small people are seen at near-native scale | Transformer attention over image features; NMS-free set prediction; can also be tiled with SAHI |
| Post-processing | NMS (with tuned IoU), plus SAHI tile-merge | **No NMS** (direct set prediction) |
| License | YOLO11 **AGPL-3.0** (or Enterprise); SAHI **MIT** | **Apache-2.0** |

These two represent the real fork in the road for this task: a **mature one-stage
CNN with a tiling wrapper** vs. a **transformer detector**. The comparison below
weighs them against the assignment's required criteria.

Note that **SAHI tiled inference is the key lever for small objects and is
model-agnostic** — it works with both models (SAHI officially supports Ultralytics
YOLO *and* RT-DETR/RF-DETR). So the comparison is not "tiling vs. no tiling"; it is
about the **base detector**, with tiling available to either.

**Note on YOLO26.** Ultralytics released **YOLO26** (January 2026) as its current
recommended generation. It is **not a separate candidate** — it is a newer version
of Model 1: same Ultralytics API, same **AGPL-3.0** license, same SAHI/fine-tuning
tooling, so a **one-line swap** from YOLO11. Its improvements are relevant to us:
**~43 % faster CPU inference** (which offsets SAHI's N× tile cost), **+2.5 box AP**
on COCO, and an **NMS-free, DFL-free** graph that exports better to edge devices.
This baseline nonetheless standardizes on **YOLO11** for two deliberate reasons:
(1) **maturity and reproducibility** — YOLO11 has a longer track record of
community examples and battle-tested SAHI integration, lowering risk in a graded
prototype; and (2) YOLO11's **explicit NMS/IoU knobs** are exactly the thresholds
the assignment asks the candidate to tune, analyze, and modify live (reqs 6 and the
technical review), whereas YOLO26's NMS-free design removes that lever. Because
YOLO26 sits in the same CNN family, choosing it would not change any verdict in the
comparison below — it is simply a **stronger instance of the selected model**, and
the pipeline is written to swap it in once validated.

---

## 2.2 Comparison against the required criteria

### (a) Suitability for small-object detection in aerial imagery

The binding constraint here is **resolution, not architecture**: a single 640-px pass on
a 4032×3024 frame shrinks a 30-px person to ~5 px and recall collapses (the failure doc 1
predicts). The fix is **tiled inference (SAHI)**, which is **model-agnostic** — so both
candidates are assessed **with tiling**, like-for-like.

- **YOLO11 + SAHI:** **Strong.** Slicing the image into overlapping ~640-px tiles
  restores near-native scale per person, then merges detections back; SAHI was built for
  exactly this and reports large small-object recall gains.
- **RT-DETR / RF-DETR + SAHI:** **Comparable.** Its attention / multi-scale design gives
  a genuine small-object edge *in principle*, but that edge is largely neutralised once
  both are tiled — tiling already makes each person large within its tile. Run tiled, it
  is on par with YOLO on this criterion.

*Verdict: **even** once both are tiled; YOLO's SAHI integration is simply more mature and
better documented. Note this criterion is **analytical** — only YOLO+SAHI was actually
run (RT-DETR was not benchmarked), so parity is a reasoned expectation, not a measured
result; a tiled head-to-head on the evaluation set would confirm it.*

### (b) Suitability for RGB and thermal input

- Neither model is trained on **thermal**. But the provided thermal is a `WhiteHot`
  8-bit image that **preserves human shape**, so a shape-driven detector can fire on
  it when the single channel is **replicated to 3 channels** and **per-frame CLAHE**
  stabilizes contrast (doc 1).
- This is a **preprocessing** decision, so it applies **equally** to both models.
  Expect **lower thermal recall** than RGB from either COCO-pretrained model, to be
  quantified in evaluation and, if needed, closed later by **fine-tuning on a
  thermal set** (see doc 8).

*Even.*

### (c) Availability and relevance of pretrained weights

- **YOLO11:** **Excellent** — Ultralytics ships COCO-pretrained weights with a
  first-class `person` class, auto-downloaded; the largest ecosystem and docs.
- **RT-DETR/RF-DETR:** **Good** — official RT-DETR and Roboflow RF-DETR COCO weights
  exist (Apache-2.0) and SAHI supports them, but the tooling and examples are
  thinner than Ultralytics'.

*Edge: YOLO11.*

### (d) Inference speed and hardware requirements

- **YOLO11:** Fastest per pass; the cost is **SAHI's N× tiles** (a 12 MP image at
  640-px tiles ≈ 30–50 tiles). Acceptable for an offline prototype counting a folder
  of images; runs acceptably even on CPU for a small set.
- **RT-DETR/RF-DETR:** Transformers are **heavier per image and more memory-hungry**;
  real-time variants exist but effectively need a GPU, and tiling multiplies that
  cost.

*Edge: YOLO11 (lighter, CPU-viable).*

### (e) Support for fine-tuning

- **YOLO11:** **Best-in-class** — one-command training/fine-tuning API, simple YOLO
  txt dataset format, abundant tutorials. More **sample-efficient** with better
  small-data tooling (heavy augmentation, one-command recipes) — the smoother on-ramp
  **when a realistic thermal set exists**.
- **RT-DETR/RF-DETR:** Fine-tunable, but a **more involved workflow** and slower
  convergence — transformers lack CNNs' built-in inductive biases, so they need
  **more data** to fine-tune well.

*(Neither is fine-tuned on the 8-image evaluation set — too small for any deep
detector; see [doc 8](8_finetuning_future_work.md).)*

*Edge: YOLO11.*

### (f) Model and code licensing

- **YOLO11: AGPL-3.0** (or paid Enterprise). Fine for this **evaluation prototype and
  internal research**, but **copyleft**: shipping it in a closed-source product
  requires open-sourcing the connected code or buying the Enterprise license. This
  is the one real disadvantage of the YOLO route. (SAHI itself is **MIT** — no added
  constraint.)
- **RT-DETR/RF-DETR: Apache-2.0** — **permissive**; the licensing-clean choice if
  closed-source deployment is a hard requirement.

*Edge: RT-DETR/RF-DETR — the single criterion where the transformer clearly wins.*

### (g) Potential future use in cloud or edge environments

- **YOLO11:** **Strong** — exports to the standard cross-platform inference formats and
  runtimes, and runs on common embedded/edge accelerators; SAHI can be toggled off or
  coarsened for a fast single-pass edge mode, giving a natural accuracy/latency dial.
- **RT-DETR/RF-DETR:** Deployable and increasingly edge-friendly, but transformer
  export and latency tuning are **more work** than YOLO's mature toolchain.

*Edge: YOLO11.*

### Summary matrix

| Criterion | **Model 1 — YOLO11 + SAHI** | Model 2 — RT-DETR/RF-DETR |
|-----------|:--:|:--:|
| Small aerial objects (tiled) | ✓✓ | ✓✓ |
| RGB + thermal (via preproc) | ✓ | ✓ |
| Pretrained weights | ✓✓ | ✓ |
| Speed / HW cost | ✓ (N× tiles, CPU-viable) | ~ (heavier, GPU) |
| Fine-tuning ease | ✓✓ | ~ |
| Licensing | AGPL (+MIT) | **Apache** |
| Cloud / edge | ✓✓ | ✓ |

---

## 2.3 Selected approach

**Primary implementation: Ultralytics YOLO11, COCO-pretrained, run with SAHI
sliced/tiled inference, applied to both modalities** (thermal replicated to
3 channels + CLAHE), filtered to the `person` class, with confidence and NMS/IoU
tuned **per modality** on the annotated evaluation sample.

### Why YOLO11 + SAHI over RT-DETR/RF-DETR

- **The two are roughly even on accuracy once both are tiled**, so the decision
  turns on the *other* criteria — and YOLO11 wins pretrained availability,
  speed/hardware cost, and fine-tuning ease.
- **Fastest path to a reliable baseline in ~30 h.** Mature weights, one `person`
  class, huge ecosystem, trivial setup — maximizing time spent on validation and
  analysis (60 %+ of the grade) rather than model plumbing.
- **CPU-viable** for a small image set, and the **smoothest fine-tuning on-ramp** if
  thermal recall proves inadequate — both matter for a tiny-dataset prototype.
- RT-DETR/RF-DETR's one clear advantage is **licensing (Apache-2.0)**, which does not
  bind an evaluation prototype.

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
  constraint — the exact trade-off this comparison surfaced.
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

## Sources

- [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11) and
  [Ultralytics License](https://www.ultralytics.com/license) — YOLO11, AGPL-3.0 vs Enterprise.
- [SAHI (obss/sahi) GitHub](https://github.com/obss/sahi) and
  [Ultralytics SAHI tiled-inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference) — MIT license, framework-agnostic slicing, supports YOLO and RT-DETR/RF-DETR.
- [SAHI paper (arXiv:2202.06934)](https://arxiv.org/abs/2202.06934) — slicing-aided inference for small-object detection.
- [RT-DETR (lyuwenyu/RT-DETR)](https://github.com/lyuwenyu/RT-DETR) and
  [RF-DETR (Roboflow)](https://blog.roboflow.com/rf-detr/) — transformer detectors, Apache-2.0.
- [UAV-DETR (PMC12349633)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349633/) — DETR variant for small-object UAV imagery.
