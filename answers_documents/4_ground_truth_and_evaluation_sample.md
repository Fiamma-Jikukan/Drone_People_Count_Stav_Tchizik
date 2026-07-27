# 4. Ground Truth and Evaluation Sample

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

This document describes the evaluation sample created from the provided data: which
images were selected and why, how they were annotated, the annotation format and
assumptions, and how the ground truth will be used to compute metrics in item #5.

---

## 4.1 Selected sample

A small, representative sample of **4 image pairs = 8 images** was selected to span
scene density while covering both modalities. Each image is annotated
**independently** (see the note on modalities below).

| Pair (timestamp_seq) | Role | Notes |
|----------------------|------|-------|
| `DJI_20260621190803_0006` | Sparse | Fewest people; tests behaviour on easy, low-density scenes. |
| `DJI_20260621190900_0005` | Medium | Moderate density on the promenade/grass. |
| `DJI_20260621190747_0004` | Dense + richest thermal | Highest thermal person count in the set. |
| `DJI_20260621190910_0006` | Densest RGB | Large seated clusters; hardest counting case. |

The 8 file names are listed in `ground_truth/evaluation_sample.txt`. This is a
deliberately small sample — the assignment does not expect a large annotation
project, only enough to support a meaningful initial evaluation.

**Why counts differ between the RGB and thermal image of a pair.** The two sensors
do **not** see the same set of people, for two reasons: (a) the thermal lens has a
**narrower field of view** (~52 mm vs ~24 mm equiv on the DJI M4T), so the thermal
frame is a zoomed-in crop covering only the centre of the RGB scene — genuinely
fewer people in frame; and (b) the two frames differ slightly in perspective. Each
image is therefore annotated and evaluated on its own, and RGB/thermal ground-truth
counts are expected to differ.

---

## 4.2 Annotation type and placement rule

Annotations are **points** — one point per person — chosen over bounding boxes
because dense RGB frames contain up to ~100 people, where drawing a box per person
is impractical for a prototype. Points still support the required detection-level
metrics via point-in-box matching (§4.5). Points are explicitly permitted by the
assignment ("bounding boxes or points").

**Placement rule:** place one point on each person's **head / upper torso** — the
part most consistently visible from an aerial oblique view and the least ambiguous
in tight clusters. Coordinates are stored as integer pixels in the **original**
image resolution.

Trade-off: points give per-image counts and precision/recall but **not** IoU-based
`mAP@0.5` (which needs boxes). `mAP` is an optional metric, so this is an accepted
limitation.

---

## 4.3 Annotation assumptions (edge cases)

For consistent, reproducible counts, a person is annotated when they are
**clearly identifiable as a person**, under these rules:

- **Occlusion:** count partially occluded people if a human shape is still
  identifiable; count each **distinguishable individual** within a dense cluster.
- **Image borders:** count people cut off by the frame edge if clearly a person.
- **Exclusions:** do **not** count statues, mannequins, reflections, or shadows.
- **Thermal-specific:** count a warm blob only when it has a **human shape**;
  ignore non-human heat sources (lamps, warmed pavement, equipment).
- **Smallness limit:** below the size at which a figure/blob cannot be reliably
  told apart from background clutter, it is **not** annotated. Such ambiguous,
  sub-identifiable targets are a documented source of count uncertainty rather
  than being force-labelled.
- **Uncertainty:** genuinely ambiguous cases are left **unmarked** and noted, so
  the ground truth stays conservative and defensible rather than guessed.

Annotations were produced **manually by the candidate** (not seeded by model
predictions), to keep the ground truth independent of the system being evaluated.

---

## 4.4 Format and tooling

### Ground-truth JSON (one file per image)

Mirrors the pipeline output schema (`src/outputs.py`) but stores points, so
predictions and ground truth are directly comparable:

```json
{
  "image_name": "DJI_20260621190910_0006_V.JPG",
  "modality": "rgb",
  "people_count": 100,
  "points": [[x, y], [x, y], ...]
}
```

`people_count` must equal `len(points)`; this is validated on load
(`ground_truth/ground_truth.py::load_ground_truth`). Files live in `ground_truth/`,
one per image, named by the image stem.

### Annotation tool

`ground_truth/annotate_points.py` is a small interactive annotator (matplotlib):

- **Controls:** left-click adds a person point; right-click / `u` undoes; scroll
  zooms around the cursor (needed for tiny people); arrow keys pan the zoomed view;
  `r` resets to the full image; `s` saves and advances; `q` skips.
- **Workflow:** `python ground_truth/annotate_points.py --input ground_truth/evaluation_sample.txt`
  opens each of the 8 images in turn and writes `ground_truth/<stem>.json`.
- **Review:** `python ground_truth/annotate_points.py --review --input ground_truth/evaluation_sample.txt`
  draws the saved points back onto each image (headless, OpenCV) into
  `ground_truth/review/` so every annotation can be visually verified.

The tool reuses the pipeline's `load_image`, `infer_modality`, and
`save_annotated_image` helpers.

---

## 4.5 How the ground truth is used (forward reference to item #5)

For each image, predictions are matched to ground-truth points:

- **Matching:** a predicted box is a **true positive (TP)** if an unmatched
  ground-truth point falls inside it; matching is greedy in order of descending
  confidence, and each point matches at most one box.
- **False positives (FP):** predicted boxes with no ground-truth point inside.
- **False negatives (FN):** ground-truth points not inside any predicted box.
- From these: **precision** = TP/(TP+FP), **recall** = TP/(TP+FN).
- **Counting error:** `|predicted_count − ground_truth_count|` per image, averaged
  into **MAE** across the sample.

RGB and thermal are evaluated separately, then compared.

---

## 4.6 AI-assistance disclosure

The annotation **tool** (`ground_truth/annotate_points.py`) and the ground-truth code
(`ground_truth/ground_truth.py`) were written with AI assistance and reviewed by the
candidate. The **annotations themselves** (the person points) are produced
manually by the candidate, so the ground truth is human-made and independent of
the detection model.
