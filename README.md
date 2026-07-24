# Person Counting in RGB and Thermal Drone Images

AI prototype that detects people in aerial **RGB** and **thermal** drone imagery and
reports the **number of people per image**. Built as an engineering prototype
(take-home assignment).

The full write-ups live in [`answers_documents/`](answers_documents); this README is
the entry point and links to them for detail.

---

## Solution overview

A function-based, nine-step pipeline:

```
load → identify modality (RGB/thermal) → preprocess → detect (YOLO11 + SAHI)
→ keep people → confidence/NMS filter → count → annotate → save JSON
```

- **Detector:** Ultralytics **YOLO11** run through **SAHI** sliced/tiled inference, so
  the many *small* people in a 12 MP aerial frame stay detectable (see [doc 2](answers_documents/2_model_research_and_selection.md)).
- **Both modalities, one pipeline:** RGB and thermal share the detector; only
  preprocessing and thresholds differ. Counting is **per image, per modality**.
- **Output per image:** an annotated image and a JSON record
  `{image_name, modality, people_count, detections: [{bbox, confidence, class}]}`.

## Project structure

| Path | Contents |
|------|----------|
| `main.py` | Entry point — runs the pipeline over an image or a directory. |
| `src/` | The nine pipeline steps, one module each (`loading`, `modality`, `preprocessing`, `detection`, `person_filter`, `filtering`, `counting`, `annotation`, `outputs`). |
| `ground_truth/` | Point-annotation tool + store + the evaluation sample's ground-truth JSON. |
| `evaluation/` | Metrics, plotting, and the evaluation runner; results in `evaluation/results/`. |
| `answers_documents/` | The written analysis (docs 1, 2, 4, 5, 6, 7). |
| `tests/` | Model-free tests for the core logic (`python -m pytest`). |
| `input_images/` | Provided RGB (`_V`) + thermal (`_T`) drone image pairs. |

## Installation

Run from the repo root with the project virtualenv active:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Model weights (`yolo11n.pt` / `yolo11x.pt`) are **downloaded automatically** by
Ultralytics on first use.

## Usage

```bash
# Count people in a folder (modality inferred from the _V / _T file names)
python main.py --input input_images

# One image, a bigger model, custom output dir
python main.py --input input_images/DJI_20260621190910_0006_V.JPG --weights yolo11x.pt -o outputs/run1

# Faster single-pass (no SAHI) and a tuned thermal threshold
python main.py --input input_images --no-sahi --thermal-conf 0.15
```

Key flags: `--weights`, `--device`, `--no-sahi`, `--rgb-conf`, `--thermal-conf`,
`--nms-iou`, `--modality`.

**Ground-truth annotation & evaluation:**

```bash
# Annotate person points (interactive), then render overlays to check them
python ground_truth/annotate_points.py --input ground_truth/evaluation_sample.txt
python ground_truth/render_annotations.py

# Produce predictions for the evaluation set (see evaluation/run_predictions.txt),
# then compute metrics, plots, and FP/FN examples into evaluation/results/
python main.py --input evaluation/eval_images --output evaluation/predictions --weights yolo11x.pt
python evaluation/evaluate.py

# Tests
python -m pytest
```

## Models considered and final choice

Two families were compared — a one-stage CNN (**YOLO11 + SAHI**) vs. a detection
transformer (**RT-DETR / RF-DETR**) — across small-object suitability, RGB+thermal,
pretrained weights, speed, fine-tuning, licensing, and edge/cloud. **YOLO11 + SAHI**
was selected for its mature pretrained `person` weights, tunable NMS, easy
fine-tuning, and CPU-viability; RT-DETR's Apache-2.0 licensing is the noted
alternative and the pipeline is written detector-agnostic. Full comparison:
[doc 2](answers_documents/2_model_research_and_selection.md).

## Preprocessing, post-processing, and key parameters

- **RGB:** near pass-through (SAHI resolution does the work).
- **Thermal:** grayscale → **CLAHE** (to stabilise per-frame auto-gain) → 3-channel.
- **SAHI tiling:** 640-px tiles, 0.2 overlap.
- **Post-processing:** keep `person` class → per-modality confidence
  (**RGB 0.25 / thermal 0.20**) → **NMS IoU 0.5** → count.
- Rationale for every choice: [doc 1](answers_documents/1_data_analysis.md).

## Annotation and validation methodology

Ground truth is **point annotations** (one point per person) on **8 images**
(4 RGB + 4 thermal pairs), created manually. A prediction is a **true positive** if
its box contains an unmatched ground-truth point (greedy by confidence); leftover
boxes are false positives, leftover points false negatives → precision / recall / F1,
and `|pred − GT|` → MAE. Details:
[doc 4](answers_documents/4_ground_truth_and_evaluation_sample.md),
[doc 5](answers_documents/5_evaluation_and_analysis.md).

## Results (YOLO11x, default thresholds)

| Modality | MAE | Precision | Recall | F1 |
|----------|----:|----------:|-------:|---:|
| **RGB**     | **5.5**  | 0.925 | 0.848 | **0.885** |
| **Thermal** | **28.0** | 0.833 | 0.250 | **0.385** |

**RGB is strong; thermal is weak** — driven by *missed* people (recall), not false
alarms. The COCO-pretrained detector has never seen `WhiteHot` thermal, so warm human
blobs don't match its learned "person" appearance. Full analysis and the
RGB-vs-thermal comparison:
[doc 5](answers_documents/5_evaluation_and_analysis.md),
[doc 6](answers_documents/6_result_analysis.md),
[doc 7](answers_documents/7_rgb_vs_thermal_comparison.md).

## Known limitations and next steps

- **Thermal needs a thermal-appropriate model** (fine-tune on FLIR/aerial thermal, or
  a thermal-native detector); lowering the thermal confidence is a cheap mitigation.
- **Quantitative ablation** of confidence / NMS / SAHI is a planned follow-up.
- **Small, single-condition sample** (one location, one dusk session): results are
  indicative, not generalisable.
- Fusion (RGB↔thermal via a fixed homography) is proposed but out of scope
  ([doc 7](answers_documents/7_rgb_vs_thermal_comparison.md)).

## AI-assistance disclosure

Significant portions of the code and documentation were written with AI assistance
and reviewed/owned by the candidate. All **ground-truth annotations** are human-made,
and all metrics are computed directly from the model predictions and that ground
truth.
