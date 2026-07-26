# 1. Data Analysis

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

This document reviews the provided image pairs, describes the challenges that
affect person counting, and explains how each challenge drives the engineering
decisions (model choice, input resolution, preprocessing, confidence
thresholds, and validation) taken in the rest of the project.

---

## 1.1 The provided dataset

The `input_images/` folder contains **13 RGB + thermal pairs (26 images)** captured
by a **DJI Mavic 3 Thermal (M4T)** drone. Modality is encoded in the file name:

| Suffix | Modality | Resolution | Notes                                   |
|--------|----------|-----------|-----------------------------------------|
| `_V`   | RGB / visual | **4032 × 3024** (12 MP) | Standard color sensor, 3 channels.      |
| `_T`   | Thermal      | **1280 × 1024** (1.3 MP) | `WhiteHot`, edge-sharpened, one channel |

Extracted from EXIF, the same for every frame:

- **Camera:** DJI M4T, both sensors on one gimbal.
- **Altitude:** ≈ **45.7 m** above the launch point (AGL) — a fairly high nadir/oblique view.
- **Capture time:** **2026-06-21, ~19:07 local** — summer **evening / low sun**, long shadows, sun glare off the sea.
- **Scene:** a seaside promenade / grass park (Caesarea harbor) with people
  sitting on picnic blankets in groups, walking on paths, and dining at café tables;
  historic stone buildings, palm trees, and open sea in the background.
- Each pair shares one timestamp, i.e. the two sensors fire **near-simultaneously**.

This is a small, homogeneous set (one location, one session, one lighting
condition). That shapes the whole approach: it is enough for a **meaningful
evaluation sample** but far too small and non-diverse to train or fine-tune a
detector on — a point that justifies using **pretrained** models (see doc 2).

---

## 1.2 Challenges that affect person counting

### (a) People are very small relative to the image

In the RGB frames a standing person is roughly **20–45 px tall** inside a
4032 × 3024 image; seated people are smaller still. That is well under 1 % of the
image height. At 45 m altitude with a 24 mm-equiv wide lens, each person occupies
only a few hundred pixels. Small objects:

- carry very little texture/feature information for a detector,
- are easily lost when an image is downscaled to a model's native input size
  (e.g. 640 px), where a 30 px person shrinks to ~5 px and effectively disappears,
- sit near the noise floor of confidence scoring, so they are the first to be
  dropped by a confidence threshold.

This is the dominant challenge. 
### (b) Dense groups, partial occlusion, overlapping detections

People gather in **tight seated clusters on blankets** (clearly visible bottom-left center
in pair DJI_20260621190710_0001). Within a cluster:

- bodies touch or overlap, so bounding boxes overlap heavily,
- **Non-Max Suppression (NMS) can merge two real people into one** when their
  boxes have high IoU, causing under-counting,
- conversely, a too-loose NMS on a single sprawled person can split into two
  boxes, causing over-counting.
- Seated/lying poses differ from the upright pedestrians that COCO-pretrained
  "person" detectors see most often, lowering recall.

### (c) Differences between RGB and thermal imagery

The two modalities behave almost oppositely:

- **RGB** has high resolution and rich texture but people are tiny, low-contrast
  against grass/stone, and buried in shadow at this sun angle.
- **Thermal** has 10× fewer pixels but people often appear as **bright warm
  blobs** that pop out of a cool background — good for detection of
  small targets, poor for classifying *what* the blob is.
- The provided thermal is **not raw radiometric**: it is a `WhiteHot`, heavily
  **edge-sharpened** 8-bit JPEG. The sharpening adds halo/edge clutter and the
  camera's **automatic gain control (AGC) re-normalizes brightness per frame** —
  so "hot = bright" is **not stable across frames** (in pair DJI_20260621190917_0007 the warm evening
  pavement leaves people barely brighter than the ground). A fixed
  brightness/threshold heuristic on thermal will therefore fail; a learned
  detector or per-frame normalization is required.

### (d) Differences in resolution, viewing angle, and field of view

The two sensors are **not aligned**:

- Different resolution (4032 × 3024 vs 1280 × 1024).
- Different **FOV**: the thermal lens is narrower (52 mm vs 24 mm equiv), so a
  thermal frame covers **only the central portion of the ground area** the RGB
  frame sees — it is a zoomed-in crop, not the same footprint.
- Small **parallax / perspective offset** because the two lenses sit a few cm
  apart on the gimbal, and the oblique angle magnifies it.

Consequence: a pixel in RGB does not map to the same pixel in thermal by a simple
resize. **Any RGB to thermal matching needs an alignment**, and
naive "count on both and add" would double-count the overlapping region.

### (e) Lighting, heat sources, background clutter, look-alikes

- **Low evening sun** → long hard shadows (a person + their shadow can read as a
  larger or second object), strong **sun glare on the sea** (a bright region a
  detector may hallucinate into), and general low contrast on shaded ground.
- **Thermal false heat sources:** sun-warmed pavement, café heat lamps, parked-car
  engines, and lamp posts can be as bright as a person and produce **false
  positives**; conversely a person on hot ground has **low contrast** and is
  missed.
- **Look-alike objects:** in RGB — traffic cones, café chairs/umbrellas, bollards,
  statues, and the vertical minaret/lamp posts have person-like proportions; in
  thermal — any warm vertical blob. These drive false positives.
- **Background clutter:** dense promenade furniture (tables, umbrellas, benches)
  fragments people and hides small targets.

### (f) Possible timing differences between paired images

EXIF gives both sensors the **same timestamp**, so within a pair the offset is
sub-second — people barely move between the RGB and thermal shot. **Between
successive pairs**, however, the drone is moving/panning and people walk, so the
scene changes frame-to-frame. Implications:

- Within a pair, alignment error comes from **geometry (FOV/parallax)**, not from
  motion — good news for future fusion.
- Across the set, the **same person appears in overlapping frames**; if images
  were ever mosaicked or counted collectively, that person would be **double
  counted**. Counting is therefore defined **per image**, as the assignment
  specifies.

---

## 1.3 How these challenges drive the engineering decisions

| Challenge | Engineering response |
|-----------|----------------------|
| Tiny people (a) | Favor a model + inference scheme that preserves small objects: **high inference resolution** and/or **tiled / sliced inference (SAHI)** instead of a single 640 px pass. Prefer models with a strong small-object track record on aerial data (see doc 2). |
| Dense/occluded groups (b) | Tune **NMS IoU carefully** (moderate, not aggressive) and prefer per-class NMS; report both under- and over-counting; consider that the *count*, not perfect boxes, is the deliverable. |
| RGB vs thermal nature (c) | Run the **same detector on both modalities but evaluate them separately**; for thermal, either use a thermal-appropriate model or **replicate the single channel to 3-ch and rely on shape**, and apply **per-frame contrast normalization (e.g. CLAHE)** to counter AGC variability. |
| Resolution / FOV / angle (d) | Choose input resolution **per modality** (don't upscale thermal beyond its real detail); keep counting **per image, per modality**; defer cross-modal matching to an optional homography-based fusion step. |
| Lighting / heat / look-alikes (e) | Set **confidence thresholds per modality** (thermal typically needs a different operating point); expect and document **false positives from cones/chairs/heat sources**; use annotated GT to pick thresholds by precision/recall rather than guessing. |
| Timing / overlap (f) | Define the task as **per-image counting**; avoid mosaicking; note cross-frame duplication as a known limitation for any future aggregate count. |

### Input resolution
Because of (a) and (d): RGB should be processed at **high resolution or in tiles**
(downscaling to 640 destroys the signal); thermal should be processed at or near
its **native 1280 × 1024** (upscaling adds no real detail and wastes compute).

### Preprocessing
- **RGB:** minimal — optional mild contrast/shadow lift; main lever is
  resolution/tiling, not pixel manipulation.
- **Thermal:** convert single-channel to 3-channel for COCO-pretrained models;
  **per-frame normalization / CLAHE** to stabilize the AGC-driven brightness
  differences seen between pairs DJI_20260621190710_0001 and DJI_20260621190917_0007.

### Confidence thresholds
Set **independently per modality**, chosen on the annotated evaluation sample by
looking at the precision/recall trade-off — not a single global default — because
the false-positive sources and target contrast differ so much between RGB and
thermal.

### Validation
The above makes a **manually annotated ground-truth sample essential**: with tiny,
clustered, look-alike-prone targets, only human-verified counts + boxes let us
measure MAE, precision, and recall and tune thresholds/NMS honestly (docs 4–5).
Because the set is small and single-condition, results will be **indicative, not
generalizable** — a limitation to state explicitly and a reason to prefer robust
pretrained models over fitting to these 13 scenes.

---

## 1.4 Summary

The defining conditions of this dataset are **very small people**, **dense seated
clusters**, and **two unaligned sensors with opposite strengths** (RGB = detail but
tiny/low-contrast targets; thermal = salient blobs but low resolution, unstable
gain, and edge-sharpening artefacts), all under **low evening light with strong
shadows and thermal clutter**. These push the design toward: a **pretrained
small-object detector**, **high-resolution or tiled inference**, **modality-specific
preprocessing and thresholds**, **careful NMS**, **per-image / per-modality
counting**, and a **small human-annotated evaluation set** to tune and validate —
with fusion deliberately deferred to optional work.
