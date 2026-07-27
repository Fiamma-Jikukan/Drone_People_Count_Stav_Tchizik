# 3. Baseline Implementation

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

The pipeline is an end-to-end, **function-based** implementation with **one module
per step** (assignment item #3's nine steps), wired together by `main.py`. This
document describes every function with input/output examples where practical.

**Design principles**
- **Function-first** — each stage is a plain function that takes data in
  and returns data out, so stages are independently testable.
- **One `src/` module per step**; `main.py` is the single **entry and output point**.
- **Detections are plain dicts** (see below), so they flow through the stages and
  serialise to JSON without conversion.
- **Heavy dependencies** (`torch`, `ultralytics`, `sahi`) are imported **lazily**
  inside `src/detection.py`, so the rest of the pipeline imports instantly.

**Core data model — a *detection*** (the internal dict passed between steps 4–8):

```python
{"bbox": [x1, y1, x2, y2], "confidence": 0.91, "class_id": 0, "class_name": "person"}
```

`bbox` is integer pixels in the original image (top-left, bottom-right).

**Pipeline flow**

```
1 loading → 2 modality → 3 preprocessing → 4 detection → 5 person filter
→ 6 confidence/NMS → 7 counting → 8 annotation → 9 outputs
```

---

## Step 1 — Loading (`src/loading.py`)

Finds and reads image files. `IMAGE_EXTENSIONS` = `{.jpg,.jpeg,.png,.bmp,.tif,.tiff}`.

### `discover_images(input_path, recursive=False)`
Expands a path into a sorted list of image `Path`s — accepts a single file or a
directory. Raises `FileNotFoundError` (missing path) or `ValueError` (a file with an
unsupported extension).

```python
discover_images("input_images")
# → [Path("input_images/DJI_..._0004_T.JPG"), Path("input_images/DJI_..._0004_V.JPG"), ...]

discover_images("input_images/DJI_..._0004_V.JPG")
# → [Path("input_images/DJI_..._0004_V.JPG")]   # single file → one-element list
```

### `load_image(image_path)`
Decodes one file into an `(H, W, 3)` uint8 **BGR** NumPy array. Uses
`np.fromfile` + `cv2.imdecode` so Unicode/Windows paths load reliably. Raises
`FileNotFoundError` if missing or undecodable.

```python
load_image("input_images/DJI_..._0004_V.JPG")   # → ndarray, shape (3024, 4032, 3), dtype uint8
load_image("input_images/DJI_..._0004_T.JPG")   # → ndarray, shape (1024, 1280, 3), dtype uint8
```

---

## Step 2 — Modality (`src/modality.py`)

Decides RGB vs thermal. Constants: `RGB="rgb"`, `THERMAL="thermal"`,
`RGB_SUFFIXES=("_V",)`, `THERMAL_SUFFIXES=("_T",)`, `DEFAULT_MODALITY="rgb"`.

### `normalize_modality(value)`
Validates/cleans an explicit modality string (case-insensitive, trims whitespace).
Raises `ValueError` on anything else.

```python
normalize_modality("Thermal")   # → "thermal"
normalize_modality(" RGB ")      # → "rgb"
normalize_modality("infrared")   # → ValueError
```

### `infer_modality(image_path, override=None, rgb_suffixes=..., thermal_suffixes=..., default=...)`
Returns `"rgb"` or `"thermal"` using priority: **explicit override → file-name suffix
→ default**. Only the file name is inspected; matching is case-insensitive; thermal is
checked first so a `_T` is never masked.

```python
infer_modality("DJI_..._0004_T.JPG")               # → "thermal"
infer_modality("DJI_..._0004_V.JPG")               # → "rgb"
infer_modality("clip_V.jpg", override="thermal")   # → "thermal"   # override wins
infer_modality("random.jpg")                        # → "rgb"       # falls back to default
```

---

## Step 3 — Preprocessing (`src/preprocessing.py`)

Modality-specific preprocessing; always returns a 3-channel BGR image the detector can
consume. Constants: `CLAHE_CLIP_LIMIT=2.0`, `CLAHE_TILE_GRID=8`.

### `preprocess(image_bgr, modality, apply_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid=8)`
Dispatcher: routes thermal to `preprocess_thermal`, everything else to `preprocess_rgb`.

```python
preprocess(rgb_img, "rgb")          # → rgb_img unchanged (pass-through)
preprocess(thermal_img, "thermal")  # → 3-channel BGR, CLAHE-normalised (see below)
```

### `preprocess_rgb(image_bgr)`
Near pass-through: returns the image as-is, only promoting a 2-D grayscale input to
3-channel BGR.

```python
preprocess_rgb(img)   # shape (3024, 4032, 3) → same (3024, 4032, 3)
```

### `preprocess_thermal(image_bgr, apply_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid=8)`
Collapses to a single intensity channel → optional **CLAHE** contrast equalisation →
replicates back to 3 identical channels.

```python
preprocess_thermal(thermal_img)                    # (1024,1280,3) → (1024,1280,3), CLAHE applied
preprocess_thermal(thermal_img, apply_clahe=False) # (1024,1280,3) → (1024,1280,3), no CLAHE
# In the output the 3 channels are identical (a replicated single channel).
```

---

## Step 4 — Detection (`src/detection.py`)

Runs YOLO11 + SAHI. Heavy libs imported lazily. Defaults: `DEFAULT_WEIGHTS="yolo11x.pt"`,
`DEFAULT_DEVICE="cpu"`, `DEFAULT_BASE_CONFIDENCE=0.10`, SAHI tiles `640×640` @ `0.2`
overlap, single-pass `imgsz=1280`.

### `load_model(weights="yolo11x.pt", device="cpu", base_confidence=0.10)`
Builds the detector once (downloads weights on first use). Returns a SAHI
`AutoDetectionModel` wrapping the Ultralytics YOLO model.

```python
model = load_model()                      # yolo11x on CPU, detection floor 0.10
model = load_model(weights="yolo11n.pt")  # fast alternative
```

### `detect(model, image_bgr, use_sahi=True, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, imgsz=1280)`
Runs detection on one preprocessed image and returns **all** detections as dicts
(every class, no confidence/person filtering yet — that's steps 5–6). With `use_sahi`
it tiles via `_detect_sahi`; otherwise a single pass via `_detect_single_pass`.

```python
detect(model, preprocessed)
# → [
#     {"bbox": [212, 832, 269, 919], "confidence": 0.42, "class_id": 0, "class_name": "person"},
#     {"bbox": [980, 410, 1020, 470], "confidence": 0.31, "class_id": 8, "class_name": "boat"},
#     ...
#   ]
```

*Internal helpers:* `_make_detection(...)` builds one detection dict with rounded
integer pixel coordinates; `_detect_sahi(...)` and `_detect_single_pass(...)` are the
two backends. Both convert the library's box coordinates to the detection-dict schema.

---

## Step 5 — Person filter (`src/person_filter.py`)

Keeps only people. Constants: `PERSON_CLASS_ID=0`, `PERSON_CLASS_NAME="person"`.

### `filter_person(detections, person_class_id=0, person_class_name="person")`
Returns a new list with only detections matching the person class **by id or name**
(robust to models that number classes differently). Order preserved.

```python
filter_person([
    {"bbox": [0,0,1,1], "confidence": 0.9, "class_id": 0, "class_name": "person"},
    {"bbox": [0,0,1,1], "confidence": 0.8, "class_id": 8, "class_name": "boat"},
])
# → [ {"bbox": [0,0,1,1], "confidence": 0.9, "class_id": 0, "class_name": "person"} ]
```

---

## Step 6 — Confidence / IoU / NMS (`src/filtering.py`)

Removes weak and duplicate boxes. Constant: `DEFAULT_NMS_IOU=0.5`.

### `filter_confidence(detections, threshold)`
Keeps detections with `confidence >= threshold` (inclusive).

```python
filter_confidence([d(0.5), d(0.49)], 0.5)   # → [d(0.5)]
```

### `iou(box_a, box_b)`
Intersection-over-Union of two `[x1,y1,x2,y2]` boxes; `0.0` when disjoint.

```python
iou([0,0,10,10], [0,0,10,10])   # → 1.0
iou([0,0,10,10], [20,20,30,30]) # → 0.0
iou([0,0,10,10], [5,0,15,10])   # → 0.333...
```

### `nms(detections, iou_threshold=0.5)`
Greedy Non-Max Suppression: keeps the highest-confidence box, suppresses any
lower-confidence box overlapping it by more than `iou_threshold`. Returns survivors,
highest-confidence first.

```python
nms([box(conf=0.5, at A), box(conf=0.9, overlapping A)], 0.5)
# → [ box(conf=0.9) ]   # the 0.9 box wins; the overlapping 0.5 box is dropped
```

### `filter_detections(detections, confidence_threshold, nms_iou=0.5)`
Convenience chain: confidence filter → NMS. This is the step-6 entry point.

```python
filter_detections(person_dets, confidence_threshold=0.25, nms_iou=0.5)
# → confident, de-duplicated person detections
```

---

## Step 7 — Counting (`src/counting.py`)

### `count_people(detections)`
The final count is the number of surviving detections.

```python
count_people([d, d, d])   # → 3
count_people([])          # → 0
```

---

## Step 8 — Annotation (`src/annotation.py`)

Colours: thermal boxes yellow `(0,255,255)`, RGB boxes green `(0,255,0)`.

### `draw_detections(image_bgr, detections, modality="rgb", count=None, box_thickness=2, show_confidence=True)`
Returns a **copy** of the image with each box (and its confidence) drawn plus a
count banner. The input image is not modified. `count` defaults to `len(detections)`.

```python
annotated = draw_detections(image, people, modality="rgb", count=3)
# → new ndarray, same shape as `image`, with 3 green boxes + banner "RGB  people: 3"
```

### `save_annotated_image(image_bgr, out_path)`
Encodes and writes an image (Unicode-safe), creating parent folders; extension picks
the format (default `.jpg`). Returns the `Path`. Raises `IOError` if encoding fails.

```python
save_annotated_image(annotated, "outputs/annotated/DJI_..._0004_V.jpg")
# → Path("outputs/annotated/DJI_..._0004_V.jpg")  (file written)
```

---

## Step 9 — Outputs (`src/outputs.py`)

Writes the structured result. Constant: `CONFIDENCE_DECIMALS=4`.

### `build_result(image_name, modality, detections)`
Assembles the per-image record in the assignment's schema, converting internal
detections (`class_id`/`class_name`) to the output form (`class`), with
`people_count = len(detections)`.

```python
build_result("DJI_..._0004_V.JPG", "rgb",
             [{"bbox": [120,85,170,210], "confidence": 0.9123, "class_id": 0, "class_name": "person"}])
# → {
#     "image_name": "DJI_..._0004_V.JPG",
#     "modality": "rgb",
#     "people_count": 1,
#     "detections": [ {"bbox": [120,85,170,210], "confidence": 0.9123, "class": "person"} ]
#   }
```

### `save_json(result, out_path)`
Writes a result dict to a pretty-printed JSON file (creating parent folders). Returns
the `Path`.

```python
save_json(result, "outputs/json/DJI_..._0004_V.json")
# → Path("outputs/json/DJI_..._0004_V.json")  (file written)
```

*Internal helper:* `_detection_to_schema(detection)` maps one internal detection to the
`{bbox, confidence, class}` output form.

---

## Orchestration — `main.py`

The single entry and output point: parses the CLI, loads the model **once**, runs
steps 1–9 per image, and writes outputs.

### `parse_args(argv=None)`
Defines the CLI. Flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--input`, `-i` | *(required)* | Image file or directory. |
| `--output`, `-o` | `outputs` | Output directory. |
| `--modality`, `-m` | inferred | Force `rgb`/`thermal`. |
| `--weights` | `yolo11x.pt` | YOLO weights. |
| `--device` | `cpu` | `cpu` / `cuda:0`. |
| `--no-sahi` | off | Disable SAHI tiling (single pass). |
| `--no-clahe` | off | Disable CLAHE thermal preprocessing. |
| `--clahe-clip` | `2.0` | CLAHE clip limit (thermal). |
| `--clahe-tile` | `8` | CLAHE tile grid N for N×N (thermal). |
| `--rgb-conf` | `0.25` | RGB confidence threshold. |
| `--thermal-conf` | `0.20` | Thermal confidence threshold. |
| `--nms-iou` | `0.5` | NMS IoU threshold. |
| `--base-conf` | `0.10` | Model detection floor. |

### `confidence_for(modality, rgb_conf, thermal_conf)`
Picks the per-modality confidence threshold.

```python
confidence_for("thermal", 0.25, 0.20)   # → 0.20
confidence_for("rgb", 0.25, 0.20)        # → 0.25
```

### `process_image(model, image_path, args)`
Runs the nine steps on **one** image (load → modality → preprocess → detect →
person filter → confidence/NMS → count → annotate+save → build+save JSON) and returns
the result dict. Writes `outputs/annotated/<stem>.jpg` and `outputs/json/<stem>.json`.

```python
process_image(model, "input_images/DJI_..._0004_V.JPG", args)
# → {"image_name": "...", "modality": "rgb", "people_count": N, "detections": [...]}
#   (+ annotated image and JSON written to the output dir)
```

### `main(argv=None)`
Parses args, configures logging, discovers images, loads the model once, then loops
`process_image` over every image with **per-image error isolation** (a bad image is
logged and skipped, not fatal), and logs a final summary. Returns a process exit code
(`0` success; `2` bad input path; `1` if nothing could be processed).

```bash
python main.py --input input_images                 # whole folder
python main.py --input img_T.JPG -m thermal -o out  # one image, forced modality
```

---

## Constants and their rationale

Every module keeps its tunables as named constants (config, not magic numbers). Why
each value was chosen:

### `src/loading.py`
- **`IMAGE_EXTENSIONS`** = `{.jpg, .jpeg, .png, .bmp, .tif, .tiff}` — a conservative
  allow-list of raster formats OpenCV can decode. `.jpg` is the dataset's format;
  `.tif/.tiff` are included to future-proof for radiometric thermal exports. The
  allow-list lets directory discovery skip non-images and reject a bad single-file input.

### `src/modality.py`
- **`RGB`** = `"rgb"`, **`THERMAL`** = `"thermal"` — canonical modality strings shared
  across the pipeline, so modality is compared against constants (no magic strings/typos).
- **`RGB_SUFFIXES`** = `("_V",)`, **`THERMAL_SUFFIXES`** = `("_T",)` — the DJI M4T
  file-naming convention (V = visual, T = thermal). Tuples so more suffixes can be added.
- **`DEFAULT_MODALITY`** = `"rgb"` — safe, **non-destructive** fallback when no suffix
  matches and no override is given: RGB preprocessing is a pass-through, whereas guessing
  thermal would grayscale-collapse a real colour image (an irreversible mistake).

### `src/preprocessing.py`
- **`CLAHE_CLIP_LIMIT`** = `2.0` — a moderate contrast cap: high enough to lift thermal
  contrast, low enough to avoid amplifying sensor noise into false texture.
- **`CLAHE_TILE_GRID`** = `8` — an 8×8 grid of local tiles: balances local adaptivity
  (helps small warm targets stand out) against noise amplification on a 1280×1024 frame.

### `src/detection.py`
- **`DEFAULT_WEIGHTS`** = `"yolo11x.pt"` — the accurate configuration; set as the default
  so a bare run reproduces the reported setup (pass `yolo11n.pt` for a fast run).
- **`DEFAULT_DEVICE`** = `"cpu"` — this environment has no GPU; deterministic. Override
  with `"cuda:0"` where a GPU exists.
- **`DEFAULT_BASE_CONFIDENCE`** = `0.10` — a **low detection floor** so the model returns
  weak candidates; the real per-modality thresholds are applied later (step 6), and a low
  floor lets a confidence sweep re-threshold saved detections without re-running the model.
- **`DEFAULT_SLICE_HEIGHT` / `DEFAULT_SLICE_WIDTH`** = `640` — match YOLO's native training
  resolution, so each SAHI tile is exactly the scale the model expects (no internal rescaling).
- **`DEFAULT_OVERLAP_HEIGHT_RATIO` / `DEFAULT_OVERLAP_WIDTH_RATIO`** = `0.2` — ~128 px of
  overlap on a 640 tile: enough to fully contain a ~30 px person straddling a tile seam,
  without exploding the tile count (and cost).
- **`DEFAULT_IMGSZ`** = `1280` — the single-pass (non-SAHI) fallback size; a compromise
  that keeps some small-object ability without tiling. Only used with `--no-sahi`.

### `src/person_filter.py`
- **`PERSON_CLASS_ID`** = `0`, **`PERSON_CLASS_NAME`** = `"person"` — the COCO "person"
  class (id 0). Matching on **both** id and name is robust to a model that renumbers classes.

### `src/filtering.py`
- **`DEFAULT_NMS_IOU`** = `0.5` — a moderate overlap threshold: low enough to remove true
  duplicates, high enough not to merge distinct adjacent people in dense clusters. A
  deliberately exposed tuning knob.

### `src/annotation.py`
- **`BOX_COLOR`** = `{thermal: (0,255,255) yellow, rgb: (0,255,0) green}`,
  **`DEFAULT_BOX_COLOR`** = `(0,255,0)` — high-visibility, modality-distinct colours (BGR):
  green reads well on grass/pavement, yellow on grayscale thermal.

### `src/outputs.py`
- **`CONFIDENCE_DECIMALS`** = `4` — enough precision to preserve score ordering/thresholding
  while keeping the JSON free of noisy long floats.

### `main.py`
- **`DEFAULT_OUTPUT_DIR`** = `"outputs"` — default write location (git-ignored).
- **`DEFAULT_RGB_CONFIDENCE`** = `0.25` — the RGB operating point; RGB detections are
  high-precision, so a moderate threshold balances precision and recall.
- **`DEFAULT_THERMAL_CONFIDENCE`** = `0.20` — set **lower than RGB** because the
  COCO-pretrained model is under-confident on thermal (warm blobs only weakly resemble a
  "person"), so a lower threshold retains real detections. This asymmetry is why the
  threshold is **per modality**.
