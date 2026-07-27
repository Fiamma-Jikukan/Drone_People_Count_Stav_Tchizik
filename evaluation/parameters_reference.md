# Parameters Reference

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Every parameter you can change — when **running the model** (`main.py`) and when
**analysing the results** afterwards. For each: purpose, default, what low/high (or
on/off) means, and its effect. For what the *metrics* mean, see
[`metrics_reference.md`](metrics_reference.md).

> **Key split:** the confidence threshold can be re-applied **after** a run (cheap
> re-analysis); everything that changes detection or preprocessing (weights, SAHI,
> CLAHE, base floor, modality) needs a **re-run** of the model.

---

## A. Running the model — `main.py` flags

### Input / output / selection

**`--input` / `-i`** *(required)* — image file or directory to process. Not a tuning
knob; more images = more work.

**`--output` / `-o`** *(default `outputs`)* — where results are written
(`json/`, `annotated/`, and `run_config.json`).

**`--modality` / `-m`** *(default: inferred from `_V`/`_T` filename)* — force `rgb` or
`thermal`. **Purpose:** for files that don't follow the DJI naming. **Gotcha:** a wrong
modality applies the wrong preprocessing — forcing `thermal` on a colour image collapses
it to grayscale (destructive); forcing `rgb` on thermal just skips CLAHE (harmless).

### Model & compute

**`--weights`** *(default `yolo11x.pt`)* — which YOLO model. **`yolo11x`** = most
accurate (used for all reported results), slow on CPU. **`yolo11n`** = fast, weaker.
*Bigger → higher recall, slower; smaller → faster, more misses.* No effect on how the
metrics are computed, only on the detections.

**`--device`** *(default `cpu`)* — `cpu` / `cuda:0`. GPU is much faster; **no accuracy
effect**, purely speed.

### Detection strategy

**`--no-sahi`** *(default: SAHI **on**)* — disable tiled inference (single pass).
- **SAHI on:** slices the image into overlapping tiles so a small person is large
  *within* its tile → **big small-object recall gain**, but **N× slower** (many tiles).
- **SAHI off:** one pass, fast, but a 30 px aerial person shrinks to ~5 px and is lost →
  recall collapses.
- **The single biggest recall lever.** Turn off only for a speed/latency mode.

**`--base-conf`** *(default `0.10`)* — the model's **detection floor**: the lowest score
the detector returns at all (before the per-modality thresholds below).
- **Lower:** returns weaker candidate detections — needed to **capture a full range for a
  later confidence sweep**; costs more post-processing.
- **Higher:** discards weak detections at the source — they can **never** be recovered
  later.
- Keep it **≤** the per-modality thresholds.

### Thermal preprocessing

**`--no-clahe`** *(default: CLAHE **on**)* — disable CLAHE thermal contrast
normalisation. **RGB is a pass-through and unaffected either way.**
- **CLAHE on:** normalises the thermal camera's per-frame auto-gain (intended to help).
- **CLAHE off:** raw thermal. **Measured to be slightly better** here — CLAHE mildly hurt.

**`--clahe-clip`** *(default `2.0`)* — CLAHE contrast cap (thermal only, when CLAHE on).
*Higher → punchier contrast but more amplified noise; lower → gentler, less noise.*

**`--clahe-tile`** *(default `8`)* — CLAHE grid, N for an N×N tiling (thermal only).
*Finer (16) → more local adaptivity + noise; coarser (4) → smoother, closer to global.*

### Post-detection filtering (the main counting knobs)

**`--rgb-conf`** *(default `0.25`)* — RGB confidence threshold.
*Higher → precision ↑, recall ↓, fewer boxes; lower → recall ↑, precision ↓.* RGB has a
broad optimum around **0.20–0.25**.

**`--thermal-conf`** *(default `0.20`)* — thermal confidence threshold. Same P/R
trade-off, but **thermal usually wants a lower value than RGB** because the model is
*under-confident* on thermal (out-of-domain scores sit near the noise floor).
**Separate from RGB on purpose** — the two modalities have different optimal operating
points.

**`--nms-iou`** *(default `0.5`)* — overlap above which two boxes are treated as the same
object and de-duplicated.
- **Higher / looser (e.g. 0.6):** keeps more overlapping boxes → duplicates, over-count
  in clusters.
- **Lower / stricter (e.g. 0.4):** merges neighbours → under-count in dense clusters
  (recall ↓).
- Matters most in **dense seated clusters**.

---

## B. After getting results — analysis parameters

### Confidence sweep — `confidence_sweep.py`

**`--thresholds`** *(default `0.10 0.15 0.20 0.25 0.30 0.35`)* — the confidence values to
**re-apply to the saved detections**, recomputing metrics at each.
- **Exact** (equivalent to re-running the pipeline at each threshold) because SAHI's
  merge and our NMS are greedy by score — no model re-run needed.
- **Only works down to the saved floor** (the run's `--base-conf` / `--*-conf`). Detections
  weaker than what was saved can't be recovered — capture at a low floor first.

**`--gt-dir` / `--pred-dir` / `--output`** — paths (ground truth, saved predictions,
output). Not tuning knobs.

### CLAHE sweep — `clahe_sweep.py`

**`--clips` / `--tiles`** *(default `1 2 4` × `4 8 16`)* — the CLAHE grid to try.
**Re-runs the model per setting** (CLAHE is pre-detection, so it can't be re-applied to
saved detections like confidence can). Plus `--thermal-conf`, `--nms-iou`, `--weights`,
`--device` fixed across the grid so CLAHE is the only variable.

### Evaluation — `evaluate.py`

`--gt-dir`, `--pred-dir`, `--images-dir`, `--output` — paths only. Scoring itself has no
tunable knobs: the **matching rule is fixed** (point-in-box, greedy by confidence,
one box ↔ one point).

---

## C. Fixed / code-level parameters (changeable in code, not via CLI)

In `src/detection.py`:
- **SAHI tile size** `640×640` — matches YOLO's native training resolution. *Smaller →
  better tiny-object recall but more tiles (slower); larger → fewer tiles, small people
  shrink.*
- **SAHI overlap** `0.2` (~128 px) — *higher → fewer tile-seam misses but more
  tiles/duplicates; lower → faster but people on seams get missed.*
- **`imgsz` `1280`** — single-pass input size, used **only** with `--no-sahi`.

In `evaluation/metrics.py`:
- **Matching rule** — point-in-box, greedy by descending confidence, one-to-one. Defines
  TP/FP/FN. A methodological choice, not a runtime parameter.

---

## Quick reference: what needs a re-run?

| Parameter | Re-run the model? | Why |
|-----------|:-----------------:|-----|
| `--rgb-conf` / `--thermal-conf` | **No** (post-hoc) | Applied after detection → re-thresholded by the sweep. |
| `--weights`, `--device` | **Yes** | Different detector / hardware. |
| `--no-sahi` | **Yes** | Changes how detection is run. |
| `--no-clahe`, `--clahe-clip`, `--clahe-tile` | **Yes** | Pre-detection preprocessing. |
| `--base-conf` | **Yes** | Sets what the detector returns (the sweep floor). |
| `--modality` | **Yes** | Changes preprocessing per image. |
| `--nms-iou` | **Yes** | NMS is applied *before* the JSON is saved, so it's baked in. |

> Practical upshot: capture once at a **low `--base-conf`** with **SAHI on** and **CLAHE
> off** (the best-known config), then explore the **confidence threshold** for free via
> the sweep. Changing anything else means another model run.
