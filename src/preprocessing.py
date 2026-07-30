"""Step 3: modality-specific preprocessing.

RGB frames are left essentially untouched: the detector handles resizing/tiling,
and the main small-object lever is inference resolution, not pixel manipulation
(see answers_documents/1_data_analysis.md). We only guarantee a 3-channel image.

Thermal frames are 8-bit ``WhiteHot`` JPEGs whose brightness is re-normalised per
frame by the camera's automatic gain control, so contrast is inconsistent across
frames. We collapse to a single channel, apply CLAHE (Contrast Limited Adaptive
Histogram Equalisation) to stabilise local contrast, then replicate to 3 channels
so a COCO-pretrained (3-channel) detector can consume it.

Functions:

  * ``preprocess``          - dispatch to the right routine for a modality.
  * ``preprocess_rgb``      - RGB preprocessing (ensure 3-channel).
  * ``preprocess_thermal``  - thermal preprocessing (grayscale -> CLAHE -> 3-channel).
"""

# cv2: OpenCV — colour conversions and the CLAHE implementation.
import cv2

# THERMAL: canonical modality name.
from .modality import THERMAL

# CLAHE contrast-clip limit: higher = stronger local contrast (and more noise).
CLAHE_CLIP_LIMIT = 2.0
# CLAHE tile grid size (NxN tiles): equalisation is computed per tile.
CLAHE_TILE_GRID = 8


def preprocess(
    image_bgr,
    modality,
    apply_clahe=True,
    clahe_clip_limit=CLAHE_CLIP_LIMIT,
    clahe_tile_grid=CLAHE_TILE_GRID,
):
    """Return a 3-channel BGR image ready for the detector.

    Args:
        image_bgr: The loaded image as a BGR (or grayscale) NumPy array.
        modality: ``"rgb"`` or ``"thermal"``.
        apply_clahe: Whether to apply CLAHE to thermal frames.
        clahe_clip_limit: CLAHE contrast-clip limit (thermal only).
        clahe_tile_grid: CLAHE tile grid size, N for an NxN grid (thermal only).

    Returns:
        A 3-channel BGR ``uint8`` array.
    """
    # Thermal frames get contrast-normalising preprocessing.
    if modality == THERMAL:
        return preprocess_thermal(image_bgr, apply_clahe, clahe_clip_limit, clahe_tile_grid)
    # Everything else is treated as RGB (pass-through).
    return preprocess_rgb(image_bgr)


def preprocess_rgb(image_bgr):
    """RGB preprocessing: guarantee a 3-channel BGR image (pixels unchanged).

    Args:
        image_bgr: A BGR or grayscale image array.

    Returns:
        A 3-channel BGR array. Always a **fresh array** (never the caller's input
        aliased), so downstream in-place edits can't corrupt the original frame.
    """
    # A grayscale (2-D) image is promoted to 3 channels so the detector input is uniform.
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    # Already 3-channel: return a copy (not the aliased input) to avoid a mutation hazard.
    return image_bgr.copy()


def preprocess_thermal(
    image_bgr,
    apply_clahe=True,
    clahe_clip_limit=CLAHE_CLIP_LIMIT,
    clahe_tile_grid=CLAHE_TILE_GRID,
):
    """Thermal preprocessing: to grayscale -> optional CLAHE -> back to 3-channel.

    Args:
        image_bgr: The thermal frame as a BGR or grayscale array.
        apply_clahe: Whether to apply CLAHE contrast equalisation.
        clahe_clip_limit: CLAHE contrast-clip limit.
        clahe_tile_grid: CLAHE tile grid size, N for an NxN grid.

    Returns:
        A 3-channel BGR array (single equalised channel replicated to 3).
    """
    # Collapse to a single intensity channel (the thermal signal is monochrome).
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Already single-channel: use as-is.
    else:
        gray = image_bgr

    # Apply CLAHE to even out the camera's per-frame gain differences.
    if apply_clahe:
        # Build the CLAHE operator with the requested clip limit and tile grid.
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
        )
        # Run the equalisation on the single channel.
        gray = clahe.apply(gray)

    # Replicate the single channel to 3-channel BGR for the 3-channel detector.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
