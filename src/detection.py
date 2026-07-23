"""Step 4: running the selected model.

The selected approach (see answers_documents/2_model_research_and_selection.md)
is Ultralytics YOLO11 run through SAHI sliced/tiled inference, so that small
aerial people stay large enough to detect. This module exposes:

  * ``load_model`` - build the detector once (downloads weights on first use).
  * ``detect``     - run detection on one preprocessed image.

A detection is a plain dict, which maps directly onto the final JSON schema::

    {"bbox": [x1, y1, x2, y2], "confidence": 0.91, "class_id": 0, "class_name": "person"}

No class or confidence filtering happens here — that is the job of the later
steps (filtering / counting), keeping detection and filtering separable.

Heavy dependencies (``sahi``, ``ultralytics``, ``torch``) are imported lazily
inside the functions, so importing this module stays cheap.
"""

# Model weights file; auto-downloaded by Ultralytics on first use.
DEFAULT_WEIGHTS = "yolo11n.pt"
# Inference device: "cpu", "cuda:0", or None to let the library choose.
DEFAULT_DEVICE = "cpu"
# Low detection floor; per-modality confidence thresholds are applied later.
DEFAULT_BASE_CONFIDENCE = 0.10

# SAHI tile height in pixels.
DEFAULT_SLICE_HEIGHT = 640
# SAHI tile width in pixels.
DEFAULT_SLICE_WIDTH = 640
# Fractional vertical overlap between neighbouring tiles.
DEFAULT_OVERLAP_HEIGHT_RATIO = 0.2
# Fractional horizontal overlap between neighbouring tiles.
DEFAULT_OVERLAP_WIDTH_RATIO = 0.2
# Inference image size for the single-pass (non-SAHI) fallback.
DEFAULT_IMGSZ = 1280


def load_model(
    weights=DEFAULT_WEIGHTS,
    device=DEFAULT_DEVICE,
    base_confidence=DEFAULT_BASE_CONFIDENCE,
):
    """Build and return the detector model handle.

    Args:
        weights: Path/name of the YOLO11 weights (auto-downloaded if absent).
        device: Inference device ("cpu", "cuda:0", or None to auto-select).
        base_confidence: Minimum score a detection needs to be returned at all.

    Returns:
        A SAHI ``AutoDetectionModel`` wrapping the Ultralytics YOLO model.
    """
    # Import lazily so this module is importable without the model stack.
    from sahi import AutoDetectionModel

    # Wrap Ultralytics YOLO11 in SAHI's generic detection-model interface.
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=base_confidence,
        device=device,
    )


def detect(
    model,
    image_bgr,
    use_sahi=True,
    slice_height=DEFAULT_SLICE_HEIGHT,
    slice_width=DEFAULT_SLICE_WIDTH,
    overlap_height_ratio=DEFAULT_OVERLAP_HEIGHT_RATIO,
    overlap_width_ratio=DEFAULT_OVERLAP_WIDTH_RATIO,
    imgsz=DEFAULT_IMGSZ,
):
    """Run detection on one preprocessed BGR image and return all detections.

    Args:
        model: The handle returned by ``load_model``.
        image_bgr: A preprocessed 3-channel BGR image.
        use_sahi: If True, use SAHI sliced inference; otherwise a single pass.
        slice_height, slice_width: SAHI tile size in pixels.
        overlap_height_ratio, overlap_width_ratio: fractional tile overlap.
        imgsz: inference size for the single-pass fallback.

    Returns:
        A list of detection dicts (unfiltered).
    """
    # SAHI tiled inference is the selected path for small aerial people.
    if use_sahi:
        return _detect_sahi(
            model, image_bgr,
            slice_height, slice_width,
            overlap_height_ratio, overlap_width_ratio,
        )
    # Single-pass fallback: faster, but weaker on tiny objects.
    return _detect_single_pass(model, image_bgr, imgsz)


def _make_detection(x1, y1, x2, y2, confidence, class_id, class_name):
    """Assemble one detection dict with clean, rounded pixel coordinates."""
    # Round float box coordinates to integer pixels and build the record.
    return {
        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        "confidence": float(confidence),
        "class_id": int(class_id),
        "class_name": str(class_name),
    }


def _detect_sahi(
    model, image_bgr,
    slice_height, slice_width,
    overlap_height_ratio, overlap_width_ratio,
):
    """SAHI sliced inference backend."""
    # Import lazily to keep module import cheap.
    from sahi.predict import get_sliced_prediction

    # Run tiled prediction; SAHI/Ultralytics expect RGB, so reverse BGR channels.
    result = get_sliced_prediction(
        image_bgr[..., ::-1],
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        verbose=0,
    )

    # Convert each SAHI object prediction into our detection dict.
    detections = []
    # Iterate over every predicted object across all merged tiles.
    for obj in result.object_prediction_list:
        # Extract the box as (x1, y1, x2, y2) in original-image pixels.
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        # Append the normalised detection record.
        detections.append(
            _make_detection(x1, y1, x2, y2, obj.score.value, obj.category.id, obj.category.name)
        )
    # Return the full, unfiltered detection list.
    return detections


def _detect_single_pass(model, image_bgr, imgsz):
    """Single-pass (non-tiled) Ultralytics inference backend."""
    # Reach the underlying Ultralytics YOLO model wrapped by SAHI.
    yolo = model.model

    # Run one forward pass on the whole image (RGB), reusing the model's confidence floor.
    results = yolo.predict(
        image_bgr[..., ::-1],
        imgsz=imgsz,
        conf=model.confidence_threshold,
        verbose=False,
    )

    # Convert Ultralytics boxes into our detection dicts.
    detections = []
    # A predict call returns one result per input image; we passed one image.
    for res in results:
        # Map class ids to human-readable names.
        names = res.names
        # Each box holds coordinates, class id, and confidence.
        for box in res.boxes:
            # Box coordinates as (x1, y1, x2, y2) floats.
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Numeric class id for this detection.
            class_id = int(box.cls[0])
            # Append the normalised detection record.
            detections.append(
                _make_detection(x1, y1, x2, y2, float(box.conf[0]), class_id, names.get(class_id, class_id))
            )
    # Return the full, unfiltered detection list.
    return detections
