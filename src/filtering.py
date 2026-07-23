"""Step 6: confidence / IoU / NMS filtering.

After keeping only people (step 5), this step removes weak and duplicate boxes:

  * ``filter_confidence`` - drop detections below a confidence threshold
    (applied per modality: RGB and thermal use different thresholds).
  * ``iou``               - Intersection-over-Union of two boxes.
  * ``nms``               - greedy Non-Max Suppression to remove duplicates,
    including near-duplicates merged across SAHI tile seams.
  * ``filter_detections`` - convenience: confidence filter, then NMS.

"""

# Default IoU threshold above which two boxes are treated as duplicates.
DEFAULT_NMS_IOU = 0.5


def filter_confidence(detections, threshold):
    """Keep detections whose confidence is at least ``threshold``.

    Args:
        detections: List of detection dicts (each with a ``confidence`` key).
        threshold: Minimum confidence to keep (inclusive).

    Returns:
        A new list with only the sufficiently confident detections.
    """
    # Keep a detection only if its confidence meets or exceeds the threshold.
    return [det for det in detections if det["confidence"] >= threshold]


def iou(box_a, box_b):
    """Intersection-over-Union of two ``[x1, y1, x2, y2]`` boxes.

    Returns:
        A float in [0, 1]; 0 when the boxes do not overlap.
    """
    # Unpack both boxes into their corner coordinates.
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Left/top edges of the intersection are the larger of the two boxes' minima.
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    # Right/bottom edges of the intersection are the smaller of the two maxima.
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Intersection width/height, clamped to zero when the boxes miss each other.
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    # Intersection area.
    inter_area = inter_w * inter_h
    # No overlap: IoU is zero.
    if inter_area == 0:
        return 0.0

    # Area of each individual box.
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    # Union area = combined area minus the double-counted intersection.
    union = area_a + area_b - inter_area
    # Guard against a degenerate zero-area union.
    if union <= 0:
        return 0.0

    # IoU is the ratio of overlap to combined coverage.
    return inter_area / union


def nms(detections, iou_threshold=DEFAULT_NMS_IOU):
    """Greedy Non-Max Suppression over detection dicts.

    Detections are considered highest-confidence first; any lower-confidence box
    overlapping a kept box by more than ``iou_threshold`` is suppressed.

    Args:
        detections: List of detection dicts (each with ``bbox`` and ``confidence``).
        iou_threshold: Overlap above which a box is treated as a duplicate.

    Returns:
        A new list of the surviving detections, ordered by descending confidence.
    """
    # Process detections from most to least confident.
    ordered = sorted(detections, key=lambda det: det["confidence"], reverse=True)
    # Detections we decide to keep.
    kept = []
    # Consider each candidate in confidence order.
    for candidate in ordered:
        # Keep it only if it does not overlap any already-kept box too much.
        if all(iou(candidate["bbox"], k["bbox"]) <= iou_threshold for k in kept):
            # No strong overlap: this is a distinct detection, keep it.
            kept.append(candidate)
    # Return the de-duplicated detections.
    return kept


def filter_detections(detections, confidence_threshold, nms_iou=DEFAULT_NMS_IOU):
    """Convenience chain: confidence filter, then NMS.

    Args:
        detections: List of (already person-only) detection dicts.
        confidence_threshold: Minimum confidence to keep.
        nms_iou: IoU threshold for NMS de-duplication.

    Returns:
        The final filtered list of detections.
    """
    # First drop low-confidence detections.
    confident = filter_confidence(detections, confidence_threshold)
    # Then remove overlapping duplicates.
    return nms(confident, nms_iou)
