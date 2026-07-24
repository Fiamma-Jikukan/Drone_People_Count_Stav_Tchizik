"""Evaluation metrics (assignment item #5).

Functions that score model predictions against the point ground truth. They
take the *results* only (no model, no images), so they are fast and fully
testable:

  * predictions come as the pipeline's output records
    ``{image_name, modality, people_count, detections: [{bbox, confidence, class}]}``
  * ground truth comes as the point records
    ``{image_name, modality, people_count, points: [[x, y], ...]}``

Matching rule (as committed in doc 4): greedily match highest-confidence predicted
boxes to unmatched ground-truth points (point-in-box), one box to one point.
  * TP - a box that contains an unmatched ground-truth point.
  * FP - a box that contains no unmatched point.
  * FN - a ground-truth point inside no box.

Functions:

  * ``point_in_box``          - is a point inside a box.
  * ``match``                 - greedy point-in-box matching for one image.
  * ``precision_recall_f1``   - the three detection scores from TP/FP/FN.
  * ``image_report``          - all per-image metrics for one image.
  * ``aggregate``             - roll a list of per-image reports into totals + MAE.
  * ``aggregate_by_modality`` - the same, split into rgb / thermal.
  * ``evaluate``              - top level: pair records by name, report + aggregate.
"""


def point_in_box(point, box):
    """Return True if ``point`` (x, y) lies inside ``box`` (x1, y1, x2, y2)."""
    # Unpack the point and the box corners.
    x, y = point
    x1, y1, x2, y2 = box
    # Inside means within both spans (edges inclusive).
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_center(box):
    """Return the (cx, cy) centre of a box."""
    # Average the opposite corners.
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def match(points, detections):
    """Greedily match detections to points (highest confidence first).

    Args:
        points: list of ground-truth ``[x, y]`` points.
        detections: list of prediction dicts, each with ``bbox`` and ``confidence``.

    Returns:
        A dict with counts and the index lists needed to visualise the result::

            {"tp", "fp", "fn",
             "matches": [(detection_index, point_index), ...],
             "fp_detections": [detection_index, ...],   # false positives
             "fn_points": [point_index, ...]}           # false negatives
    """
    # Process detections from most to least confident.
    order = sorted(range(len(detections)), key=lambda i: detections[i]["confidence"], reverse=True)
    # Track which ground-truth points have already been claimed.
    matched_points = set()
    # Recorded (detection, point) pairs and the boxes that matched nothing.
    matches = []
    fp_detections = []

    # Walk the detections in confidence order.
    for det_index in order:
        # The current box.
        box = detections[det_index]["bbox"]
        # Unmatched points that fall inside this box.
        candidates = [
            p for p in range(len(points))
            if p not in matched_points and point_in_box(points[p], box)
        ]
        # If none, this box is a false positive.
        if not candidates:
            fp_detections.append(det_index)
            continue
        # Otherwise claim the candidate nearest the box centre (a stable tie-break).
        cx, cy = _box_center(box)
        best = min(candidates, key=lambda p: (points[p][0] - cx) ** 2 + (points[p][1] - cy) ** 2)
        matched_points.add(best)
        matches.append((det_index, best))

    # Any ground-truth point never claimed is a false negative.
    fn_points = [p for p in range(len(points)) if p not in matched_points]

    # Return the counts plus the index lists.
    return {
        "tp": len(matches),
        "fp": len(fp_detections),
        "fn": len(fn_points),
        "matches": matches,
        "fp_detections": fp_detections,
        "fn_points": fn_points,
    }


def precision_recall_f1(tp, fp, fn):
    """Return (precision, recall, f1) from true/false positive/negative counts."""
    # Precision = correct detections / all detections (guard against no detections).
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    # Recall = correct detections / all ground-truth people (guard against no GT).
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # F1 = harmonic mean of precision and recall (guard against both being zero).
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def image_report(gt_record, pred_record):
    """Compute all per-image metrics for one image.

    Args:
        gt_record: a ground-truth record (with ``points``).
        pred_record: a prediction record (with ``detections``).

    Returns:
        A metrics dict for the image, including the raw ``match`` detail (kept so
        false positives / negatives can be visualised later).
    """
    # The ground-truth points and the predicted detections.
    points = gt_record["points"]
    detections = pred_record["detections"]
    # Match them.
    result = match(points, detections)
    # Detection scores from the match counts.
    precision, recall, f1 = precision_recall_f1(result["tp"], result["fp"], result["fn"])
    # Counts and the absolute counting error.
    gt_count = len(points)
    pred_count = len(detections)
    # Assemble the per-image report.
    return {
        "image_name": gt_record["image_name"],
        "modality": gt_record["modality"],
        "gt_count": gt_count,
        "pred_count": pred_count,
        "abs_error": abs(pred_count - gt_count),
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "match": result,
    }


def aggregate(reports):
    """Roll a list of per-image reports into overall metrics.

    Returns totals, micro-averaged precision/recall/F1 (from summed TP/FP/FN), and
    the Mean Absolute counting Error. Returns zeros for an empty list.
    """
    # An empty set has no metrics.
    if not reports:
        return {"images": 0, "gt_total": 0, "pred_total": 0, "mae": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    # Sum the match counts across images.
    tp = sum(r["tp"] for r in reports)
    fp = sum(r["fp"] for r in reports)
    fn = sum(r["fn"] for r in reports)
    # Micro-averaged detection scores.
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    # Mean absolute counting error.
    mae = sum(r["abs_error"] for r in reports) / len(reports)
    # Assemble the aggregate.
    return {
        "images": len(reports),
        "gt_total": sum(r["gt_count"] for r in reports),
        "pred_total": sum(r["pred_count"] for r in reports),
        "mae": mae,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_by_modality(reports):
    """Aggregate reports separately per modality (the assignment scores rgb/thermal apart)."""
    # Build one aggregate per modality that actually appears.
    result = {}
    for modality in ("rgb", "thermal"):
        subset = [r for r in reports if r["modality"] == modality]
        if subset:
            result[modality] = aggregate(subset)
    return result


def evaluate(gt_records, pred_records):
    """Top level: pair ground-truth and prediction records by name and score them.

    Args:
        gt_records: iterable of ground-truth records.
        pred_records: iterable of prediction records.

    Returns:
        ``{"per_image": [...], "by_modality": {...}, "overall": {...}}``.

    Raises:
        KeyError: if a ground-truth image has no matching prediction.
    """
    # Index predictions by image name for lookup.
    preds_by_name = {p["image_name"]: p for p in pred_records}
    # Build a per-image report for every ground-truth record.
    reports = []
    for gt in gt_records:
        name = gt["image_name"]
        # Every ground-truth image must have a prediction to score against.
        if name not in preds_by_name:
            raise KeyError(f"No prediction found for ground-truth image: {name}")
        reports.append(image_report(gt, preds_by_name[name]))
    # Return per-image reports plus per-modality and overall aggregates.
    return {
        "per_image": reports,
        "by_modality": aggregate_by_modality(reports),
        "overall": aggregate(reports),
    }
