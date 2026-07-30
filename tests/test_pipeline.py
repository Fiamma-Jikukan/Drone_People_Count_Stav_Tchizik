"""Test suite for the person-counting pipeline (core, model-free logic).

Covers the counting-critical steps and common failure cases without loading the
detection model, so the whole suite runs in a fraction of a second. Organised by
pipeline step. Run with:

    python -m pytest
"""

# json: verify the JSON output round-trips.
import json

# cv2 / numpy: build and check small test images.
import cv2
import numpy as np
# pytest: fixtures (tmp_path) and exception assertions.
import pytest

# The pipeline steps under test.
from src.loading import discover_images, load_image
from src.modality import infer_modality, normalize_modality
from src.preprocessing import preprocess, preprocess_thermal
from src.person_filter import filter_person
from src.filtering import filter_confidence, filter_detections, iou, nms
from src.counting import count_people
from src.annotation import draw_detections, save_annotated_image
from src.outputs import build_result, save_json
from ground_truth import (
    build_ground_truth,
    load_ground_truth,
    load_ground_truth_dir,
    save_ground_truth,
)
from evaluation import (
    aggregate,
    aggregate_by_modality,
    evaluate,
    image_report,
    match,
    point_in_box,
    precision_recall_f1,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _write_image(path, shape=(8, 8, 3)):
    """Write a small blank image to ``path``."""
    ok, buffer = cv2.imencode(".jpg", np.zeros(shape, dtype=np.uint8))
    buffer.tofile(str(path))


def _person_det(class_id, class_name):
    """A detection dict with a given class (for the person filter)."""
    return {"bbox": [0, 0, 1, 1], "confidence": 0.9, "class_id": class_id, "class_name": class_name}


def _box_det(x1, y1, x2, y2, confidence):
    """A person detection dict with a given box and confidence."""
    return {"bbox": [x1, y1, x2, y2], "confidence": confidence, "class_id": 0, "class_name": "person"}


# --------------------------------------------------------------------------
# Step 1: loading
# --------------------------------------------------------------------------

def test_discover_single_file(tmp_path):
    image = tmp_path / "a_V.jpg"
    _write_image(image)
    assert discover_images(image) == [image]


def test_discover_directory_is_sorted_and_filtered(tmp_path):
    for name in ["b_V.jpg", "a_T.jpg"]:
        _write_image(tmp_path / name)
    (tmp_path / "notes.txt").write_text("ignore me")
    found = [p.name for p in discover_images(tmp_path)]
    assert found == ["a_T.jpg", "b_V.jpg"]  # sorted; non-image excluded


def test_discover_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path / "nope")


def test_discover_unsupported_file_raises(tmp_path):
    bad = tmp_path / "doc.pdf"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        discover_images(bad)


def test_discover_empty_directory_returns_empty(tmp_path):
    assert discover_images(tmp_path) == []


def test_discover_txt_manifest(tmp_path):
    # A .txt manifest lists image paths (comments/blanks ignored) — no file duplication.
    img_b = tmp_path / "b_V.jpg"
    img_a = tmp_path / "a_T.jpg"
    _write_image(img_b)
    _write_image(img_a)
    manifest = tmp_path / "sample.txt"
    manifest.write_text(f"# a sample\n{img_b}\n\n{img_a}\n", encoding="utf-8")
    found = [p.name for p in discover_images(manifest)]
    assert found == ["a_T.jpg", "b_V.jpg"]  # sorted, both resolved


def test_discover_manifest_missing_file_raises(tmp_path):
    manifest = tmp_path / "sample.txt"
    manifest.write_text(f"{tmp_path / 'ghost_V.jpg'}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        discover_images(manifest)


def test_load_image_roundtrip(tmp_path):
    path = tmp_path / "x_V.png"
    _write_image(path, (6, 6, 3))
    image = load_image(path)
    assert image.shape == (6, 6, 3)


def test_load_image_corrupt_file_raises(tmp_path):
    bad = tmp_path / "broken_V.jpg"
    bad.write_bytes(b"not really a jpeg")
    with pytest.raises(FileNotFoundError):
        load_image(bad)


# --------------------------------------------------------------------------
# Step 2: modality
# --------------------------------------------------------------------------

def test_thermal_suffix():
    assert infer_modality("DJI_20260621190710_0001_T.JPG") == "thermal"


def test_rgb_suffix():
    assert infer_modality("DJI_20260621190710_0001_V.JPG") == "rgb"


def test_suffix_is_case_insensitive():
    assert infer_modality("scene_t.jpg") == "thermal"
    assert infer_modality("scene_v.jpg") == "rgb"


def test_override_wins_over_name():
    assert infer_modality("clearly_V.jpg", override="thermal") == "thermal"


def test_unknown_name_falls_back_to_default():
    assert infer_modality("random_photo.jpg") == "rgb"


def test_custom_default():
    assert infer_modality("random_photo.jpg", default="thermal") == "thermal"


def test_only_filename_is_inspected():
    assert infer_modality("/some/dir_V/scene_T.JPG") == "thermal"


def test_normalize_modality_accepts_and_cleans():
    assert normalize_modality(" Thermal ") == "thermal"


def test_normalize_modality_rejects_bad_value():
    with pytest.raises(ValueError):
        normalize_modality("infrared")


# --------------------------------------------------------------------------
# Step 3: preprocessing
# --------------------------------------------------------------------------

def test_rgb_passthrough_preserves_pixels_without_aliasing():
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    out = preprocess(image, "rgb")
    # RGB is a pixel pass-through: same content ...
    assert np.array_equal(out, image)
    # ... but a fresh array, so mutating it never corrupts the caller's frame.
    assert out is not image
    out[0, 0] = 255
    assert image[0, 0].sum() == 0


def test_thermal_output_is_three_channel():
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = preprocess(image, "thermal")
    assert out.ndim == 3 and out.shape[2] == 3


def test_thermal_clahe_changes_pixels():
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    with_clahe = preprocess_thermal(image, apply_clahe=True)
    without = preprocess_thermal(image, apply_clahe=False)
    assert not np.array_equal(with_clahe, without)


def test_thermal_channels_are_replicated_gray():
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = preprocess_thermal(image, apply_clahe=True)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


# --------------------------------------------------------------------------
# Step 5: person filter
# --------------------------------------------------------------------------

def test_keeps_only_people():
    dets = [_person_det(0, "person"), _person_det(8, "boat"), _person_det(14, "bird")]
    kept = filter_person(dets)
    assert len(kept) == 1
    assert kept[0]["class_name"] == "person"


def test_matches_by_name_even_with_odd_id():
    assert len(filter_person([_person_det(99, "person")])) == 1


def test_preserves_order():
    dets = [_person_det(0, "person"), _person_det(2, "car"), _person_det(0, "person")]
    assert len(filter_person(dets)) == 2


def test_empty_input():
    assert filter_person([]) == []


# --------------------------------------------------------------------------
# Step 6: confidence / IoU / NMS filtering
# --------------------------------------------------------------------------

def test_iou_identical_is_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_is_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == 50 / 150


def test_confidence_threshold_is_inclusive():
    kept = filter_confidence([_box_det(0, 0, 1, 1, 0.5), _box_det(0, 0, 1, 1, 0.49)], 0.5)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.5


def test_nms_suppresses_overlapping_lower_confidence():
    kept = nms([_box_det(1, 1, 11, 11, 0.5), _box_det(0, 0, 10, 10, 0.9)], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


def test_nms_keeps_disjoint_boxes():
    kept = nms([_box_det(0, 0, 10, 10, 0.9), _box_det(100, 100, 110, 110, 0.8)], iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_keeps_both_when_overlap_below_threshold():
    # IoU here is 1/3, which is <= 0.5, so both survive.
    kept = nms([_box_det(0, 0, 10, 10, 0.9), _box_det(5, 0, 15, 10, 0.8)], iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_empty_input():
    assert nms([], 0.5) == []


def test_filter_detections_drops_weak_and_duplicate():
    dets = [
        _box_det(0, 0, 10, 10, 0.9),     # kept
        _box_det(1, 1, 11, 11, 0.6),     # duplicate of the above -> suppressed
        _box_det(50, 50, 60, 60, 0.1),   # below threshold -> dropped
    ]
    kept = filter_detections(dets, confidence_threshold=0.25, nms_iou=0.5)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


# --------------------------------------------------------------------------
# Step 7: counting
# --------------------------------------------------------------------------

def test_counts_detections():
    assert count_people([{"a": 1}, {"a": 2}, {"a": 3}]) == 3


def test_empty_is_zero():
    assert count_people([]) == 0


# --------------------------------------------------------------------------
# Step 8: annotation
# --------------------------------------------------------------------------

def test_draw_does_not_mutate_input():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [_box_det(5, 5, 20, 20, 0.9)]
    out = draw_detections(image, dets, modality="rgb", count=1)
    assert out.shape == image.shape
    assert np.array_equal(image, np.zeros((50, 50, 3), dtype=np.uint8))  # original untouched
    assert not np.array_equal(out, image)  # something was drawn


def test_save_creates_file_and_roundtrips(tmp_path):
    image = np.full((12, 12, 3), 128, dtype=np.uint8)
    out_path = save_annotated_image(image, tmp_path / "sub" / "annotated.jpg")
    assert out_path.is_file()
    reloaded = cv2.imdecode(np.fromfile(str(out_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert reloaded.shape == (12, 12, 3)


# --------------------------------------------------------------------------
# Step 9: structured output
# --------------------------------------------------------------------------

def test_build_result_matches_schema_and_renames_class():
    dets = [{"bbox": [1, 2, 3, 4], "confidence": 0.912345, "class_id": 0, "class_name": "person"}]
    result = build_result("a_V.JPG", "rgb", dets)
    assert result == {
        "image_name": "a_V.JPG",
        "modality": "rgb",
        "people_count": 1,
        "detections": [
            # class_id dropped, class_name -> "class", confidence rounded to 4 dp.
            {"bbox": [1, 2, 3, 4], "confidence": 0.9123, "class": "person"},
        ],
    }


def test_build_result_empty_has_zero_count():
    result = build_result("a_T.JPG", "thermal", [])
    assert result["people_count"] == 0
    assert result["detections"] == []


def test_save_json_roundtrip(tmp_path):
    result = build_result("a.jpg", "thermal", [])
    path = save_json(result, tmp_path / "sub" / "a.json")
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == result


# --------------------------------------------------------------------------
# Ground truth (item #4)
# --------------------------------------------------------------------------

def test_build_ground_truth_derives_count_and_rounds_points():
    record = build_ground_truth("a_V.JPG", "rgb", [(10.4, 20.6), (30, 40)])
    assert record == {
        "image_name": "a_V.JPG",
        "modality": "rgb",
        "people_count": 2,
        "points": [[10, 21], [30, 40]],  # rounded to int pixels
    }


def test_build_ground_truth_empty_is_zero():
    record = build_ground_truth("a_T.JPG", "thermal", [])
    assert record["people_count"] == 0
    assert record["points"] == []


def test_ground_truth_save_and_load_roundtrip(tmp_path):
    record = build_ground_truth("a_V.JPG", "rgb", [(1, 2), (3, 4), (5, 6)])
    path = save_ground_truth(record, tmp_path / "gt" / "a_V.json")
    assert path.is_file()
    assert load_ground_truth(path) == record


def test_load_ground_truth_count_mismatch_raises(tmp_path):
    bad = {"image_name": "x.JPG", "modality": "rgb", "people_count": 5, "points": [[1, 2]]}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(path)


def test_load_ground_truth_missing_key_raises(tmp_path):
    bad = {"image_name": "x.JPG", "modality": "rgb", "points": []}  # no people_count
    path = tmp_path / "x.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(path)


def test_load_ground_truth_dir_sorted(tmp_path):
    save_ground_truth(build_ground_truth("b_V.JPG", "rgb", [(1, 1)]), tmp_path / "b_V.json")
    save_ground_truth(build_ground_truth("a_T.JPG", "thermal", []), tmp_path / "a_T.json")
    records = load_ground_truth_dir(tmp_path)
    assert [r["image_name"] for r in records] == ["a_T.JPG", "b_V.JPG"]


def test_load_ground_truth_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth_dir(tmp_path / "nope")


# --------------------------------------------------------------------------
# Evaluation metrics (item #5)
# --------------------------------------------------------------------------

def _gt(image_name, modality, points):
    """A ground-truth record."""
    return {"image_name": image_name, "modality": modality, "people_count": len(points), "points": points}


def _pred(image_name, modality, detections):
    """A prediction record."""
    return {"image_name": image_name, "modality": modality, "people_count": len(detections), "detections": detections}


# --- point_in_box ---

def test_point_in_box_inside_and_outside():
    assert point_in_box([5, 5], [0, 0, 10, 10]) is True
    assert point_in_box([15, 5], [0, 0, 10, 10]) is False


def test_point_in_box_edges_inclusive():
    assert point_in_box([0, 0], [0, 0, 10, 10]) is True
    assert point_in_box([10, 10], [0, 0, 10, 10]) is True


# --- match ---

def test_match_box_over_two_points_is_one_tp_one_fn():
    # One box covering two points can only claim one of them.
    m = match([[4, 5], [6, 5]], [_box_det(0, 0, 10, 10, 0.9)])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)


def test_match_two_boxes_one_point_higher_confidence_wins():
    m = match([[5, 5]], [_box_det(0, 0, 10, 10, 0.4), _box_det(0, 0, 10, 10, 0.95)])
    assert (m["tp"], m["fp"]) == (1, 1)
    assert m["matches"] == [(1, 0)]      # the 0.95 detection (index 1) claims the point
    assert m["fp_detections"] == [0]     # the 0.4 detection is the false positive


def test_match_empty_box_is_false_positive():
    m = match([], [_box_det(0, 0, 1, 1, 0.9)])
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 0)


def test_match_lonely_point_is_false_negative():
    m = match([[100, 100]], [_box_det(0, 0, 10, 10, 0.9)])
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_match_empty_inputs():
    m = match([], [])
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 0)


# --- precision / recall / F1 ---

def test_prf_normal_case():
    precision, recall, f1 = precision_recall_f1(8, 2, 4)
    assert precision == 0.8
    assert round(recall, 4) == 0.6667
    assert round(f1, 4) == 0.7273


def test_prf_zero_division_guards():
    assert precision_recall_f1(0, 0, 5) == (0.0, 0.0, 0.0)   # no detections
    assert precision_recall_f1(0, 5, 0) == (0.0, 0.0, 0.0)   # no ground truth
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)   # nothing at all


# --- image_report ---

def test_image_report_fields():
    gt = _gt("a_V.JPG", "rgb", [[5, 5], [6, 6], [100, 100]])
    pred = _pred("a_V.JPG", "rgb", [_box_det(0, 0, 10, 10, 0.9), _box_det(200, 200, 210, 210, 0.8)])
    r = image_report(gt, pred)
    assert (r["gt_count"], r["pred_count"], r["abs_error"]) == (3, 2, 1)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 2)


# --- aggregate ---

def test_aggregate_mae_and_micro_scores():
    reports = [
        {"modality": "rgb", "gt_count": 10, "pred_count": 8, "abs_error": 2, "tp": 8, "fp": 0, "fn": 2},
        {"modality": "rgb", "gt_count": 5, "pred_count": 9, "abs_error": 4, "tp": 5, "fp": 4, "fn": 0},
    ]
    a = aggregate(reports)
    assert a["images"] == 2 and a["mae"] == 3.0
    assert (a["tp"], a["fp"], a["fn"]) == (13, 4, 2)
    assert round(a["precision"], 4) == round(13 / 17, 4)   # micro-averaged
    assert round(a["recall"], 4) == round(13 / 15, 4)


def test_aggregate_empty_is_zeros():
    a = aggregate([])
    assert a["images"] == 0 and a["mae"] == 0.0 and a["precision"] == 0.0


def test_aggregate_by_modality_splits():
    reports = [
        {"modality": "rgb", "gt_count": 3, "pred_count": 3, "abs_error": 0, "tp": 3, "fp": 0, "fn": 0},
        {"modality": "thermal", "gt_count": 4, "pred_count": 1, "abs_error": 3, "tp": 1, "fp": 0, "fn": 3},
    ]
    by = aggregate_by_modality(reports)
    assert set(by) == {"rgb", "thermal"}
    assert by["rgb"]["images"] == 1 and by["thermal"]["mae"] == 3.0


# --- evaluate ---

def test_evaluate_pairs_by_name_and_aggregates():
    gt = [_gt("a_V.JPG", "rgb", [[5, 5]]), _gt("a_T.JPG", "thermal", [[5, 5]])]
    pred = [_pred("a_V.JPG", "rgb", [_box_det(0, 0, 10, 10, 0.9)]), _pred("a_T.JPG", "thermal", [])]
    res = evaluate(gt, pred)
    assert len(res["per_image"]) == 2
    assert res["by_modality"]["rgb"]["recall"] == 1.0     # the one RGB person was found
    assert res["by_modality"]["thermal"]["recall"] == 0.0  # the one thermal person was missed


def test_evaluate_missing_prediction_raises():
    gt = [_gt("a_V.JPG", "rgb", [[5, 5]])]
    with pytest.raises(KeyError):
        evaluate(gt, [])


# --------------------------------------------------------------------------
# Orchestration: process_image end-to-end (mocked detector)
# --------------------------------------------------------------------------

def test_process_image_end_to_end_with_mocked_detector(tmp_path, monkeypatch):
    # Exercises steps 1-9 (load -> modality -> preprocess -> detect -> person
    # filter -> confidence/NMS -> count -> annotate -> JSON) with the model stubbed.
    import main

    # A real input image on disk; modality is inferred from the _V suffix -> rgb.
    img = tmp_path / "scene_0001_V.jpg"
    _write_image(img, shape=(64, 64, 3))
    out_dir = tmp_path / "out"

    # Stub the detector to return a fixed set, bypassing the model entirely.
    fake = [
        _box_det(4, 4, 20, 24, 0.90),                    # person, kept
        _box_det(30, 30, 50, 55, 0.50),                  # person, kept (>= rgb 0.25)
        _box_det(40, 41, 45, 46, 0.10),                  # person, dropped (< 0.25)
        {"bbox": [10, 10, 25, 25], "confidence": 0.95,   # not a person -> dropped
         "class_id": 2, "class_name": "car"},
    ]
    monkeypatch.setattr(main, "detect", lambda *a, **k: [dict(d) for d in fake])

    args = main.parse_args(["--input", str(img), "--output", str(out_dir)])
    result = main.process_image(model=object(), image_path=img, args=args)

    # Only the two confident person boxes survive the filter chain.
    assert result["modality"] == "rgb"
    assert result["image_name"] == "scene_0001_V.jpg"
    assert result["people_count"] == 2
    assert len(result["detections"]) == 2
    # Output schema keeps bbox/confidence/class only.
    assert set(result["detections"][0]) == {"bbox", "confidence", "class"}

    # The JSON record was written and matches the returned result.
    json_path = out_dir / "json" / "scene_0001_V.json"
    assert json_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8")) == result

    # An annotated image was written and is a decodable 3-channel image.
    annotated_path = out_dir / "annotated" / "scene_0001_V.jpg"
    assert annotated_path.is_file()
    decoded = load_image(annotated_path)
    assert decoded.ndim == 3 and decoded.shape[2] == 3


# --------------------------------------------------------------------------
# Plots: smoke test that plot_all renders every chart (item #5)
# --------------------------------------------------------------------------

def test_plot_all_writes_every_chart(tmp_path):
    from evaluation import plots
    gt = [_gt("a_V.JPG", "rgb", [[5, 5]]), _gt("a_T.JPG", "thermal", [[5, 5]])]
    pred = [_pred("a_V.JPG", "rgb", [_box_det(0, 0, 10, 10, 0.9)]), _pred("a_T.JPG", "thermal", [])]
    results = evaluate(gt, pred)
    written = plots.plot_all(results, tmp_path / "plots")
    # All five charts rendered to real PNG files.
    assert len(written) == 5
    for path in written:
        assert path.is_file() and path.suffix == ".png"
