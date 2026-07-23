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

def test_rgb_passthrough_returns_same_array():
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    assert preprocess(image, "rgb") is image  # RGB is a pass-through


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
