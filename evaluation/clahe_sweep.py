"""CLAHE clip/tile sweep (ablation, item #6).

CLAHE is a *preprocessing* step, so — unlike the confidence sweep — each setting
changes the detector's input and must be re-run. To stay cheap, this loads the model
**once** and loops the (clip, tile) grid (plus a CLAHE-off reference) over the thermal
images, scoring each setting against the ground truth. Thermal-only (RGB is unaffected
by CLAHE).

Consistency check: the "off" row should reproduce the CLAHE-off run, and "clip=2.0
tile=8" should reproduce the CLAHE-on baseline (both at the same thermal threshold).

Usage:
    python evaluation/clahe_sweep.py
    python evaluation/clahe_sweep.py --clips 1 2 4 --tiles 4 8 16 --thermal-conf 0.20
"""

# argparse / csv / sys / pathlib: CLI + table IO.
import argparse
import csv
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Pipeline stages reused for prediction.
from src.loading import load_image
from src.preprocessing import preprocess_thermal
from src.detection import load_model, detect
from src.person_filter import filter_person
from src.filtering import filter_detections
from ground_truth import load_ground_truth_dir
from evaluation.metrics import evaluate

# Default grid for the sweep.
DEFAULT_CLIPS = [1.0, 2.0, 4.0]
DEFAULT_TILES = [4, 8, 16]


def predict_thermal(model, image_path, apply_clahe, clip, tile, thermal_conf, nms_iou):
    """Predict people on one thermal image with a given CLAHE setting."""
    # Load and preprocess with the requested CLAHE configuration.
    image = load_image(image_path)
    preprocessed = preprocess_thermal(image, apply_clahe=apply_clahe,
                                      clahe_clip_limit=clip, clahe_tile_grid=tile)
    # Detect, keep people, then apply the (fixed) confidence + NMS filtering.
    raw = detect(model, preprocessed, use_sahi=True)
    people = filter_person(raw)
    return filter_detections(people, confidence_threshold=thermal_conf, nms_iou=nms_iou)


def score_setting(model, gt_records, images_dir, apply_clahe, clip, tile, thermal_conf, nms_iou):
    """Run one CLAHE setting over all thermal images and return the aggregate metrics."""
    # Build a prediction record per thermal ground-truth image.
    preds = []
    for gt in gt_records:
        people = predict_thermal(model, Path(images_dir) / gt["image_name"],
                                 apply_clahe, clip, tile, thermal_conf, nms_iou)
        preds.append({"image_name": gt["image_name"], "modality": "thermal",
                      "people_count": len(people), "detections": people})
    # Score against the ground truth and return the thermal aggregate.
    return evaluate(gt_records, preds)["by_modality"]["thermal"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sweep CLAHE clip/tile on the thermal images.")
    parser.add_argument("--gt-dir", default="ground_truth", help="Ground-truth JSON directory.")
    parser.add_argument("--images-dir", default="input_images", help="Directory holding the source images.")
    parser.add_argument("--output", "-o", default="evaluation/clahe_sweep", help="Output directory.")
    parser.add_argument("--weights", default="yolo11x.pt", help="YOLO weights.")
    parser.add_argument("--device", default="cpu", help='Device, e.g. "cpu" or "cuda:0".')
    parser.add_argument("--thermal-conf", type=float, default=0.20, help="Fixed thermal confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--clips", type=float, nargs="+", default=DEFAULT_CLIPS, help="CLAHE clip limits to try.")
    parser.add_argument("--tiles", type=int, nargs="+", default=DEFAULT_TILES, help="CLAHE tile grid sizes to try.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Ground truth, restricted to thermal images.
    gt_records = [g for g in load_ground_truth_dir(args.gt_dir) if g["modality"] == "thermal"]
    if not gt_records:
        print("No thermal ground-truth records found.")
        return 1

    # Load the model once (low floor so the fixed threshold applies cleanly).
    model = load_model(weights=args.weights, device=args.device, base_confidence=0.10)

    # Settings: a CLAHE-off reference, then the full clip x tile grid.
    settings = [("off", False, None, None)]
    for clip in args.clips:
        for tile in args.tiles:
            settings.append((f"clip={clip} tile={tile}", True, clip, tile))

    # Score every setting.
    rows = []
    print(f"{'setting':<20} {'P':>6} {'R':>6} {'F1':>6} {'MAE':>7}")
    for label, apply_clahe, clip, tile in settings:
        # For the "off" reference, clip/tile are irrelevant (use placeholders).
        agg = score_setting(model, gt_records, args.images_dir, apply_clahe,
                            clip or 2.0, tile or 8, args.thermal_conf, args.nms_iou)
        rows.append({"setting": label, "clahe": "on" if apply_clahe else "off",
                     "clip": clip if apply_clahe else "", "tile": tile if apply_clahe else "",
                     "precision": round(agg["precision"], 4), "recall": round(agg["recall"], 4),
                     "f1": round(agg["f1"], 4), "mae": round(agg["mae"], 4)})
        print(f"{label:<20} {agg['precision']:>6.3f} {agg['recall']:>6.3f} {agg['f1']:>6.3f} {agg['mae']:>7.2f}")

    # Write the table.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clahe_sweep.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "clahe", "clip", "tile",
                                                    "precision", "recall", "f1", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out / 'clahe_sweep.csv'} (thermal conf {args.thermal_conf}, NMS {args.nms_iou})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
