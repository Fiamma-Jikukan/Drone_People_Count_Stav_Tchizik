"""Evaluation runner (assignment item #5): score predictions against ground truth.

Loads the ground-truth point records and the pipeline's prediction records
(already produced by ``main.py``), computes the metrics
(``evaluation.metrics.evaluate``), and writes the results:

  * ``metrics_per_image.csv`` - one row per image.
  * ``summary.json``          - per-modality + overall aggregates.
  * ``plots/*.png``           - the result charts (``evaluation.plots``).
  * ``examples/*.png``        - per-image TP / FP / FN visualisations.

This runner does NOT run the model; predictions are read from disk.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --pred-dir evaluation/predictions/json --output evaluation/results
"""

# argparse / csv / json / pathlib: CLI, table + record IO.
import argparse
import csv
import json
import sys
from pathlib import Path

# Make the repo root importable when run as `python evaluation/evaluate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cv2: draw the TP/FP/FN example overlays.
import cv2

# Reuse image loading / saving, the ground-truth loader, metrics, and plots.
from src.loading import load_image
from src.annotation import save_annotated_image
from ground_truth import load_ground_truth_dir
from evaluation.metrics import evaluate
from evaluation import plots

# Overlay colours in BGR.
TP_BOX = (0, 200, 0)        # green   - correct detection
FP_BOX = (0, 0, 255)        # red     - false positive
MATCHED_POINT = (255, 128, 0)  # blue - ground-truth point that was found
FN_POINT = (0, 255, 255)    # yellow  - ground-truth point that was missed

# CSV columns for the per-image table.
CSV_FIELDS = ["image_name", "modality", "gt_count", "pred_count", "abs_error",
              "tp", "fp", "fn", "precision", "recall", "f1"]


def load_predictions(pred_dir):
    """Load every prediction JSON record from a directory."""
    # Read and parse each .json file, sorted for stable order.
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(pred_dir).glob("*.json"))]


def write_per_image_csv(reports, out_path):
    """Write one row per image with the per-image metrics."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in reports:
            # Round the score columns; keep the rest as-is.
            writer.writerow({
                **{k: r[k] for k in ("image_name", "modality", "gt_count", "pred_count", "abs_error", "tp", "fp", "fn")},
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
            })
    return path


def write_summary_json(results, out_path):
    """Write the per-modality and overall aggregates (without the per-image detail)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"by_modality": results["by_modality"], "overall": results["overall"]}
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def draw_example(image_bgr, detections, points, match, modality, report):
    """Draw predicted boxes (TP green / FP red) and GT points (matched blue / FN yellow)."""
    # Work on a copy.
    canvas = image_bgr.copy()
    # Which detection indices were matched (TP) vs false positives.
    matched_dets = {det_index for det_index, _ in match["matches"]}
    # Draw every predicted box, coloured by whether it matched a point.
    for index, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = TP_BOX if index in matched_dets else FP_BOX
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    # Point marker radius scales gently with image size.
    height, width = canvas.shape[:2]
    radius = max(4, round(min(height, width) * 0.005))
    # Ground-truth points found by a box.
    for _, point_index in match["matches"]:
        x, y = points[point_index]
        cv2.circle(canvas, (int(x), int(y)), radius, MATCHED_POINT, -1)
    # Ground-truth points missed entirely (false negatives).
    for point_index in match["fn_points"]:
        x, y = points[point_index]
        cv2.circle(canvas, (int(x), int(y)), radius, FN_POINT, -1)
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 0), 1)
    # A banner summarising the image.
    banner = (f"{modality.upper()}  GT {report['gt_count']}  pred {report['pred_count']}  "
              f"TP {report['tp']} FP {report['fp']} FN {report['fn']}")
    cv2.putText(canvas, banner, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, banner, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_examples(results, gt_by_name, pred_by_name, images_dir, out_dir):
    """Write a TP/FP/FN overlay image for every evaluated image."""
    saved = []
    for report in results["per_image"]:
        name = report["image_name"]
        image = load_image(Path(images_dir) / name)
        detections = pred_by_name[name]["detections"]
        points = gt_by_name[name]["points"]
        overlay = draw_example(image, detections, points, report["match"], report["modality"], report)
        saved.append(save_annotated_image(overlay, Path(out_dir) / f"{Path(name).stem}.png"))
    return saved


def parse_args(argv=None):
    """Define and parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Score predictions against the point ground truth.")
    parser.add_argument("--gt-dir", default="ground_truth", help="Directory of ground-truth JSON files.")
    parser.add_argument("--pred-dir", default="evaluation/predictions/json", help="Directory of prediction JSON files.")
    parser.add_argument("--images-dir", default="input_images", help="Directory holding the source images.")
    parser.add_argument("--output", "-o", default="evaluation/results", help="Where to write metrics, plots, examples.")
    return parser.parse_args(argv)


def main(argv=None):
    """Compute metrics from saved predictions and write all result artefacts."""
    args = parse_args(argv)

    # Load ground truth and predictions.
    gt_records = load_ground_truth_dir(args.gt_dir)
    pred_records = load_predictions(args.pred_dir)
    # Only score ground-truth images that have a prediction (supports subset runs,
    # e.g. a thermal-only lower-floor experiment).
    predicted = {p["image_name"] for p in pred_records}
    gt_records = [g for g in gt_records if g["image_name"] in predicted]
    # Index both by image name for the example overlays.
    gt_by_name = {g["image_name"]: g for g in gt_records}
    pred_by_name = {p["image_name"]: p for p in pred_records}

    # Compute all metrics.
    results = evaluate(gt_records, pred_records)

    # Write the numeric outputs.
    out = Path(args.output)
    reports = sorted(results["per_image"], key=lambda r: (r["modality"], r["image_name"]))
    write_per_image_csv(reports, out / "metrics_per_image.csv")
    write_summary_json(results, out / "summary.json")
    # Parameter subtitle for the charts, from the run's recorded config (if present).
    config_path = Path(args.pred_dir).parent / "run_config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    subtitle = plots.format_run_config(run_config)
    # Draw the result charts and the per-image FP/FN overlays.
    plots.plot_all(results, out / "plots", subtitle=subtitle)
    save_examples(results, gt_by_name, pred_by_name, args.images_dir, out / "examples")

    # Print a compact summary to the console.
    for modality in ("rgb", "thermal"):
        if modality in results["by_modality"]:
            a = results["by_modality"][modality]
            print(f"{modality:7s}: MAE={a['mae']:.2f}  P={a['precision']:.3f} "
                  f"R={a['recall']:.3f} F1={a['f1']:.3f}")
    print(f"Wrote metrics, plots, and examples under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
