"""Confidence-threshold sweep (ablation, item #6).

Takes one low-floor prediction run (e.g. ``evaluation/third_run``) and re-applies a
range of confidence thresholds to the *saved* detections, recomputing the metrics at
each — no further model runs. This is exact because SAHI's merge and our NMS are
greedy by score (see evaluation/ablation_plan.md).

Outputs a per-modality sweep table (CSV) and line charts of precision / recall / F1 /
MAE vs threshold.

Usage:
    python evaluation/confidence_sweep.py
    python evaluation/confidence_sweep.py --pred-dir evaluation/third_run/json --output evaluation/third_run/sweep
"""

# argparse / csv / json / sys / pathlib: CLI + record IO.
import argparse
import csv
import json
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# matplotlib (headless) for the sweep charts.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the ground-truth loader, metrics, and the shared plot style.
from ground_truth import load_ground_truth_dir
from evaluation.metrics import evaluate
from evaluation.plots import MODALITY_COLOR, _save, _style

# Confidence thresholds to sweep.
DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
# Metrics to chart (key, axis label, y-limit).
METRICS = [("recall", "recall", (0, 1)), ("precision", "precision", (0, 1)),
           ("f1", "F1", (0, 1)), ("mae", "MAE", None)]


def load_predictions(pred_dir):
    """Load prediction records from a directory."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(pred_dir).glob("*.json"))]


def filter_predictions(preds, threshold):
    """Return copies of the prediction records keeping only detections >= threshold."""
    filtered = []
    for pred in preds:
        dets = [d for d in pred["detections"] if d["confidence"] >= threshold]
        filtered.append({**pred, "detections": dets, "people_count": len(dets)})
    return filtered


def run_sweep(gt_records, pred_records, thresholds):
    """Return one row per (threshold, modality) with the aggregate metrics."""
    rows = []
    for threshold in thresholds:
        results = evaluate(gt_records, filter_predictions(pred_records, threshold))
        for modality, agg in results["by_modality"].items():
            rows.append({
                "threshold": threshold, "modality": modality,
                "precision": agg["precision"], "recall": agg["recall"],
                "f1": agg["f1"], "mae": agg["mae"],
            })
    return rows


def write_csv(rows, out_path):
    """Write the sweep rows to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "modality", "precision", "recall", "f1", "mae"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "precision": round(row["precision"], 4),
                             "recall": round(row["recall"], 4), "f1": round(row["f1"], 4),
                             "mae": round(row["mae"], 4)})
    return path


def plot_metric(rows, metric, ylabel, ylim, thresholds, out_path):
    """Line chart of one metric vs threshold, one line per modality."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for modality in ("rgb", "thermal"):
        series = [r for r in rows if r["modality"] == modality]
        if not series:
            continue
        xs = thresholds
        ys = [next(r[metric] for r in series if r["threshold"] == t) for t in thresholds]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=6,
                color=MODALITY_COLOR[modality], label=modality.upper())
    ax.set_xlabel("confidence threshold")
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax, f"{ylabel} vs confidence threshold", ylabel)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, out_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Confidence-threshold sweep over saved predictions.")
    parser.add_argument("--gt-dir", default="ground_truth", help="Ground-truth JSON directory.")
    parser.add_argument("--pred-dir", default="evaluation/third_run/json", help="Low-floor prediction JSON directory.")
    parser.add_argument("--output", "-o", default="evaluation/third_run/sweep", help="Output directory.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS, help="Thresholds to sweep.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gt_records = load_ground_truth_dir(args.gt_dir)
    pred_records = load_predictions(args.pred_dir)
    # Only score ground-truth images that have a prediction (supports subset runs,
    # e.g. a thermal-only lower-floor experiment).
    predicted = {p["image_name"] for p in pred_records}
    gt_records = [g for g in gt_records if g["image_name"] in predicted]

    thresholds = sorted(args.thresholds)
    rows = run_sweep(gt_records, pred_records, thresholds)

    out = Path(args.output)
    write_csv(rows, out / "sweep.csv")
    for key, label, ylim in METRICS:
        plot_metric(rows, key, label, ylim, thresholds, out / f"{key}_vs_threshold.png")

    # Print a compact table to the console.
    print(f"{'thr':>5} {'mod':<7} {'P':>6} {'R':>6} {'F1':>6} {'MAE':>7}")
    for row in rows:
        print(f"{row['threshold']:>5.2f} {row['modality']:<7} {row['precision']:>6.3f} "
              f"{row['recall']:>6.3f} {row['f1']:>6.3f} {row['mae']:>7.2f}")
    print(f"Wrote sweep table and charts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
