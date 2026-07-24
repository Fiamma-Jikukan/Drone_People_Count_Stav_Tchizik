"""Evaluation of the pipeline against the point ground truth (assignment item #5).

  * ``metrics`` - pure functions that score prediction records against ground-truth
    records (matching, precision/recall/F1, MAE).

The metric functions are re-exported here so callers can write
``from evaluation import evaluate, image_report``.
"""

# Re-export the metric helpers at the package level.
from .metrics import (
    aggregate,
    aggregate_by_modality,
    evaluate,
    image_report,
    match,
    point_in_box,
    precision_recall_f1,
)
