"""Ground-truth evaluation sample (assignment item #4).

Bundles everything ground-truth related in one place:

  * ``ground_truth`` (module) - build / save / load the ground-truth JSON records.
  * ``annotate_points``       - interactive point annotator + headless review overlays.
  * ``evaluation_sample.txt`` - the list of images in the evaluation sample.
  * the per-image ``*.json`` annotations (produced by the annotator).

The record functions are re-exported here so callers can simply write
``from ground_truth import build_ground_truth, load_ground_truth``.
"""

# Re-export the ground-truth record helpers at the package level.
from .ground_truth import (
    build_ground_truth,
    load_ground_truth,
    load_ground_truth_dir,
    save_ground_truth,
)
