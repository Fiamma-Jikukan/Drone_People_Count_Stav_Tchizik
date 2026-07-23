"""Ground-truth records for the evaluation sample (assignment item #4).

Ground truth is stored as one JSON file per image, mirroring the pipeline output
schema (``src/outputs.py``) but holding manually placed **points** (one per
person) instead of predicted boxes, so predictions and ground truth are directly
comparable in item #5.

Ground-truth JSON schema::

    {
      "image_name": "DJI_20260621190910_0006_V.JPG",
      "modality": "rgb",
      "people_count": 102,
      "points": [[x, y], [x, y], ...]
    }

Points are integer pixel coordinates in the original image; ``people_count`` must
equal ``len(points)`` (checked on load).

Functions:

  * ``build_ground_truth``     - assemble a ground-truth record from points.
  * ``save_ground_truth``      - write a record as JSON.
  * ``load_ground_truth``      - read and validate one record.
  * ``load_ground_truth_dir``  - load every record in a directory.
"""

# json: serialise / parse the records.
import json
# pathlib.Path: build paths and iterate a directory.
from pathlib import Path

# Required keys every ground-truth record must contain.
REQUIRED_KEYS = ("image_name", "modality", "people_count", "points")


def build_ground_truth(image_name, modality, points):
    """Assemble a ground-truth record from a list of points.

    Args:
        image_name: The image's file name (not full path).
        modality: ``"rgb"`` or ``"thermal"``.
        points: Iterable of ``(x, y)`` pixel coordinates, one per person.

    Returns:
        A dict matching the ground-truth JSON schema.
    """
    # Normalise every point to a two-element list of ints.
    clean_points = [[int(round(x)), int(round(y))] for x, y in points]
    # Assemble the record; the count is derived from the points.
    return {
        "image_name": image_name,
        "modality": modality,
        "people_count": len(clean_points),
        "points": clean_points,
    }


def save_ground_truth(record, out_path):
    """Write a ground-truth record to a JSON file.

    Args:
        record: A record from ``build_ground_truth``.
        out_path: Destination ``.json`` path.

    Returns:
        The ``Path`` written to.
    """
    # Normalise to a Path object.
    path = Path(out_path)
    # Ensure the destination directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the record as pretty-printed JSON.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    # Return the path we wrote.
    return path


def load_ground_truth(path):
    """Read and validate one ground-truth record.

    Args:
        path: Path to a ground-truth ``.json`` file.

    Returns:
        The record dict.

    Raises:
        ValueError: if a required key is missing or ``people_count`` does not
            match the number of points.
    """
    # Normalise to a Path object.
    path = Path(path)
    # Read and parse the JSON file.
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)

    # Every required key must be present.
    for key in REQUIRED_KEYS:
        if key not in record:
            raise ValueError(f"Ground-truth file {path} is missing key '{key}'.")

    # The stored count must agree with the number of points (guards typos/edits).
    if record["people_count"] != len(record["points"]):
        raise ValueError(
            f"Ground-truth file {path}: people_count={record['people_count']} "
            f"but found {len(record['points'])} points."
        )

    # Return the validated record.
    return record


def load_ground_truth_dir(directory):
    """Load every ground-truth record directly under a directory.

    Args:
        directory: Folder containing ground-truth ``.json`` files.

    Returns:
        A list of records, sorted by file name.

    Raises:
        FileNotFoundError: if the directory does not exist.
    """
    # Normalise to a Path object.
    path = Path(directory)
    # Fail early if the directory is missing.
    if not path.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {path}")
    # Load each JSON file (top-level only), sorted for deterministic order.
    return [load_ground_truth(json_file) for json_file in sorted(path.glob("*.json"))]
