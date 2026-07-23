"""Step 9: saving structured output in JSON / CSV format.

Builds the per-image result record in the schema the assignment specifies and
writes it out:

  * ``build_result``  - assemble the per-image result dict.
  * ``save_json``     - write one image's result as JSON.

The per-image JSON schema::

    {
      "image_name": "DJI_example.jpg",
      "modality": "rgb",
      "people_count": 18,
      "detections": [
        {"bbox": [120, 85, 170, 210], "confidence": 0.91, "class": "person"}
      ]
    }
"""

# json: serialise the per-image result.
import json
# pathlib.Path: build output paths and create parent folders.
from pathlib import Path

# Number of decimals to round confidence to in the output.
CONFIDENCE_DECIMALS = 4


def _detection_to_schema(detection):
    """Convert an internal detection dict to the output schema.

    Internal detections carry ``class_id`` / ``class_name``; the output schema
    keeps only ``bbox``, ``confidence`` and ``class``.
    """
    # Keep only the fields the schema defines, renaming class_name -> class.
    return {
        "bbox": detection["bbox"],
        "confidence": round(float(detection["confidence"]), CONFIDENCE_DECIMALS),
        "class": detection["class_name"],
    }


def build_result(image_name, modality, detections):
    """Assemble the per-image result record.

    Args:
        image_name: The image's file name (not full path).
        modality: ``"rgb"`` or ``"thermal"``.
        detections: The final filtered list of person detection dicts.

    Returns:
        A dict matching the per-image JSON schema.
    """
    # Assemble the record; people_count is the number of final detections.
    return {
        "image_name": image_name,
        "modality": modality,
        "people_count": len(detections),
        "detections": [_detection_to_schema(det) for det in detections],
    }


def save_json(result, out_path):
    """Write one image's result dict to a JSON file.

    Args:
        result: A result dict from ``build_result``.
        out_path: Destination ``.json`` path.

    Returns:
        The ``Path`` written to.
    """
    # Normalise to a Path object.
    path = Path(out_path)
    # Ensure the destination directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the result as pretty-printed JSON.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    # Return the path we wrote.
    return path
