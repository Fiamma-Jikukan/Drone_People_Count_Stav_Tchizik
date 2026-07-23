"""Step 8: saving an annotated image for visual review.

Draws each detection box (with its confidence) and a people-count banner onto a
copy of the image, then writes it to disk. Two functions:

  * ``draw_detections``       - return an annotated copy of an image.
  * ``save_annotated_image``  - encode and write an image (Unicode-path safe).
"""

# pathlib.Path: build output paths and create parent folders.
from pathlib import Path

# cv2: OpenCV — drawing primitives and image encoding.
import cv2

# THERMAL: canonical modality name, used to pick the box colour.
from .modality import THERMAL

# Box colours in BGR: thermal boxes yellow, everything else (RGB) green.
BOX_COLOR = {THERMAL: (0, 255, 255), "rgb": (0, 255, 0)}
# Fallback box colour when the modality is unknown.
DEFAULT_BOX_COLOR = (0, 255, 0)


def draw_detections(
    image_bgr,
    detections,
    modality="rgb",
    count=None,
    box_thickness=2,
    show_confidence=True,
):
    """Return a copy of the image with detection boxes and a count banner.

    Args:
        image_bgr: The original (unpreprocessed) image to annotate.
        detections: Final list of person detection dicts (with ``bbox`` /
            ``confidence``).
        modality: ``"rgb"`` or ``"thermal"`` (selects the box colour and banner).
        count: People count to show; defaults to ``len(detections)``.
        box_thickness: Rectangle line thickness in pixels.
        show_confidence: Whether to print each box's confidence score.

    Returns:
        A new annotated BGR image (the input is not modified).
    """
    # Work on a copy so the caller's image is left untouched.
    canvas = image_bgr.copy()
    # Promote grayscale to 3-channel so coloured boxes are visible.
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    # Pick the box colour for this modality.
    color = BOX_COLOR.get(modality, DEFAULT_BOX_COLOR)

    # Draw every detection box.
    for det in detections:
        # Unpack the integer box corners.
        x1, y1, x2, y2 = det["bbox"]
        # Draw the bounding rectangle.
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
        # Optionally label the box with its confidence score.
        if show_confidence:
            # Format the confidence to two decimals.
            label = f"{det['confidence']:.2f}"
            # Place the label just above the box's top-left corner.
            cv2.putText(
                canvas, label, (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    # Use the given count, or fall back to the number of detections drawn.
    total = count if count is not None else len(detections)
    # Compose the banner text.
    banner = f"{modality.upper()}  people: {total}"
    # Draw a thick black outline first for readability on any background.
    cv2.putText(canvas, banner, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    # Draw the white banner text on top of the outline.
    cv2.putText(canvas, banner, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    # Return the annotated copy.
    return canvas


def save_annotated_image(image_bgr, out_path):
    """Encode and write an image to disk, creating parent folders as needed.

    Uses ``cv2.imencode`` + ``tofile`` so non-ASCII / Unicode output paths
    (common on Windows) are written reliably.

    Args:
        image_bgr: The image array to save.
        out_path: Destination path; extension selects the format (default .jpg).

    Returns:
        The ``Path`` written to.

    Raises:
        IOError: if the image could not be encoded.
    """
    # Normalise to a Path object.
    path = Path(out_path)
    # Ensure the destination directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Choose the encoding format from the extension, defaulting to JPEG.
    ext = path.suffix if path.suffix else ".jpg"
    # Encode the image into an in-memory buffer of the chosen format.
    ok, buffer = cv2.imencode(ext, image_bgr)
    # Fail clearly if encoding did not succeed.
    if not ok:
        raise IOError(f"Failed to encode image for: {path}")
    # Write the raw buffer to disk (Unicode-path safe).
    buffer.tofile(str(path))
    # Return the path we wrote.
    return path
