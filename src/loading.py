"""Step 1: loading a single image or a directory of images.

Two functions:

  * ``discover_images`` - expand an input path (one file or a folder) into a
    sorted list of image file paths.
  * ``load_image``      - read one image file into a BGR NumPy array.

Both validate their input and raise clear errors, so later steps can assume they
receive a real, decodable image.
"""

# Path: object-oriented filesystem paths (exists/is_file/suffix/glob, etc.).
from pathlib import Path

# cv2: OpenCV — decoding image files into arrays and image processing.
import cv2
# numpy: numerical arrays; images are represented as (H, W, 3) uint8 arrays.
import numpy as np

# Image file types we accept as input.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_images(input_path, recursive=False):
    """Return a sorted list of image files for a single file or a directory.

    Args:
        input_path: Path to one image file or a directory of images.
        recursive: If ``True`` and ``input_path`` is a directory, also search
            sub-directories.

    Returns:
        A sorted list of image ``Path`` objects. May be empty if a directory
        contains no supported images.

    Raises:
        FileNotFoundError: if ``input_path`` does not exist.
        ValueError: if ``input_path`` is a file with an unsupported extension.
    """
    # Normalize the input into a Path object so we get its helper methods.
    path = Path(input_path)

    # Fail early if the path points to nothing on disk.
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    # Case 1: the input is a single file.
    if path.is_file():
        # Reject files whose extension is not a supported image type.
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            # Raise a clear error listing the accepted extensions.
            raise ValueError(
                f"Unsupported image type '{path.suffix}' for file: {path}. "
                f"Supported: {sorted(IMAGE_EXTENSIONS)}"
            )
        # Return the single valid image wrapped in a one-element list.
        return [path]

    # Case 2: the input is a directory — list entries recursively or top-level only.
    candidates = path.rglob("*") if recursive else path.glob("*")
    # Keep only image files, sorted for deterministic ordering.
    images = sorted(
        f for f in candidates
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    # Return the collected image paths (may be empty).
    return images


def load_image(image_path):
    """Load one image file as a 3-channel BGR NumPy array.

    Uses ``np.fromfile`` + ``cv2.imdecode`` rather than ``cv2.imread`` so that
    non-ASCII / Unicode file paths (common on Windows) load reliably.

    Args:
        image_path: Path to an image file.

    Returns:
        The image as an ``(H, W, 3)`` uint8 BGR array.

    Raises:
        FileNotFoundError: if the file does not exist or cannot be decoded.
    """
    # Normalize the input into a Path object.
    path = Path(image_path)
    # Fail early if the file is missing, before trying to read any bytes.
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    # Read the raw file bytes into a NumPy buffer (handles Unicode paths).
    raw = np.fromfile(str(path), dtype=np.uint8)
    # Decode the byte buffer into a BGR image; guard against an empty file.
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    # Raise if decoding produced nothing (corrupt or unsupported content).
    if image is None:
        raise FileNotFoundError(f"Could not decode image (corrupt or unsupported): {path}")
    # Return the decoded image as an (H, W, 3) BGR array.
    return image
