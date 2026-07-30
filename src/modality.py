"""Step 2: identifying image modality (RGB vs thermal).

The drone (DJI M4T) encodes modality in the file name: a ``_V`` suffix marks the
visual/RGB frame and ``_T`` marks the thermal frame. This module decides an
image's modality using, in priority order:

  1. an explicit override passed by the caller,
  2. the file-name naming convention (the DJI ``_V`` / ``_T`` suffixes),
  3. a configured default.

Two functions:

  * ``normalize_modality`` - validate/clean an explicit modality string.
  * ``infer_modality``     - decide the modality for a given image path.
"""

# logging: warn when a file's modality can't be inferred and we fall back to a default.
import logging
# Path: object-oriented filesystem paths (we only inspect the file name here).
from pathlib import Path

# Module logger (propagates to the root logging config set up by main.py).
logger = logging.getLogger(__name__)

# Canonical modality names, used across the whole pipeline.
RGB = "rgb"
# Thermal modality name.
THERMAL = "thermal"

# File-name suffixes (before the extension) that mark an RGB / visual frame.
RGB_SUFFIXES = ("_V",)
# File-name suffixes that mark a thermal frame.
THERMAL_SUFFIXES = ("_T",)
# Modality assumed when no suffix matches and no override is given.
DEFAULT_MODALITY = RGB


def normalize_modality(value):
    """Validate and normalise an explicit modality string.

    Args:
        value: A modality name such as ``"rgb"`` or ``"Thermal"``.

    Returns:
        The lower-cased modality, one of ``"rgb"`` / ``"thermal"``.

    Raises:
        ValueError: if ``value`` is not a recognised modality.
    """
    # Lower-case and strip surrounding whitespace so inputs like " RGB " work.
    normalized = str(value).strip().lower()
    # Reject anything that is not one of the two supported modalities.
    if normalized not in (RGB, THERMAL):
        raise ValueError(
            f"Unknown modality {value!r}; expected one of {(RGB, THERMAL)}."
        )
    # Return the clean, validated modality name.
    return normalized


def infer_modality(
    image_path,
    override=None,
    rgb_suffixes=RGB_SUFFIXES,
    thermal_suffixes=THERMAL_SUFFIXES,
    default=DEFAULT_MODALITY,
):
    """Decide whether an image is RGB or thermal.

    Args:
        image_path: Path to the image; only its file name is inspected.
        override: If given, this modality is used directly (after validation).
        rgb_suffixes: File-name suffixes that indicate an RGB frame.
        thermal_suffixes: File-name suffixes that indicate a thermal frame.
        default: Modality returned when no suffix matches.

    Returns:
        ``"rgb"`` or ``"thermal"``.

    Raises:
        ValueError: if ``override`` or ``default`` is not a valid modality.
    """
    # An explicit override always wins — validate and return it.
    if override is not None:
        return normalize_modality(override)

    # Take the file name without its extension, upper-cased for case-insensitive matching.
    stem = Path(image_path).stem.upper()

    # Thermal is checked first so a thermal suffix is never masked by an RGB one.
    for suffix in thermal_suffixes:
        # Match when the name ends with this thermal suffix (e.g. "..._T").
        if stem.endswith(suffix.upper()):
            return THERMAL

    # Otherwise look for an RGB / visual suffix (e.g. "..._V").
    for suffix in rgb_suffixes:
        # Match when the name ends with this RGB suffix.
        if stem.endswith(suffix.upper()):
            return RGB

    # No recognised suffix: warn (so a mis-named file isn't silently mis-classified)
    # and fall back to the configured default.
    logger.warning("No modality suffix (_V/_T) in %r; defaulting to %s. "
                   "Pass --modality to set it explicitly.", Path(image_path).name, default)
    return normalize_modality(default)
