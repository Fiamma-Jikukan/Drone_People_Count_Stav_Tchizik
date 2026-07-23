"""Render ground-truth points onto their images as clear, high-visibility markers.

The annotator's ``--review`` overlay draws small filled dots, which can be hard to
see. This standalone script reads ground-truth JSON files and produces cleaner
images: each person is a hollow ring with a dark contrast outline (visible on both
bright and dark backgrounds), plus an optional index number.

Usage:
    # render every ground-truth JSON in ground_truth/ into ground_truth/render/
    python ground_truth/render_annotations.py

    # render one file, with numbered markers
    python ground_truth/render_annotations.py --input ground_truth/DJI_..._V.json --number
"""

# argparse: parse command-line arguments.
import argparse
# sys / pathlib: path handling and making the repo root importable.
import sys
from pathlib import Path

# Make the repo root importable when run as `python ground_truth/render_annotations.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cv2: draw the markers.
import cv2

# Reuse the pipeline's image loading / saving and the ground-truth loader.
from src.loading import load_image
from src.annotation import save_annotated_image
from ground_truth import load_ground_truth

# Marker colours in BGR: thermal yellow, RGB green.
MARKER_COLOR = {"thermal": (0, 255, 255), "rgb": (0, 255, 0)}
# Fallback colour when the modality is unknown.
DEFAULT_MARKER_COLOR = (0, 255, 0)


def resolve_gt_files(input_arg):
    """Return the ground-truth JSON files to render from a file or a directory."""
    # Normalise to a Path object.
    path = Path(input_arg)
    # A single .json file: render just that one.
    if path.is_file() and path.suffix.lower() == ".json":
        return [path]
    # A directory: render every top-level .json, sorted for stable order.
    if path.is_dir():
        return sorted(path.glob("*.json"))
    # Anything else is an error.
    raise FileNotFoundError(f"No ground-truth JSON found at: {path}")


def find_image(image_name, images_dir):
    """Locate the source image for a ground-truth record."""
    # The record stores the original file name; look for it under images_dir.
    candidate = Path(images_dir) / image_name
    # Fail clearly if the source image is missing.
    if not candidate.is_file():
        raise FileNotFoundError(f"Source image not found: {candidate}")
    # Return the resolved image path.
    return candidate


def draw_markers(image_bgr, points, modality, count, number=False, radius=None):
    """Return a copy of the image with each point drawn as a high-visibility ring."""
    # Work on a copy so the original is untouched.
    canvas = image_bgr.copy()
    # Pick the marker colour for this modality.
    color = MARKER_COLOR.get(modality, DEFAULT_MARKER_COLOR)
    # Auto-size the ring radius from the image size unless one was given.
    height, width = canvas.shape[:2]
    ring = radius if radius else max(7, round(min(height, width) * 0.006))

    # Draw every point.
    for index, (x, y) in enumerate(points, start=1):
        # Integer pixel centre.
        cx, cy = int(x), int(y)
        # Dark outer ring for contrast on any background.
        cv2.circle(canvas, (cx, cy), ring + 1, (0, 0, 0), 3, cv2.LINE_AA)
        # Coloured ring on top.
        cv2.circle(canvas, (cx, cy), ring, color, 2, cv2.LINE_AA)
        # Small solid centre dot to mark the exact point.
        cv2.circle(canvas, (cx, cy), 2, color, -1, cv2.LINE_AA)
        # Optionally label each marker with its index (useful for verifying counts).
        if number:
            text = str(index)
            origin = (cx + ring + 2, cy - 2)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Compose the count banner.
    banner = f"GT {modality.upper()}  people: {count}"
    # Draw a black outline then white text for readability.
    cv2.putText(canvas, banner, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, banner, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    # Return the rendered copy.
    return canvas


def render_file(gt_path, images_dir, output_dir, number=False, radius=None):
    """Render one ground-truth JSON file to an annotated image and save it."""
    # Load and validate the ground-truth record.
    record = load_ground_truth(gt_path)
    # Load the matching source image.
    image = load_image(find_image(record["image_name"], images_dir))
    # Draw the markers.
    rendered = draw_markers(image, record["points"], record["modality"], record["people_count"], number, radius)
    # Save alongside using the image stem (Unicode-safe writer).
    out_path = save_annotated_image(rendered, Path(output_dir) / f"{Path(record['image_name']).stem}.png")
    # Report what we wrote.
    print(f"rendered -> {out_path} ({record['people_count']} people)")
    return out_path


def parse_args(argv=None):
    """Define and parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Render ground-truth points onto their images.")
    # A ground-truth JSON file or a directory of them.
    parser.add_argument("--input", "-i", default="ground_truth",
                        help="Ground-truth .json file or directory (default: ground_truth/).")
    # Where the original images live.
    parser.add_argument("--images-dir", default="input_images", help="Directory holding the source images.")
    # Where rendered images are written.
    parser.add_argument("--output", "-o", default="ground_truth/render", help="Output directory.")
    # Whether to number each marker.
    parser.add_argument("--number", action="store_true", help="Draw an index number next to each marker.")
    # Optional fixed marker radius.
    parser.add_argument("--radius", type=int, default=None, help="Marker radius in pixels (default: auto).")
    return parser.parse_args(argv)


def main(argv=None):
    """Render every requested ground-truth file."""
    # Parse the command line.
    args = parse_args(argv)
    # Find the ground-truth files to render.
    files = resolve_gt_files(args.input)
    # Nothing to do if there are no ground-truth files yet.
    if not files:
        print(f"No ground-truth JSON found for input: {args.input}")
        return 1
    # Render each file in turn.
    for gt_path in files:
        render_file(gt_path, args.images_dir, args.output, args.number, args.radius)
    return 0


# Run main when executed as a script.
if __name__ == "__main__":
    raise SystemExit(main())
