"""Interactive point annotator for the evaluation ground truth (item #4).

Lets you place one point per person on each evaluation image and saves the result
as ground-truth JSON (see ``ground_truth/ground_truth.py``). A headless ``--review`` mode
draws the saved points back onto each image so annotations can be checked.

Controls (annotation window):
    left-click        add a person point
    right-click / u   undo the last point
    scroll            zoom in/out around the cursor
    arrow keys        pan the (zoomed) view
    r                 reset the view to the full image
    s                 save this image's points and move to the next
    q                 skip this image without saving

Usage:
    # annotate the 8 evaluation images listed in the manifest
    python ground_truth/annotate_points.py --input ground_truth/evaluation_sample.txt

    # write overlay PNGs of the saved points for visual verification
    python ground_truth/annotate_points.py --review --input ground_truth/evaluation_sample.txt
"""

# argparse: parse command-line arguments.
import argparse
# sys / pathlib: path handling and making the repo root importable.
import sys
from pathlib import Path

# Make the repo root importable when run as `python ground_truth/annotate_points.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cv2 / numpy: draw review overlays and handle image arrays.
import cv2
import numpy as np

# Reuse the pipeline's loading, modality, and saving helpers, plus the GT store.
from src.loading import discover_images, load_image
from src.modality import infer_modality
from src.annotation import save_annotated_image
from ground_truth import build_ground_truth, load_ground_truth, save_ground_truth

# Point colours in BGR: thermal yellow, RGB green.
POINT_COLOR = {"thermal": (0, 255, 255), "rgb": (0, 255, 0)}
# Fallback colour when the modality is unknown.
DEFAULT_POINT_COLOR = (0, 255, 0)
# Fraction of the visible view to shift per arrow-key press (panning).
PAN_FRACTION = 0.2


def resolve_inputs(input_arg):
    """Return the list of image paths from a manifest file, a directory, or a file.

    A ``.txt`` manifest holds one image path per line (blank lines and lines
    starting with ``#`` are ignored). Anything else is passed to ``discover_images``.
    """
    # Normalise to a Path object.
    path = Path(input_arg)
    # A .txt file is treated as a manifest of image paths.
    if path.is_file() and path.suffix.lower() == ".txt":
        # Collect non-empty, non-comment lines as image paths.
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                images.append(Path(stripped))
        return images
    # Otherwise defer to the normal image discovery (single file or directory).
    return discover_images(path)


def draw_points(image_bgr, points, modality, count):
    """Return a copy of the image with each ground-truth point drawn as a dot."""
    # Work on a copy so the original is untouched.
    canvas = image_bgr.copy()
    # Pick the dot colour for this modality.
    color = POINT_COLOR.get(modality, DEFAULT_POINT_COLOR)
    # Scale the dot radius a little with image size so dots are visible but small.
    height, width = canvas.shape[:2]
    radius = max(3, round(min(height, width) * 0.004))
    # Draw every point as a filled dot with a thin dark outline.
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1)
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 0), 1)
    # Compose the ground-truth count banner.
    banner = f"GT {modality.upper()}  people: {count}"
    # Draw a black outline then white text for readability on any background.
    cv2.putText(canvas, banner, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, banner, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    # Return the annotated copy.
    return canvas


def review_image(image_path, gt_dir, review_dir, modality_override=None):
    """Headless: draw a saved ground-truth file's points onto its image and save it.

    Returns the overlay path, or ``None`` if no ground-truth file exists yet.
    """
    # The ground-truth file shares the image's name stem.
    stem = Path(image_path).stem
    gt_path = Path(gt_dir) / f"{stem}.json"
    # Skip images that have not been annotated yet.
    if not gt_path.is_file():
        print(f"no ground truth for {Path(image_path).name} (expected {gt_path}); skipping")
        return None
    # Load and validate the ground-truth record.
    record = load_ground_truth(gt_path)
    # Load the original image and decide the modality (prefer the stored one).
    image = load_image(image_path)
    modality = record.get("modality") or infer_modality(image_path, override=modality_override)
    # Draw the points and save the overlay as a PNG for visual review.
    overlay = draw_points(image, record["points"], modality, record["people_count"])
    out_path = save_annotated_image(overlay, Path(review_dir) / f"{stem}.png")
    print(f"review -> {out_path} ({record['people_count']} people)")
    return out_path


def annotate_image(image_path, output_dir, modality_override=None):
    """Open one image in a matplotlib window and let the user place person points."""
    # Lazy import so this module (and --review) work without matplotlib/a display.
    import matplotlib.pyplot as plt

    # Free the arrow keys and 'r' from matplotlib's default navigation shortcuts
    # so our own pan / reset handlers are the only thing bound to them.
    for keymap, drop in (("keymap.back", "left"), ("keymap.forward", "right"), ("keymap.home", "r")):
        plt.rcParams[keymap] = [key for key in plt.rcParams[keymap] if key != drop]

    # Load the image and decide its modality.
    image = load_image(image_path)
    modality = infer_modality(image_path, override=modality_override)
    # Convert BGR to RGB so matplotlib shows correct colours.
    rgb = image[..., ::-1]

    # Points placed so far, as [x, y] pixel coordinates.
    points = []

    # Set up the figure and show the image.
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    # Remember the full-image view so 'r' can restore it after zoom/pan.
    home_xlim = ax.get_xlim()
    home_ylim = ax.get_ylim()
    ax.set_title(
        f"{Path(image_path).name} [{modality}]  |  click=add  u/right=undo  "
        f"scroll=zoom  arrows=pan  r=reset  s=save+next  q=skip"
    )
    # A scatter layer for the placed points.
    scatter = ax.scatter([], [], s=30, c="red", marker="+")

    def redraw():
        # Update the scatter with current points (or clear it when empty).
        scatter.set_offsets(points if points else np.empty((0, 2)))
        # Show the running count on the x-axis label.
        ax.set_xlabel(f"{len(points)} people")
        fig.canvas.draw_idle()

    def on_click(event):
        # Ignore clicks outside the image axes.
        if event.inaxes != ax or event.xdata is None:
            return
        # Left button adds a point; right button undoes the last one.
        if event.button == 1:
            points.append([event.xdata, event.ydata])
            redraw()
        elif event.button == 3 and points:
            points.pop()
            redraw()

    def pan(dx_fraction, dy_fraction):
        # Shift the current view by a fraction of the visible span in each axis.
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        step_x = (xhi - xlo) * dx_fraction
        step_y = abs(yhi - ylo) * dy_fraction
        ax.set_xlim(xlo + step_x, xhi + step_x)
        ax.set_ylim(ylo + step_y, yhi + step_y)
        fig.canvas.draw_idle()

    def on_key(event):
        # 'u' undoes the last point.
        if event.key == "u" and points:
            points.pop()
            redraw()
        # Arrow keys pan the (zoomed) view; up moves toward the top of the image.
        elif event.key == "left":
            pan(-PAN_FRACTION, 0)
        elif event.key == "right":
            pan(PAN_FRACTION, 0)
        elif event.key == "up":
            pan(0, -PAN_FRACTION)
        elif event.key == "down":
            pan(0, PAN_FRACTION)
        # 'r' resets the view back to the full image.
        elif event.key == "r":
            ax.set_xlim(home_xlim)
            ax.set_ylim(home_ylim)
            fig.canvas.draw_idle()
        # 's' saves the ground truth for this image and closes the window.
        elif event.key == "s":
            record = build_ground_truth(Path(image_path).name, modality, points)
            out_path = save_ground_truth(record, Path(output_dir) / f"{Path(image_path).stem}.json")
            print(f"saved {out_path} ({record['people_count']} people)")
            plt.close(fig)
        # 'q' skips this image without saving.
        elif event.key == "q":
            print(f"skipped {Path(image_path).name}")
            plt.close(fig)

    def on_scroll(event):
        # Zoom in/out around the cursor by rescaling the axis limits.
        if event.inaxes != ax or event.xdata is None:
            return
        # Zoom factor: shrink the view on scroll-up, grow it on scroll-down.
        scale = 1 / 1.2 if event.button == "up" else 1.2
        x, y = event.xdata, event.ydata
        # Rescale both axes about the cursor position.
        ax.set_xlim([x + (edge - x) * scale for edge in ax.get_xlim()])
        ax.set_ylim([y + (edge - y) * scale for edge in ax.get_ylim()])
        fig.canvas.draw_idle()

    # Connect the event handlers.
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    # Draw the initial (empty) state and hand control to the GUI event loop.
    redraw()
    plt.show()


def parse_args(argv=None):
    """Define and parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Annotate person points for the evaluation ground truth.")
    # Manifest file, image, or directory to annotate.
    parser.add_argument("--input", "-i", default="ground_truth/evaluation_sample.txt",
                        help="Manifest .txt, image file, or directory (default: the evaluation manifest).")
    # Where ground-truth JSON (and review overlays) are written.
    parser.add_argument("--output", "-o", default="ground_truth", help="Ground-truth output directory.")
    # Optional modality override; otherwise inferred from the file name.
    parser.add_argument("--modality", "-m", choices=["rgb", "thermal"], default=None,
                        help="Force modality instead of inferring from file names.")
    # Review mode: draw saved points onto images instead of annotating.
    parser.add_argument("--review", action="store_true", help="Write overlay PNGs of saved points (headless).")
    return parser.parse_args(argv)


def main(argv=None):
    """Annotate (or review) every input image."""
    # Parse the command line.
    args = parse_args(argv)
    # Resolve the list of images to work on.
    images = resolve_inputs(args.input)
    # Nothing to do if the input yielded no images.
    if not images:
        print(f"No images found for input: {args.input}")
        return 1

    # Review mode: write overlay PNGs and exit.
    if args.review:
        review_dir = Path(args.output) / "review"
        for image_path in images:
            review_image(image_path, args.output, review_dir, args.modality)
        return 0

    # Annotation mode: open each image in turn.
    print("Controls: click=add, right-click/u=undo, scroll=zoom, arrows=pan, r=reset, s=save+next, q=skip")
    for image_path in images:
        annotate_image(image_path, args.output, args.modality)
    return 0


# Run main when executed as a script.
if __name__ == "__main__":
    raise SystemExit(main())
