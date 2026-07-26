"""Entry point: count people in RGB and thermal drone images.

This is the single entry *and* output point. It parses the command line, loads
the detector once, then runs the nine pipeline steps over each image and writes
the outputs (annotated image + per-image JSON).

Pipeline steps (one module each, under ``src/``):
    1 loading -> 2 modality -> 3 preprocessing -> 4 detection ->
    5 person filter -> 6 confidence/NMS -> 7 counting -> 8 annotation -> 9 outputs

Examples:
    # Every image in a folder (modality inferred from the _V / _T file names)
    python main.py --input input_images

    # One image, forced modality, custom output dir
    python main.py --input input_images/DJI_20260621190710_0001_T.JPG -m thermal -o outputs/run1

    # Faster single-pass (no SAHI tiling) and a tuned thermal threshold
    python main.py --input input_images --no-sahi --thermal-conf 0.15
"""

# argparse: parse command-line arguments.
import argparse
# logging: progress and summary messages.
import logging
# pathlib.Path: build output paths from image names.
from pathlib import Path

# Step 1: find and load images.
from src.loading import discover_images, load_image
# Step 2: decide RGB vs thermal.
from src.modality import infer_modality
# Step 3: modality-specific preprocessing.
from src.preprocessing import preprocess
# Step 4: build the model and run detection (import its defaults too).
from src.detection import DEFAULT_BASE_CONFIDENCE, DEFAULT_DEVICE, DEFAULT_WEIGHTS, detect, load_model
# Step 5: keep only people.
from src.person_filter import filter_person
# Step 6: confidence + NMS filtering (import its default IoU too).
from src.filtering import DEFAULT_NMS_IOU, filter_detections
# Step 7: count the final detections.
from src.counting import count_people
# Step 8: draw and save the annotated image.
from src.annotation import draw_detections, save_annotated_image
# Step 9: build the result record and save it as JSON.
from src.outputs import build_result, save_json

# Default output directory for JSON and annotated images.
DEFAULT_OUTPUT_DIR = "outputs"
# Default per-modality confidence thresholds (thermal usually needs a lower one).
DEFAULT_RGB_CONFIDENCE = 0.25
DEFAULT_THERMAL_CONFIDENCE = 0.20

# Module logger.
logger = logging.getLogger("person_count")


def parse_args(argv=None):
    """Define and parse the command-line arguments."""
    # Create the argument parser.
    parser = argparse.ArgumentParser(description="Count people in RGB and thermal drone images.")
    # Input image file or directory (required).
    parser.add_argument("--input", "-i", required=True, help="Image file or directory of images.")
    # Where to write outputs.
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    # Optional modality override; otherwise inferred from the file name.
    parser.add_argument("--modality", "-m", choices=["rgb", "thermal"], default=None,
                        help="Force modality instead of inferring from file names.")
    # Model weights (auto-downloaded by Ultralytics if absent).
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO11 weights file.")
    # Inference device.
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='Device, e.g. "cpu" or "cuda:0".')
    # Disable SAHI tiled inference (single-pass; faster, weaker on small people).
    parser.add_argument("--no-sahi", action="store_true", help="Disable SAHI tiled inference.")
    # Disable CLAHE thermal preprocessing (for the CLAHE on/off ablation).
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE thermal preprocessing.")
    # Per-modality confidence thresholds and NMS IoU (the main tuning knobs).
    parser.add_argument("--rgb-conf", type=float, default=DEFAULT_RGB_CONFIDENCE, help="RGB confidence threshold.")
    parser.add_argument("--thermal-conf", type=float, default=DEFAULT_THERMAL_CONFIDENCE, help="Thermal confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, help="NMS IoU threshold.")
    # Detection floor: the lowest score the model returns at all (below the per-modality
    # thresholds). Lower it to capture weak detections for a confidence sweep.
    parser.add_argument("--base-conf", type=float, default=DEFAULT_BASE_CONFIDENCE, help="Model detection floor.")
    # Parse and return the namespace.
    return parser.parse_args(argv)


def confidence_for(modality, rgb_conf, thermal_conf):
    """Pick the confidence threshold for a modality."""
    # Thermal uses its own threshold; everything else uses the RGB one.
    return thermal_conf if modality == "thermal" else rgb_conf


def process_image(model, image_path, args):
    """Run the nine pipeline steps on one image and write its outputs.

    Returns:
        The per-image result dict (also written to JSON).
    """
    # Step 1: load the image as a BGR array.
    image = load_image(image_path)
    # Step 2: decide the modality (explicit override or file-name convention).
    modality = infer_modality(image_path, override=args.modality)
    # Step 3: apply modality-specific preprocessing (CLAHE on thermal unless --no-clahe).
    preprocessed = preprocess(image, modality, apply_clahe=not args.no_clahe)
    # Step 4: run detection (SAHI tiled unless --no-sahi).
    raw = detect(model, preprocessed, use_sahi=not args.no_sahi)
    # Step 5: keep only detections classified as people.
    people = filter_person(raw)
    # Step 6: drop low-confidence boxes (per modality) and de-duplicate with NMS.
    threshold = confidence_for(modality, args.rgb_conf, args.thermal_conf)
    people = filter_detections(people, confidence_threshold=threshold, nms_iou=args.nms_iou)
    # Step 7: the count is the number of surviving detections.
    count = count_people(people)

    # Prepare a name stem for the output files.
    stem = Path(image_path).stem
    # Step 8: draw and save the annotated image for visual review.
    annotated = draw_detections(image, people, modality=modality, count=count)
    save_annotated_image(annotated, Path(args.output) / "annotated" / f"{stem}.jpg")
    # Step 9: build the structured result and save it as JSON.
    result = build_result(Path(image_path).name, modality, people)
    save_json(result, Path(args.output) / "json" / f"{stem}.json")

    # Return the result.
    return result


def main(argv=None):
    """Parse arguments, run the pipeline over all inputs, and report a summary."""
    # Parse the command line.
    args = parse_args(argv)
    # Configure basic logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    # Step 1 (discovery): expand the input into a list of image paths.
    try:
        images = discover_images(args.input)
    except (FileNotFoundError, ValueError) as error:
        # Report a bad input path/type and exit with an error code.
        logger.error("%s", error)
        return 2
    # Nothing to do if the directory held no supported images.
    if not images:
        logger.error("No supported images found under: %s", args.input)
        return 1

    # Announce the run configuration.
    logger.info("Found %d image(s); weights=%s sahi=%s -> %s",
                len(images), args.weights, not args.no_sahi, args.output)
    # Load the detector once, before the loop (weights download on first use).
    model = load_model(weights=args.weights, device=args.device, base_confidence=args.base_conf)

    # Track results and failures across the run.
    results = []
    failures = 0
    # Process each image, keeping going if one fails.
    for index, image_path in enumerate(images, start=1):
        try:
            result = process_image(model, image_path, args)
        except Exception as error:
            # Log the failure and continue with the next image.
            logger.exception("Failed on %s: %s", image_path, error)
            failures += 1
            continue
        # Record the result and log a one-line summary for this image.
        results.append(result)
        logger.info("[%d/%d] %s (%s): %d people",
                    index, len(images), result["image_name"], result["modality"], result["people_count"])

    # Final summary across the whole run.
    total_people = sum(result["people_count"] for result in results)
    logger.info("Done. %d/%d processed, %d total people, %d failure(s).",
                len(results), len(images), total_people, failures)
    # Non-zero exit only if nothing succeeded.
    return 1 if failures and not results else 0


# Run main when executed as a script.
if __name__ == "__main__":
    raise SystemExit(main())
