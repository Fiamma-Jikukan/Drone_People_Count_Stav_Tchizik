"""Plot the evaluation results (assignment item #5).

Takes the dict returned by ``evaluation.metrics.evaluate`` and draws the relevant
graphs, saving each as a PNG. No model or images needed here - it consumes the
metrics only.

Each chart carries a **subtitle listing the run parameters** (from the run's
``run_config.json``), so a graph is self-documenting about how it was produced.

Charts:
  * counts per image      - ground-truth vs predicted count (grouped bars).
  * absolute error        - per-image counting error (bars, coloured by modality).
  * detection scores      - precision / recall / F1 per modality (grouped bars).
  * detection outcomes    - TP / FP / FN per modality (grouped bars).
  * MAE                   - mean absolute counting error per modality (bars).

Colours use the Okabe-Ito colour-blind-safe palette, with one entity->colour
mapping kept consistent across charts (rgb = blue, thermal = orange).
"""

# textwrap: wrap the (possibly long) parameter subtitle across lines.
import textwrap
# matplotlib with the non-interactive Agg backend so PNGs render without a display.
import matplotlib
matplotlib.use("Agg")
# pyplot: the plotting API.
import matplotlib.pyplot as plt
# pathlib.Path: build output paths and create folders.
from pathlib import Path

# Consistent modality colours (Okabe-Ito: blue / orange).
MODALITY_COLOR = {"rgb": "#0072B2", "thermal": "#E69F00"}
# Ground-truth vs predicted colours (neutral "truth" grey, model green).
GT_COLOR = "#7F7F7F"
PRED_COLOR = "#009E73"
# Recessive grid / ink / muted colours.
GRID_COLOR = "#DDDDDD"
INK = "#222222"
MUTED = "#666666"
# Wrap width (characters) for the parameter subtitle.
SUBTITLE_WRAP = 64


def format_run_config(config):
    """Turn a run-config dict (from ``run_config.json``) into a parameter subtitle.

    Accepts the argparse namespace of a ``main.py`` run (as a dict). Returns a
    compact one-line string of *all* the parameters that shaped the run, e.g.
    ``"yolo11x.pt · cpu · SAHI on · CLAHE off · conf rgb 0.25 / thermal 0.20 · NMS 0.5 · floor 0.10"``.
    Returns ``""`` if no config is available.
    """
    # No config recorded → no subtitle.
    if not config:
        return ""
    # Build the parameter pieces in a readable order.
    parts = [str(config.get("weights", "?")), str(config.get("device", "?"))]
    parts.append("SAHI on" if not config.get("no_sahi") else "SAHI off")
    if config.get("no_clahe"):
        parts.append("CLAHE off")
    else:
        parts.append(f"CLAHE on (clip {config.get('clahe_clip')}, tile {config.get('clahe_tile')})")
    parts.append(f"conf rgb {config.get('rgb_conf')} / thermal {config.get('thermal_conf')}")
    parts.append(f"NMS {config.get('nms_iou')}")
    parts.append(f"floor {config.get('base_conf')}")
    # Note a forced modality if one was given.
    if config.get("modality"):
        parts.append(f"modality forced={config['modality']}")
    return "  ·  ".join(parts)


def _short_name(image_name):
    """Shorten a DJI file name to a compact label like '0004_V'."""
    # Drop the extension and split on underscores.
    parts = Path(image_name).stem.split("_")
    # Use the sequence number and the modality suffix when the name has that shape.
    if len(parts) >= 4:
        return f"{parts[-2]}_{parts[-1]}"
    # Otherwise fall back to the stem.
    return Path(image_name).stem


def _style(ax, ylabel):
    """Apply a clean, recessive style to an axes (title handled by `_titles`)."""
    # Y-axis label in muted ink.
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    # Drop the top and right spines; soften the rest.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    # A light horizontal grid behind the bars.
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    # Tick label colour.
    ax.tick_params(colors=INK, labelsize=9)


def _titles(fig, ax, title, subtitle):
    """Set the bold main title (figure) and the parameter subtitle (axes)."""
    # Main title at the top of the figure.
    fig.suptitle(title, color=INK, fontsize=12, fontweight="bold", y=0.99)
    # Parameter subtitle just above the plot, wrapped and muted.
    if subtitle:
        ax.set_title("\n".join(textwrap.wrap(subtitle, SUBTITLE_WRAP)),
                     color=MUTED, fontsize=7.5, pad=6)


def _label_bars(ax, bars, fmt="{:.0f}"):
    """Write each bar's value just above it (selective direct labels)."""
    # Annotate every bar with its height.
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)


def _save(fig, out_path):
    """Save and close a figure, creating the parent folder."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Leave headroom at the top for the figure suptitle.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def _sorted_images(results):
    """Per-image reports sorted by modality then name (rgb group, then thermal)."""
    return sorted(results["per_image"], key=lambda r: (r["modality"], r["image_name"]))


def plot_counts_per_image(results, out_path, subtitle=""):
    """Grouped bars: ground-truth vs predicted count for each image."""
    reports = _sorted_images(results)
    labels = [_short_name(r["image_name"]) for r in reports]
    gt = [r["gt_count"] for r in reports]
    pred = [r["pred_count"] for r in reports]

    # Bar positions for the two grouped series.
    x = range(len(reports))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7, len(reports) * 1.1), 4.7))
    bars_gt = ax.bar([i - width / 2 for i in x], gt, width, label="Ground truth", color=GT_COLOR)
    bars_pred = ax.bar([i + width / 2 for i in x], pred, width, label="Predicted", color=PRED_COLOR)

    # Axis labels, ticks, style, titles, value labels, legend.
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _style(ax, "people")
    _titles(fig, ax, "People count: ground truth vs predicted (per image)", subtitle)
    _label_bars(ax, bars_gt)
    _label_bars(ax, bars_pred)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, out_path)


def plot_abs_error_per_image(results, out_path, subtitle=""):
    """Bars: absolute counting error per image, coloured by modality."""
    reports = _sorted_images(results)
    labels = [_short_name(r["image_name"]) for r in reports]
    errors = [r["abs_error"] for r in reports]
    colors = [MODALITY_COLOR[r["modality"]] for r in reports]

    fig, ax = plt.subplots(figsize=(max(7, len(reports) * 0.9), 4.7))
    bars = ax.bar(range(len(reports)), errors, color=colors)
    ax.set_xticks(range(len(reports)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _style(ax, "|predicted - ground truth|")
    _titles(fig, ax, "Absolute counting error per image", subtitle)
    _label_bars(ax, bars)
    # Legend from the modality colours actually present.
    _modality_legend(ax, results)
    return _save(fig, out_path)


def plot_detection_scores(results, out_path, subtitle=""):
    """Grouped bars: precision / recall / F1 per modality."""
    return _grouped_by_modality(
        results, out_path,
        keys=["precision", "recall", "f1"],
        labels=["Precision", "Recall", "F1"],
        title="Detection scores by modality", ylabel="score (0-1)",
        fmt="{:.2f}", ylim=(0, 1), subtitle=subtitle,
    )


def plot_outcomes(results, out_path, subtitle=""):
    """Grouped bars: TP / FP / FN per modality."""
    return _grouped_by_modality(
        results, out_path,
        keys=["tp", "fp", "fn"],
        labels=["True positives", "False positives", "False negatives"],
        title="Detection outcomes by modality", ylabel="count",
        fmt="{:.0f}", ylim=None, subtitle=subtitle,
    )


def plot_mae(results, out_path, subtitle=""):
    """Bars: mean absolute counting error per modality."""
    modalities = [m for m in ("rgb", "thermal") if m in results["by_modality"]]
    values = [results["by_modality"][m]["mae"] for m in modalities]
    colors = [MODALITY_COLOR[m] for m in modalities]

    fig, ax = plt.subplots(figsize=(5.0, 4.7))
    bars = ax.bar(modalities, values, color=colors, width=0.55)
    _style(ax, "mean |pred - GT|")
    _titles(fig, ax, "Mean absolute counting error (MAE)", subtitle)
    _label_bars(ax, bars, fmt="{:.2f}")
    return _save(fig, out_path)


def _grouped_by_modality(results, out_path, keys, labels, title, ylabel, fmt, ylim, subtitle=""):
    """Shared helper: grouped bars over `keys`, one bar group per modality."""
    modalities = [m for m in ("rgb", "thermal") if m in results["by_modality"]]
    x = range(len(keys))
    width = 0.8 / max(1, len(modalities))

    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    # One set of bars per modality, offset side by side.
    for index, modality in enumerate(modalities):
        offset = (index - (len(modalities) - 1) / 2) * width
        values = [results["by_modality"][modality][k] for k in keys]
        bars = ax.bar([i + offset for i in x], values, width,
                      label=modality.upper(), color=MODALITY_COLOR[modality])
        _label_bars(ax, bars, fmt=fmt)

    # Axis labels, ticks, optional limit, style, titles, legend.
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax, ylabel)
    _titles(fig, ax, title, subtitle)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, out_path)


def _modality_legend(ax, results):
    """Add a modality legend using proxy handles for the colours present."""
    from matplotlib.patches import Patch
    modalities = sorted({r["modality"] for r in results["per_image"]})
    handles = [Patch(color=MODALITY_COLOR[m], label=m.upper()) for m in modalities]
    ax.legend(handles=handles, frameon=False, fontsize=9)


def plot_all(results, out_dir, subtitle=""):
    """Draw every chart into ``out_dir`` (with a shared parameter subtitle)."""
    out = Path(out_dir)
    # Produce each chart, collecting the written paths.
    return [
        plot_counts_per_image(results, out / "counts_per_image.png", subtitle),
        plot_abs_error_per_image(results, out / "abs_error_per_image.png", subtitle),
        plot_detection_scores(results, out / "detection_scores.png", subtitle),
        plot_outcomes(results, out / "detection_outcomes.png", subtitle),
        plot_mae(results, out / "mae.png", subtitle),
    ]
