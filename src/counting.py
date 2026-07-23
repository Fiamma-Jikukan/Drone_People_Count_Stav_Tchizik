"""Step 7: counting the final detections.

After person filtering (step 5) and confidence/NMS filtering (step 6), the number
of people in an image is simply the number of remaining detections. Giving this
its own function keeps the pipeline readable and provides a single place to change
the counting rule later (e.g. if counting ever needed weighting or grouping).

One function: ``count_people``.
"""


def count_people(detections):
    """Return the final people count for an image.

    Args:
        detections: The final list of person detection dicts (already filtered).

    Returns:
        The number of people as an int.
    """
    # The count is simply how many detections survived filtering.
    return len(detections)
