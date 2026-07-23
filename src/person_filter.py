"""Step 5: filtering detections to objects classified as people.

The detector (step 4) returns detections for every COCO class it finds
(person, boat, bird, ...). This step keeps only the ones classified as people.

A detection is treated as a person if it matches the target either by class id
(COCO "person" is id 0) or by class name ("person"). Matching on both makes the
filter robust if a different model numbers its classes differently.

One function: ``filter_person``.
"""

# COCO class id for "person".
PERSON_CLASS_ID = 0
# COCO class name for "person".
PERSON_CLASS_NAME = "person"


def filter_person(
    detections,
    person_class_id=PERSON_CLASS_ID,
    person_class_name=PERSON_CLASS_NAME,
):
    """Return only the detections classified as people.

    Args:
        detections: A list of detection dicts from step 4, each with
            ``class_id`` and ``class_name`` keys.
        person_class_id: Class id that counts as a person.
        person_class_name: Class name that counts as a person.

    Returns:
        A new list containing only the person detections (order preserved).
    """
    # Collect the detections that are classified as people.
    people = []
    # Check each detection in turn.
    for det in detections:
        # A detection is a person if its class id or class name matches the target.
        if det["class_id"] == person_class_id or det["class_name"] == person_class_name:
            # Keep this detection.
            people.append(det)
    # Return only the person detections.
    return people
