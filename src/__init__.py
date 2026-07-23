"""Person counting in RGB and thermal drone images.

A small, reproducible, function-based computer-vision pipeline. Each module
corresponds to one step of the baseline described in the assignment (section 3):

    1. loading      - load a single image or a directory of images
    2. modality     - identify image modality (rgb / thermal)
    3. preprocessing- modality-specific preprocessing
    4. detection    - run the selected model
    5. person_filter- keep only objects classified as people
    6. filtering    - confidence / IoU / NMS filtering
    7. counting     - count the final detections
    8. annotation   - save an annotated image
    9. outputs      - save structured JSON / CSV output

See the design rationale in ``answers_documents/`` (docs 1 and 2).
"""

__version__ = "0.1.0"
