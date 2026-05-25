# Architecture Rationale

Quality assurance here means detecting suspicious pose annotations before training or evaluation. The checks reduce label noise; they do not replace human review.

Check families:

- schema checks: required COCO fields and keypoint counts
- spatial checks: keypoints inside plausible body/bounding-box regions
- temporal checks: motion consistency across ordered frames
- visualization: reviewer-facing flagged cases

Upgrade path:

- add a MediaPipe or YOLOv8-pose wrapper for model-assisted review
- add reviewer feedback loop
- add per-keypoint confusion/error summaries
