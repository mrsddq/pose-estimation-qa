# Model Card: Pose Estimation QA

## Dataset
Target benchmark: COCO 2017 Keypoints validation annotations.

## Model
MediaPipe Pose is wrapped and remapped from 33 landmarks to the COCO 17-keypoint format.

## Evaluation
Primary metric: OKS-based COCO AP. Secondary metrics: AP50 and AR. QA output also reports schema and spatial plausibility flags.

## Limitations
MediaPipe is not trained inside this repository. COCO benchmarking requires careful handling of person detection and multiperson images.

## Ethical Considerations
Pose estimation can reveal sensitive movement and identity-adjacent information. Use consented or public data.
