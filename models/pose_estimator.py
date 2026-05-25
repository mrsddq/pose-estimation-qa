from __future__ import annotations

import numpy as np


MEDIAPIPE_TO_COCO = {
    0: 0,
    2: 1,
    5: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}


class MediaPipePoseEstimator:
    """MediaPipe 33-landmark pose wrapper that returns COCO 17 keypoints."""

    def __init__(self, complexity: int = 1, min_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("mediapipe is required for MediaPipePoseEstimator.") from exc
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=complexity,
            min_detection_confidence=min_confidence,
        )

    def predict(self, image_bgr) -> np.ndarray | None:
        height, width = image_bgr.shape[:2]
        image_rgb = image_bgr[:, :, ::-1]
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        keypoints = np.zeros((17, 3), dtype=np.float32)
        landmarks = results.pose_landmarks.landmark
        for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
            lm = landmarks[mp_idx]
            keypoints[coco_idx] = [lm.x * width, lm.y * height, lm.visibility]
        return keypoints
