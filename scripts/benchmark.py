from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import yaml

from models import MediaPipePoseEstimator


def main(config: str) -> Path:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("pycocotools is required for OKS benchmarking.") from exc

    coco = COCO(cfg["data"]["annotations"])
    estimator = MediaPipePoseEstimator(
        complexity=int(cfg["model"]["complexity"]),
        min_confidence=float(cfg["model"]["min_confidence"]),
    )
    detections = []
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs([image_id])[0]
        image_path = Path(cfg["data"]["images_dir"]) / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        keypoints = estimator.predict(image)
        if keypoints is None:
            continue
        detections.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints.reshape(-1).tolist(),
                "score": float(keypoints[:, 2].mean()),
            }
        )

    output_json = Path(cfg["evaluation"]["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(detections), encoding="utf-8")
    coco_dt = coco.loadRes(str(output_json))
    evaluator = COCOeval(coco, coco_dt, "keypoints")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    output_csv = Path(cfg["evaluation"]["output_csv"])
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in zip(["AP", "AP50", "AP75", "AP_medium", "AP_large", "AR"], evaluator.stats[:6]):
            writer.writerow({"metric": metric, "value": float(value)})
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MediaPipe pose on COCO keypoints.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    print(main(args.config))
