"""Validate COCO annotations against schema and their actual image dimensions."""
import argparse
import json
import math
from pathlib import Path
import yaml


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_schema(ann, cfg):
    if not isinstance(ann, dict):
        return False, "annotation must be an object"
    for field in cfg["schema"]["required_fields"]:
        if field not in ann:
            return False, f"missing field: {field}"
    if any(not isinstance(ann.get(key), int) or isinstance(ann.get(key), bool) for key in ("id", "image_id")):
        return False, "id and image_id must be integers"
    points = ann.get("keypoints")
    if not isinstance(points, list) or len(points) % 3:
        return False, "keypoints must be a flat list of x/y/visibility triples"
    if not all(_finite(value) for value in points):
        return False, "keypoints must be finite numbers"
    if any(v not in (0, 1, 2) for v in points[2::3]):
        return False, "COCO visibility must be 0, 1 or 2"
    count = ann.get("num_keypoints")
    if not isinstance(count, int) or isinstance(count, bool) or count != sum(v > 0 for v in points[2::3]):
        return False, "num_keypoints must equal the number of labelled points"
    if len(points) // 3 > cfg["schema"].get("max_keypoints", 17):
        return False, "too many keypoints"
    if count < cfg["schema"]["min_keypoints"]:
        return False, "too few keypoints"
    box = ann.get("bbox")
    if not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box):
        return False, "bbox must contain four finite numbers"
    if box[2] <= 0 or box[3] <= 0 or box[2] * box[3] < cfg["schema"]["min_bbox_area"]:
        return False, "bbox area is too small or dimensions are nonpositive"
    return True, "ok"


def check_spatial(ann, cfg, img_h, img_w):
    if not _finite(img_h) or not _finite(img_w) or img_h <= 0 or img_w <= 0:
        return False, "image dimensions must be positive and finite"
    x, y, w, h = ann["bbox"]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img_w or y + h > img_h:
        return False, "bbox is outside image bounds"
    margin = cfg["spatial"]["bbox_margin"]
    for px, py, visibility in zip(ann["keypoints"][::3], ann["keypoints"][1::3], ann["keypoints"][2::3]):
        if visibility == 0:
            continue
        if not (0 <= px < img_w and 0 <= py < img_h):
            return False, "labelled keypoint is outside image bounds"
        if not (x - w * margin <= px <= x + w * (1 + margin) and y - h * margin <= py <= y + h * (1 + margin)):
            return False, "labelled keypoint is outside bbox"
    return True, "ok"


def run_qa(inp_dir, out_dir, cfg):
    source, destination = Path(inp_dir).resolve(), Path(out_dir).resolve()
    if source == destination:
        raise ValueError("Input and output directories must differ")
    paths = sorted(source.glob("*.json"))
    if not paths:
        raise ValueError("No annotation JSON files found")
    flag, reject = cfg["output"]["flag_threshold"], cfg["output"]["reject_threshold"]
    if not 0 < flag <= reject <= 1:
        raise ValueError("Thresholds must satisfy 0 < flag <= reject <= 1")
    buckets = {"accepted": [], "flagged": [], "rejected": []}
    images, categories, report, ids = {}, {}, [], set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        image_map = {item["id"]: item for item in payload.get("images", [])}
        for key, image in image_map.items():
            if key in images and images[key] != image:
                raise ValueError(f"Conflicting image metadata for ID {key}")
            images[key] = image
        for category in payload.get("categories", []):
            key = category["id"]
            if key in categories and categories[key] != category:
                raise ValueError(f"Conflicting category metadata for ID {key}")
            categories[key] = category
        for position, ann in enumerate(payload.get("annotations", [])):
            valid, reason = check_schema(ann, cfg)
            identifier = ann.get("id") if isinstance(ann, dict) else None
            reasons = []
            if not valid:
                status, reasons = "rejected", [reason]
            elif identifier in ids:
                raise ValueError(f"Duplicate annotation ID: {identifier}")
            else:
                image = image_map.get(ann["image_id"])
                if image is None:
                    status, reasons = "rejected", ["image_id has no image metadata"]
                else:
                    spatial_ok, spatial_reason = check_spatial(ann, cfg, image.get("height"), image.get("width"))
                    fail_rate = 0 if spatial_ok else 0.5
                    status = "rejected" if fail_rate >= reject else "flagged" if fail_rate >= flag else "accepted"
                    if not spatial_ok:
                        reasons.append(spatial_reason)
                ids.add(identifier)
            buckets[status].append(ann)
            report.append({"source": path.name, "position": position, "annotation_id": identifier, "status": status, "reasons": reasons})
    destination.mkdir(parents=True, exist_ok=True)
    for status, annotations in buckets.items():
        (destination / f"{status}.json").write_text(json.dumps({"images": list(images.values()), "categories": list(categories.values()), "annotations": annotations}, indent=2), encoding="utf-8")
    summary = {"counts": {key: len(value) for key, value in buckets.items()}, "checks": ["schema", "spatial"], "annotations": report}
    (destination / "qa_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/qa.yaml")
    a = p.parse_args()
    print(json.dumps(run_qa(a.input, a.output, yaml.safe_load(Path(a.config).read_text()))["counts"]))
