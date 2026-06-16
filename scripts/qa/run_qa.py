"""Run full annotation QA pipeline.
Usage: python scripts/qa/run_qa.py --input data/raw_annotations/ --output data/qa_output/
"""
import argparse, json, yaml, numpy as np
from pathlib import Path


def check_schema(ann, cfg):
    for field in cfg["schema"]["required_fields"]:
        if field not in ann:
            return False, f"missing field: {field}"
    if not isinstance(ann["keypoints"], list) or len(ann["keypoints"]) % 3 != 0:
        return False, "keypoints must be a flat list of x/y/visibility triples"
    if len(ann["keypoints"]) // 3 > cfg["schema"].get("max_keypoints", 17):
        return False, "too many keypoints"
    if not isinstance(ann["bbox"], list) or len(ann["bbox"]) != 4:
        return False, "bbox must contain x, y, width, height"
    if ann["bbox"][2] * ann["bbox"][3] < cfg["schema"]["min_bbox_area"]:
        return False, "bbox area is too small"
    if ann["num_keypoints"] < cfg["schema"]["min_keypoints"]:
        return False, "too few keypoints"
    return True, "ok"


def check_spatial(ann, cfg, img_h, img_w):
    kps = np.array(ann["keypoints"]).reshape(-1, 3)
    visible = kps[kps[:, 2] > cfg["spatial"]["min_confidence"]]
    if len(visible) < 2:
        return True, "insufficient visible keypoints to check"
    x1, y1, w, h = ann["bbox"]
    if x1 < 0 or y1 < 0 or w <= 0 or h <= 0 or x1 + w > img_w or y1 + h > img_h:
        return False, "bbox is outside image bounds"
    margin = cfg["spatial"]["bbox_margin"]
    for kp in visible:
        if not (x1 - w*margin <= kp[0] <= x1 + w*(1+margin) and
                y1 - h*margin <= kp[1] <= y1 + h*(1+margin)):
            return False, f"keypoint ({kp[0]:.0f},{kp[1]:.0f}) outside bbox"
    return True, "ok"


def run_qa(inp_dir, out_dir, cfg):
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    inp_dir, out_dir = Path(inp_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted, flagged, rejected = [], [], []

    for ann_file in sorted(inp_dir.glob("*.json")):
        with open(ann_file) as f:
            data = json.load(f)
        for ann in tqdm(data.get("annotations", []), desc=ann_file.name):
            fails = 0
            schema_ok, _ = check_schema(ann, cfg)
            if not schema_ok:
                fails += 1
            spatial_ok, _ = check_spatial(ann, cfg, 1080, 1920)
            if not spatial_ok:
                fails += 1
            total_checks = 2
            fail_rate = fails / total_checks
            if fail_rate >= cfg["output"]["reject_threshold"]:
                rejected.append(ann)
            elif fail_rate >= cfg["output"]["flag_threshold"]:
                flagged.append(ann)
            else:
                accepted.append(ann)

    print(f"Accepted: {len(accepted)}  Flagged: {len(flagged)}  Rejected: {len(rejected)}")
    json.dump({"annotations": accepted}, open(out_dir / "accepted.json", "w"), indent=2)
    json.dump({"annotations": flagged},  open(out_dir / "flagged.json",  "w"), indent=2)
    json.dump({"annotations": rejected}, open(out_dir / "rejected.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw_annotations/")
    p.add_argument("--output", default="data/qa_output/")
    p.add_argument("--config", default="configs/qa.yaml")
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    run_qa(a.input, a.output, cfg)
