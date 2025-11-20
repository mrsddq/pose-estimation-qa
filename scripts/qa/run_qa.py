"""Run full annotation QA pipeline.
Usage: python scripts/qa/run_qa.py --input data/raw_annotations/ --output data/qa_output/
"""
import argparse, json, yaml, numpy as np
from pathlib import Path
from tqdm import tqdm


def check_schema(ann, cfg):
    for field in cfg["schema"]["required_fields"]:
        if field not in ann:
            return False, f"missing field: {field}"
    if ann["num_keypoints"] < cfg["schema"]["min_keypoints"]:
        return False, "too few keypoints"
    return True, "ok"


def check_spatial(ann, cfg, img_h, img_w):
    kps = np.array(ann["keypoints"]).reshape(-1, 3)
    visible = kps[kps[:, 2] > cfg["spatial"]["min_confidence"]]
    if len(visible) < 2:
        return True, "insufficient visible keypoints to check"
    x1, y1, w, h = ann["bbox"]
    margin = cfg["spatial"]["bbox_margin"]
    for kp in visible:
        if not (x1 - w*margin <= kp[0] <= x1 + w*(1+margin) and
                y1 - h*margin <= kp[1] <= y1 + h*(1+margin)):
            return False, f"keypoint ({kp[0]:.0f},{kp[1]:.0f}) outside bbox"
    return True, "ok"


def run_qa(inp_dir, out_dir, cfg):
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
