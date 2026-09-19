import copy
import json
import pytest
from scripts.qa.run_qa import run_qa, check_schema, check_spatial
from scripts.evaluation.evaluate_precision import retention_report

CFG = {"schema": {"required_fields": ["id", "image_id", "keypoints", "num_keypoints", "bbox"], "min_keypoints": 1, "max_keypoints": 17, "min_bbox_area": 1}, "spatial": {"bbox_margin": 0.1}, "output": {"flag_threshold": 0.5, "reject_threshold": 0.8}}
ANN = {"id": 1, "image_id": 7, "keypoints": [15, 15, 2], "num_keypoints": 1, "bbox": [10, 10, 30, 30]}

@pytest.mark.parametrize("patch", [{"keypoints": [float("nan"), 0, 2]}, {"bbox": [1, 1, -20, -20]}, {"num_keypoints": 2}, {"keypoints": [2, 2, 0.4]}])
def test_schema_rejects_invalid_numbers_and_counts(patch):
    assert not check_schema({**ANN, **patch}, CFG)[0]


def test_single_keypoint_still_checks_bbox():
    assert not check_spatial({**ANN, "bbox": [90, 90, 30, 30]}, CFG, 100, 100)[0]


def test_malformed_annotation_does_not_crash_and_real_dimensions_apply(tmp_path):
    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    payload = {"images": [{"id": 7, "width": 32, "height": 32}], "categories": [{"id": 1, "name": "person"}], "annotations": [ANN, {"id": 2}]}
    (source / "a.json").write_text(json.dumps(payload))
    report = run_qa(source, output, CFG)
    assert report["counts"] == {"accepted": 0, "flagged": 1, "rejected": 1}
    assert report["annotations"][0]["reasons"] == ["bbox is outside image bounds"]
    assert json.loads((output / "flagged.json").read_text())["images"] == payload["images"]
    assert retention_report(source, output)["retention_rate"] == 0
    with pytest.raises(ValueError, match="differ"):
        run_qa(source, source, CFG)


def test_missing_image_metadata_is_rejected(tmp_path):
    source = tmp_path / "in"; source.mkdir()
    (source / "a.json").write_text(json.dumps({"annotations": [ANN]}))
    assert run_qa(source, tmp_path / "out", CFG)["counts"]["rejected"] == 1
