from scripts.qa.run_qa import check_schema, check_spatial


CFG = {
    "schema": {
        "required_fields": ["id", "image_id", "keypoints", "num_keypoints", "bbox"],
        "min_keypoints": 5,
        "max_keypoints": 17,
        "min_bbox_area": 100,
    },
    "spatial": {"min_confidence": 0.3, "bbox_margin": 0.1},
}


def _annotation():
    keypoints = []
    for i in range(17):
        keypoints.extend([20 + i, 25 + i, 1.0])
    return {
        "id": 1,
        "image_id": 10,
        "keypoints": keypoints,
        "num_keypoints": 17,
        "bbox": [10, 10, 80, 100],
    }


def test_check_schema_accepts_valid_annotation():
    ok, reason = check_schema(_annotation(), CFG)

    assert ok
    assert reason == "ok"


def test_check_schema_rejects_malformed_keypoints():
    ann = _annotation()
    ann["keypoints"] = [1, 2]

    ok, reason = check_schema(ann, CFG)

    assert not ok
    assert "triples" in reason


def test_check_spatial_rejects_bbox_outside_image():
    ann = _annotation()
    ann["bbox"] = [900, 900, 200, 200]

    ok, reason = check_spatial(ann, CFG, img_h=1000, img_w=1000)

    assert not ok
    assert "outside image bounds" in reason
