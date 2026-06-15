# Portfolio Evidence Plan

This project should be shown as a pose-annotation quality assurance toolkit. Do not claim precision or reviewer-time improvements until a labelled dataset run is documented.

## Reproducible Demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_portfolio_contract.py"
python -m scripts.qa.run_qa --input data/raw_annotations --output data/qa_output --config configs/qa.yaml
python -m scripts.visualize.visualize_flags --annotations data/qa_output/flagged.json
python -m scripts.evaluation.evaluate_precision --before data/raw_annotations --after data/qa_output
```

## Evidence To Capture

| Artifact | Portfolio Use |
|---|---|
| `assets/flagged-keypoints.png` | Shows spatially suspicious annotations. |
| `assets/temporal-jump.png` | Shows a temporal consistency failure. |
| `outputs/reports/qa_summary.json` | Records flagged, rejected, and accepted counts. |
| `docs/QA_REPORT_TEMPLATE.md` | Captures precision, recall, and reviewer workload changes. |
| `docs/RESULTS.md` | Summarizes only verified QA runs. |

## Demo Narrative

1. Start with COCO-style annotation schema.
2. Run schema, spatial, and temporal checks.
3. Show flagged examples and explain the review decision.
4. Report reviewer-facing metrics such as false positive rate and workload reduction.

## Evidence Checklist Before Pinning

- [ ] Shareable annotation sample identified.
- [ ] Flagged example image added to `assets/`.
- [ ] Real QA summary added to `docs/RESULTS.md`.
- [ ] CI badge green on the latest commit.
- [ ] Threshold limitations documented.
