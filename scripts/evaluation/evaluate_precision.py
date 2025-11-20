"""Compare annotation quality before and after QA.
Usage: python scripts/evaluation/evaluate_precision.py --before data/raw_annotations/ --after data/qa_output/
"""
import argparse, json
from pathlib import Path


def count_annotations(folder):
    total = 0
    for f in Path(folder).glob("*.json"):
        data = json.load(open(f))
        total += len(data.get("annotations", []))
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    a = p.parse_args()
    before = count_annotations(a.before)
    after_acc = count_annotations(str(Path(a.after)))
    print(f"Before QA: {before} annotations")
    print(f"After QA (accepted): {after_acc} annotations")
    if before > 0:
        removed = before - after_acc
        print(f"Removed (flagged/rejected): {removed} ({removed/before*100:.1f}%)")
        print(f"Precision improvement estimate: compute OKS against GT if available")
