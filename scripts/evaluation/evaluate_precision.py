"""Count retained annotations; precision requires independently labelled ground truth."""
import argparse
import json
from pathlib import Path


def count_annotations(folder):
    return sum(len(json.loads(p.read_text(encoding="utf-8")).get("annotations", [])) for p in Path(folder).glob("*.json") if p.name != "qa_report.json")


def retention_report(before, after):
    total = count_annotations(before)
    counts = {}
    for status in ("accepted", "flagged", "rejected"):
        path = Path(after) / f"{status}.json"
        counts[status] = len(json.loads(path.read_text(encoding="utf-8"))["annotations"])
    if sum(counts.values()) != total:
        raise ValueError("QA output count does not match the input annotation count")
    return {"input": total, **counts, "retention_rate": counts["accepted"] / total if total else None,
            "precision": None, "note": "Retention is not precision; independent reference labels are required."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    a = p.parse_args()
    print(json.dumps(retention_report(a.before, a.after), indent=2))
