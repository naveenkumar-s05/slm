"""Validate dataset integrity, schema conformance and split isolation."""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from src.model.prompting import CATEGORIES, parse_response

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"
REQUIRED_FIELDS = ("id", "source", "question", "category", "risk", "advice", "action", "target", "messages", "split")


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return {token for token in normalise(text).split() if len(token) > 2}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_split(name: str) -> list[dict[str, Any]]:
    path = SPLIT_DIR / f"{name}.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    splits = {name: load_split(name) for name in ("train", "val", "test")}
    for name, records in splits.items():
        if not records:
            errors.append(f"{name} split is empty")

    ids: set[str] = set()
    for name, records in splits.items():
        for record in records:
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                errors.append(f"{name}:{record.get('id', '?')} missing fields {missing}")
                continue
            if record["id"] in ids:
                errors.append(f"duplicate id {record['id']}")
            ids.add(record["id"])
            if record["split"] != name:
                errors.append(f"{record['id']} split field {record['split']} != {name}")
            parsed = parse_response(record["target"])
            if not parsed["valid"]:
                errors.append(f"{record['id']} target invalid: {parsed['errors']}")
            if parsed["category"] != record["category"]:
                errors.append(f"{record['id']} category mismatch")
            if parsed["risk"] != record["risk"]:
                errors.append(f"{record['id']} risk mismatch")
            if record["messages"][-1]["content"] != record["target"]:
                errors.append(f"{record['id']} assistant message does not match target")

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    for left_name, left, right_name, right in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        left_keys = {normalise(record["question"]) for record in left}
        right_keys = {normalise(record["question"]) for record in right}
        overlap = left_keys & right_keys
        if overlap:
            errors.append(f"exact question overlap between {left_name} and {right_name}: {len(overlap)}")
        left_tokens = [(record["id"], token_set(record["question"])) for record in left]
        for record in right:
            tokens = token_set(record["question"])
            for other_id, other_tokens in left_tokens:
                if jaccard(tokens, other_tokens) >= 0.85:
                    errors.append(f"near-duplicate across {left_name}/{right_name}: {other_id} ~ {record['id']}")
                    break

    for name, records in splits.items():
        categories = {record["category"] for record in records}
        missing = set(CATEGORIES) - categories
        if missing:
            messages = f"{name} missing categories: {sorted(missing)}"
            if name == "test":
                errors.append(messages)
            else:
                warnings.append(messages)

    test_categories = {record["category"] for record in test}
    if "unsafe-request" not in test_categories:
        errors.append("test split has no unsafe-request cases")
    if len(test) < 50:
        warnings.append(f"test split is small: {len(test)} records")

    report = {
        "ok": not errors,
        "counts": {name: len(records) for name, records in splits.items()},
        "categories": {
            name: {category: sum(1 for record in records if record["category"] == category) for category in CATEGORIES}
            for name, records in splits.items()
        },
        "errors": errors,
        "warnings": warnings,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({k: report[k] for k in ("ok", "counts", "errors", "warnings")}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
