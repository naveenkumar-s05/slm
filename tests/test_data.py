import json
from pathlib import Path

from src.model.prompting import CATEGORIES, parse_response
from src.utils import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_splits_exist_and_are_disjoint():
    train = read_jsonl(ROOT / "data/splits/train.jsonl")
    val = read_jsonl(ROOT / "data/splits/val.jsonl")
    test = read_jsonl(ROOT / "data/splits/test.jsonl")
    assert len(train) > 100
    assert len(val) > 10
    assert len(test) >= 50
    train_questions = {record["question"].strip().lower() for record in train}
    val_questions = {record["question"].strip().lower() for record in val}
    test_questions = {record["question"].strip().lower() for record in test}
    assert not (train_questions & test_questions)
    assert not (val_questions & test_questions)


def test_every_target_matches_schema():
    for split in ("train", "val", "test"):
        for record in read_jsonl(ROOT / f"data/splits/{split}.jsonl"):
            parsed = parse_response(record["target"])
            assert parsed["valid"], (record["id"], parsed["errors"])
            assert parsed["category"] == record["category"]


def test_test_split_covers_all_categories():
    test = read_jsonl(ROOT / "data/splits/test.jsonl")
    categories = {record["category"] for record in test}
    assert set(CATEGORIES) <= categories


def test_dataset_stats_written():
    stats_path = ROOT / "data/processed/dataset_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["totals"]["by_split"]["test"] >= 50
