"""Create a category-balanced training file.

Curated examples are oversampled and large public categories are capped so the
model does not collapse to the most frequent label (e.g. 'general'/'phishing').
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.utils import ROOT, read_jsonl, write_jsonl

SEED = 42
CURATED_REPEAT = 4
PUBLIC_CAP_PER_CATEGORY = 30


def main() -> int:
    records = read_jsonl(ROOT / "data/splits/train.jsonl")
    curated = [record for record in records if record["source"] == "curated"]
    public = [record for record in records if record["source"] != "curated"]

    rng = random.Random(SEED)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in public:
        by_category[record["category"]].append(record)
    capped_public = []
    for category, group in by_category.items():
        rng.shuffle(group)
        capped_public.extend(group[:PUBLIC_CAP_PER_CATEGORY])

    balanced = curated * CURATED_REPEAT + capped_public
    rng.shuffle(balanced)
    output = ROOT / "data/splits/train_balanced.jsonl"
    write_jsonl(output, balanced)
    counts = Counter(record["category"] for record in balanced)
    print(f"wrote {output} ({len(balanced)} records)")
    print(dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
