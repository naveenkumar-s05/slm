"""Apply deployment guardrails to saved predictions and recompute metrics.

This separates model-only results from the guarded production system.
Run: python -m src.evaluation.apply_guardrails --tag final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.metrics import compute_metrics
from src.model.prompting import apply_guardrails
from src.utils import ROOT, read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="final")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    predictions = read_jsonl(output_dir / f"{args.tag}_predictions.jsonl")
    guarded = []
    triggered = 0
    for item in predictions:
        parsed = apply_guardrails(item["question"], item["parsed"], item["raw_response"])
        triggered += int(bool(parsed.get("guardrail_triggered")))
        guarded.append(
            {
                **item,
                "parsed": parsed,
                "raw_response": parsed.get("raw", item["raw_response"]) if parsed.get("guardrail_triggered") else item["raw_response"],
                "guardrail_triggered": bool(parsed.get("guardrail_triggered")),
            }
        )

    metrics = compute_metrics(guarded)
    write_jsonl(output_dir / f"{args.tag}_guarded_predictions.jsonl", guarded)
    write_json(output_dir / f"{args.tag}_guarded_metrics.json", metrics)
    print(f"guardrail triggered: {triggered}/{len(guarded)}")
    for key in (
        "format_valid_rate",
        "category_accuracy",
        "risk_accuracy",
        "task_score",
        "unsafe_request_recall",
        "safety_flag_rate",
    ):
        print(f"{key}: {metrics[key]}")
    print(f"wrote {output_dir / (args.tag + '_guarded_metrics.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
