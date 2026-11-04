"""Build the baseline / adapted / guarded comparison table from saved results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils import ROOT, write_json

VARIANTS = [
    ("Zero-shot baseline", "experiments/results/baseline_metrics.json", "experiments/results/baseline_run.json"),
    ("QLoRA v2 (raw)", "experiments/results/final_metrics.json", "experiments/results/final_run.json"),
    ("QLoRA v2 + guardrails", "experiments/results/final_guarded_metrics.json", "experiments/results/final_run.json"),
    ("QLoRA v2 smoke (16 prompts)", "experiments/results/v2_smoke_metrics.json", "experiments/results/v2_smoke_run.json"),
]


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def peak_vram(run: dict[str, Any]) -> float:
    resources = run.get("inference_resources", {})
    return float(resources.get("inference_peak_reserved_mb", 0.0))


def build_row(label: str, metrics: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": label,
        "count": metrics["count"],
        "format_valid_rate": metrics["format_valid_rate"],
        "category_accuracy": metrics["category_accuracy"],
        "risk_accuracy": metrics["risk_accuracy"],
        "joint_category_risk_accuracy": metrics["joint_category_risk_accuracy"],
        "task_score": metrics["task_score"],
        "advice_rouge_l": metrics["advice_rouge_l"],
        "unsafe_request_recall": metrics["unsafe_request_recall"],
        "safety_flag_rate": metrics["safety_flag_rate"],
        "latency_mean_ms": metrics["latency_ms"]["mean"],
        "latency_p95_ms": metrics["latency_ms"]["p95"],
        "peak_vram_mb": peak_vram(run),
        "batch_size": run.get("batch_size"),
        "max_new_tokens": run.get("generation", {}).get("max_new_tokens"),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| System | n | Format | Category | Risk | Joint | Task | ROUGE-L | Unsafe recall | Safety flags | Mean latency | p95 latency | Peak VRAM |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['count']} | {row['format_valid_rate']:.1%} | "
            f"{row['category_accuracy']:.1%} | {row['risk_accuracy']:.1%} | "
            f"{row['joint_category_risk_accuracy']:.1%} | {row['task_score']:.1%} | "
            f"{row['advice_rouge_l']:.4f} | {row['unsafe_request_recall']:.1%} | "
            f"{row['safety_flag_rate']:.1%} | {row['latency_mean_ms']:.0f} ms | "
            f"{row['latency_p95_ms']:.0f} ms | {row['peak_vram_mb']:.0f} MB |"
        )
    return header + "\n".join(lines) + "\n"


def main() -> None:
    rows = []
    for label, metrics_path, run_path in VARIANTS:
        rows.append(build_row(label, load_json(metrics_path), load_json(run_path)))

    baseline = rows[0]
    adapted = rows[1]
    guarded = rows[2]
    payload = {
        "rows": rows,
        "deltas": {
            "format_valid_rate_pp": round((adapted["format_valid_rate"] - baseline["format_valid_rate"]) * 100, 2),
            "category_accuracy_pp": round((adapted["category_accuracy"] - baseline["category_accuracy"]) * 100, 2),
            "task_score_pp": round((adapted["task_score"] - baseline["task_score"]) * 100, 2),
            "advice_rouge_l": round(adapted["advice_rouge_l"] - baseline["advice_rouge_l"], 4),
            "guardrail_unsafe_recall_gain_pp": round(
                (guarded["unsafe_request_recall"] - adapted["unsafe_request_recall"]) * 100, 2
            ),
        },
    }
    write_json(ROOT / "experiments/results/final_comparison.json", payload)

    table = markdown_table(rows)
    attrs = payload["deltas"]
    note = (
        "\n## Key changes\n\n"
        f"- Schema validity: **{baseline['format_valid_rate']:.1%} → {adapted['format_valid_rate']:.1%}** "
        f"(+{attrs['format_valid_rate_pp']:.1f} pp).\n"
        f"- Category accuracy: **{baseline['category_accuracy']:.1%} → {adapted['category_accuracy']:.1%}** "
        f"(+{attrs['category_accuracy_pp']:.1f} pp).\n"
        f"- Task score: **{baseline['task_score']:.1%} → {adapted['task_score']:.1%}** "
        f"(+{attrs['task_score_pp']:.1f} pp).\n"
        f"- Unsafe-request recall: **{adapted['unsafe_request_recall']:.1%} → {guarded['unsafe_request_recall']:.1%}** "
        "after the deterministic input/output guardrail.\n"
        "- Quantized 4-bit inference used ~1.2 GB loaded VRAM and peaked at ~1.75 GB reserved, "
        "fitting the 4 GB RTX 2050.\n"
        f"- Latency is hardware-bound: full 84-prompt run averaged **{adapted['latency_mean_ms']:.0f} ms/request**; "
        f"the 16-prompt batched smoke run averaged **{rows[3]['latency_mean_ms']:.0f} ms/request**.\n"
        "- The smoke row is indicative only (n=16) and is not directly comparable to the full held-out test.\n"
    )
    report = "# Benchmark Evidence\n\n" + table + note
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/benchmark_evidence.md").write_text(report, encoding="utf-8")
    print(table)
    print(f"wrote {ROOT / 'reports/benchmark_evidence.md'}")


if __name__ == "__main__":
    main()
