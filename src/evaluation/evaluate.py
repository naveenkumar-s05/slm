"""Evaluate a model variant on the held-out test split.

Examples:
  python -m src.evaluation.evaluate --tag baseline
  python -m src.evaluation.evaluate --tag final --adapter models/cyberaware-qlora
  python -m src.evaluation.evaluate --tag final_fp16 --adapter models/cyberaware-qlora --no-quantized
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from src.evaluation.metrics import compute_metrics
from src.model.inference import SLMGenerator
from src.model.loading import load_adapter_model, load_base_model, load_tokenizer
from src.utils import (
    ROOT,
    cuda_memory_mb,
    hardware_info,
    load_yaml,
    now_iso,
    read_jsonl,
    set_seed,
    write_json,
    write_jsonl,
    ResourceMonitor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-quantized", action="store_true")
    parser.add_argument("--output-dir", default="experiments/results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    eval_cfg = load_yaml(args.eval_config)
    test_path = Path(args.test_path or eval_cfg["test_path"])
    if not test_path.is_absolute():
        test_path = ROOT / test_path

    records = read_jsonl(test_path)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no test records")

    gen_cfg = eval_cfg.get("generation", {})
    seed = int(eval_cfg.get("seed", 42))
    set_seed(seed)

    print(f"[{args.tag}] loading tokenizer and model (quantized={not args.no_quantized}) ...")
    tokenizer = load_tokenizer(model_cfg)
    if args.adapter:
        model = load_adapter_model(model_cfg, args.adapter, quantized=not args.no_quantized)
    else:
        model = load_base_model(model_cfg, quantized=not args.no_quantized)
    model_loaded_memory = cuda_memory_mb()

    generator = SLMGenerator(
        tokenizer,
        model,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 220)),
        min_new_tokens=int(gen_cfg.get("min_new_tokens", 10)),
        do_sample=bool(gen_cfg.get("do_sample", False)),
        num_beams=int(gen_cfg.get("num_beams", 1)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
        max_input_tokens=1024,
    )

    print(f"[{args.tag}] warmup ...")
    generator.generate_batch([records[0]["question"]])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = ResourceMonitor()
    monitor.start()
    predictions = []
    started = time.perf_counter()
    for index in range(0, len(records), args.batch_size):
        batch = records[index : index + args.batch_size]
        results = generator.generate_batch([record["question"] for record in batch])
        for record, result in zip(batch, results):
            predictions.append(
                {
                    "id": record["id"],
                    "source": record.get("source"),
                    "question": record["question"],
                    "gold": {
                        "category": record["category"],
                        "risk": record["risk"],
                        "advice": record["advice"],
                        "action": record["action"],
                    },
                    "raw_response": result["raw_response"],
                    "parsed": result["parsed"],
                    "latency_ms": result["latency_ms"],
                    "batch_latency_ms": result["batch_latency_ms"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
            )
        done = min(index + args.batch_size, len(records))
        print(f"[{args.tag}] {done}/{len(records)}", flush=True)
    wall_time = time.perf_counter() - started
    resources = monitor.stop()

    metrics = compute_metrics(predictions)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"{args.tag}_predictions.jsonl", predictions)
    write_json(output_dir / f"{args.tag}_metrics.json", metrics)

    run_info = {
        "tag": args.tag,
        "created_utc": now_iso(),
        "base_model": model_cfg["base_model"],
        "model_license": model_cfg.get("license"),
        "adapter": args.adapter,
        "quantized": not args.no_quantized,
        "batch_size": args.batch_size,
        "test_path": str(test_path.relative_to(ROOT)).replace("\\", "/"),
        "count": len(records),
        "generation": gen_cfg,
        "seed": seed,
        "wall_time_s": round(wall_time, 2),
        "throughput_requests_per_s": round(len(records) / wall_time, 3) if wall_time else None,
        "model_loaded_memory": model_loaded_memory,
        "inference_resources": resources,
        "hardware": hardware_info(),
        "metrics_summary": {
            key: metrics[key]
            for key in (
                "format_valid_rate",
                "category_accuracy",
                "risk_accuracy",
                "joint_category_risk_accuracy",
                "task_score",
                "advice_rouge_l",
                "unsafe_request_recall",
                "latency_ms",
            )
        },
    }
    write_json(output_dir / f"{args.tag}_run.json", run_info)

    print("\n=== METRICS ===")
    for key in (
        "count",
        "format_valid_rate",
        "category_accuracy",
        "risk_accuracy",
        "joint_category_risk_accuracy",
        "task_score",
        "advice_rouge_l",
        "action_rouge_l",
        "advice_token_f1",
        "unsafe_request_recall",
        "safety_flag_rate",
    ):
        print(f"{key}: {metrics[key]}")
    print("latency_ms:", metrics["latency_ms"])
    print("resources:", resources)
    print(f"wrote {output_dir / (args.tag + '_metrics.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
