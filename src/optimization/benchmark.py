"""Benchmark inference configurations: quantization, batching and token limits.

Run: python -m src.optimization.benchmark --limit 12
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

from src.evaluation.metrics import compute_metrics
from src.model.inference import SLMGenerator
from src.model.loading import load_adapter_model, load_tokenizer
from src.utils import (
    ROOT,
    ResourceMonitor,
    cuda_memory_mb,
    hardware_info,
    load_yaml,
    now_iso,
    read_jsonl,
    set_seed,
    write_json,
)


def run_config(name, model_cfg, adapter, records, batch_size, max_new_tokens, quantized, seed):
    set_seed(seed)
    print(f"\n=== {name} (quantized={quantized}, batch={batch_size}, max_new={max_new_tokens}) ===")
    tokenizer = load_tokenizer(model_cfg)
    model = load_adapter_model(model_cfg, adapter, quantized=quantized)
    device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = any(str(device).lower() in {"cpu", "disk"} for device in device_map.values())
    loaded_memory = cuda_memory_mb()

    if offloaded:
        result = {
            "name": name,
            "quantized": quantized,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "fits_in_4gb_vram": False,
            "device_map": {str(key): str(value) for key, value in device_map.items()},
            "model_loaded_memory": loaded_memory,
            "error": "model required CPU/disk offload on the 4 GB GPU",
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    generator = SLMGenerator(tokenizer, model, max_new_tokens=max_new_tokens, repetition_penalty=1.05)
    generator.generate_batch([records[0]["question"]])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()
    predictions = []
    for index in range(0, len(records), batch_size):
        batch = records[index : index + batch_size]
        results = generator.generate_batch([record["question"] for record in batch])
        for record, item in zip(batch, results):
            predictions.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "gold": {
                        "category": record["category"],
                        "risk": record["risk"],
                        "advice": record["advice"],
                        "action": record["action"],
                    },
                    "raw_response": item["raw_response"],
                    "parsed": item["parsed"],
                    "latency_ms": item["latency_ms"],
                    "output_tokens": item["output_tokens"],
                }
            )
    wall_time = time.perf_counter() - started
    resources = monitor.stop()
    metrics = compute_metrics(predictions)

    result = {
        "name": name,
        "quantized": quantized,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "num_prompts": len(records),
        "fits_in_4gb_vram": True,
        "model_loaded_memory": loaded_memory,
        "inference_resources": resources,
        "wall_time_s": round(wall_time, 2),
        "throughput_requests_per_s": round(len(records) / wall_time, 3),
        "latency_ms": metrics["latency_ms"],
        "format_valid_rate": metrics["format_valid_rate"],
        "category_accuracy": metrics["category_accuracy"],
        "risk_accuracy": metrics["risk_accuracy"],
        "task_score": metrics["task_score"],
        "advice_rouge_l": metrics["advice_rouge_l"],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default="models/cyberaware-qlora-v2")
    parser.add_argument("--test-path", default="data/splits/test.jsonl")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", default="experiments/results/optimization_benchmark.json")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    records = read_jsonl(ROOT / args.test_path)[: args.limit]
    seed = 42

    configs = [
        {
            "name": "pre_optimization_4bit_batch1",
            "batch_size": 1,
            "max_new_tokens": 220,
            "quantized": True,
        },
        {
            "name": "optimized_4bit_batch8",
            "batch_size": 8,
            "max_new_tokens": 140,
            "quantized": True,
        },
        {
            "name": "optimized_4bit_batch8_short",
            "batch_size": 8,
            "max_new_tokens": 100,
            "quantized": True,
        },
    ]
    results = []
    for config in configs:
        try:
            results.append(
                run_config(
                    config["name"],
                    model_cfg,
                    args.adapter,
                    records,
                    config["batch_size"],
                    config["max_new_tokens"],
                    config["quantized"],
                    seed,
                )
            )
        except Exception as exc:  # noqa: BLE001 - benchmark must record failures
            results.append({"name": config["name"], "error": f"{type(exc).__name__}: {exc}"})
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "created_utc": now_iso(),
        "adapter": args.adapter,
        "num_prompts": len(records),
        "hardware": hardware_info(),
        "results": results,
    }
    write_json(ROOT / args.output, payload)
    print("\n=== OPTIMIZATION SUMMARY ===")
    for result in results:
        if result.get("error"):
            print(f"{result['name']}: ERROR {result['error']}")
            continue
        print(
            f"{result['name']}: latency_mean={result['latency_ms']['mean']} ms, "
            f"throughput={result['throughput_requests_per_s']} req/s, "
            f"peak_vram={result.get('inference_resources', {}).get('inference_peak_allocated_mb')} MB, "
            f"task_score={result['task_score']}"
        )
    print(f"wrote {ROOT / args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
