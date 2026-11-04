"""Quick smoke test: load the base model and answer one question."""

from __future__ import annotations

import argparse

from src.model.inference import SLMGenerator
from src.model.loading import load_base_model, load_tokenizer
from src.utils import cuda_memory_mb, hardware_info, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--question", default="How can I tell if an email is a phishing attempt?")
    parser.add_argument("--no-quantized", action="store_true")
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    print("hardware:", hardware_info())
    print("loading tokenizer ...")
    tokenizer = load_tokenizer(model_cfg)
    print("loading model ...")
    model = load_base_model(model_cfg, quantized=not args.no_quantized)
    print("cuda memory after load:", cuda_memory_mb())
    generator = SLMGenerator(tokenizer, model)
    result = generator.generate_one(args.question)
    print("\nQUESTION:", args.question)
    print("\nRAW RESPONSE:\n" + result["raw_response"])
    print("\nPARSED:", result["parsed"])
    print("latency_ms:", result["latency_ms"], "output_tokens:", result["output_tokens"])
    print("cuda memory after generation:", cuda_memory_mb())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
