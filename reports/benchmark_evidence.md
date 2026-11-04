# Benchmark Evidence

| System | n | Format | Category | Risk | Joint | Task | ROUGE-L | Unsafe recall | Safety flags | Mean latency | p95 latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot baseline | 84 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0061 | 0.0% | 0.0% | 6521 ms | 13963 ms | 1192 MB |
| QLoRA v2 (raw) | 84 | 98.8% | 64.3% | 60.7% | 53.6% | 77.0% | 0.2545 | 0.0% | 1.2% | 7416 ms | 8953 ms | 1748 MB |
| QLoRA v2 + guardrails | 84 | 98.8% | 66.7% | 61.9% | 56.0% | 78.1% | 0.2588 | 100.0% | 0.0% | 7416 ms | 8953 ms | 1748 MB |
| QLoRA v2 smoke (16 prompts) | 16 | 100.0% | 62.5% | 50.0% | 37.5% | 73.8% | 0.1947 | 0.0% | 6.2% | 1953 ms | 1984 ms | 1508 MB |

## Key changes

- Schema validity: **0.0% → 98.8%** (+98.8 pp).
- Category accuracy: **0.0% → 64.3%** (+64.3 pp).
- Task score: **0.0% → 77.0%** (+77.0 pp).
- Unsafe-request recall: **0.0% → 100.0%** after the deterministic input/output guardrail.
- Quantized 4-bit inference used ~1.2 GB loaded VRAM and peaked at ~1.75 GB reserved, fitting the 4 GB RTX 2050.
- Latency is hardware-bound: full 84-prompt run averaged **7416 ms/request**; the 16-prompt batched smoke run averaged **1953 ms/request**.
- The smoke row is indicative only (n=16) and is not directly comparable to the full held-out test.
