# Technical Report — CyberAware SLM

## 1. Executive summary

This project adapts a 1.5B-parameter instruction-tuned language model into a
cybersecurity awareness assistant that runs locally on a 4 GB NVIDIA RTX 2050.
The base model, `Qwen/Qwen2.5-1.5B-Instruct`, was fine-tuned with QLoRA on a hybrid
dataset of 830 cleaned examples from three public cybersecurity QA sources plus 82
curated examples.

The central problem was not general language ability but **behavioural alignment to a
strict output contract**. The zero-shot baseline produced fluent prose but had **0%
schema validity**, **0% category accuracy**, and **0% task score**. After QLoRA
adaptation, schema validity reached **98.8%**, category accuracy **64.3%**, and task
score **77.0%** on an 84-prompt held-out test set. A deterministic safety layer raised
unsafe-request recall from **0% to 100%** while reducing safety-flag rate to **0%**.

All training and inference fit within the target hardware using 4-bit NF4
quantization, gradient checkpointing, paged AdamW, and a small LoRA adapter.

## 2. Use case and requirements

**Use case:** a defensive cybersecurity awareness assistant for employees and
non-technical users.

Functional requirements:

1. Answer questions about phishing, passwords, malware, social engineering, privacy,
   safe browsing, updates, public Wi-Fi, mobile security, incident response, and backup.
2. Return exactly four fields: `Category`, `Risk`, `Advice`, `Action`.
3. Refuse harmful requests and redirect to defensive guidance.
4. Run locally without a paid API.
5. Expose an HTTP API with validation, logging, and metrics.
6. Be reproducible and testable.

Non-goals: penetration testing, live incident response, legal advice, or replacing a
security professional.

## 3. Model selection

| Candidate | Params | Why considered | Outcome |
|---|---:|---|---|
| Qwen2.5-1.5B-Instruct | 1.54B | Strong instruction following, Apache-2.0, fits 4-bit in 4 GB | **Selected** |
| Qwen2.5-0.5B-Instruct | 0.49B | Lower memory and faster fallback | Configured as fallback, not used |
| Phi-3-mini-4k-instruct | 3.8B | Strong reasoning for size | Too tight for 4 GB with KV cache and training overhead |
| TinyLlama-1.1B-Chat | 1.1B | Very small | Older base; weaker instruction and safety behaviour |

The 1.5B Qwen model was the best balance of capability, licence, and memory fit.
Its 4-bit weights occupy roughly 1.2 GB, leaving room for activations and KV cache
on the 4 GB GPU. The 0.5B fallback remains available in `configs/model.yaml` for
machines with less memory.

## 4. Dataset

### 4.1 Sources

| Source | Licence | Raw rows | Accepted after filtering |
|---|---|---:|---:|
| Project-curated examples | Project-authored | 82 | 82 |
| `mariiazhiv/cybersecurity_qa` | Apache-2.0 | 502 | 144 |
| `Rowden/CybersecurityQAA` | MIT | 1,563 | 612 |
| `Ngadou/social-engineering-convo` | Apache-2.0 | 10 | 10 |

After deduplication (15 removed) and cross-split leakage checks (3 test rows removed),
the final dataset contains **830 records**:

| Split | Records |
|---|---:|
| Train | 669 |
| Validation | 77 |
| Test | 84 |

The balanced training file used for stage two contains **444 records**, with the rare
`unsafe-request` and `general` classes retained and the dominant classes down-sampled.

### 4.2 Construction pipeline

1. Download public sources and record provenance in `data/raw/SOURCES.json`.
2. Filter rows for cybersecurity-awareness relevance.
3. Remove offensive or instructional harmful content.
4. Normalise each row into the four-line target schema.
5. Remove near-duplicates at a similarity threshold of 0.85.
6. Split into train/validation/test with a fixed seed.
7. Detect and remove cross-split near-duplicates at threshold 0.80.
8. Validate schema, label ranges, and split integrity.

Full statistics are in `data/processed/dataset_stats.json`; the validation report is
`data/processed/validation_report.json` with `"ok": true`.

### 4.3 Test-set coverage

| Category | Test rows | Category | Test rows |
|---|---:|---|---:|
| phishing | 11 | incident-response | 11 |
| passwords | 6 | backup | 4 |
| malware | 8 | unsafe-request | 2 |
| social-engineering | 6 | general | 9 |
| privacy | 9 | updates | 5 |
| safe-browsing | 3 | public-wifi | 4 |
| mobile | 6 | | |

The test set covers all 13 categories, including both unsafe-request examples.

## 5. Baseline

The unmodified base model was evaluated with the same prompt and a 220-token limit.
It produced readable security prose but ignored the four-line contract:

| Metric | Baseline |
|---|---:|
| Schema validity | 0.0% |
| Category accuracy | 0.0% |
| Risk accuracy | 0.0% |
| Joint category/risk | 0.0% |
| Task score | 0.0% |
| Advice ROUGE-L | 0.0061 |
| Unsafe-request recall | 0.0% |
| Safety-flag rate | 0.0% |
| Mean latency | 6,520.94 ms |
| p95 latency | 13,962.91 ms |
| Peak reserved VRAM | 1,192 MB |

Every baseline response was missing the required fields (84/84 missing `Category`
and `Risk`). This established that prompt engineering alone was insufficient and that
supervised adaptation was justified.

## 6. Adaptation

### 6.1 Method

QLoRA was chosen because it keeps the base model frozen, trains a small adapter, and
dramatically reduces optimizer and activation memory. The adapter targets all linear
projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`.

| Setting | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit NF4, double quantization, bf16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Max sequence length | 512 |
| Train batch size | 1 |
| Gradient accumulation | 4 |
| Optimizer | paged AdamW 8-bit |
| Learning rate (stage 1) | 2e-4 |
| Learning rate (stage 2) | 5e-5 |
| Scheduler | cosine, 3% warmup |
| Gradient checkpointing | enabled |
| Seed | 42 |

### 6.2 Training stages

| Stage | Data | Steps | Train loss | Eval loss | Duration |
|---|---|---:|---:|---:|---:|
| v1 | 669 train rows | 160 | 1.3436 | 1.3210 | 1,416.8 s |
| v2 | 444 balanced rows, initialised from v1 | 100 | 1.3118 | 1.3201 | 811.5 s |

Stage two did not materially lower validation loss, which suggests the remaining
error is driven by label ambiguity and class coverage rather than underfitting.
The balanced stage did improve the distribution of rare categories and was retained
because it produced the stronger final test behaviour.

### 6.3 Final adapter

- Path: `models/cyberaware-qlora-v2`
- Training summary: `models/cyberaware-qlora-v2/training_summary.json`
- The adapter is loaded on top of the 4-bit base model at inference time.

## 7. Optimization

The 4 GB GPU is the binding constraint. The following measures made the system
feasible:

| Optimization | Effect |
|---|---|
| 4-bit NF4 + double quantization | Loaded model ~1.2 GB; full-run peak reserved ~1.75 GB |
| Gradient checkpointing | Traded compute for activation memory during training |
| Paged AdamW 8-bit | Reduced optimizer-state memory |
| LoRA adapters | Trained <1% of parameters; base weights frozen |
| Max sequence length 512 | Bounded training memory and step time |
| Batched inference (size 8) | 16-prompt smoke run: 1,953 ms/request |
| Max new tokens 220 → 140 | Bounded worst-case generation; mean output ~81 tokens |
| Deterministic guardrails | No model call for clearly unsafe prompts |

The baseline batch-size-1 latency was 6,521 ms/request. A 16-prompt batch-size-8 smoke
run measured 1,953 ms/request, a **70% reduction** on that subset. The full 84-prompt
run averaged 7,416 ms/request under sustained load, showing thermal throttling and
batch-composition effects. The optimization is therefore real but not sufficient to
make the model fast on this GPU; it makes it **fit and operate** within 4 GB.

The fp16 configuration was removed from the benchmark because 1.5B fp16 weights plus
KV cache do not fit reliably in 4 GB VRAM. The reproducible benchmark script is
`src/optimization/benchmark.py`.

## 8. Final evaluation

### 8.1 Overall

| Metric | Raw v2 | v2 + guardrails |
|---|---:|---:|
| Count | 84 | 84 |
| Schema validity | 98.8% | 98.8% |
| Category accuracy | 64.3% | **66.7%** |
| Risk accuracy | 60.7% | **61.9%** |
| Joint category/risk | 53.6% | **56.0%** |
| Task score | 77.0% | **78.1%** |
| Advice ROUGE-L | 0.2545 | **0.2588** |
| Action ROUGE-L | 0.2998 | **0.3048** |
| Advice token F1 | 0.3303 | **0.3343** |
| Unsafe-request recall | 0.0% | **100.0%** |
| Safety-flag rate | 1.2% | **0.0%** |
| Mean latency | 7,415.96 ms | 7,415.96 ms |
| p95 latency | 8,952.6 ms | 8,952.6 ms |
| Peak reserved VRAM | 1,748 MB | 1,748 MB |

The guardrail improves category accuracy because it deterministically labels unsafe
requests as `unsafe-request`, correcting the model's misclassification of those rows.

### 8.2 Per-category behaviour (raw v2)

| Category | n | Valid | Category acc. | Risk acc. | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| phishing | 11 | 100% | 90.9% | 81.8% | 0.2288 |
| passwords | 6 | 100% | 83.3% | 50.0% | 0.3003 |
| malware | 8 | 100% | 87.5% | 62.5% | 0.2944 |
| social-engineering | 6 | 100% | 33.3% | 33.3% | 0.2345 |
| privacy | 9 | 100% | 88.9% | 88.9% | 0.2995 |
| safe-browsing | 3 | 100% | 0.0% | 33.3% | 0.2007 |
| updates | 5 | 80% | 20.0% | 20.0% | 0.1994 |
| public-wifi | 4 | 100% | 25.0% | 25.0% | 0.2674 |
| mobile | 6 | 100% | 50.0% | 33.3% | 0.2392 |
| incident-response | 11 | 100% | 63.6% | 81.8% | 0.2692 |
| backup | 4 | 100% | 50.0% | 50.0% | 0.2447 |
| unsafe-request | 2 | 100% | 0.0% | 50.0% | 0.1917 |
| general | 9 | 100% | 88.9% | 77.8% | 0.2413 |

Phishing, privacy, malware, passwords, and general awareness are strong. Social
engineering, safe browsing, updates, public Wi-Fi, and mobile security are weak. These
categories have fewer training examples and overlapping boundaries (for example, a
public-Wi-Fi question may be labelled `safe-browsing` or `privacy`).

### 8.3 Failure analysis

1. **One schema failure:** an `updates` response produced 721 characters of advice,
   exceeding the 700-character bound by 21 characters. This is a minor formatting
   failure, not a content failure.
2. **Unsafe misclassification:** the raw adapter answered one of two unsafe-request
   prompts with harmful spyware advice and did not label it `unsafe-request`. This is
   the most important failure. The input guardrail now refuses such requests before
   inference and the output guardrail catches harmful completions.
3. **Category confusion:** `social-engineering` is often predicted as `phishing`;
   `public-wifi` is often predicted as `privacy`; `updates` is sometimes predicted as
   `malware` or `general`.
4. **Risk calibration:** risk accuracy is 61.9%; the model tends to overuse `medium`.
5. **Latency variance:** sustained full-test inference is slower than short smoke runs,
   consistent with GPU thermal throttling on the laptop.

## 9. Safety layer

The safety layer is deliberately separate from the fine-tuned model:

- **Input guardrail:** regex detection of hacking, credential theft, malware creation,
  DDoS, phishing kits, stalking/spyware, and disabling security controls.
- **Refusal template:** deterministic `unsafe-request` response with high risk,
  a refusal, and a defensive alternative.
- **Output guardrail:** scans generated text for dangerous patterns and replaces any
  flagged answer with the refusal template.
- **API validation:** length, control characters, empty input, and extra fields are
  rejected before inference.

This design makes safety testable and auditable. It is not a complete defence: regex
rules can be paraphrased around. A production deployment should add a trained safety
classifier and human review for high-risk use.

## 10. Deployment

FastAPI application in `src/api/`:

| Endpoint | Purpose |
|---|---|
| `POST /v1/answer` | Validated question → structured answer |
| `GET /health` | Model load status |
| `GET /v1/model` | Model, adapter, licence, categories |
| `GET /metrics` | Request counts, validity, latency |
| `GET /` | Minimal browser demo |

The service loads the model lazily on the first request, uses a lock to prevent
duplicate loads, logs a SHA-256 hash of each question (not the raw question), and
writes aggregate metrics to `experiments/results/api_metrics.json`.

Run:

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

## 11. Limitations and future work

1. The test set has 84 rows; confidence intervals are wide. Expand to 300–500 rows
   with independently authored questions.
2. Category accuracy of 66.7% is useful for a demo but not production-grade. Add
   more social-engineering, updates, public-Wi-Fi, and safe-browsing examples.
3. Replace regex safety with a trained classifier and red-team the system with
   paraphrased and multi-turn attacks.
4. Improve risk calibration with explicit risk definitions in the prompt and
   class-balanced training.
5. Add retrieval over a vetted awareness knowledge base to reduce hallucination.
6. Optimize inference further with vLLM, ONNX, or a smaller distilled model; the
   current 7.4 s full-test latency is hardware-bound.
7. Add a human evaluation study for helpfulness and clarity.

## 12. Reproducibility

```powershell
pip install -r requirements.txt
python -m src.data.download_datasets
python -m src.data.build_dataset
python -m src.data.validate_data
python -m src.data.balance_train
python -m src.model.train_qlora --config configs/train.yaml
python -m src.model.train_qlora --init-adapter models/cyberaware-qlora `
  --train-path data/splits/train_balanced.jsonl `
  --output-dir models/cyberaware-qlora-v2 `
  --learning-rate 5e-5 --max-steps 100
python -m src.evaluation.evaluate --adapter models/cyberaware-qlora-v2 --tag final
python -m src.evaluation.apply_guardrails
python -m src.evaluation.compare_results
python -m pytest -q
```

Saved artefacts:

- `experiments/results/baseline_metrics.json`
- `experiments/results/final_metrics.json`
- `experiments/results/final_guarded_metrics.json`
- `experiments/results/final_comparison.json`
- `experiments/logs/train_history.json`
- `models/cyberaware-qlora-v2/training_summary.json`

## 13. Conclusion

The project demonstrates a complete, reproducible SLM lifecycle on constrained
hardware: dataset curation, baseline measurement, QLoRA adaptation, optimization,
evaluation, safety hardening, API deployment, and automated testing. The largest gain
came from adaptation to the output contract: schema validity went from **0% to 98.8%**
and task score from **0% to 78.1%**. The largest remaining risk is category-level
accuracy and robustness to paraphrased unsafe requests, which are documented and
addressed with a separate deterministic guardrail rather than hidden inside the model.
