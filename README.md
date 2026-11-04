# CyberAware — Domain-Adapted Small Language Model

A locally deployable cybersecurity awareness assistant built by fine-tuning
**Qwen2.5-1.5B-Instruct** with **QLoRA** on a hybrid, licence-cleared dataset.
The model answers everyday security questions in a fixed, machine-checkable
four-line format and refuses harmful requests through deterministic guardrails.

Built for an environment with a **4 GB NVIDIA RTX 2050**, 16 GB RAM, and no paid API.

## Results (84-prompt held-out test)

| System | Schema validity | Category | Risk | Task score | ROUGE-L | Unsafe recall | Safety flags | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0061 | 0.0% | 0.0% | 6521 ms |
| QLoRA v2 (raw) | 98.8% | 64.3% | 60.7% | 77.0% | 0.2545 | 0.0% | 1.2% | 7416 ms |
| **QLoRA v2 + guardrails** | **98.8%** | **66.7%** | **61.9%** | **78.1%** | **0.2588** | **100.0%** | **0.0%** | 7416 ms |

The zero-shot model produced fluent prose but never matched the required schema.
QLoRA adaptation raised schema validity to 98.8% and category accuracy to 66.7%.
The deterministic guardrail fixed the one remaining raw-model safety failure on the
held-out unsafe prompts. Full evidence: [`reports/benchmark_evidence.md`](reports/benchmark_evidence.md).

## Use case

**AI Cybersecurity Awareness Assistant** for non-technical users:

- Recognise phishing, social engineering, unsafe Wi-Fi, and malware risks.
- Give plain-language protective advice and one concrete next action.
- Refuse requests for hacking, credential theft, malware, or evasion.
- Return a stable schema so the API can validate, log, and display responses.

Response contract:

```text
Category: phishing
Risk: high
Advice: Check the sender address and hover over links before clicking. Unexpected urgency or requests for credentials are warning signs. When in doubt, contact the organisation through a known channel.
Action: Report the message to your IT or security team and delete it.
```

## Architecture

```text
User question
     |
     v
Input validation (Pydantic, length/control chars)
     |
     v
Unsafe-request guardrail (regex + deterministic refusal)
     |
     v
QLoRA adapter on Qwen2.5-1.5B-Instruct (4-bit NF4)
     |
     v
Schema parser + output safety guardrail
     |
     v
FastAPI JSON response + JSONL audit log + metrics
```

## Repository layout

```text
configs/                 model, training, evaluation, API settings
data/
  raw/                   downloaded source metadata
  processed/             combined dataset and statistics
  splits/                train / val / test JSONL (no leakage)
models/
  cyberaware-qlora/      first adapter (training artefact)
  cyberaware-qlora-v2/   final adapter used by the API
src/
  api/                   FastAPI app, schemas, inference service
  data/                  curation, download, build, validation, balancing
  evaluation/            metrics, evaluation runner, guardrails, comparison
  model/                 loading, prompting, inference, QLoRA training
  optimization/          benchmark script
  utils.py               config, IO, hardware and memory helpers
tests/                   pytest suite (API, schema, metrics, data integrity)
experiments/results/     saved predictions and metrics
reports/                 technical report, model selection, dataset report
```

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For offline use after the base model has been cached:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:TOKENIZERS_PARALLELISM="false"
```

## Reproduce the data

```powershell
python -m src.data.download_datasets
python -m src.data.build_dataset
python -m src.data.validate_data
python -m src.data.balance_train
```

Current split sizes: **train 669 / val 77 / test 84**. The balanced training file has
444 records and is used for the second training stage.

## Train

First adapter:

```powershell
python -m src.model.train_qlora --config configs/train.yaml
```

Second stage (balanced data, lower learning rate, starts from adapter v1):

```powershell
python -m src.model.train_qlora `
  --init-adapter models/cyberaware-qlora `
  --train-path data/splits/train_balanced.jsonl `
  --output-dir models/cyberaware-qlora-v2 `
  --learning-rate 5e-5 `
  --max-steps 100
```

Training used 4-bit NF4, LoRA `r=16`, `alpha=32`, max sequence length 512,
gradient checkpointing, paged AdamW 8-bit, and batch size 1 with gradient
accumulation 4. Total training time was about 37 minutes on the RTX 2050.

## Evaluate

```powershell
python -m src.evaluation.evaluate --adapter models/cyberaware-qlora-v2 --tag final
python -m src.evaluation.apply_guardrails
python -m src.evaluation.compare_results
```

Outputs are written to `experiments/results/`. The comparison script regenerates
`reports/benchmark_evidence.md`.

## Run the API

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>

```powershell
curl -X POST http://127.0.0.1:8000/v1/answer `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"How can I tell if an email is phishing?\"}'
```

Example response:

```json
{
  "request_id": "8b0f...",
  "answer": {
    "category": "phishing",
    "risk": "high",
    "advice": "Check the sender address and hover over links before clicking.",
    "action": "Report the message to your IT or security team."
  },
  "raw_response": "Category: phishing\nRisk: high\n...",
  "valid": true,
  "guardrail_triggered": false,
  "latency_ms": 2140.5,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "warnings": []
}
```

Endpoints:

- `POST /v1/answer` — validated question, structured answer
- `GET /health` — service and model load status
- `GET /v1/model` — model, adapter, licence, categories
- `GET /metrics` — request counts, valid rate, latency
- `GET /` — minimal browser demo

Requests are logged to `experiments/logs/api_requests.jsonl` with a question hash,
not the raw question, to reduce accidental personal-data retention.

## Tests

```powershell
python -m pytest -q
```

The suite covers schema parsing, unsafe-request detection, guardrails, API
validation/error handling, metrics, split integrity, and label coverage.

## Optimization evidence

- **4-bit NF4 + double quantization**: ~1.2 GB loaded VRAM; ~1.75 GB peak reserved
  during the full inference run, fitting the 4 GB GPU.
- **Batched inference** (batch size 8): 16-prompt smoke run averaged 1953 ms/request
  vs 6521 ms for the batch-size-1 baseline. The full 84-prompt run averaged 7416 ms
  under sustained load, so latency is still hardware-bound.
- **Token cap** reduced from 220 to 140, cutting worst-case generation length while
  keeping mean output at ~81 tokens.
- **Deterministic guardrails** add negligible latency because unsafe requests are
  refused before inference.

`python -m src.optimization.benchmark --limit 12` is provided to reproduce a
batch-size/token-cap comparison. The fp16 configuration was removed because
1.5B fp16 weights plus KV cache do not fit reliably in 4 GB VRAM.

## Design decisions

- **Qwen2.5-1.5B-Instruct** fits 4-bit on 4 GB, follows instructions well, and is
  Apache-2.0. A 0.5B fallback is configured for lower-memory machines.
- **QLoRA** trains <1% of parameters and keeps the base model frozen.
- **Hybrid data** combines three public cybersecurity QA sources with 82 curated
  examples for schema, safety, and category coverage.
- **Guardrails are separate from the model** so the safety layer is deterministic,
  testable, and not confused with adaptation quality.
- **Held-out test set is never used for training or hyperparameter selection.**

## Limitations

- 84 test prompts are enough for a clear baseline/adaptation comparison but not for
  production-grade confidence intervals.
- Category accuracy is 66.7%; social engineering, safe browsing, updates, and public
  Wi-Fi remain weaker than phishing and privacy.
- The 4 GB GPU limits batch size and makes inference slow; this is a local demo, not
  a high-throughput service.
- Regex guardrails are intentionally conservative and can miss paraphrased attacks.
  A production system should add a trained safety classifier.
- The model can still produce incomplete or imprecise advice; users should treat it
  as awareness guidance, not professional incident response.

## Data licences and attribution

- `mariiazhiv/cybersecurity_qa` — Apache-2.0
- `Rowden/CybersecurityQAA` — MIT
- `Ngadou/social-engineering-convo` — Apache-2.0
- 82 project-authored curated examples
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` — Apache-2.0

See [`reports/dataset_report.md`](reports/dataset_report.md) for filtering and
split details. Public source rows were filtered for awareness relevance; offensive
instructional content was removed.

## Submission checklist

- [x] Domain-specific SLM and use case
- [x] Hybrid dataset with provenance and licences
- [x] QLoRA adaptation and saved adapter
- [x] Baseline and final evaluation on a held-out test set
- [x] Optimization evidence and hardware measurements
- [x] Guardrails and failure analysis
- [x] FastAPI deployment with validation and logging
- [x] Automated tests
- [x] Technical report and model/dataset reports
- [ ] Push to GitHub and submit the repository URL
