# Model Card — CyberAware QLoRA v2

## Model details

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Adapter:** `models/cyberaware-qlora-v2`
- **Method:** QLoRA (4-bit NF4 base, LoRA rank 16, alpha 32)
- **Parameters trained:** <1% of the base model
- **Licence:** Apache-2.0 (base model); project adapter released for educational use
- **Language:** English
- **Task:** Defensive cybersecurity awareness question answering

## Intended use

- Employee security-awareness education
- Demonstrating domain adaptation and QLoRA on consumer hardware
- Local, offline question answering with structured output
- API integration for demos and coursework

## Out-of-scope use

- Penetration testing or offensive security
- Live incident response, forensics, or legal advice
- Medical, financial, or safety-critical decisions
- Autonomous security operations
- Any request for hacking, malware, credential theft, evasion, or surveillance

## Output contract

```text
Category: <one of 13 allowed labels>
Risk: <low|medium|high>
Advice: <2–4 plain-language sentences>
Action: <one concrete next step>
```

## Training data

830 cleaned records from three public cybersecurity QA datasets and 82 curated
examples. Splits: 669 train / 77 validation / 84 test. See
`reports/dataset_report.md` for provenance and licences.

## Evaluation

Held-out test (n=84), greedy decoding, max 140 new tokens:

| Metric | Value |
|---|---:|
| Schema validity | 98.8% |
| Category accuracy | 66.7% (with guardrails) |
| Risk accuracy | 61.9% |
| Task score | 78.1% |
| Advice ROUGE-L | 0.2588 |
| Unsafe-request recall | 100.0% |
| Safety-flag rate | 0.0% |
| Mean latency | 7,416 ms on RTX 2050 4 GB |

## Safety

- Input regex guardrail refuses clearly harmful requests before inference.
- Output guardrail replaces dangerous completions with a fixed refusal.
- The guardrail is deterministic and separately testable.
- Regex rules are not a complete defence; paraphrased attacks may bypass them.

## Limitations

- Small test set; wide confidence intervals.
- Weak categories: social engineering, safe browsing, updates, public Wi-Fi, mobile.
- Risk labels are not well calibrated.
- Slow inference on 4 GB hardware.
- English-only.
- Can produce incorrect or incomplete security advice; not a substitute for a
  qualified professional.

## Ethical considerations

The dataset is filtered to remove offensive instructional content. The model is
designed to refuse harmful requests and redirect to defensive guidance. Logging stores
question hashes rather than raw questions to reduce personal-data retention.

## Contact

Project coursework artefact. Use at your own risk; no warranty.
