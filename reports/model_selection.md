# Model Selection Report

## 1. Constraints

| Constraint | Value |
|---|---|
| GPU | NVIDIA RTX 2050, **4 GB VRAM** |
| System RAM | 15.7 GB |
| CPU | Intel Core i5-12450H, 12 threads |
| Cost | Local, no paid API |
| Data | 830 cleaned examples |
| Output | Structured four-line response |
| Safety | Must refuse harmful requests |

The model must fit in 4 GB **with KV cache and generation headroom**, not merely load.
This rules out 7B models and makes even 3–4B models risky without aggressive offload.

## 2. Candidates

| Candidate | Params | Licence | 4-bit fit | Instruction quality | Decision |
|---|---:|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.54B | Apache-2.0 | ~1.2 GB loaded; ~1.75 GB peak | Strong | **Selected** |
| Qwen2.5-0.5B-Instruct | 0.49B | Apache-2.0 | Very comfortable | Weaker | Fallback |
| Phi-3-mini-4k-instruct | 3.8B | MIT | Tight; KV cache risk | Strong reasoning | Rejected |
| TinyLlama-1.1B-Chat | 1.1B | Apache-2.0 | Comfortable | Older, weaker | Rejected |
| Llama-3.2-1B-Instruct | 1.24B | Llama licence | Comfortable | Good | Rejected for licence simplicity |
| Gemma-2-2B-it | 2.6B | Gemma licence | Tight | Good | Rejected for memory/licence |

## 3. Why Qwen2.5-1.5B-Instruct

1. **Memory fit:** 4-bit NF4 weights load at about 1.2 GB; the full evaluation peaked
   at 1.75 GB reserved, leaving headroom for batching and KV cache.
2. **Instruction following:** handles a strict four-line format after lightweight
   supervised adaptation.
3. **Licence:** Apache-2.0 is straightforward for an internship submission.
4. **Ecosystem:** first-class support in Transformers, PEFT, and bitsandbytes.
5. **Fallback:** the 0.5B variant uses the same code path if memory is tighter.

## 4. Why QLoRA

- Base weights stay frozen; only a small adapter is trained.
- 4-bit NF4 base + bf16 compute reduces memory versus full fine-tuning.
- Adapters are portable and easy to save, version, and reload.
- Training 260 steps total took about 37 minutes on the target GPU.
- The final adapter is ~74 MB, far smaller than a merged model.

## 5. Why not full fine-tuning or a larger model

- Full fine-tuning a 1.5B model needs optimizer and activation memory that exceeds
  4 GB.
- A 3.8B model at 4-bit is close to the limit before KV cache; generation would be
  slow and prone to OOM.
- The task is narrow and schema-driven; a 1.5B model is sufficient to demonstrate
  adaptation, and the baseline/final gap is already large.

## 6. Selected configuration

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
quantization: 4-bit NF4, double quant, bf16 compute
lora: r=16, alpha=32, dropout=0.05
targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
max_seq_length: 512
train_batch_size: 1
gradient_accumulation: 4
optimizer: paged_adamw_8bit
learning_rate: 2e-4 (stage 1), 5e-5 (stage 2)
```

## 7. Evidence of fit

| Measurement | Value |
|---|---:|
| 4-bit model loaded | ~1.17 GB allocated |
| Full-run peak reserved | 1,748 MB |
| Training peak reserved | 4,586 MB (includes allocator reserve; completed successfully) |
| Baseline mean latency | 6,521 ms |
| Adapted mean latency (full test) | 7,416 ms |
| Adapted smoke latency (16 prompts, batch 8) | 1,953 ms |

The training summary records a high reserved value because PyTorch's caching allocator
holds memory; the run completed without OOM. Inference stays within the 4 GB device.

## 8. Decision summary

Qwen2.5-1.5B-Instruct + QLoRA is the only candidate that simultaneously satisfies the
4 GB memory limit, Apache-2.0 licensing, strong instruction following, and practical
training time. The 0.5B fallback is retained in configuration but was not needed.
