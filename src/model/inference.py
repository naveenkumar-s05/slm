"""Batched text generation for the CyberAware assistant."""

from __future__ import annotations

import time
from typing import Any

import torch

from src.model.prompting import build_messages, parse_response


class SLMGenerator:
    def __init__(
        self,
        tokenizer,
        model,
        *,
        max_new_tokens: int = 220,
        min_new_tokens: int = 10,
        do_sample: bool = False,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,
        max_input_tokens: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.max_input_tokens = max_input_tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, questions: list[str]) -> dict[str, torch.Tensor]:
        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(question),
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in questions
        ]
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )

    @torch.inference_mode()
    def generate_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        if not questions:
            return []
        inputs = self._encode(questions)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        generated = output_ids[:, input_length:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_lengths = [int((row != self.tokenizer.pad_token_id).sum()) for row in generated]

        results = []
        per_request_ms = elapsed_ms / len(questions)
        for question, text, output_tokens in zip(questions, texts, output_lengths):
            parsed = parse_response(text)
            results.append(
                {
                    "question": question,
                    "raw_response": text.strip(),
                    "parsed": parsed,
                    "latency_ms": round(per_request_ms, 2),
                    "batch_latency_ms": round(elapsed_ms, 2),
                    "input_tokens": int(input_length),
                    "output_tokens": output_tokens,
                    "tokens_per_second": round(output_tokens / (per_request_ms / 1000.0), 2)
                    if per_request_ms > 0
                    else None,
                }
            )
        return results

    def generate_one(self, question: str) -> dict[str, Any]:
        return self.generate_batch([question])[0]
