"""QLoRA adaptation for the CyberAware assistant.

Run:  python -m src.model.train_qlora
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

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
)


class ChatDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_seq_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.skipped = 0
        for record in records:
            item = self._encode(record, tokenizer, max_seq_length)
            if item is None:
                self.skipped += 1
            else:
                self.items.append(item)

    @staticmethod
    def _encode(record: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, list[int]] | None:
        messages = record["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + target_ids
        if len(full_ids) <= len(prompt_ids):
            return None
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if len(full_ids) > max_seq_length:
            full_ids = full_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]
        return {"input_ids": full_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = ROOT / (args.output_dir or train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = ROOT / train_cfg.get("logging_dir", "experiments/logs")
    logging_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(ROOT / (args.train_path or train_cfg["train_path"]))
    val_records = read_jsonl(ROOT / train_cfg["val_path"])

    print("loading tokenizer and 4-bit base model ...")
    tokenizer = load_tokenizer(model_cfg)
    if args.init_adapter:
        print(f"continuing from adapter: {args.init_adapter}")
        model = load_adapter_model(model_cfg, args.init_adapter, quantized=True)
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model = load_base_model(model_cfg, quantized=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_cfg = train_cfg["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["lora_alpha"]),
            lora_dropout=float(lora_cfg["lora_dropout"]),
            bias=str(lora_cfg.get("bias", "none")),
            task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
            target_modules=list(lora_cfg["target_modules"]),
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("cuda memory after LoRA attach:", cuda_memory_mb())

    train_dataset = ChatDataset(train_records, tokenizer, int(train_cfg["max_seq_length"]))
    val_dataset = ChatDataset(val_records, tokenizer, int(train_cfg["max_seq_length"]))
    print(f"train examples: {len(train_dataset)} (skipped {train_dataset.skipped})")
    print(f"val examples: {len(val_dataset)} (skipped {val_dataset.skipped})")

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        run_name=train_cfg.get("run_name", "cyberaware-qlora"),
        max_steps=int(args.max_steps or train_cfg["max_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(args.learning_rate or train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=str(train_cfg.get("eval_strategy", "steps")),
        eval_steps=int(train_cfg.get("eval_steps", 40)),
        save_strategy=str(train_cfg.get("save_strategy", "steps")),
        save_steps=int(train_cfg.get("save_steps", 40)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        prediction_loss_only=True,
        seed=seed,
        data_seed=seed,
    )
    training_args = TrainingArguments(**training_kwargs)
    collator = CausalCollator(tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("starting training ...")
    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    duration = time.perf_counter() - started

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    history = [entry for entry in trainer.state.log_history]
    write_json(logging_dir / "train_history.json", {"log_history": history})
    summary = {
        "created_utc": now_iso(),
        "base_model": model_cfg["base_model"],
        "adapter_path": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "duration_s": round(duration, 1),
        "train_metrics": {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))},
        "final_train_loss": next(
            (entry["loss"] for entry in reversed(history) if "loss" in entry), None
        ),
        "last_eval_loss": next(
            (entry["eval_loss"] for entry in reversed(history) if "eval_loss" in entry), None
        ),
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train_config": train_cfg,
        "hardware": hardware_info(),
        "cuda_memory_after_load": cuda_memory_mb(),
    }
    write_json(output_dir / "training_summary.json", summary)
    print("\nDONE")
    print(f"duration_s: {summary['duration_s']}")
    print(f"final_train_loss: {summary['final_train_loss']}")
    print(f"last_eval_loss: {summary['last_eval_loss']}")
    print(f"adapter: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
