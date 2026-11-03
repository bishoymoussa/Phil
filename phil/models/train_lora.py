"""LoRA SFT for PhiLLM direct-answer training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from phil.models.prompting import format_sft_example
from phil.utils.io import ensure_dir, read_jsonl, write_json


def _messages_to_text(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            # naive fallback
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            return "\n".join(parts)


def build_text_dataset(path: str, tokenizer, limit: int | None = None):
    from datasets import Dataset

    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    texts = []
    for r in rows:
        fmt = format_sft_example(r)
        texts.append(_messages_to_text(tokenizer, fmt["messages"]))
    return Dataset.from_dict({"text": texts})


def train(
    model_name: str,
    train_path: str,
    dev_path: str,
    out_dir: str,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: float = 1.0,
    lr: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 1024,
    seed: int = 42,
    limit_train: int | None = None,
    limit_dev: int | None = None,
) -> Path:
    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    ensure_dir(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_ds = build_text_dataset(train_path, tokenizer, limit=limit_train)
    eval_ds = build_text_dataset(dev_path, tokenizer, limit=limit_dev)

    use_bf16 = dtype == torch.bfloat16
    # Prefer modern SFTConfig; fall back to TrainingArguments kwargs
    try:
        from trl import SFTConfig

        args = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=use_bf16,
            fp16=not use_bf16 and torch.cuda.is_available(),
            max_seq_length=max_seq_len,
            seed=seed,
            report_to=[],
            dataset_text_field="text",
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except TypeError:
        targs = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=use_bf16,
            fp16=not use_bf16 and torch.cuda.is_available(),
            seed=seed,
            report_to=[],
        )
        trainer = SFTTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_len,
        )

    train_result = trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {
        "train_loss": float(getattr(train_result, "training_loss", None) or 0.0),
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "train_path": train_path,
        "dev_path": dev_path,
        "n_train": len(train_ds),
        "n_dev": len(eval_ds),
    }
    write_json(Path(out_dir) / "train_metrics.json", metrics)
    return Path(out_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-dev", type=int, default=None)
    args = p.parse_args(argv)
    out = train(
        args.model,
        args.train,
        args.dev,
        args.out,
        lora_r=args.lora_r,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        limit_train=args.limit_train,
        limit_dev=args.limit_dev,
    )
    print(f"saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
