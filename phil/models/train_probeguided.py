"""Probe-guided LoRA: SFT + auxiliary linear heads on mid-layer activations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phil.models.prompting import build_messages
from phil.utils.io import ensure_dir, read_jsonl, write_json


@dataclass
class AuxTargets:
    acceleration: np.ndarray  # (dim,)
    future_pos: np.ndarray
    future_vel: np.ndarray


def compute_aux_targets(example: dict[str, Any]) -> AuxTargets | None:
    """Derived quantities for the first object (teaching signal)."""
    if example["task_type"] != "T1":
        return None
    o = example["gold_spec"]["objects"][0]
    m = float(o["mass"])
    f = np.asarray(o.get("force") or [0.0] * example["gold_spec"]["dim"], dtype=np.float32)
    a = f / m
    ans = example["gold_answer"]["objects"][0]
    return AuxTargets(
        acceleration=a.astype(np.float32),
        future_pos=np.asarray(ans["position"], dtype=np.float32),
        future_vel=np.asarray(ans["velocity"], dtype=np.float32),
    )


def train_probeguided(
    model_name: str,
    train_path: str,
    dev_path: str,
    out_dir: str,
    *,
    layer_index: int = -4,
    aux_weight: float = 0.1,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_r: int = 16,
    batch_limit: int | None = 5000,
    seed: int = 42,
    device: str = "cuda",
) -> Path:
    """Lightweight custom loop: CE on answer tokens + MSE aux heads."""
    import torch
    import torch.nn as nn
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(base, peft_config)
    hidden = model.config.hidden_size
    # assume up to 2D targets (pad to 2)
    head_a = nn.Linear(hidden, 2).to(model.device, dtype=torch.float32)
    head_x = nn.Linear(hidden, 2).to(model.device, dtype=torch.float32)
    head_v = nn.Linear(hidden, 2).to(model.device, dtype=torch.float32)

    train_rows = [r for r in read_jsonl(train_path) if r["task_type"] == "T1"]
    if batch_limit:
        idx = rng.choice(len(train_rows), size=min(batch_limit, len(train_rows)), replace=False)
        train_rows = [train_rows[i] for i in idx]

    params = [p for p in model.parameters() if p.requires_grad] + list(head_a.parameters()) + list(
        head_x.parameters()
    ) + list(head_v.parameters())
    opt = AdamW(params, lr=lr)

    model.train()
    step = 0
    losses = []
    for epoch in range(epochs):
        rng.shuffle(train_rows)
        for ex in tqdm(train_rows, desc=f"epoch {epoch}"):
            aux = compute_aux_targets(ex)
            if aux is None:
                continue
            messages = build_messages(ex, mode="direct", fewshot=False, enable_thinking=False)
            import json as _json

            completion = _json.dumps(ex["gold_answer"], separators=(",", ":"))
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = prompt + completion
            tok = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
            tok = {k: v.to(model.device) for k, v in tok.items()}
            # mask prompt tokens from CE
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            labels = tok["input_ids"].clone()
            labels[:, :prompt_len] = -100

            out = model(**tok, labels=labels, output_hidden_states=True)
            ce = out.loss
            hs = out.hidden_states[layer_index][0, -1].float()  # last token of full seq
            # better: last prompt token
            hs = out.hidden_states[layer_index][0, prompt_len - 1].float()

            def pad2(x: np.ndarray) -> torch.Tensor:
                t = torch.zeros(2, device=model.device, dtype=torch.float32)
                t[: len(x)] = torch.tensor(x, device=model.device, dtype=torch.float32)
                return t

            pred_a = head_a(hs)
            pred_x = head_x(hs)
            pred_v = head_v(hs)
            mse = (
                nn.functional.mse_loss(pred_a, pad2(aux.acceleration))
                + nn.functional.mse_loss(pred_x, pad2(aux.future_pos))
                + nn.functional.mse_loss(pred_v, pad2(aux.future_vel))
            )
            loss = ce + aux_weight * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step % 100 == 0:
                print(f"step {step} loss={np.mean(losses[-100:]):.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(
        {"a": head_a.state_dict(), "x": head_x.state_dict(), "v": head_v.state_dict(), "layer": layer_index},
        Path(out_dir) / "aux_heads.pt",
    )
    write_json(
        Path(out_dir) / "train_metrics.json",
        {
            "mean_loss": float(np.mean(losses)) if losses else None,
            "steps": step,
            "aux_weight": aux_weight,
            "layer_index": layer_index,
            "model_name": model_name,
            "seed": seed,
            "n_train": len(train_rows),
        },
    )
    return Path(out_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    out = train_probeguided(
        args.model,
        args.train,
        args.dev,
        args.out,
        epochs=args.epochs,
        aux_weight=args.aux_weight,
        batch_limit=args.limit,
        seed=args.seed,
    )
    print(f"saved probe-guided adapter to {out}")


if __name__ == "__main__":
    main()
