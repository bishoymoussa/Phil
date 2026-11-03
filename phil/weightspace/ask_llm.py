"""Ask an LLM to read tiny simulator weights (RQ5)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from phil.utils.io import ensure_dir, read_json, write_json, write_jsonl
from phil.weightspace.serialize import build_read_prompt, weights_to_text


LAW_RE = re.compile(r"LAW\s*=\s*([A-Za-z_]+)", re.I)
PARAM_RE = re.compile(r"PARAM\s*=\s*([-+0-9.eE]+)")


def parse_answer(text: str) -> dict[str, Any]:
    law = None
    m = LAW_RE.search(text or "")
    if m:
        law = m.group(1).lower()
    param = None
    m2 = PARAM_RE.search(text or "")
    if m2:
        try:
            param = float(m2.group(1))
        except ValueError:
            param = None
    return {"law": law, "param": param, "raw": text}


def ask_hf(
    model_name: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    enable_thinking: bool = True,
    device: str = "cuda",
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    messages = [
        {"role": "system", "content": "You carefully read neural network weights."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    if device.startswith("cuda"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def evaluate_catalog(
    catalog_path: str,
    model_name: str,
    out_path: str,
    *,
    include_controls: bool = True,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    root = Path(catalog_path).parent
    rows = []
    correct = 0
    n = 0
    for law, meta in catalog.items():
        variants = [("true", meta["path"])]
        if include_controls:
            arch = meta["arch"]
            variants.append(("shuffled", str(root / f"{law}_{arch}_shuffled.npz")))
            variants.append(("random", str(root / f"{law}_{arch}_random.npz")))
        for kind, path in variants:
            prompt = build_read_prompt(weights_to_text(path))
            raw = ask_hf(model_name, prompt, enable_thinking=enable_thinking)
            parsed = parse_answer(raw)
            ok = kind == "true" and parsed["law"] == law
            if kind == "true":
                n += 1
                correct += int(ok)
            rows.append(
                {
                    "law_true": law,
                    "kind": kind,
                    "path": path,
                    "pred_law": parsed["law"],
                    "pred_param": parsed["param"],
                    "correct_law": ok if kind == "true" else None,
                    "raw": raw,
                    "model": model_name,
                }
            )
            print(f"{law}/{kind}: pred={parsed['law']} ok={ok}")
    summary = {
        "model": model_name,
        "accuracy_true": correct / n if n else None,
        "n_true": n,
        "n_rows": len(rows),
    }
    ensure_dir(Path(out_path).parent)
    write_jsonl(out_path, rows)
    write_json(str(Path(out_path).with_suffix(".summary.json")), summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--out", required=True)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    args = p.parse_args(argv)
    evaluate_catalog(
        args.catalog,
        args.model,
        args.out,
        include_controls=not args.no_controls,
        enable_thinking=not args.no_thinking,
    )


if __name__ == "__main__":
    main()
