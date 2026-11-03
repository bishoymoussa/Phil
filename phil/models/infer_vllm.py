"""Batch inference with vLLM (Explorer) or HuggingFace generate (fallback)."""

from __future__ import annotations

import argparse
import json
from typing import Any

from phil.models.prompting import build_messages
from phil.utils.io import ensure_dir, read_jsonl, write_jsonl


def _apply_chat_template(tokenizer, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3/3.5 may support enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def infer_vllm(
    model_name: str,
    examples: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    fewshot: bool = True,
    enable_thinking: bool = False,
    mode: str = "direct",
    tensor_parallel: int = 1,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel,
        dtype="auto",
    )
    prompts = []
    for ex in examples:
        messages = build_messages(ex, mode=mode, fewshot=fewshot, enable_thinking=enable_thinking)
        prompts.append(_apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking))
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    rows = []
    for ex, out in zip(examples, outputs):
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        rows.append(
            {
                "id": ex["id"],
                "raw_output": text,
                "final_answer": text,
                "called": False,
                "spec_str": None,
                "spec_valid": None,
                "p_sim": None,
                "n_tokens": n_tokens,
                "model": model_name,
            }
        )
    return rows


def infer_hf(
    model_name: str,
    examples: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    fewshot: bool = True,
    enable_thinking: bool = False,
    mode: str = "direct",
    adapter_path: str | None = None,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    rows = []
    for ex in examples:
        messages = build_messages(ex, mode=mode, fewshot=fewshot, enable_thinking=enable_thinking)
        prompt = _apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
        inputs = tokenizer(prompt, return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        rows.append(
            {
                "id": ex["id"],
                "raw_output": text,
                "final_answer": text,
                "called": False,
                "spec_str": None,
                "spec_valid": None,
                "p_sim": None,
                "n_tokens": int(gen.numel()),
                "model": model_name,
                "adapter": adapter_path,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fewshot", action="store_true", default=True)
    p.add_argument("--no-fewshot", action="store_true")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--adapter", default=None)
    p.add_argument("--tp", type=int, default=1)
    args = p.parse_args(argv)

    examples = read_jsonl(args.data)
    if args.limit:
        examples = examples[: args.limit]
    fewshot = not args.no_fewshot
    if args.backend == "vllm":
        rows = infer_vllm(
            args.model,
            examples,
            max_tokens=args.max_tokens,
            fewshot=fewshot,
            enable_thinking=args.thinking,
            tensor_parallel=args.tp,
        )
    else:
        rows = infer_hf(
            args.model,
            examples,
            max_tokens=args.max_tokens,
            fewshot=fewshot,
            enable_thinking=args.thinking,
            adapter_path=args.adapter,
        )
    ensure_dir(__import__("pathlib").Path(args.out).parent)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
