"""Extract hidden states at number tokens and last prompt token."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from phil.models.prompting import build_messages
from phil.utils.io import ensure_dir, read_jsonl, write_json


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def derived_targets(example: dict[str, Any]) -> dict[str, float]:
    o = example["gold_spec"]["objects"][0]
    m = float(o["mass"])
    f0 = float((o.get("force") or [0.0])[0])
    a = f0 / m
    ans = example["gold_answer"]
    if example["task_type"] == "T1":
        xk = float(ans["objects"][0]["position"][0])
        vk = float(ans["objects"][0]["velocity"][0])
    else:
        xk = float("nan")
        vk = float("nan")
    v0 = float(o["velocity"][0])
    return {
        "m": m,
        "x0": float(o["position"][0]),
        "v0": v0,
        "f": f0,
        "a": a,
        "xk": xk,
        "vk": vk,
        "momentum": m * v0,
        "ke": 0.5 * m * v0 * v0,
    }


def extract_hidden_states(
    model_name: str,
    data_path: str,
    out_dir: str,
    *,
    adapter_path: str | None = None,
    limit: int | None = 500,
    device: str = "cuda",
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_dir(out_dir)
    rows = read_jsonl(data_path)
    rows = [r for r in rows if r["task_type"] == "T1"]
    if limit:
        rows = rows[:limit]

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

    n_layers = model.config.num_hidden_layers + 1  # embeddings + layers
    # store last-prompt-token states per layer
    last_tok = []
    targets = []
    ids = []

    for ex in rows:
        messages = build_messages(ex, mode="direct", fewshot=False, enable_thinking=False)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tok = tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(model.device) for k, v in tok.items()}
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple (n_layers+1,) each [1, seq, hidden]
        layer_vecs = []
        for h in out.hidden_states:
            layer_vecs.append(h[0, -1].float().cpu().numpy())
        last_tok.append(np.stack(layer_vecs, axis=0))  # (L, H)
        targets.append(derived_targets(ex))
        ids.append(ex["id"])

    arr = np.stack(last_tok, axis=0)  # (N, L, H)
    np.savez_compressed(
        Path(out_dir) / "hidden_last_prompt.npz",
        hidden=arr,
        ids=np.array(ids),
    )
    # targets as columns
    keys = list(targets[0].keys())
    y = np.array([[t[k] for k in keys] for t in targets], dtype=np.float64)
    np.savez_compressed(Path(out_dir) / "targets.npz", y=y, keys=np.array(keys))
    write_json(
        Path(out_dir) / "meta.json",
        {
            "model": model_name,
            "adapter": adapter_path,
            "n": len(ids),
            "n_layers": int(arr.shape[1]),
            "hidden": int(arr.shape[2]),
            "data": data_path,
            "keys": keys,
        },
    )
    return Path(out_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args(argv)
    out = extract_hidden_states(args.model, args.data, args.out, adapter_path=args.adapter, limit=args.limit)
    print(f"wrote hidden states to {out}")


if __name__ == "__main__":
    main()
