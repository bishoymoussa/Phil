"""Simulator-calling agent loop for RQ3."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from phil.models.prompting import SYSTEM_AGENT, build_messages
from phil.sim.spec import SpecError, extract_sim_block, parse_spec
from phil.sim.simulator import run
from phil.utils.io import ensure_dir, read_jsonl, write_jsonl


def decide_call_from_text(text: str) -> tuple[bool, str | None]:
    block = extract_sim_block(text)
    return (block is not None, block)


def run_sim_safe(spec_str: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        spec = parse_spec(spec_str)
        result = run(spec)
        return True, result, None
    except SpecError as e:
        return False, None, str(e)
    except Exception as e:  # noqa: BLE001
        return False, None, f"runtime: {e}"


def build_observation_message(result: dict[str, Any]) -> str:
    return "Simulator observation:\n" + json.dumps(
        {
            "objects": [
                {"id": o["id"], "position": o["position"], "velocity": o["velocity"]}
                for o in result["objects"]
            ],
            "t": result["t"],
            "n_collisions": result["n_collisions"],
        },
        separators=(",", ":"),
    ) + "\nNow give the final answer."


def oracle_predictions(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        result = run(ex["gold_spec"])
        if ex["task_type"] == "T1":
            ans = {
                "objects": [
                    {"id": o["id"], "position": o["position"], "velocity": o["velocity"]}
                    for o in result["objects"]
                ]
            }
            final = json.dumps(ans, separators=(",", ":"))
        elif ex["task_type"] == "T2":
            final = f"{float(ex['gold_answer']):.6g}"
        else:
            final = str(ex["gold_answer"])
        rows.append(
            {
                "id": ex["id"],
                "raw_output": final,
                "final_answer": final,
                "called": True,
                "spec_str": json.dumps(ex["gold_spec"]),
                "spec_valid": True,
                "p_sim": 1.0,
                "n_tokens": 0,
                "system": "oracle",
            }
        )
    return rows


def always_call_targets(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SFT targets that always emit gold <sim> then answer from observation."""
    rows = []
    for ex in examples:
        result = run(ex["gold_spec"])
        spec_json = json.dumps(ex["gold_spec"], separators=(",", ":"))
        if ex["task_type"] == "T1":
            final = json.dumps(
                {
                    "objects": [
                        {"id": o["id"], "position": o["position"], "velocity": o["velocity"]}
                        for o in result["objects"]
                    ]
                },
                separators=(",", ":"),
            )
        elif ex["task_type"] == "T2":
            final = f"{float(ex['gold_answer']):.6g}"
        else:
            final = str(ex["gold_answer"])
        rows.append(
            {
                "id": ex["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_AGENT + " /no_think"},
                    {"role": "user", "content": ex["text"]},
                    {"role": "assistant", "content": f"<sim>{spec_json}</sim>"},
                    {"role": "user", "content": build_observation_message(result)},
                    {"role": "assistant", "content": final},
                ],
            }
        )
    return rows


def label_calls_from_errors(
    gold_path: str,
    pred_path: str,
    out_path: str,
    tol: float = 0.05,
) -> list[dict[str, Any]]:
    """I5: label CALL if relative error > tol / incorrect."""
    from phil.eval.eval import score_example

    gold = {r["id"]: r for r in read_jsonl(gold_path)}
    preds = read_jsonl(pred_path)
    labels = []
    for pr in preds:
        g = gold[pr["id"]]
        sc = score_example(g, pr, tol=tol)
        label = "ANSWER" if sc["correct"] else "CALL"
        labels.append(
            {
                "id": g["id"],
                "rel_err": sc.get("rel_err"),
                "label": label,
                "correct": sc["correct"],
            }
        )
    write_jsonl(out_path, labels)
    return labels


def build_selective_sft(
    gold_path: str,
    labels_path: str,
    out_path: str,
) -> int:
    gold = {r["id"]: r for r in read_jsonl(gold_path)}
    labels = read_jsonl(labels_path)
    rows = []
    for lab in labels:
        ex = gold[lab["id"]]
        if lab["label"] == "ANSWER":
            from phil.models.prompting import format_sft_example

            rows.append(format_sft_example(ex))
        else:
            # CALL: gold sim then answer
            rows.extend(always_call_targets([ex]))
    write_jsonl(out_path, rows)
    return len(rows)


def infer_agent_hf(
    model_name: str,
    examples: list[dict[str, Any]],
    *,
    adapter_path: str | None = None,
    force_call: bool = False,
    theta: float | None = None,
    max_tokens: int = 512,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Two-pass agent: optionally emit <sim>, run it, then answer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    def generate(messages: list[dict[str, str]]) -> tuple[str, int, float | None]:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # optional p(<sim>) via first-token scores
        p_sim = None
        with torch.no_grad():
            if theta is not None:
                out_scores = model(**inputs)
                logits = out_scores.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                # approximate: probability mass on tokens that decode to '<' or '<sim'
                sim_ids = tokenizer.encode("<sim>", add_special_tokens=False)
                if sim_ids:
                    p_sim = float(probs[sim_ids[0]].item())
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        return text, int(gen.numel()), p_sim

    rows = []
    for ex in examples:
        messages = build_messages(ex, mode="agent", fewshot=False, enable_thinking=False)
        if force_call:
            # force first generation to be gold-style call request; still let model write spec
            messages[0]["content"] += " You must call the simulator first."
        text1, n1, p_sim = generate(messages)
        called, spec_str = decide_call_from_text(text1)
        if force_call and not called:
            # fallback: use whatever JSON-looking content, else mark invalid
            called = True
            spec_str = text1
        if theta is not None and p_sim is not None:
            called = p_sim > theta
            if not called:
                rows.append(
                    {
                        "id": ex["id"],
                        "raw_output": text1,
                        "final_answer": text1,
                        "called": False,
                        "spec_str": None,
                        "spec_valid": None,
                        "p_sim": p_sim,
                        "n_tokens": n1,
                        "system": f"selective_theta_{theta}",
                    }
                )
                continue
        if called and spec_str is not None:
            ok, result, err = run_sim_safe(spec_str)
            if ok and result is not None:
                messages = messages + [
                    {"role": "assistant", "content": f"<sim>{spec_str}</sim>"},
                    {"role": "user", "content": build_observation_message(result)},
                ]
                text2, n2, _ = generate(messages)
                rows.append(
                    {
                        "id": ex["id"],
                        "raw_output": text1 + "\n" + text2,
                        "final_answer": text2,
                        "called": True,
                        "spec_str": spec_str,
                        "spec_valid": True,
                        "p_sim": p_sim,
                        "n_tokens": n1 + n2,
                        "system": "always_call" if force_call else "selective",
                    }
                )
            else:
                rows.append(
                    {
                        "id": ex["id"],
                        "raw_output": text1,
                        "final_answer": text1,
                        "called": True,
                        "spec_str": spec_str,
                        "spec_valid": False,
                        "p_sim": p_sim,
                        "n_tokens": n1,
                        "error": err,
                        "system": "always_call" if force_call else "selective",
                    }
                )
        else:
            rows.append(
                {
                    "id": ex["id"],
                    "raw_output": text1,
                    "final_answer": text1,
                    "called": False,
                    "spec_str": None,
                    "spec_valid": None,
                    "p_sim": p_sim,
                    "n_tokens": n1,
                    "system": "llm_only",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("oracle")
    o.add_argument("--data", required=True)
    o.add_argument("--out", required=True)

    a = sub.add_parser("always-call-targets")
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)

    l = sub.add_parser("label-calls")
    l.add_argument("--gold", required=True)
    l.add_argument("--pred", required=True)
    l.add_argument("--out", required=True)

    s = sub.add_parser("selective-sft")
    s.add_argument("--gold", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)

    i = sub.add_parser("infer")
    i.add_argument("--model", required=True)
    i.add_argument("--data", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--adapter", default=None)
    i.add_argument("--force-call", action="store_true")
    i.add_argument("--theta", type=float, default=None)
    i.add_argument("--limit", type=int, default=None)

    args = p.parse_args(argv)
    if args.cmd == "oracle":
        rows = oracle_predictions(read_jsonl(args.data))
        write_jsonl(args.out, rows)
        print(f"wrote {len(rows)} -> {args.out}")
    elif args.cmd == "always-call-targets":
        rows = always_call_targets(read_jsonl(args.data))
        write_jsonl(args.out, rows)
        print(f"wrote {len(rows)} -> {args.out}")
    elif args.cmd == "label-calls":
        rows = label_calls_from_errors(args.gold, args.pred, args.out)
        n_call = sum(1 for r in rows if r["label"] == "CALL")
        print(f"wrote {len(rows)} labels ({n_call} CALL) -> {args.out}")
    elif args.cmd == "selective-sft":
        n = build_selective_sft(args.gold, args.labels, args.out)
        print(f"wrote {n} -> {args.out}")
    elif args.cmd == "infer":
        data = read_jsonl(args.data)
        if args.limit:
            data = data[: args.limit]
        rows = infer_agent_hf(
            args.model,
            data,
            adapter_path=args.adapter,
            force_call=args.force_call,
            theta=args.theta,
        )
        write_jsonl(args.out, rows)
        print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
