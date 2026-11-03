"""Prompt construction for PhiLLM tasks."""

from __future__ import annotations

from typing import Any


SYSTEM_DIRECT = (
    "You are PhiLLM, a careful physics assistant. "
    "Solve Newtonian point-mass problems exactly. "
    "Use SI units. Do not call tools unless asked. "
    "For state prediction, reply with JSON only."
)

SYSTEM_AGENT = (
    "You are PhiLLM, a careful physics assistant with access to a simulator tool. "
    "If you need the simulator, emit exactly one block:\n"
    "<sim>{...JSON spec...}</sim>\n"
    "Otherwise answer directly. Spec schema: "
    "{dim, dt, horizon, objects:[{id,mass,position,velocity,force}], walls?, collisions?}."
)

FEWSHOT_T1 = """Example:
Time step Δt = 0.1 s. Predict the state after 1 steps.
Object o0 has mass 2 kg, position 0 m, velocity 1 m/s, and constant force 0 N.
Return JSON with keys objects (list of {id, position, velocity}), using SI units (m, m/s).
Answer:
{"objects":[{"id":"o0","position":[0.1],"velocity":[1.0]}]}
"""


def build_messages(
    example: dict[str, Any],
    *,
    mode: str = "direct",
    fewshot: bool = True,
    enable_thinking: bool = False,
) -> list[dict[str, str]]:
    system = SYSTEM_AGENT if mode == "agent" else SYSTEM_DIRECT
    user = example["text"]
    if fewshot and example.get("task_type") == "T1":
        user = FEWSHOT_T1 + "\nNow solve:\n" + user
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Qwen3.5 thinking control is often done via chat template kwargs; keep flag in content hint
    if not enable_thinking:
        messages[0]["content"] += " /no_think"
    return messages


def format_sft_example(example: dict[str, Any]) -> dict[str, str]:
    """Build supervised fine-tuning prompt/completion pair."""
    messages = build_messages(example, mode="direct", fewshot=False, enable_thinking=False)
    ans = example["gold_answer"]
    if example["task_type"] == "T1":
        import json

        completion = json.dumps(ans, separators=(",", ":"))
    elif example["task_type"] == "T2":
        completion = f"{float(ans):.6g}"
    else:
        completion = str(ans)
    return {
        "messages": messages + [{"role": "assistant", "content": completion}],
        "id": example["id"],
    }
