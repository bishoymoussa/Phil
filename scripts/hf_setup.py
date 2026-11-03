"""Create public Hugging Face dataset/model repos for PhiLLM."""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi, create_repo


REPOS = [
    ("dataset", "bishoygaloaa/phillm-data"),
    ("dataset", "bishoygaloaa/phillm-results"),
    ("model", "bishoygaloaa/phillm-qwen3.5-2b-lora"),
    ("model", "bishoygaloaa/phillm-qwen3.5-2b-probeguided"),
    ("model", "bishoygaloaa/phillm-qwen3.5-2b-selective"),
    ("space", "bishoygaloaa/phillm-demo"),
]


def main() -> None:
    api = HfApi()
    me = api.whoami()
    print("HF user:", me["name"])
    for kind, repo_id in REPOS:
        repo_type = {"dataset": "dataset", "model": "model", "space": "space"}[kind]
        try:
            url = create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True, space_sdk="gradio" if kind == "space" else None)
            print(f"ok {repo_type}: {repo_id} -> {url}")
        except Exception as e:  # noqa: BLE001
            print(f"fail {repo_id}: {e}")


if __name__ == "__main__":
    main()
