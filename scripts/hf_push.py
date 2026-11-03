"""Push local artifacts to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--path-in-repo", default=None)
    p.add_argument("--commit-message", default="PhiLLM upload")
    args = p.parse_args(argv)
    api = HfApi()
    info = upload_folder(
        folder_path=args.folder,
        repo_id=args.repo,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )
    print(info)


if __name__ == "__main__":
    main()
