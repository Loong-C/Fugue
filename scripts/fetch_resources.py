from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@dataclass(frozen=True)
class GitResource:
    name: str
    url: str
    target: Path
    branch: str | None = None


GIT_RESOURCES = [
    GitResource(
        name="bach-wtc-fugues",
        url="https://github.com/humdrum-tools/bach-wtc-fugues.git",
        target=RAW / "humdrum" / "bach-wtc-fugues",
    ),
    GitResource(
        name="bach-wtc",
        url="https://github.com/humdrum-tools/bach-wtc.git",
        target=RAW / "humdrum" / "bach-wtc",
        branch="main",
    ),
    GitResource(
        name="jsb-chorales-dataset",
        url="https://github.com/czhuang/JSB-Chorales-dataset.git",
        target=RAW / "jsb-chorales-dataset",
    ),
]


MODEL_URLS = {
    "coconet-checkpoint.zip": "https://download.magenta.tensorflow.org/models/coconet/checkpoint.zip",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_or_update(resource: GitResource) -> None:
    resource.target.parent.mkdir(parents=True, exist_ok=True)
    if (resource.target / ".git").exists():
        run(["git", "fetch", "--depth", "1", "origin"], cwd=resource.target)
        branch = resource.branch or "master"
        run(["git", "checkout", branch], cwd=resource.target)
        run(["git", "pull", "--ff-only", "--depth", "1", "origin", branch], cwd=resource.target)
        return

    cmd = ["git", "clone", "--depth", "1"]
    if resource.branch is not None:
        cmd.extend(["--branch", resource.branch])
    cmd.extend([resource.url, str(resource.target)])
    run(cmd)


def download_model(name: str, url: str) -> None:
    target = RAW / "models" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Already exists: {target}")
    else:
        print(f"Downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)

    if target.suffix == ".zip":
        out_dir = target.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(out_dir)
        print(f"Extracted to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch public research resources for Fugue.")
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also download larger pretrained model checkpoints.",
    )
    args = parser.parse_args()

    for resource in GIT_RESOURCES:
        clone_or_update(resource)

    if args.include_models:
        for name, url in MODEL_URLS.items():
            download_model(name, url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

