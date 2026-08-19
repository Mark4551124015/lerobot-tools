"""Shared knowledge of the LeRobot v2.1 on-disk layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_v21_datasets(root: Path) -> list[Path]:
    """Return one v2.1 root, or every nested root with ``meta/info.json``."""
    root = root.expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted({path.parent.parent.resolve() for path in root.rglob("meta/info.json")})


def video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def cache_path(cache_root: Path, episode_index: int, chunk_size: int, video_key: str) -> Path:
    chunk = int(episode_index) // int(chunk_size)
    return (
        cache_root
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{int(episode_index):06d}.frames_jpeg.lmdb"
    )


def video_path(dataset_root: Path, episode_index: int, chunk_size: int, video_key: str) -> Path:
    chunk = int(episode_index) // int(chunk_size)
    return (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{int(episode_index):06d}.mp4"
    )


def resolve_cache_root(
    dataset_root: Path, input_root: Path, output_root: Path | None, num_datasets: int
) -> Path:
    """Mirror nested datasets when a common external output root is requested."""
    if output_root is None:
        return dataset_root / "lmdb"
    if num_datasets == 1:
        return output_root
    try:
        return output_root / dataset_root.relative_to(input_root) / "lmdb"
    except ValueError:
        return output_root / dataset_root.name / "lmdb"
