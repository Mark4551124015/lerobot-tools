"""Convert LeRobot v3 task-sharded datasets into v2.1 episode-oriented datasets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .layout import discover_v21_datasets, read_json, video_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 task-sharded roots into LeRobot v2.1 episode files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source_root", type=Path, help="A v3 dataset root or parent containing v3 datasets.")
    parser.add_argument("dest_root", type=Path, help="New v2.1 output root; nested inputs are mirrored here.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Episodes per v2.1 chunk; defaults to source chunks_size.")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel ffmpeg trim jobs.")
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace each existing destination dataset.")
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _find_one_parquet(directory: Path) -> Path:
    candidates = sorted(directory.rglob("*.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one parquet below {directory}, found {len(candidates)}")
    return candidates[0]


def _load_tasks(path: Path) -> OrderedDict[int, str]:
    table = pd.read_parquet(path).reset_index().rename(columns={"index": "task"})
    if not {"task", "task_index"}.issubset(table.columns):
        raise ValueError(f"Unexpected tasks.parquet columns: {list(table.columns)}")
    return OrderedDict(
        (int(row.task_index), str(row.task)) for row in table.sort_values("task_index").itertuples(index=False)
    )


def _tasks_from_value(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def _first_value(value: Any) -> Any:
    """Handle scalar, list, NumPy array, and pandas values from v3 statistics."""
    normalized = _jsonable(value)
    return normalized[0] if isinstance(normalized, list) else normalized


def _stat_payload(row: pd.Series) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for column, value in row.items():
        if not column.startswith("stats/"):
            continue
        _, feature, statistic = column.split("/", 2)
        output.setdefault(feature, {})[statistic] = _jsonable(value)
    return output


def _trim_video(job: dict[str, Any], codec: str, crf: int, preset: str, threads: int) -> None:
    duration = float(job["end"]) - float(job["start"])
    if duration <= 0:
        raise ValueError(f"Invalid video duration for {job['source']}")
    job["destination"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", str(max(1, threads)),
        "-ss", f"{job['start']:.6f}", "-t", f"{duration:.6f}", "-i", str(job["source"]),
        "-an", "-c:v", codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", str(job["destination"]),
    ]
    subprocess.run(command, check=True)


def _empty_or_create(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {path}; use --overwrite to replace it")
        shutil.rmtree(path)
    (path / "meta").mkdir(parents=True)


def _output_info(template: dict[str, Any], episodes: int, frames: int, tasks: int, views: int, chunk_size: int, codec: str) -> dict[str, Any]:
    output = copy.deepcopy(template)
    output.update(
        {
            "codebase_version": "v2.1", "total_episodes": episodes, "total_frames": frames,
            "total_tasks": tasks, "total_videos": episodes * views,
            "total_chunks": math.ceil(episodes / chunk_size) if episodes else 0,
            "chunks_size": chunk_size, "splits": {"train": f"0:{episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
    )
    for feature in output.get("features", {}).values():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            feature.setdefault("info", {})["video.codec"] = codec
    return output


def _convert_dataset(source: Path, destination: Path, args: argparse.Namespace) -> None:
    _empty_or_create(destination, args.overwrite)
    info = read_json(source / "meta" / "info.json")
    chunk_size = int(args.chunk_size or info["chunks_size"])
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    episode_table = pd.read_parquet(_find_one_parquet(source / "meta" / "episodes"))
    episode_table = episode_table.sort_values("episode_index").reset_index(drop=True)
    if args.max_episodes is not None:
        episode_table = episode_table.iloc[: args.max_episodes]
    task_by_id = _load_tasks(source / "meta" / "tasks.parquet")
    keys = video_keys(info)
    source_data: dict[tuple[int, int], pd.DataFrame] = {}
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    trim_jobs: list[dict[str, Any]] = []
    total_frames = 0

    for _, row in tqdm(episode_table.iterrows(), total=len(episode_table), desc=f"convert {source.name}"):
        episode_index, length = int(row["episode_index"]), int(row["length"])
        episode_chunk = episode_index // chunk_size
        data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if data_key not in source_data:
            source_data[data_key] = pd.read_parquet(
                source / "data" / f"chunk-{data_key[0]:03d}" / f"file-{data_key[1]:03d}.parquet"
            )
        episode = source_data[data_key].iloc[int(row["dataset_from_index"]):int(row["dataset_to_index"])].copy()
        episode.reset_index(drop=True, inplace=True)
        episode["episode_index"] = episode_index
        for column in ("frame_index", "index"):
            if column in episode:
                episode[column] = range(len(episode))
        output_data = destination / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        output_data.parent.mkdir(parents=True, exist_ok=True)
        episode.to_parquet(output_data, index=False)

        tasks = _tasks_from_value(row.get("tasks"))
        if not tasks:
            task_index = int(_first_value(row["stats/task_index/min"]))
            tasks = [task_by_id[task_index]]
        output_episodes.append({"episode_index": episode_index, "tasks": tasks, "length": length})
        output_stats.append({"episode_index": episode_index, "stats": _stat_payload(row)})
        total_frames += length
        for key in keys:
            trim_jobs.append(
                {
                    "source": source / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4",
                    "destination": destination / "videos" / f"chunk-{episode_chunk:03d}" / key / f"episode_{episode_index:06d}.mp4",
                    "start": float(row[f"videos/{key}/from_timestamp"]),
                    "end": float(row[f"videos/{key}/to_timestamp"]),
                }
            )

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_trim_video, job, args.video_codec, args.video_crf, args.video_preset, args.ffmpeg_threads) for job in trim_jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="trim videos"):
            future.result()
    _write_jsonl(destination / "meta" / "tasks.jsonl", [{"task_index": key, "task": value} for key, value in task_by_id.items()])
    _write_jsonl(destination / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(destination / "meta" / "episodes_stats.jsonl", output_stats)
    _write_json(destination / "meta" / "info.json", _output_info(info, len(output_episodes), total_frames, len(task_by_id), len(keys), chunk_size, args.video_codec))


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    source_root, dest_root = args.source_root.expanduser().resolve(), args.dest_root.expanduser().resolve()
    datasets = discover_v21_datasets(source_root)
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
    if not datasets:
        raise FileNotFoundError(f"No LeRobot source dataset found below {source_root}")
    for source in datasets:
        relative = Path(".") if source == source_root else source.relative_to(source_root)
        destination = dest_root / relative
        print(f"{source} -> {destination}")
        _convert_dataset(source, destination, args)


if __name__ == "__main__":
    main()
