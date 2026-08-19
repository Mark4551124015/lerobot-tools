"""Command line builder for fast LeRobot JPEG-frame LMDB caches."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .cache import read_metadata, write_frames
from .layout import cache_path, discover_v21_datasets, read_json, resolve_cache_root, video_keys, video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JPEG-in-LMDB frame caches for LeRobot v2.1 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_root", type=Path, help="One v2.1 dataset root or a parent containing datasets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Cache root. By default caches are written to <dataset>/lmdb; nested datasets are mirrored.",
    )
    parser.add_argument(
        "--decoder",
        "--video-backend",
        dest="decoder",
        choices=("auto", "cv2", "ffmpeg"),
        default="auto",
        help="Video decoding backend. auto tries OpenCV first (fast) then ffmpeg.",
    )
    parser.add_argument("--jpeg-quality", "--frame-jpeg-quality", dest="jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--jpeg-subsampling",
        "--frame-jpeg-subsampling",
        dest="jpeg_subsampling",
        choices=(0, 1, 2),
        type=int,
        default=0,
        help="Pillow JPEG subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0.",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Optionally keep an evenly sampled frame subset.")
    parser.add_argument("--jobs", "--num-workers", dest="jobs", type=int, default=1, help="Parallel video jobs.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Only cache the first N episodes in each dataset.")
    parser.add_argument("--video-key", action="append", default=None, help="Only cache this video feature; repeatable.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing caches.")
    parser.add_argument("--skip-bad-videos", action="store_true", help="Record failures and continue processing.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without decoding or writing.")
    return parser.parse_args()


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _ffprobe(path: Path) -> tuple[int, int, float]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", str(path),
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    stream = json.loads(output)["streams"][0]
    fps = _fraction(stream.get("avg_frame_rate")) or _fraction(stream.get("r_frame_rate"))
    return int(stream["width"]), int(stream["height"]), fps


def _decode_cv2(path: Path) -> tuple[np.ndarray, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed; run `pip install mtl-tool[cv2]`.") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"OpenCV decoded zero frames from {path}")
    return np.stack(frames), fps


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, float]:
    width, height, fps = _ffprobe(path)
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    one_frame = width * height * 3
    usable = (len(result.stdout) // one_frame) * one_frame
    if usable == 0:
        raise RuntimeError(f"ffmpeg decoded zero frames from {path}")
    frames = np.frombuffer(result.stdout[:usable], dtype=np.uint8).reshape(-1, height, width, 3).copy()
    return frames, fps


def decode_video(path: Path, decoder: str) -> tuple[np.ndarray, float, str]:
    """Decode with the requested backend; automatic mode intentionally prefers cv2."""
    errors: list[str] = []
    backends = ("cv2", "ffmpeg") if decoder == "auto" else (decoder,)
    for backend in backends:
        try:
            frames, fps = _decode_cv2(path) if backend == "cv2" else _decode_ffmpeg(path)
            return frames, fps, backend
        except Exception as exc:  # Preserve both failure causes for useful diagnostics.
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(f"Unable to decode {path}. " + " | ".join(errors))


def sample_frame_ids(num_frames: int, source_fps: float, target_fps: float | None) -> np.ndarray:
    if target_fps is None or target_fps <= 0 or source_fps <= 0 or target_fps >= source_fps:
        return np.arange(num_frames, dtype=np.int64)
    indices = np.arange(0.0, num_frames, source_fps / target_fps, dtype=np.float64)
    return np.unique(np.clip(np.round(indices).astype(np.int64), 0, num_frames - 1))


def _episode_count(dataset_root: Path, info: dict[str, Any]) -> int:
    declared = int(info["total_episodes"])
    episodes = dataset_root / "meta" / "episodes.jsonl"
    if not episodes.exists():
        return declared
    with episodes.open("r", encoding="utf-8") as handle:
        actual = sum(bool(line.strip()) for line in handle)
    return min(declared, actual)


def _cache_is_compatible(path: Path) -> bool:
    if not (path / "data.mdb").is_file():
        return False
    try:
        metadata = read_metadata(path)
        return metadata.get("format") == "jpeg" and int(metadata["shape"][0]) > 0
    except Exception:
        return False


def _build_one(task: dict[str, Any]) -> dict[str, Any]:
    frames, detected_fps, used_decoder = decode_video(task["video_path"], task["decoder"])
    source_fps = detected_fps if detected_fps > 0 else float(task["info_fps"])
    source_indices = sample_frame_ids(len(frames), source_fps, task["target_fps"])
    selected = frames[source_indices]
    metadata = write_frames(
        selected,
        task["cache_path"],
        quality=task["jpeg_quality"],
        subsampling=task["jpeg_subsampling"],
        extra_metadata={
            "source_video": str(task["video_path"]),
            "source_fps": source_fps,
            "target_fps": task["target_fps"],
            "source_frame_indices": source_indices.tolist(),
            "decoder": used_decoder,
        },
    )
    return {**task, "frames": len(selected), "decoder_used": used_decoder, "metadata": metadata}


def lerobot_lmdb_build(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    decoder: str = "auto",
    jpeg_quality: int = 95,
    jpeg_subsampling: int = 0,
    target_fps: float | None = None,
    jobs: int = 1,
    video_key: str | list[str] | None = None,
    max_episodes: int | None = None,
    overwrite: bool = False,
    skip_bad_videos: bool = False,
    dry_run: bool = False,
) -> None:
    """Build compatible LeRobot JPEG-in-LMDB caches from Python.

    This is the programmatic counterpart to the ``mtl_tool.lerobot_lmdb_build`` command.
    ``video_key`` may be one video feature or a list of features.
    """
    keys = [video_key] if isinstance(video_key, str) else video_key
    args = argparse.Namespace(
        dataset_root=Path(dataset_root),
        output_root=Path(output_root) if output_root is not None else None,
        decoder=decoder,
        jpeg_quality=jpeg_quality,
        jpeg_subsampling=jpeg_subsampling,
        target_fps=target_fps,
        jobs=jobs,
        video_key=keys,
        max_episodes=max_episodes,
        overwrite=overwrite,
        skip_bad_videos=skip_bad_videos,
        dry_run=dry_run,
    )
    _run_build(args)


def _run_build(args: argparse.Namespace) -> None:
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    input_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    datasets = discover_v21_datasets(input_root)
    if not datasets:
        raise FileNotFoundError(f"No LeRobot v2.1 dataset found below {input_root}; expected meta/info.json")

    tasks: list[dict[str, Any]] = []
    skipped = 0
    for dataset_root in datasets:
        info = read_json(dataset_root / "meta" / "info.json")
        chunk_size = int(info["chunks_size"])
        keys = video_keys(info)
        if args.video_key:
            unknown = set(args.video_key) - set(keys)
            if unknown:
                raise ValueError(f"{dataset_root}: unknown --video-key values: {sorted(unknown)}")
            keys = [key for key in keys if key in args.video_key]
        cache_root = resolve_cache_root(dataset_root, input_root, output_root, len(datasets))
        count = _episode_count(dataset_root, info)
        if args.max_episodes is not None:
            count = min(count, args.max_episodes)
        print(f"dataset={dataset_root}\n  cache_root={cache_root}\n  episodes={count}, video_keys={keys}")
        for episode_index in range(count):
            for key in keys:
                source = video_path(dataset_root, episode_index, chunk_size, key)
                destination = cache_path(cache_root, episode_index, chunk_size, key)
                if not source.exists():
                    message = f"Missing video: {source}"
                    if args.skip_bad_videos:
                        print(f"[skip] {message}")
                        skipped += 1
                        continue
                    raise FileNotFoundError(message)
                if not args.overwrite and _cache_is_compatible(destination):
                    skipped += 1
                    continue
                tasks.append(
                    {
                        "video_path": source, "cache_path": destination, "decoder": args.decoder,
                        "info_fps": float(info.get("fps", 0.0)), "target_fps": args.target_fps,
                        "jpeg_quality": args.jpeg_quality, "jpeg_subsampling": args.jpeg_subsampling,
                    }
                )

    print(f"planned={len(tasks)}  existing_or_skipped={skipped}  decoder={args.decoder}  jpeg=Q{args.jpeg_quality}")
    if args.dry_run or not tasks:
        return

    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor, tqdm(total=len(tasks), desc="LMDB caches") as progress:
        futures = {executor.submit(_build_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                progress.set_postfix_str(f"{result['decoder_used']} {result['frames']} frames")
            except Exception as exc:
                failures.append((task["video_path"], exc))
                if not args.skip_bad_videos:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"Failed while caching {task['video_path']}: {exc}") from exc
            finally:
                progress.update(1)
    if failures:
        print(f"Completed with {len(failures)} skipped videos:")
        for path, error in failures:
            print(f"  {path}: {error}")


def main() -> None:
    _run_build(parse_args())


if __name__ == "__main__":
    main()
