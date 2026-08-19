"""Stable JPEG-frame LMDB format used by this project and its training loader."""

from __future__ import annotations

import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import lmdb
import numpy as np
from PIL import Image

META_KEY = b"__meta__"


def frame_key(frame_index: int) -> bytes:
    """Return the cache key used by the original project implementation."""
    return f"frame/{int(frame_index):08d}".encode("ascii")


def encode_jpeg_rgb(frame: np.ndarray, quality: int, subsampling: int = 0) -> bytes:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected uint8 HWC RGB frame, got shape={frame.shape}, dtype={frame.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(frame).save(
        buffer, format="JPEG", quality=int(quality), subsampling=int(subsampling)
    )
    return buffer.getvalue()


def decode_jpeg_rgb(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def read_metadata(cache_dir: str | Path) -> dict[str, Any]:
    env = lmdb.open(str(cache_dir), readonly=True, lock=False, readahead=False, subdir=True)
    try:
        with env.begin(buffers=True) as transaction:
            raw = transaction.get(META_KEY)
            if raw is None:
                raise RuntimeError(f"Missing {META_KEY!r} in LMDB cache: {cache_dir}")
            return json.loads(bytes(raw).decode("utf-8"))
    finally:
        env.close()


def read_frame(cache_dir: str | Path, frame_index: int) -> np.ndarray:
    env = lmdb.open(str(cache_dir), readonly=True, lock=False, readahead=False, subdir=True)
    try:
        with env.begin(buffers=True) as transaction:
            payload = transaction.get(frame_key(frame_index))
            if payload is None:
                raise KeyError(f"Frame {frame_index} is not present in {cache_dir}")
            return decode_jpeg_rgb(bytes(payload))
    finally:
        env.close()


def iter_frames(cache_dir: str | Path) -> Iterator[np.ndarray]:
    metadata = read_metadata(cache_dir)
    for frame_index in range(int(metadata["shape"][0])):
        yield read_frame(cache_dir, frame_index)


def _initial_map_size(frames: np.ndarray) -> int:
    # JPEG usually occupies much less than raw RGB, but this leaves enough headroom for any image.
    raw_size = int(frames.nbytes)
    return max(raw_size * 2 + 64 * 1024 * 1024, 256 * 1024 * 1024)


def write_frames(
    frames: np.ndarray,
    cache_dir: str | Path,
    *,
    quality: int = 95,
    subsampling: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write RGB frames in the compatible JPEG-in-LMDB format.

    A sibling temporary directory is used, so a cancelled or failed build never leaves a
    partially readable cache at the final path.
    """
    if not 1 <= int(quality) <= 100:
        raise ValueError("quality must be in [1, 100]")
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 NHWC RGB frames, got shape={frames.shape}, dtype={frames.dtype}")
    if len(frames) == 0:
        raise ValueError("Cannot create an empty frame cache")

    cache_dir = Path(cache_dir)
    temporary = cache_dir.with_name(f"{cache_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "format": "jpeg",
        "quality": int(quality),
        "subsampling": int(subsampling),
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layout": "frame/{index:08d}",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    env = lmdb.open(str(temporary), map_size=_initial_map_size(frames), subdir=True, meminit=False)
    try:
        encoded_bytes = 0
        with env.begin(write=True) as transaction:
            transaction.put(META_KEY, json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        for start in range(0, len(frames), 128):
            encoded = [encode_jpeg_rgb(frame, quality, subsampling) for frame in frames[start : start + 128]]
            while True:
                try:
                    with env.begin(write=True) as transaction:
                        for offset, payload in enumerate(encoded):
                            transaction.put(frame_key(start + offset), payload)
                    encoded_bytes += sum(map(len, encoded))
                    break
                except lmdb.MapFullError:
                    env.set_mapsize(env.info()["map_size"] * 2)
        env.sync()
    finally:
        env.close()

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    temporary.rename(cache_dir)
    metadata["jpeg_bytes"] = encoded_bytes
    return metadata
