"""A small PyTorch Dataset for the JPEG-in-LMDB cache format."""

from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import lmdb
import numpy as np
from PIL import Image

from .cache import frame_key, read_metadata
from .layout import cache_path, read_json, video_keys


class LmdbFrameDataset:
    """Index individual cached RGB frames from a LeRobot v2.1 dataset.

    The class deliberately has no import-time Torch dependency. Install the ``examples`` extra
    and instantiate it in a PyTorch process. Each DataLoader worker owns its own bounded LMDB
    environment cache, which avoids sharing unsafe handles across process boundaries.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        cache_root: str | Path | None = None,
        video_key: str | list[str] | None = None,
        max_episodes: int | None = None,
        transform: Callable[[np.ndarray], Any] | None = None,
        env_cache_size: int = 32,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("LmdbFrameDataset requires PyTorch. Install `mtl-tool[examples]`.") from exc
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve() if cache_root else self.dataset_root / "lmdb"
        self.info = read_json(self.dataset_root / "meta" / "info.json")
        available = video_keys(self.info)
        requested = [video_key] if isinstance(video_key, str) else video_key
        self.video_keys = available if requested is None else list(requested)
        unknown = set(self.video_keys) - set(available)
        if unknown:
            raise ValueError(f"Unknown video keys: {sorted(unknown)}; available: {available}")
        self.chunk_size = int(self.info["chunks_size"])
        episode_count = int(self.info["total_episodes"])
        if max_episodes is not None:
            episode_count = min(episode_count, int(max_episodes))
        self.transform = transform
        self.env_cache_size = max(0, int(env_cache_size))
        self._envs: OrderedDict[str, lmdb.Environment] = OrderedDict()
        self.samples: list[tuple[Path, int, int, str]] = []

        for episode_index in range(episode_count):
            for key in self.video_keys:
                path = cache_path(self.cache_root, episode_index, self.chunk_size, key)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing cache {path}. Build it with `mtl_tool.lerobot_lmdb_build {self.dataset_root}`."
                    )
                frame_count = int(read_metadata(path)["shape"][0])
                self.samples.extend((path, episode_index, frame_index, key) for frame_index in range(frame_count))

    def __len__(self) -> int:
        return len(self.samples)

    def _environment(self, path: Path) -> lmdb.Environment:
        key = str(path)
        cached = self._envs.get(key)
        if cached is not None:
            self._envs.move_to_end(key)
            return cached
        env = lmdb.open(key, readonly=True, lock=False, readahead=False, subdir=True, max_readers=2048)
        if self.env_cache_size > 0:
            self._envs[key] = env
            while len(self._envs) > self.env_cache_size:
                _, old = self._envs.popitem(last=False)
                old.close()
        return env

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded in __init__
            raise ImportError("LmdbFrameDataset requires PyTorch.") from exc
        path, episode_index, frame_index, key = self.samples[index]
        env = self._environment(path)
        try:
            with env.begin(buffers=True) as transaction:
                payload = transaction.get(frame_key(frame_index))
                if payload is None:
                    raise KeyError(f"Missing frame {frame_index} in {path}")
                with Image.open(io.BytesIO(bytes(payload))) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
        finally:
            if self.env_cache_size <= 0:
                env.close()
        image_value = self.transform(rgb) if self.transform else torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return {
            "image": image_value,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "video_key": key,
        }

    def close(self) -> None:
        while self._envs:
            _, env = self._envs.popitem(last=False)
            env.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_envs"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
