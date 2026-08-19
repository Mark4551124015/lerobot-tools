from __future__ import annotations

import numpy as np

from mtl_tool.cache import frame_key, read_frame, read_metadata, write_frames


def test_jpeg_lmdb_round_trip_and_compatible_keys(tmp_path):
    frames = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    cache = tmp_path / "episode_000000.frames_jpeg.lmdb"

    result = write_frames(frames, cache, quality=95, extra_metadata={"decoder": "cv2"})

    assert (cache / "data.mdb").is_file()
    assert frame_key(7) == b"frame/00000007"
    assert result["format"] == "jpeg"
    metadata = read_metadata(cache)
    assert metadata["shape"] == [3, 8, 10, 3]
    assert metadata["decoder"] == "cv2"
    restored = read_frame(cache, 1)
    assert restored.shape == (8, 10, 3)
    assert restored.dtype == np.uint8
    assert restored[..., 0].mean() > 245
