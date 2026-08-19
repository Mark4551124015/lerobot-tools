#!/usr/bin/env python3
"""Minimal, runnable PyTorch DataLoader example for a built LMDB cache."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from lerobot_tools import LmdbFrameDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--video-key", action="append", default=None)
    args = parser.parse_args()

    dataset = LmdbFrameDataset(
        args.dataset_root, cache_root=args.cache_root, video_key=args.video_key
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"frames={len(dataset):,}, batches≈{len(loader):,}")
    for batch_index, batch in enumerate(loader):
        # uint8 NCHW RGB; normalise or augment here as required by your model.
        print(
            f"batch={batch_index} image={tuple(batch['image'].shape)} "
            f"episodes={batch['episode_index'][:3].tolist()} keys={batch['video_key'][:3]}"
        )
        if batch_index + 1 >= args.batches:
            break
    dataset.close()


if __name__ == "__main__":
    main()
