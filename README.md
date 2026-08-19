# mtl_tool

为 [LeRobot](https://github.com/huggingface/lerobot) episode 数据集提供快速、可随机读取的 JPEG-in-LMDB 帧缓存，以及 RobotWin 风格 LeRobot v3 到 v2.1 的转换工具。

它来自本项目的训练数据路径，LMDB 格式与原有 loader 直接兼容：`__meta__` 保存元数据，帧以 `frame/00000000` 形式保存 RGB JPEG。缓存保留原始 RGB 尺寸；训练时再 resize/augment，避免把某个模型的图像几何固化进数据集。

## 功能

- 自动扫描一个 v2.1 数据集，或其下的多个数据集。
- 默认 `auto` decoder：优先使用更快的 OpenCV (`cv2`)，失败时自动退回到 `ffmpeg`。
- 显式选择 `cv2` 或 `ffmpeg`，并记录实际 decoder、源 FPS 和采样 frame id 到 LMDB 元数据。
- 可设置 JPEG 质量（默认 `95`）和 chroma subsampling；写入使用原子目录替换，半途失败不会留下可误读的缓存。
- 示例 PyTorch `DataLoader`：每个 worker 都有自己的 LRU LMDB 句柄缓存。
- 转换 v3 task-sharded 布局为 v2.1 episode parquet/video 布局，然后可直接构建 LMDB。

## 安装

Python 3.10+：

```bash
git clone https://github.com/<your-org>/mtl_tool.git
cd mtl_tool
pip install -e '.[cv2,convert,examples]'
```

`ffmpeg`/`ffprobe` 仅在使用 `--decoder ffmpeg`、自动 decoder 回退，或执行 v3 转换时需要。例如 Ubuntu：`sudo apt install ffmpeg`。

## 1. 从 LeRobot v2.1 视频构建 LMDB

最常用的命令：

```bash
mtl_tool.lerobot_lmdb_build /data/lerobot/my_dataset \
  --decoder auto \
  --jpeg-quality 95 \
  --jobs 8
```

默认输出为 `/data/lerobot/my_dataset/lmdb`。目录结构如下，和本项目的训练 loader 兼容：

```text
lmdb/
└── chunk-000/
    └── observation.images.top/
        └── episode_000000.frames_jpeg.lmdb/
            ├── data.mdb
            └── lock.mdb
```

常见用法：

```bash
# 先确认会处理哪些视频，不写任何数据
mtl_tool.lerobot_lmdb_build /data/lerobot --dry-run

# 把多数据集根目录的缓存统一写到高速本地盘；保留原有子目录结构
mtl_tool.lerobot_lmdb_build /data/lerobot --output-root /scratch/lerobot-lmdb --jobs 16

# 明确用 OpenCV（若不存在或损坏的视频应立即失败）
mtl_tool.lerobot_lmdb_build /data/lerobot/my_dataset --decoder cv2

# 更小的缓存；quality=90, 4:2:0。视觉训练通常建议先比较指标再降质量。
mtl_tool.lerobot_lmdb_build /data/lerobot/my_dataset --jpeg-quality 90 --jpeg-subsampling 2

# 只构建一个视角 / 试跑少量 episode
mtl_tool.lerobot_lmdb_build /data/lerobot/my_dataset \
  --video-key observation.images.top --max-episodes 10

# 重建已有缓存；坏视频记录后继续
mtl_tool.lerobot_lmdb_build /data/lerobot/my_dataset --overwrite --skip-bad-videos
```

默认不使用 `--target-fps`，因此 cache key 与 episode 的原始 frame index 一一对应，最适合现有 LeRobot 训练代码。使用 `--target-fps` 时，缓存中的 key 变为连续的采样序号；原始 frame id 在 `__meta__.source_frame_indices` 中保存，训练代码需要据此映射，不能直接把原始 frame index 当作 LMDB key。

## 2. DataLoader 示例

先安装 `examples` extra 并完成构建：

```bash
python examples/dataloader.py /data/lerobot/my_dataset \
  --batch-size 64 --workers 8 --batches 3
```

batch 的 `image` 是 `uint8` 的 `NCHW RGB` Tensor，另有 `episode_index`、`frame_index` 和 `video_key`。可以在模型前归一化，也可以给 `LmdbFrameDataset(..., transform=...)` 传入自己的 NumPy RGB transform。

```python
from torch.utils.data import DataLoader
from mtl_tool import LmdbFrameDataset, lerobot_lmdb_build

dataset = LmdbFrameDataset("/data/lerobot/my_dataset")
loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, persistent_workers=True)
for batch in loader:
    images = batch["image"].float().div_(255)  # N, C, H, W
    # train(images)
```

同一能力也可直接作为 Python API 调用：

```python
from mtl_tool import convert, lerobot_lmdb_build

convert("/data/robotwin_v3", "/data/robotwin_v21", jobs=8)
lerobot_lmdb_build("/data/robotwin_v21", decoder="auto", jpeg_quality=95, jobs=8)
```

## 3. LeRobot v3 → v2.1

此转换器面向 task-sharded v3 数据：共享 parquet / MP4 文件加 `meta/episodes/*.parquet` 的时间范围。输出为 v2.1 的逐 episode parquet、逐 episode MP4 和 `meta/*.jsonl`：

```bash
mtl_tool.convert /data/robotwin_v3 /data/robotwin_v21 \
  --jobs 8 --ffmpeg-threads 1

# 验证完转换后，再构建 LMDB
mtl_tool.lerobot_lmdb_build /data/robotwin_v21 --decoder auto --jobs 8
```

转换器不会覆盖输出目录，除非显式传入 `--overwrite`。`--max-datasets` 和 `--max-episodes` 可用于先做小规模验证。视频裁剪会重编码（默认 `libx264`, CRF 18）；可用 `--video-codec`、`--video-crf` 和 `--video-preset` 控制。

## 兼容性与限制

- 输入 v2.1 需要 `meta/info.json`、`videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4`，并在 metadata 中使用 `chunks_size` 与 video features。
- v3 转换器假设源路径为 `data/chunk-XXX/file-XXX.parquet`、`videos/<video_key>/chunk-XXX/file-XXX.mp4`，并包含 `meta/episodes/*.parquet`、`meta/tasks.parquet`。这是 RobotWin 任务分片导出的布局。
- LMDB 是“解码视频后再 JPEG 编码”的有损缓存。质量 95 / 4:4:4 是默认的保真取向；请针对你的模型验证低质量或 4:2:0 的影响。
- 多进程/多机训练可并发只读同一 cache；每个 DataLoader worker 会独立打开 LMDB，避免跨进程复用句柄。

## 开发

```bash
pip install -e '.[cv2,convert,examples,dev]'
pytest -q
ruff check src tests examples
```

本仓库不包含数据集、模型权重或 ffmpeg 二进制文件。
