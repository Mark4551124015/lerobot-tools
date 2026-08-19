# lerobot-tools

LeRobot 数据集转换与 LMDB I/O 加速工具集。它包含两个可组合的能力：

1. 将 RobotWin 风格、task-sharded 的 LeRobot v3 数据转换为 episode-oriented 的 v2.1 数据；
2. 将 v2.1 视频预解码为 JPEG-in-LMDB 帧缓存，降低训练时的随机视频 I/O 开销。

> Powered by MTL Lab — Zexin Feng

## 为什么使用它

- `auto` decoder 优先使用通常更快的 OpenCV/cv2，失败时自动回退 ffmpeg。
- JPEG 质量、色度采样、并行度、视角和处理 episode 范围均可控制。
- 写入使用临时目录和原子替换：中断不会留下半成品 cache。
- LMDB 帧 key 与本项目现有训练 loader 兼容：`frame/00000000`，并以 `__meta__` 保存元数据。
- 缓存保留原始 RGB 尺寸；resize、crop 与 augmentation 应在训练/推理阶段完成。
- 自带可运行的 PyTorch DataLoader 示例，每个 worker 独立维护 LMDB 句柄缓存。

## 安装

需要 Python 3.10+。从源码安装完整功能：

```bash
git clone https://github.com/<your-org>/lerobot-tools.git
cd lerobot-tools
pip install -e '.[cv2,convert,examples]'
```

依赖分组：

| Extra | 用途 |
| --- | --- |
| `cv2` | 默认优先的视频 decoder（OpenCV） |
| `convert` | v3→v2.1 所需的 pandas、pyarrow |
| `examples` | DataLoader 示例所需的 PyTorch |
| `dev` | 测试与 lint 所需的 pytest、ruff |

`--decoder ffmpeg`、`auto` 回退以及 v3 转换都需要系统中的 `ffmpeg` 与 `ffprobe`。Ubuntu 示例：`sudo apt install ffmpeg`。

安装完成后：

```bash
lerobot_tools --help
lerobot_tools lmdb-build --help
lerobot_tools convert --help
```

## 命令行快速开始

### 1. 从 LeRobot v2.1 构建 LMDB

v2.1 数据集至少应包含 `meta/info.json` 和逐 episode 视频：

```text
my_dataset/
├── meta/info.json
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/<video_key>/episode_000000.mp4
```

最常用的构建命令：

```bash
lerobot_tools lmdb-build /data/my_dataset \
  --decoder auto \
  --jpeg-quality 95 \
  --jobs 8
```

默认输出是 `/data/my_dataset/lmdb`：

```text
lmdb/
└── chunk-000/
    └── observation.images.top/
        └── episode_000000.frames_jpeg.lmdb/
            ├── data.mdb
            └── lock.mdb
```

常见选项：

```bash
# 只查看任务计划，不解码、不写数据
lerobot_tools lmdb-build /data/lerobot --dry-run

# 多数据集根目录：缓存写入本地 SSD，并镜像数据集相对路径
lerobot_tools lmdb-build /data/lerobot \
  --output-root /scratch/lerobot-lmdb --jobs 16

# 强制 cv2；若打不开视频则直接报错
lerobot_tools lmdb-build /data/my_dataset --decoder cv2

# 更小的 cache：JPEG 质量 90、4:2:0。请先在任务指标上验证质量损失。
lerobot_tools lmdb-build /data/my_dataset \
  --jpeg-quality 90 --jpeg-subsampling 2

# 仅处理一个视角和前 10 条 episode
lerobot_tools lmdb-build /data/my_dataset \
  --video-key observation.images.top --max-episodes 10

# 健康 cache 默认跳过；显式重建并跳过坏视频
lerobot_tools lmdb-build /data/my_dataset --overwrite --skip-bad-videos
```

`--jpeg-subsampling`：`0=4:4:4`（默认）、`1=4:2:2`、`2=4:2:0`。默认 `quality=95, subsampling=0` 优先保真。

### 关于 `--target-fps`

默认不传 `--target-fps`，cache key 与原始 episode frame index 一一对应，可直接给现有训练 loader 使用。

使用 `--target-fps` 后，LMDB key 是“采样后的连续帧序号”；原始视频 frame index 保存为 `__meta__.source_frame_indices`。训练代码必须根据它做映射，不能再直接用原始 frame index 作为 LMDB key。

### 2. LeRobot v3 → v2.1

转换器面向如下 v3 task-sharded 布局：

```text
source_root/
├── meta/info.json
├── meta/tasks.parquet
├── meta/episodes/*.parquet
├── data/chunk-XXX/file-XXX.parquet
└── videos/<video_key>/chunk-XXX/file-XXX.mp4
```

```bash
lerobot_tools convert /data/robotwin_v3 /data/robotwin_v21 \
  --jobs 8 --ffmpeg-threads 1

# 先小规模验证
lerobot_tools convert /data/robotwin_v3 /tmp/robotwin_v21_check \
  --max-datasets 1 --max-episodes 10

# 转换成功后构建 LMDB
lerobot_tools lmdb-build /data/robotwin_v21 --jobs 8
```

输出为逐 episode parquet、逐 episode MP4，以及 v2.1 所需的 `meta/*.jsonl`。视频默认重编码为 `libx264` / CRF 18 / `veryfast`；可使用 `--video-codec`、`--video-crf`、`--video-preset` 调整。已有输出目录默认会报错，只有 `--overwrite` 才会替换它。

## Python API

```python
from lerobot_tools import convert, lerobot_lmdb_build

convert(
    "/data/robotwin_v3",
    "/data/robotwin_v21",
    jobs=8,
    ffmpeg_threads=1,
)

lerobot_lmdb_build(
    "/data/robotwin_v21",
    decoder="auto",        # auto | cv2 | ffmpeg
    jpeg_quality=95,        # 1..100
    jpeg_subsampling=0,     # 0 | 1 | 2
    jobs=8,
)
```

`lerobot_lmdb_build` 支持：`output_root`、`decoder`、`jpeg_quality`、`jpeg_subsampling`、`target_fps`、`jobs`、`video_key`、`max_episodes`、`overwrite`、`skip_bad_videos`、`dry_run`。

`convert` 支持：`chunk_size`、`video_codec`、`video_crf`、`video_preset`、`jobs`、`ffmpeg_threads`、`max_datasets`、`max_episodes`、`overwrite`。

## PyTorch DataLoader

完成 LMDB 构建后可直接运行：

```bash
python examples/dataloader.py /data/my_dataset \
  --batch-size 64 --workers 8 --batches 3
```

或集成到训练：

```python
from torch.utils.data import DataLoader
from lerobot_tools import LmdbFrameDataset

dataset = LmdbFrameDataset("/data/my_dataset")
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
)

for batch in loader:
    # uint8 NCHW RGB
    images = batch["image"].float().div_(255)
    # train(images)
```

返回 batch 包含 `image`、`episode_index`、`frame_index` 与 `video_key`。每个 DataLoader worker 均维护独立、有限大小的 LMDB environment LRU cache，避免跨进程复用句柄。

## 兼容性与限制

- LMDB cache 只包含 RGB 视频帧，不包含 parquet、action、文本或模型权重。
- JPEG cache 是有损格式；调低质量或使用 4:2:0 前，请针对自己的训练/评测任务验证。
- 支持多进程/多机并发只读；训练期间不要重建同一 cache 目录。
- v2.1 dataset 通过 `meta/info.json` 中的 `chunks_size` 与 `features[*].dtype == "video"` 定位视频。

## 开发

```bash
pip install -e '.[cv2,convert,examples,dev]'
pytest -q
ruff check src tests examples
```

MIT License. 本仓库不包含数据集、模型权重或 ffmpeg 二进制文件。
