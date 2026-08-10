# Pointmap 数据压缩方案

## 背景

IK 逆解 / Web 录制产生的数据集体积过大，训练时 CPU 与内存顶不住。
根因是 **pointmap 用 float32 原始存储 + RGB 用 parquet 原始 uint8 存储**，两者都没有压缩。

实测每帧体积（480×640 分辨率）：

| 特征     | 当前存储           | 每帧大小    | 占比 |
| -------- | ------------------ | ----------- | ---- |
| RGB top  | parquet 原始 uint8 | 0.92 MB     | 20%  |
| pointmap | float32 原始       | 3.69 MB     | 80%  |
| **合计** |                    | **4.61 MB** |      |

5000 帧 = 23 GB，2 万帧 = 92 GB。pointmap 占 80%，是主要矛盾。

## 可行性确认（已核对 LeRobot 源码）

两点关键结论，决定方案能否落地：

1. **LeRobotDataset 支持任意 numpy dtype 的 feature 存储**。
   - `is_valid_numpy_dtype_string`（`utils/utils.py:182`）只调 `np.dtype()` 校验，无白名单。
   - `float16` / `int16` / `int8` 全部合法，parquet 原生无损读写。
   - `__getitem__` 经 `torch.tensor(...)` 转换，**保留存储 dtype**（存 float16 返回 `torch.float16`）。
   - 约束：feature 必须声明为数值 dtype（如 `"float16"`），**不能用 `"image"`/`"video"`**，否则进图像/视频管线。

2. **视频编码按 feature 逐个决定，可混合**。
   - 只有 `dtype=="video"` 的 feature 进视频编码，`dtype=="image"` 的进图像管线。
   - pointmap 声明成数值 dtype 后，**完全绕开视频/图像处理**，保持度量精度。
   - pointmap **不能**用 H.264 编码：编码器要求 uint8[0,255] 或 float[0,1]（`image_writer.py:98`），pointmap 含负值（如 -0.5m）会直接报错，且有损压缩破坏度量精度。

## 最终方案（已实施）：RGB 视频编码 + pointmap float16 + 降采样

经实测发现，仅靠 float16 不足以解决内存问题（HF Arrow 表的嵌套 list 结构有
~1.6x 膨胀开销，5000 帧 pointmap 仍需 14 GB）。追加**降采样**到 240×320 后
才真正可控。pointmap 是平滑几何场，且 SigLIP 内部会 resize 到 512×512，半
分辨率无损。

| 特征     | 存储方式     | dtype       | 分辨率      | 每帧磁盘 | 每帧内存(Arrow) |
| -------- | ------------ | ----------- | ----------- | -------- | --------------- |
| RGB top  | mp4 视频编码 | `"video"`   | 480×640     | ~0.03 MB | ~0（不入表）    |
| pointmap | parquet 原始 | `"float16"` | **240×320** | ~0.18 MB | **0.73 MB**     |

### 实测压缩效果（64 帧基准）

| 版本                              | 磁盘       | pointmap Arrow 内存     | 5000 帧外推 |
| --------------------------------- | ---------- | ----------------------- | ----------- |
| v1 原始（image+float32 全分辨率） | 36 MB      | 307 MB（4.8 MB/帧）     | **23.4 GB** |
| v2 video+float16 全分辨率         | 17 MB      | 187 MB（2.9 MB/帧）     | 14.3 GB     |
| **v3 video+float16+降采样**       | **4.7 MB** | **47 MB（0.73 MB/帧）** | **3.6 GB**  |

- 磁盘压缩：**7.7×**（36M → 4.7M）
- 内存压缩：**6.5×**（23.4GB → 3.6GB @ 5000帧）
- 5000 帧总内存 ~3.6GB pointmap + ~1GB 固定开销 ≈ 4.6GB，15GB 内存机器安全

## 实施点（已完成）

### 改动的文件

| 文件                   | 改动                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data_gen/scripted.py` | RGB dtype `"image"`→`"video"`；pointmap dtype `"float32"`→`"float16"`；pointmap shape `(480,640,3)`→`POINTMAP_STORE_SHAPE=(240,320,3)`；新增 `_downsample_pointmap()`；`add_frame()` 降采样+float16；`LeRobotDataset.create` `use_videos=False`→`True` |
| `data_gen/reach_ik.py` | `LeRobotDataset.create` `use_videos=False`→`True`（features/add_frame 复用 scripted）                                                                                                                                                                  |

### 关键约束（实测确认）

1. **视频编码触发条件**：feature 必须声明 `dtype:"video"`（不是 `"image"`），
   否则 `video_keys` 为空，`use_videos=True` 不生效。这是 v1 失败的根因。

2. **dtype/shape 严格校验**（`feature_utils.py:319`）：`add_frame` 传入的数组
   必须精确匹配声明的 dtype 和 shape。所以 pointmap 要 `_downsample_pointmap()`
   后再 `.astype(np.float16)`。

3. **pointmap 必须用数值 dtype**：不能用 `"image"`/`"video"`，否则进视频管线，
   含负值（如 -0.5m）会报错，且有损压缩破坏度量精度。

4. **训练时策略侧零改动**：`prepare_pointmaps`（`modeling_smolvla.py`）里
   `pm / radius` 除法时 PyTorch 自动把 float16 升回 float32；SigLIP 内部 resize
   到 512×512 会把 240×320 上采样，模型无感。

5. **降采样无损性**：pointmap 是稠密几何场，非纹理。SigLIP 反正 resize 到
   512×512，原生 480×640 的精度本就被丢弃。240×320 保留了所有空间结构。

## 验证清单

实施后跑一遍以下检查：

- [ ] `dataset_features()` 返回的 pointmap dtype 为 `"float16"`
- [ ] 生成 1 个 episode 的最小数据集，无 dtype 校验报错
- [ ] 数据集 meta/info.json 里 pointmap feature dtype 显示 float16
- [ ] 用 `LeRobotDataset(path)[0]` 读回一帧，pointmap 是 `torch.float16`，数值合理（非全 0、非 NaN）
- [ ] 数据集体积相比 float32 版本减半
- [ ] RGB 以 mp4 形式存储（data/ 下出现 videos/ 目录或 .mp4 文件）
- [ ] SmolVLA 训练能加载该数据集，forward 不报 dtype 错误

## 进阶选项（当前方案仍不够时）

当前 video+float16+降采样 已将 5000 帧 pointmap 内存压到 3.6 GB。若数据规模
更大（>2 万帧）或内存更小，可追加：

1. **进一步降采样**：把 `POINTMAP_STORE_SHAPE` 从 (240,320,3) 降到 (160,212,3)，
   再省 ~2.4×。pointmap 足够平滑，160×212 仍保留主要几何结构。

2. **pointmap int8 量化**：XYZ 量化到 int8（精度 ~5mm，clamp ±1.27m）。
   需评估双臂最远点（~0.86m）是否在量程内。体积再砍 2×。

3. **pointmap 不进 Arrow 表**：改为像视频一样存独立文件（.npy/.npz），按需读取。
   这能彻底避开 HF datasets 的嵌套 list 膨胀（当前 1.6x 开销），但需改
   LeRobotDataset 核心，工作量大。

组合极限（160×212 + int8）：pointmap 从 4.8 MB/帧 → 0.1 MB/帧，2 万帧 < 2 GB。
