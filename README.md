# Tree_Segment

本目录为 **自包含** 实现：不从 `Tree_Segment` 目录外 `import` 任何项目代码文件（只依赖第三方库 `numpy/opencv-python/laspy`）。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 一条命令跑通（基于 222_2025-11-25-155532 数据）

```bash
python3 "/Users/yjc/project/沄视/project/plant_classfication/222_2025-11-25-155532/Tree_Segment/scripts/run_tree_pano_pipeline.py" \
  --stage all \
  --tree-las "/Users/yjc/project/沄视/project/plant_classfication/222_2025-11-25-155532/tree2.las" \
  --pano-image-dir "/Users/yjc/project/沄视/project/plant_classfication/222_2025-11-25-155532/panoramicImage" \
  --pano-poses "/Users/yjc/project/沄视/project/plant_classfication/222_2025-11-25-155532/panoramicPoses.csv" \
  --out-dir "/Users/yjc/project/沄视/project/plant_classfication/222_2025-11-25-155532/Tree_Segment/output" \
  --max-dist 10 \
  --max-angle-deg 90 \
  --downsample-step 50 \
  --flip-v \
  --morph-kernel 9 --dilate-iter 2 --close-iter 2 \
  --refine convex_hull
```

## 代码结构

```
Tree_Segment/
  requirements.txt
  README.md
  scripts/
    run_tree_pano_pipeline.py          # 单入口：select/project/refine/all
    lib/
      __init__.py
      poses.py                         # 读取 panoramicPoses.csv，构建 basename->pose 索引
      quaternion.py                    # 四元数共轭/旋转
      las_utils.py                     # 读 tree.las、center/bbox、下采样
      projection_equirect.py           # 等距柱状投影 world->(u,v)
      mask_ops.py                      # 点->mask、mask优化、抠图输出
```

## 脚本作用与基本原理

### 入口脚本：`scripts/run_tree_pano_pipeline.py`

该脚本提供三阶段流水线，核心输入是：
- **tree.las**：树的点云（用于计算树中心 + 作为投影点）
- **panoramicImage/**：全景等距柱状图片目录
- **panoramicPoses.csv**：全景图片对应位姿（camera-to-world）

支持三阶段流水线：
1. **筛图**（距离 + 视角）：输出 `selected_frames.jsonl`，可选复制筛选图片
   - 可选 **遮挡剔除**：基于全局 `map.las`，若相机到目标树的视线之间存在大量其它点云遮挡则剔除该帧
2. **投影生成掩码**：tree 点云下采样后投影到等距柱状全景图，输出 `masks_raw/*.png`
3. **掩码优化 + 抠图**：轮廓整体化/平滑，输出 `masks_refined/*.png` 与 `segmented/*.png`

#### 阶段 (1) 筛图：距离 & 视角（可见性近似）
- **距离**：相机位置 `cam_xyz` 到树中心 `tree_center` 的欧氏距离  
  \[
  dist = \|cam\_xyz - tree\_center\|
  \]
- **视角**：将“相机指向树中心”的单位向量从世界系旋转到相机系，取相机前向分量 `d_cam.z`  
  - 相机系约定：**+Z 前方、+X 右、+Y 上**
  - `q_c2w` 是 camera-to-world，所以 world-to-camera 用共轭 `q_w2c = conj(q_c2w)`
  - 若设最大视角 `max_angle`，等价判据为：
    \[
    d\_{cam}.z \ge \cos(max\_angle)
    \]
  - 你之前的 `d_cam.z > 0` 对应 `max_angle = 90°`

输出 `selected_frames.jsonl`（每行一个 frame，包含：图片路径、位姿、dist、angle）。

#### （可选）遮挡剔除：基于 map.las 的“视线遮挡”过滤
某些情况下，虽然距离/视角满足，但从相机看过去**树被其它物体遮挡**，图像主体变成了其它物体。  
此时可提供全局点云 `--map-las`，对每个候选帧做近似遮挡判断（XY 平面）：  
- 在相机到树方向的**水平扇形**内截取 map 点云（默认 FOV=100°、半径=10m）  
- 仅保留位于“相机与树之间”的点，并且在视线附近的“管状区域”内（默认管道半径 1m）  
- 将这些“可能遮挡点”做 **XY 体素占用**统计（默认体素 0.2m）  
- 若遮挡体素数 / 树体素数 > 阈值（默认 0.4）则剔除该帧  

遮挡统计会写入 `selected_frames.jsonl` 字段：`occl_vox_xy_count/tree_vox_xy_total/occl_ratio`，便于你回看为什么被过滤。

#### 阶段 (2) 投影：点云 → 像素 → raw mask
- 点云（tree.las）先下采样（`--downsample-step` 或 `--downsample-ratio`）
- 每个点按等距柱状（equirectangular）模型投影到像素：
  - 先把世界点方向转到相机系，得到单位向量 `d_cam=(dx,dy,dz)`
  - 方位角/俯仰角：
    \[
    \theta = atan2(dx, dz), \quad \phi = asin(dy)
    \]
  - 像素坐标：
    \[
    u = \frac{\theta + \pi}{2\pi}W, \quad v = \frac{\pi/2 - \phi}{\pi}H
    \]
- mask 生成：对落点 `(u,v)` 置 255，然后用形态学操作把稀疏点连接成区域（dilate + close）

输出 `masks_raw/*.png`（单通道 0/255）。

#### 阶段 (3) 掩码优化 + 抠图保存
- 从 `masks_raw` 中找外轮廓，取 **最大轮廓**，再做“整体化”：
  - `largest_contour`：直接填充最大轮廓
  - `convex_hull`：对最大轮廓取凸包（更容易变成一个整体）
  - `approx`：对轮廓做多边形近似（更平滑）
- 抠图：将 refined mask 应用于原图，**只保留掩码内像素**（背景置黑）并保存

输出：
- `masks_refined/*.png`
- `segmented/*.png`
- 可选 `--write-rgba` 输出 `segmented/*_rgba.png`（mask 作为 alpha）

## 使用方法（分阶段）

### 只做筛图（生成 selected_frames.jsonl，可选复制图片）

```bash
python3 ".../Tree_Segment/scripts/run_tree_pano_pipeline.py" \
  --stage select \
  --tree-las ".../tree2.las" \
  --pano-image-dir ".../panoramicImage" \
  --pano-poses ".../panoramicPoses.csv" \
  --out-dir ".../Tree_Segment/output" \
  --max-dist 10 --max-angle-deg 90 \
  --copy-selected
```

### 只做筛图（含遮挡剔除：需要 map.las）

```bash
python3 ".../Tree_Segment/scripts/run_tree_pano_pipeline.py" \
  --stage select \
  --tree-las ".../tree2.las" \
  --pano-image-dir ".../panoramicImage" \
  --pano-poses ".../panoramicPoses.csv" \
  --out-dir ".../Tree_Segment/output" \
  --max-dist 10 --max-angle-deg 90 \
  --map-las ".../map.las" \
  --occl-fov-deg 100 --occl-radius-m 10 \
  --occl-voxel-size 0.20 --occl-thr 0.40 \
  --occl-tube-radius-m 1.0
```

### 只做投影掩码（依赖已有 selected_frames.jsonl）

```bash
python3 ".../Tree_Segment/scripts/run_tree_pano_pipeline.py" \
  --stage project \
  --tree-las ".../tree2.las" \
  --pano-image-dir ".../panoramicImage" \
  --pano-poses ".../panoramicPoses.csv" \
  --out-dir ".../Tree_Segment/output" \
  --downsample-step 50 \
  --flip-v \
  --morph-kernel 9 --dilate-iter 2 --close-iter 2
```

### 只做掩码优化 + 抠图（依赖已有 masks_raw）

```bash
python3 ".../Tree_Segment/scripts/run_tree_pano_pipeline.py" \
  --stage refine \
  --tree-las ".../tree2.las" \
  --pano-image-dir ".../panoramicImage" \
  --pano-poses ".../panoramicPoses.csv" \
  --out-dir ".../Tree_Segment/output" \
  --refine convex_hull \
  --write-rgba
```

## 输出目录结构
- `output/selected_frames.jsonl`：筛选后帧清单（含图片路径、位姿、距离、视角）
- `output/masks_raw/*.png`：投影点 + 形态学后的 raw 掩码
- `output/masks_refined/*.png`：轮廓整体化后的 refined 掩码
- `output/segmented/*.png`：抠图结果（背景置黑，仅保留掩码内区域）

## 参数建议（经验值）
- **速度/效果折中**：先用 `--downsample-step 50~200` 快速看效果，再降低 step 提升细节
- **掩码更连贯**：增大 `--morph-kernel` 或增加 `--dilate-iter/--close-iter`
- **更“整体”的轮廓**：用 `--refine convex_hull`（默认），如果想更贴边可试 `largest_contour`

