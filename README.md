# 目标检测与多目标追踪分析系统

基于 **YOLOv5 + DeepSORT** 的视频目标检测、多目标跟踪与数量统计系统。可对视频中的目标（以行人、车辆为例）进行逐帧检测、跨帧关联跟踪，并按跟踪 ID 去重统计目标总数，输出带跟踪框与统计信息的可视化结果视频。

系统采用轻量级 HOG 特征提取方案替代原始 DeepSORT 的重型 ReID 网络，普通 CPU 即可运行，适合课程实训与效果演示。可通过修改类别配置与训练数据，扩展适配水下目标检测追踪等各类场景。

## ✨ 功能特性

- **目标检测**：YOLOv5 实时检测，支持 COCO 80 类目标
- **多目标跟踪**：DeepSORT 算法跨帧关联，为每个目标分配稳定跟踪 ID
- **分类统计**：按跟踪 ID 去重统计行人、车辆总数（同一目标只计一次，避免重复计数）
- **可视化输出**：
  - 行人红色框 / 车辆蓝色框，带跟踪 ID 标签
  - 画面右上角实时显示统计总数
  - 结果视频自动保存至 `runs/track/exp*` 目录
- **灵活参数**：命令行可配置权重路径、视频路径、推理尺寸、置信度阈值、运行设备等

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────┐
│                 main.py（主程序）            │
│   1.路径检查  2.设备选择  3.工具函数          │
├──────────────────────────────────────────────┤
│  目标检测层：YOLOv5（yolov5/models+utils）   │
│    DetectMultiBackend + non_max_suppression  │
├──────────────────────────────────────────────┤
│  目标跟踪层：DeepSORT（deep_sort/）          │
│    HOG特征提取 → NearestNeighbor匹配 →       │
│    卡尔曼滤波预测/更新 → 跟踪ID管理          │
├──────────────────────────────────────────────┤
│  输出层：分类统计 + 画框标注 + 视频保存      │
└──────────────────────────────────────────────┘
```

## 📁 目录结构

```
underwater-target-tracking/
├── main.py                # 主程序（检测+跟踪+统计+可视化）
├── test_video.mp4         # 测试视频
├── deep_sort/             # DeepSORT 多目标跟踪库
│   ├── deep_sort/         # 核心算法（检测/卡尔曼/匹配/跟踪器）
│   ├── application_util/  # 工具模块
│   ├── tools/             # 辅助工具
│   └── LICENSE
├── yolov5/                # YOLOv5 目标检测框架
│   ├── models/            # 网络模型定义
│   ├── utils/             # 工具函数
│   ├── data/              # 数据集配置
│   ├── detect.py          # 官方检测脚本
│   └── requirements.txt   # YOLOv5 依赖
└── samples/               # 运行效果示例视频
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 权重文件 `yolov5s.pt`（需手动下载）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型权重

将 `yolov5s.pt` 放入 `yolov5/` 目录：

```bash
# 官方下载地址
https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### 4. 运行

```bash
python main.py
```

指定视频与设备：

```bash
# CPU 运行
python main.py --source test_video.mp4 --device cpu

# GPU 运行
python main.py --source test_video.mp4 --device 0

# 自定义参数
python main.py --source 视频路径 --conf-thres 0.3 --iou-thres 0.5 --imgsz 640
```

运行结束后，结果视频保存在 `runs/track/exp*/test_video.mp4`，控制台输出最终统计结果。

## ⚙️ 核心实现说明

| 模块 | 说明 |
|------|------|
| `select_device` | 手动实现的设备选择，兼容 CPU/CUDA 多卡 |
| `scale_coords` / `clip_coords` | 检测框坐标从推理尺寸映射回原图 |
| `simple_feature_extractor` | 基于 HOG 的轻量特征提取（64×128 窗口），替代重型 ReID 网络 |
| 类别配置 | `PERSON_CLASS_ID=0`（行人）、`VEHICLE_CLASS_IDS=[2,3,5,6,7]`（车辆），可自定义 |
| 去重统计 | `counted_*_ids` 集合按跟踪 ID 去重，同一目标仅计数一次 |
| IOU 匹配 | 跟踪框与检测框 IOU>0.5 时关联类别，绘制对应颜色框 |

## ⚠️ 注意事项

- 首次运行需确保 `yolov5s.pt` 权重已下载至 `yolov5/` 目录，否则程序会报错并提示下载地址
- 默认 CPU 运行，如需 GPU 加速请安装对应 CUDA 版本的 PyTorch
- 类别统计仅覆盖行人（ID 0）与车辆（ID 2/3/5/6/7），其他类别目标只跟踪不统计

## 📄 许可

本项目集成 YOLOv5（AGPL-3.0）与 DeepSORT（MIT）开源组件，请遵循各自许可证条款。项目主体代码仅供学习交流使用。
