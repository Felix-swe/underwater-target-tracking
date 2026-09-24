import argparse
import os
import sys
import glob  # 新增：补充glob导入
import re  # 新增：补充re导入
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# ===================== 1. 路径配置与检查 =====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# ---------------------- YOLOv5配置（手动下载后路径） ----------------------
YOLOV5_DIR = os.path.join(PROJECT_ROOT, "yolov5")  # YOLOv5源码文件夹路径
WEIGHTS_PATH = os.path.join(YOLOV5_DIR, "yolov5s.pt")  # 手动下载的权重文件路径
# 检查权重是否存在
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"\n❌ 未找到YOLOv5权重文件！请将yolov5s.pt放到以下路径：\n{WEIGHTS_PATH}\n"
        "手动下载地址：https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    )
sys.path.append(YOLOV5_DIR)

# ---------------------- DeepSORT配置 ----------------------
DEEPSORT_DIR = os.path.join(PROJECT_ROOT, "deep_sort")
if not os.path.exists(DEEPSORT_DIR):
    raise FileNotFoundError(
        f"\n❌ 未找到deep_sort文件夹！路径：{DEEPSORT_DIR}\n"
        "下载地址：https://github.com/nwojke/deep_sort/archive/refs/heads/master.zip"
    )
sys.path.append(DEEPSORT_DIR)


# ===================== 2. 手动实现设备选择函数（替代YOLOv5的select_device） =====================
def select_device(device='', batch_size=0):
    """
    手动实现设备选择，兼容不同版本YOLOv5
    device: 'cpu'/'cuda'/'0'/'0,1'
    """
    # 检查CUDA是否可用
    cuda = torch.cuda.is_available() and device.lower() != 'cpu'
    if cuda:
        devices = device.split(',') if device else '0'
        device = f'cuda:{devices[0]}' if len(devices) > 0 else 'cuda'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    # 打印设备信息
    print(f"使用设备：{device}")
    return torch.device(device)


# ===================== 3. 工具函数 =====================
def clip_coords(boxes, img_shape):
    """裁剪检测框到图像范围内"""
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """调整检测框坐标到原始图像尺寸"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


# ===================== 4. 手动实现check_img_size（替代YOLOv5的check_img_size） =====================
def check_img_size(img_size, s=32, floor=0):
    """
    校验推理尺寸为stride的整数倍
    img_size: 输入尺寸
    s: stride值
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    # 调整为stride的整数倍
    img_size = list(img_size)
    img_size[0] = max(floor * s, int(np.ceil(img_size[0] / s) * s))
    img_size[1] = max(floor * s, int(np.ceil(img_size[1] / s) * s))
    return tuple(img_size)


# ===================== 5. 手动实现increment_path（替代YOLOv5的increment_path） =====================
def increment_path(path, exist_ok=False, sep=''):
    """
    自动递增保存路径，如exp→exp1→exp2
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}").with_suffix(suffix)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===================== 6. 轻量级特征提取器（DeepSORT专用） =====================
def simple_feature_extractor(image, bboxes):
    """HOG特征提取，适配DeepSORT跟踪"""
    features = []
    for bbox in bboxes:
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h

        # 边界检查
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        if x2 <= x1 or y2 <= y1:
            features.append(np.zeros(128))
            continue

        # 提取HOG特征
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (64, 128))
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        feat = hog.compute(crop_gray).flatten()[:128]
        features.append(feat)
    return features


# ===================== 7. 核心配置（类别/颜色/统计规则） =====================
# COCO类别映射
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 类别ID配置（可自定义增减）
PERSON_CLASS_ID = 0  # 行人
VEHICLE_CLASS_IDS = [2, 3, 5, 6, 7]  # 车辆：car(2)/motorcycle(3)/bus(5)/train(6)/truck(7)

# 颜色配置（OpenCV为BGR格式）
COLOR_PERSON = (0, 0, 255)  # 行人框：红色
COLOR_VEHICLE = (255, 0, 0)  # 车辆框：蓝色
COLOR_TEXT_BG = (0, 0, 0)  # 统计文字背景：黑色
COLOR_TEXT_FONT = (0, 255, 255)  # 统计文字：黄色（醒目）
COLOR_LABEL = (255, 255, 255)  # 跟踪ID文字：白色


# ===================== 8. 核心跟踪与统计函数 =====================
@torch.no_grad()
def run(
        weights=WEIGHTS_PATH,  # 手动下载的权重路径
        source=os.path.join(PROJECT_ROOT, "test_video.mp4"),  # 测试视频路径
        imgsz=(640, 640),  # 推理尺寸
        conf_thres=0.25,  # 检测置信度阈值
        iou_thres=0.45,  # NMS IoU阈值
        device="cpu",  # 运行设备（cpu/cuda）
        save_video=True  # 是否保存结果视频
):
    # ---------------------- 导入必要的YOLOv5模块（仅保留核心） ----------------------
    from models.common import DetectMultiBackend
    from utils.dataloaders import LoadImages
    from utils.general import non_max_suppression

    # ---------------------- 初始化统计变量（基于跟踪ID去重） ----------------------
    counted_person_ids = set()  # 已统计的行人ID（避免重复计数）
    counted_vehicle_ids = set()  # 已统计的车辆ID
    total_persons = 0  # 最终行人总数
    total_vehicles = 0  # 最终车辆总数

    # ---------------------- 初始化设备和模型 ----------------------
    device = select_device(device)  # 使用手动实现的设备选择
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 使用手动实现的尺寸校验

    # ---------------------- 初始化DeepSORT跟踪器 ----------------------
    max_cosine_distance = 0.6
    nn_budget = 100
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # ---------------------- 加载视频数据 ----------------------
    source = str(source)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_writer = None

    # 获取视频参数
    video_fps = 30
    video_width, video_height = 640, 480
    if hasattr(dataset, 'cap') and dataset.cap.isOpened():
        video_fps = dataset.cap.get(cv2.CAP_PROP_FPS)
        video_width = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(dataset, 'cap') else 0

    # ---------------------- 结果保存路径 ----------------------
    save_dir = increment_path(Path(PROJECT_ROOT) / "runs" / "track" / "exp", exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_video_path = str(save_dir / Path(source).name)

    # ---------------------- 逐帧处理视频 ----------------------
    print(f"\n开始处理视频：{source}")
    print(f"视频总帧数：{total_frames}")
    print(f"结果将保存至：{save_video_path}\n")

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # 预处理图像（归一化、维度扩展）
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]  # 扩展batch维度

        # YOLOv5目标检测
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)


        for i, det in enumerate(pred):
            im0 = im0s.copy()  # 原始图像副本（用于绘制）

            if len(det):
                # 调整检测框坐标到原始图像尺寸
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 提取检测框、置信度、类别ID
                bboxes = []  # 检测框：[x1, y1, w, h]
                scores = []  # 置信度
                class_ids = []  # 类别ID
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    bboxes.append([float(x1), float(y1), float(w), float(h)])
                    scores.append(float(conf))
                    class_ids.append(int(cls))

                # 提取特征并更新跟踪器
                features = simple_feature_extractor(im0, bboxes)
                detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
                tracker.predict()
                tracker.update(detections)

                # 绘制跟踪框 + 统计目标数量
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue  # 跳过未确认/过期的跟踪框

                    track_id = track.track_id
                    bbox = track.to_tlbr()  # 跟踪框坐标：(x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, bbox)

                    # 匹配跟踪框对应的类别ID
                    cls_id = -1
                    min_iou = 0.5  # IOU阈值（匹配跟踪框和检测框）
                    for idx, (bbox_det, cls) in enumerate(zip(bboxes, class_ids)):
                        x1_det, y1_det, w_det, h_det = bbox_det
                        x2_det = x1_det + w_det
                        y2_det = y1_det + h_det
                        # 计算IOU
                        inter_x1 = max(x1, x1_det)
                        inter_y1 = max(y1, y1_det)
                        inter_x2 = min(x2, x2_det)
                        inter_y2 = min(y2, y2_det)
                        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        area_det = w_det * h_det
                        area_track = (x2 - x1) * (y2 - y1)
                        iou = inter_area / (area_det + area_track - inter_area + 1e-6)
                        if iou > min_iou:
                            min_iou = iou
                            cls_id = cls

                    # 分类统计 + 绘制不同颜色框
                    if cls_id == PERSON_CLASS_ID:
                        # 行人：红色框
                        draw_color = COLOR_PERSON
                        # 仅首次出现时统计
                        if track_id not in counted_person_ids:
                            counted_person_ids.add(track_id)
                            total_persons += 1
                        label = f"ID:{track_id} person"

                    elif cls_id in VEHICLE_CLASS_IDS:
                        # 车辆：蓝色框
                        draw_color = COLOR_VEHICLE
                        # 仅首次出现时统计
                        if track_id not in counted_vehicle_ids:
                            counted_vehicle_ids.add(track_id)
                            total_vehicles += 1
                        label = f"ID:{track_id} {COCO_CLASSES[cls_id]}"

                    else:
                        # 其他类别：跳过
                        continue

                    # 绘制跟踪框
                    cv2.rectangle(im0, (x1, y1), (x2, y2), draw_color, 2)
                    # 绘制跟踪ID标签（带背景）
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_y = y1 - label_size[1] - 4 if y1 - label_size[1] - 4 > 0 else y1 + label_size[1] + 4
                    cv2.rectangle(im0, (x1, label_y - label_size[1] - 2),
                                  (x1 + label_size[0], label_y + 2), draw_color, -1)
                    cv2.putText(im0, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1)

            # ---------------------- 绘制统计文字（右上角，醒目显示） ----------------------
            stats_text = f"行人总数：{total_persons} | 车辆总数：{total_vehicles}"
            # 1. 绘制黑色背景框（防止文字与背景融合）
            text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = im0.shape[1] - text_size[0] - 20  # 右上角X坐标（距右20像素）
            text_y = 40  # 右上角Y坐标（距上40像素）
            cv2.rectangle(im0, (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), COLOR_TEXT_BG, -1)
            # 2. 绘制黄色统计文字
            cv2.putText(im0, stats_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT_FONT, 3)

            # ---------------------- 保存结果视频 ----------------------
            if save_video:
                if vid_writer is None:
                    # 初始化视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter(save_video_path, fourcc, video_fps, (im0.shape[1], im0.shape[0]))
                vid_writer.write(im0)

            # 打印帧处理进度
            if (frame_idx + 1) % 100 == 0:
                print(f"已处理帧：{frame_idx + 1}/{total_frames} | 当前统计：行人{total_persons} 车辆{total_vehicles}")

    # ---------------------- 结束处理，释放资源 ----------------------
    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()

    # ---------------------- 输出最终统计结果 ----------------------
    print("\n==================== 统计结果 ====================")
    print(f"行人总数：{total_persons}")
    print(f"车辆总数：{total_vehicles}")
    print(f"结果视频保存路径：{save_video_path}")
    print("===================================================")


# ===================== 9. DeepSORT模块导入 =====================
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker


# ===================== 10. 参数解析与主函数 =====================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH, help="YOLOv5权重路径（手动下载）")
    parser.add_argument("--source", type=str, default=os.path.join(PROJECT_ROOT, "test_video.mp4"), help="测试视频路径")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备（cpu/cuda）")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="推理尺寸")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 适配(height, width)
    return opt


def main(opt):
    # 运行核心跟踪统计函数
    run(
        weights=opt.weights,
        source=opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)