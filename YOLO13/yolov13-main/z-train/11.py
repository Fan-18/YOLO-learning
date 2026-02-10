import os
import json
import glob
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

def validate_match(pred_json, gt_json):
    """诊断函数：检查预测值和真值是否能对应上"""
    print("\n" + "="*20 + " 诊断信息 " + "="*20)
    
    with open(pred_json) as f:
        preds = json.load(f)
    with open(gt_json) as f:
        gts = json.load(f)
        
    if not preds:
        print("❌ 预测文件为空！")
        return False
        
    # 1. 检查 Image ID 格式
    pred_id_example = preds[0]['image_id']
    gt_id_example = gts['images'][0]['id']
    
    print(f"预测文件 Image ID 示例: '{pred_id_example}' (类型: {type(pred_id_example).__name__})")
    print(f"真值文件 Image ID 示例: '{gt_id_example}' (类型: {type(gt_id_example).__name__})")
    
    if pred_id_example != gt_id_example:
        print("⚠️ 警告: Image ID 格式不一致！这会导致 mAP 接近 0。")
        # 尝试判断是否仅仅是后缀的区别
        if str(pred_id_example).split('.')[0] == str(gt_id_example).split('.')[0]:
            print("  -> 看起来只是文件后缀的区别，脚本将尝试自动修复匹配。")
    else:
        print("✅ Image ID 格式一致。")

    # 2. 检查坐标范围 (Box Coordinate Check)
    # 找一张同时存在于 pred 和 gt 的图片
    common_id = None
    for img in gts['images']:
        # 尝试找到匹配的 ID
        if any(p['image_id'] == img['id'] for p in preds):
            common_id = img['id']
            break
            
    if common_id is None:
        print("❌ 致命错误: 无法找到任何匹配的图片 ID！pycocotools 无法进行评估。")
        return False
        
    p_box = next(p['bbox'] for p in preds if p['image_id'] == common_id)
    g_box = next(g['bbox'] for g in gts['annotations'] if g['image_id'] == common_id)
    
    print(f"\nID 为 '{common_id}' 的图片坐标抽样:")
    print(f"  预测框 (xywh): {p_box}")
    print(f"  真值框 (xywh): {g_box}")
    
    # 简单检查数值量级
    if p_box[2] < 1.0 and g_box[2] > 10.0:
        print("❌ 坐标系不匹配：预测值看起来是归一化的 (0-1)，而真值是绝对像素。")
    elif p_box[2] > 10.0 and g_box[2] < 1.0:
        print("❌ 坐标系不匹配：预测值是绝对像素，真值看起来是归一化的。")
    else:
        print("✅ 坐标数值量级看起来都在绝对像素范围内 (或者都一致)。")
        
    print("="*50 + "\n")
    return True

def get_coco_metrics_fixed(predictions_path, data_yaml_path):
    # --- 1. 读取配置 ---
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    # 处理路径
    val_path = data_cfg.get('val')
    if not os.path.isabs(val_path):
        # 尝试拼接 dataset root
        if 'path' in data_cfg:
            val_path = os.path.join(data_cfg['path'], val_path)
        else:
            # 假设相对于 yaml 文件
            val_path = os.path.join(os.path.dirname(data_yaml_path), val_path)

    # 修正：VisDrone 有时 images/val 这种结构，确保指向图片目录
    if not os.path.exists(val_path) and os.path.exists(val_path + '/images'):
         val_path = val_path + '/images'

    print(f"验证集图片路径: {val_path}")

    # 读取预测文件以确定 ID 格式
    with open(predictions_path, 'r') as f:
        preds = json.load(f)
        # 创建一个 查找表: filename_stem -> image_id_in_pred
        # 这样无论 prediction 用的是全名还是无后缀，我们都能强行匹配
        pred_id_map = {}
        for p in preds:
            pid = p['image_id']
            # 假设 pid 是字符串，去掉可能的后缀
            key = str(pid).rsplit('.', 1)[0] 
            pred_id_map[key] = pid # 保存原始 ID

    gt_json_path = predictions_path.replace('predictions.json', 'ground_truth_fixed.json')
    
    images = []
    annotations = []
    ann_id = 0
    
    img_files = sorted(glob.glob(os.path.join(val_path, '*.*')))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    img_files = [x for x in img_files if x.lower().endswith(valid_extensions)]

    print(f"正在生成修正版 GT JSON ({len(img_files)} 张图片)...")
    
    match_count = 0
    for img_file in tqdm(img_files):
        filename = os.path.basename(img_file)
        file_stem = os.path.splitext(filename)[0]
        
        # --- 核心修复逻辑：强行匹配 ID ---
        # 我们用文件名(无后缀)去预测结果里找对应的 ID
        if file_stem in pred_id_map:
            image_id = pred_id_map[file_stem] # 使用预测文件中用过的那个 ID
            match_count += 1
        else:
            # 如果预测里没有这张图，说明没检测到目标，或者ID格式差异巨大
            # 为了保证评估运行，我们使用 stem，但这张图在 eval 时会被认为漏检
            image_id = file_stem 

        try:
            with Image.open(img_file) as img:
                w, h = img.size
        except:
            continue

        images.append({
            "id": image_id,
            "file_name": filename,
            "height": h,
            "width": w
        })

        # 读取 Label
        # 尝试多种 label 目录结构
        # 1. .../images/val/x.jpg -> .../labels/val/x.txt
        label_file = img_file.replace('/images/', '/labels/').replace(os.path.splitext(filename)[1], '.txt')
        # 2. 简单的同级目录替换
        if not os.path.exists(label_file):
             label_file = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'labels', file_stem + '.txt')

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # YOLO (Normalized) -> COCO (Absolute xywh)
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        w_abs = bw * w
                        h_abs = bh * h
                        
                        annotations.append({
                            "id": ann_id,
                            "image_id": image_id, # 必须和 images 里的 id 一致
                            "category_id": cls_id, 
                            "bbox": [x, y, w_abs, h_abs],
                            "area": w_abs * h_abs,
                            "iscrowd": 0
                        })
                        ann_id += 1
    
    print(f"图片 ID 匹配成功率: {match_count}/{len(img_files)}")

    # 写入 JSON
    # 类别 ID 0-9
    names = data_cfg['names']
    if isinstance(names, list):
        categories = [{"id": i, "name": n} for i, n in enumerate(names)]
    else:
        categories = [{"id": k, "name": v} for k, v in names.items()]

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(gt_json_path, 'w') as f:
        json.dump(coco_format, f)

    # --- 诊断 ---
    if not validate_match(predictions_path, gt_json_path):
        return

    # --- 评估 ---
    print("开始调用 pycocotools 评估...")
    cocoGt = COCO(gt_json_path)
    cocoDt = cocoGt.loadRes(predictions_path)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    # 请修改这里的路径
    pred_path = '/home/jack/桌面/runs/detect/val12/predictions.json'
    yaml_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml'
    
    get_coco_metrics_fixed(pred_path, yaml_path)