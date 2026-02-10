import json
import os
import glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= 🔧 配置区域 =================
# 1. 模型路径
model_path = '/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone/38_yolo11n_p2_detX_yamlv2_VisDrone_1024_bs643/weights/best.pt'

# 2. 数据集配置文件
data_yaml = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml'

# 3. 验证集根目录 (必须包含 images 和 labels 文件夹)
dataset_root = '/home/jack/11/1027/data/VisDrone/VisDrone2019-DET-val'
# ===============================================

def run_final_small_map_eval():
    print(f"🚀 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # -----------------------------------------------------------
    # 第一步：运行模型推理 (生成 predictions.json)
    # -----------------------------------------------------------
    print("\n" + "="*50)
    print("🤖 1. 正在运行模型推理...")
    
    # 使用回调函数捕获验证器，以便获取输出路径
    holder = {}
    model.add_callback("on_val_end", lambda v: holder.update({'validator': v}))
    
    # save_json=True 是必须的
    model.val(data=data_yaml, split='val', save_json=True)
    
    validator = holder.get('validator')
    if not validator:
        print("❌ 错误: 无法获取验证结果路径。")
        return
        
    pred_json_path = os.path.join(validator.save_dir, 'predictions.json')
    if not os.path.exists(pred_json_path):
        print(f"❌ 错误: 未找到预测文件 {pred_json_path}")
        return

    # -----------------------------------------------------------
    # 第二步：生成 Ground Truth (读取硬盘 TXT + 整数 ID)
    # -----------------------------------------------------------
    print("\n" + "="*50)
    print("📄 2. 正在生成标准 COCO Ground Truth...")
    
    images_dir = os.path.join(dataset_root, 'images')
    labels_dir = os.path.join(dataset_root, 'labels')
    
    # 建立映射表：文件名(stem) -> 唯一整数 ID (1, 2, 3...)
    # 这是为了防止 pycocotools 因为字符串 ID 而崩溃
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
    stem_to_int = {}
    
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(12)]
    }
    
    ann_id = 1
    processed_count = 0
    
    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        stem = Path(filename).stem
        int_id = idx + 1
        stem_to_int[stem] = int_id
        
        try:
            # 读取图片尺寸
            with Image.open(img_path) as img:
                w, h = img.size
            
            coco_gt["images"].append({
                "id": int_id,
                "file_name": filename,
                "width": w,
                "height": h
            })
            
            # 读取对应的 TXT 标签 (YOLO 格式: class cx cy w h)
            txt_path = os.path.join(labels_dir, stem + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, nw, nh = map(float, parts[1:])
                            
                            # 坐标还原: Normalized -> Pixel Top-Left
                            pixel_w = nw * w
                            pixel_h = nh * h
                            x_min = (cx * w) - (pixel_w / 2)
                            y_min = (cy * h) - (pixel_h / 2)
                            
                            coco_gt["annotations"].append({
                                "id": ann_id,
                                "image_id": int_id,
                                "category_id": cls_id,
                                "bbox": [x_min, y_min, pixel_w, pixel_h],
                                "area": pixel_w * pixel_h,
                                "iscrowd": 0
                            })
                            ann_id += 1
            processed_count += 1
        except Exception as e:
            continue

    # 保存 GT 文件 (可选，用于调试)
    gt_path = os.path.join(validator.save_dir, 'visdrone_final_gt.json')
    with open(gt_path, 'w') as f: json.dump(coco_gt, f)
    print(f"✅ GT 生成完毕: {len(coco_gt['annotations'])} 个框")

    # -----------------------------------------------------------
    # 第三步：计算 mAP (应用 ID-1 修正)
    # -----------------------------------------------------------
    print("\n" + "="*50)
    print("📊 3. 正在计算最终指标 (应用 ID-1 修正)...")
    
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
    
    fixed_preds = []
    
    for p in preds:
        stem = Path(str(p['image_id'])).stem
        
        # 只处理我们在 GT 中见过的图片
        if stem in stem_to_int:
            new_p = p.copy()
            # 1. 修正 Image ID (转为整数)
            new_p['image_id'] = stem_to_int[stem]
            # 2. 修正 Category ID (VisDrone 1-10 -> YOLO 0-9)
            # 这是之前 mAP 为 0 的核心原因，必须减 1
            new_p['category_id'] = p['category_id'] - 1
            
            # 只保留有效的 0-9 类别
            if 0 <= new_p['category_id'] <= 11:
                fixed_preds.append(new_p)
    
    if not fixed_preds:
        print("❌ 错误: 没有生成有效的预测框！")
        return

    try:
        # 调用 COCO API 进行评估
        cocoGt = COCO(gt_path)
        cocoDt = cocoGt.loadRes(fixed_preds)
        
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        print("\n" + "="*60)
        cocoEval.summarize()
        print("="*60)
        
        # 提取核心数据
        map_total = cocoEval.stats[0]
        map_small = cocoEval.stats[3]
        
        print(f"\n🏆 最终论文数据 (Final Results):")
        print(f"   Total mAP (50-95): {map_total:.4f} ")
        print(f"   Small mAP (<32²) : {map_small:.4f}  <--- ✅ 小目标精度")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 评估计算出错: {e}")

if __name__ == '__main__':
    run_final_small_map_eval()