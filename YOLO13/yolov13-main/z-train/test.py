from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import glob

# ================= 配置区域 =================
# 1. 基础路径配置
base_path = '/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone/1_yolo11n.pt_VisDrone-640'
model_path = os.path.join(base_path, 'weights/best.pt')
data_yaml = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml'

# 指定验证结果保存位置
save_project = base_path      # 结果保存在原训练目录下
save_name = 'visdrone_eval'   # 子文件夹名称
# ===========================================

# 2. 加载模型
print(f"🚀 Loading model from: {model_path}")
model = YOLO(model_path)

# 3. 运行验证模式
# project 和 name 结合，结果会保存在: {base_path}/visdrone_eval/
print("running model.val()...")
metrics = model.val(
    data=data_yaml, 
    split='val',         
    save_json=True,      # 必须：生成预测 JSON
    project=save_project,# 指定保存的主目录
    name=save_name       # 指定保存的子目录名
)

# --- 打印 Ultralytics 默认指标 ---
names = model.names
print(f"\n{'Class':<20} | {'mAP50-95':<10}")
print("-" * 35)
for i, m in enumerate(metrics.box.maps):
    print(f"{names[i]:<20} | {m:.4f}")

# ==============================================================================
# 4. 核心修改：自动定位文件并调用 PyCOCOTools 计算 Small/Medium/Large mAP
# ==============================================================================
print("\n" + "="*60)
print("正在进行 COCO 风格评估 (Target: Small Object mAP)...")
print("="*60)

# A. 确定预测文件路径 (Auto)
# 路径结构是: project/name/predictions.json
pred_json_path = os.path.join(save_project, save_name, 'predictions.json')

if not os.path.exists(pred_json_path):
    print(f"❌ 错误: 未找到预测文件: {pred_json_path}")
    print("可能是 model.val() 没有成功生成 JSON，请检查显存或数据集配置。")
else:
    print(f"✅ 锁定预测文件: {pred_json_path}")

    # B. 确定标注文件 (Ground Truth) (Auto Search)
    # Ultralytics 第一次运行 save_json=True 时，会在数据集目录下生成一个 .json
    # 我们根据 yaml 路径反推数据集目录，并搜索 json
    gt_json_path = None
    
    # 假设 VisDrone 数据集在 yaml 文件的同级或上级目录
    # 这里我们尝试在几个常见位置搜索 VisDrone_val.json 或类似的 json
    search_dirs = [
        os.path.dirname(data_yaml),  # yaml 同级
        os.path.join(os.path.dirname(data_yaml), '../VisDrone'), # 假设在 datasets/VisDrone
        '/home/jack/11/1027/YOLO13/yolov13-main/datasets/VisDrone' # 绝对路径猜测
    ]

    print("🔍 正在搜索 Ground Truth JSON...")
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # 搜索该目录下所有带 'val' 的 json 文件，且排除 predictions.json
            candidates = glob.glob(os.path.join(search_dir, '**', '*val*.json'), recursive=True)
            candidates = [f for f in candidates if 'predictions' not in f and 'instance' not in f]
            
            if candidates:
                # 通常取找到的第一个，或者找名字里带 VisDrone 的
                gt_json_path = candidates[0]
                break
    
    # 如果没搜到，给用户一个手动填写的机会
    if not gt_json_path:
        # TODO: 如果脚本报错找不到 GT，请在这里手动填入你的真实路径
        # gt_json_path = "/home/jack/.../datasets/VisDrone/annotations/VisDrone_val.json"
        pass

    if gt_json_path and os.path.exists(gt_json_path):
        print(f"✅ 锁定真值文件: {gt_json_path}")
        
        try:
            # 调用 COCO API
            cocoGt = COCO(gt_json_path)
            cocoDt = cocoGt.loadRes(pred_json_path)
            
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            print("\n" + "-"*60)
            cocoEval.summarize() # <--- 这里会打印包含 area=small 的详细表格
            print("-"*60)
            
            # 单独高亮打印
            print(f"\n🏆 最终结果 - 小目标 mAP (area < 32x32): {cocoEval.stats[3]:.4f}")
            
        except Exception as e:
            print(f"❌ COCOEval 计算出错: {e}")
            print("可能是类别 ID 不对应 (VisDrone TXT ID vs COCO JSON ID)。")
    else:
        print("⚠️  警告: 未找到 Ground Truth JSON 文件。")
        print("   Ultralytics 可能已经将其生成在了数据集目录下，但脚本没搜到。")
        print("   请手动查看 model.val() 上方的日志，找到 'Converting ... to COCO format' 那一行，")
        print("   并将生成的路径填入代码中的 `gt_json_path` 变量。")
# python z-train/test.py 