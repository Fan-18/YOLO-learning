from ultralytics import YOLO
import torch
import numpy as np
import random
import os

# 1. 随机种子设置（保持你原有的，很好）
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 2. 在这里应用种子
    seed_everything(42)

    # 3. 加载模型
    # 注意：你用 nano 的权重(pt)加载到自定义 yaml 结构中
    # Ultralytics 会自动匹配能用的权重，不匹配的(如FEM模块)会随机初始化，这是正常的 Transfer Learning。
    # model = YOLO(r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/models/v13/yolov13-p2-nop5v2.yaml')

    # model = YOLO(r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/models/11/yolo11-p2-FEM-SPDv2.yaml')
    model = YOLO('/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone/30_yolo11n_p2_FEM_SPD_yamlv2_det_VisDrone_1024_bc8/weights/best.pt')
    model.load("yolo11n.pt")

    # 4. 训练配置
    results = model.train(
        data=r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml',
        
        epochs=300,
        imgsz=1024,      # VisDrone 必须大图，1024 没问题
        
        # --- 针对 RTX 5090 的关键修改 ---
        batch=8,        # 建议从 64 开始尝试，如果显存还剩很多，可以加到 96 或 128
        device=0,
        workers=16,      # 5090 跑得快，需要更多 CPU 线程来预处理图片，防止 CPU 瓶颈
        
        # --- 针对实验复现 ---
        seed=42,
        # deterministic=True,
        
        # --- 针对自定义模块 ---
        amp=True,       # 如果你的 FEM 模块容易 NaN，保持 False；否则建议 True 以提速
        
        resume=True,   # 从上次中断处继续训练

        patience=50,
        project='/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone',
        # project='/home/jack/11/1027/YOLO13/yolov13-main/runs/YOLOv13_VisDrone',

        name='30_yolo11n_p2_FEM_SPD_yamlv2_det_VisDrone_1024_bc8' # 改名标记 batch size

        
    )

if __name__ == '__main__':
    # 所有的逻辑都要包在这里面！
    main()

# python z-train/detect_train.py | tee train_log.txt