import os
import sys
from pathlib import Path
import torch
import yaml

# 将项目根目录添加到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train import train, parse_opt
from utils.general import LOGGER, colorstr, increment_path
from utils.callbacks import Callbacks

def run_visdrone_training():
    # 1. 基础配置解析
    opt = parse_opt(known=True)
    
    # 2. 针对 FFCA-YOLO 和 VisDrone 的核心参数修改
    # 注意：确保你的软链接 /data/VisDrone 已经配置好
    opt.data = str(ROOT / 'data/VisDrone.yaml') 
    
    # 指向 FFCA-YOLO 论文中提到的核心配置文件 [cite: 2208, 2328]
    # 默认使用 yolov5m 基础上的改进版
    opt.cfg = str(ROOT / 'models/FFCA-YOLO.yaml') 
    
    # 权重设置：从头开始训练以验证 FFCA 模块的增益 [cite: 1845]
    opt.weights = '' 
    
    # 训练超参数优化
    opt.epochs = 300            # 建议训练300轮以保证收敛 [cite: 1841]
    opt.batch_size = 8          # 根据你 RTX 5090 的显存情况，1024分辨率建议设为 8-16
    opt.imgsz = 1024            # VisDrone 小目标多，必须使用高分辨率 [cite: 1841]
    
    # 硬件设置
    opt.device = '0'            # 使用你的第一块显卡 (RTX 5090)
    opt.workers = 4             # 5090 性能强，可以适当调高数据读取线程
    
    # 路径与命名
    opt.project = str(ROOT / 'runs/train_visdrone')
    opt.name = '1_FFCA_YOLO_VisDrone_1024'
    opt.exist_ok = True


    # --- 关键修复：手动初始化 save_dir ---
    # 这行代码会生成如 runs/train_visdrone/FFCA_YOLO_VisDrone2 这样的目录
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    # 创建目录
    os.makedirs(opt.save_dir, exist_ok=True)
    # ------------------------------------
    
    # 保存周期
    opt.save_period = 10        # 每10轮保存一次权重，防止意外中断
    
   # 3. 加载超参数
    hyp_path = ROOT / 'data/hyps/hyp.scratch-low.yaml'
    with open(hyp_path, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # 关键修复：将路径加载为字典
    
    LOGGER.info(colorstr('VisDrone Training: ') + f'Model={opt.cfg}, ImgSize={opt.imgsz}, SaveDir={opt.save_dir}')

    # 4. 执行训练
    try:
        # 传入加载好的 hyp 字典，而不是路径
        train(hyp, opt, torch.device('cuda:0'), Callbacks())
    except Exception as e:
        import traceback
        LOGGER.error(f"训练启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    
    run_visdrone_training()