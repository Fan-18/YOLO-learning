"""
可视化显示当前网络每个模块的输出特征图
"""

import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from gradcam_utils import YOLO11GradCAM, draw_on_image, preprocess_image 

path = '/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone/8_yolo11n_p2_VisDrone_1024'
weights_path = f'{path}/weights/best.pt'
# img_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-train/images/0000039_05625_d_0000062.jpg'
# img_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-train/images/0000046_00000_d_0000087.jpg'
# img_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-train/images/0000068_03174_d_0000009.jpg'
# img_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-train/images/0000072_05564_d_0000007.jpg'
img_path = '/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-test-dev/images/0000105_02931_d_0000074.jpg'



output_dir = f'{path}/vis_results/pic3'
os.makedirs(output_dir, exist_ok=True)

# 1. 加载模型
yolo_model = YOLO(weights_path) 
model = yolo_model.model.eval().to('cuda') 

# 2. 配置多层查看目标
target_layers_dict = {
    "Backbone_P2_Out": model.model[2],
    "Backbone_P3_Out": model.model[4],
    "Backbone_P4_Out": model.model[6],
    "Backbone_SPPF":   model.model[7],
    "Backbone_C2PSA":  model.model[8],
    "TD_P3_Fusion":    model.model[11],
    "TD_P2_Fusion":    model.model[14],
    "BU_P3_Output":    model.model[17],
    "BU_P4_Output":    model.model[20],
}

# 3. 推理并生成【纯检测框对比图】
raw_img = cv2.imread(img_path)
results = yolo_model(raw_img)[0]

# 创建一个用于对比的纯框图
# 使用 label="" 且 txt_color=(0,0,0,0) 等技巧隐藏标签，或者直接调用 box_label 时不传文字
annotator_only_boxes = Annotator(raw_img.copy(), line_width=2)
if results.boxes:
    for d in results.boxes:
        conf = float(d.conf)
        if conf > 0.25:
            # 关键修改：只画框，不传任何 label 文字
            c = int(d.cls)
            annotator_only_boxes.box_label(d.xyxy[0], label="", color=colors(c, True))

# 保存纯框图
pure_boxes_path = os.path.join(output_dir, 'AAA_detection_boxes_only.jpg')
cv2.imwrite(pure_boxes_path, annotator_only_boxes.result())
print(f"检测框对比图已保存: {pure_boxes_path}")

# 4. 预处理张量用于热力图
input_tensor = preprocess_image(raw_img, target_size=(320, 320)).to('cuda')

# 5. 循环生成【纯净热力图】
for i, (name, layer) in enumerate(target_layers_dict.items()):
    # 获取该层在 model.model 中的实际索引数字 (例如 2, 4, 11 等)
    # 我们可以通过 list(target_layers_dict.values()) 来反查，或者手动硬编码
    # 这里我们直接从字典的顺序生成一个序号，方便文件夹排序
    
    # 初始化 CAM 工具
    cam_tool = YOLO11GradCAM(model, layer)
    
    # 生成热力图
    heatmap = cam_tool.generate_heatmap(input_tensor)
    
    # 叠加颜色
    pure_heatmap_img = draw_on_image(raw_img.copy(), heatmap)
    
    # 修改后的命名规则：序号_层名_pure_heatmap.jpg
    # {:02d} 确保序号是 01, 02 这种格式，方便在文件夹里按名称排序
    file_name = f"{i:02d}_{name}_pure_heatmap.jpg"
    save_path = os.path.join(output_dir, file_name)
    
    cv2.imwrite(save_path, pure_heatmap_img)
    print(f"已保存层级 {i}: {save_path}")
    
    cam_tool.remove_hooks()

print("\n可视化任务全部完成！")


# # 生成每个卷积核的特征图
# results = model.predict(
#     source='/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/data/VisDrone/VisDrone2019-DET-train/images/0000002_00005_d_0000014.jpg', 
#     visualize=True,   # 开启特征图可视化
#     save=True,        # 保存预测结果图
#     project='/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone/8_yolo11n_p2_VisDrone_1024/vis_results', # 自定义大目录名称
#     name='experiment_1',      # 自定义子目录名称
#     exist_ok=True             # 如果目录已存在，不新建目录而是直接覆盖
# )


# python z-train/visualize_test.py