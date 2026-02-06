from ultralytics import YOLO

# 1. 加载训练好的模型
model = YOLO('/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone/2_yolo11n.yaml_VisDrone-640/weights/best.pt')

# 2. 运行验证模式，但指定在测试集上运行
# split='test' 会自动读取 yaml 中的 test 路径
metrics = model.val(data='/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml', split='test')

# 获取类别名称字典
names = model.names

# 打印所有类别的 mAP50-95
print(f"{'Class':<20} | {'mAP50-95':<10}")
print("-" * 35)
for i, m in enumerate(metrics.box.maps):
    # i 是类别索引，m 是该类别的 mAP50-95
    print(f"{names[i]:<20} | {m:.4f}")

# 3. 打印结果
print("Test mAP50:", metrics.box.map50)
print("Test mAP50-95:", metrics.box.map)

print(metrics.speed) 
# 输出示例: {'preprocess': 0.1, 'inference': 3.5, 'loss': 0.0, 'postprocess': 0.5}