from ultralytics import YOLO
model = YOLO("/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/models/11/yolo11-p2.yaml")
print(model.info()) # 检查是否有 4 个 Detection Layers
print(f"Model strides: {model.model.stride}")