from ultralytics import YOLO
 
# model = YOLO(r'ultralytics/cfg/models/v13/yolov13.yaml') 
model = YOLO(r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/models/11/yolo11-p2.yaml')
# model = YOLO(r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/models/11/yolo11m.yaml')

model.load("yolo11n.pt")

# model.load('yolo13l.pt') # 权重文件名，官网下载
# model.load('/home/jack/11/1027/YOLO13/yolov13-main/yolov13l.pt')
results = model.train(
    data=r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml', # 数据yaml文件

    epochs=300,
    batch=8,
    device=0,
    # lr0=0.01,  # 设置较小的学习率
    workers=4,
    imgsz=1024,

    patience=50, # 如果连续 50 轮效果没有改进，则提前停止训练

    project='/home/jack/11/1027/YOLO13/yolov13-main/runs/VisDrone',
    name='8_yolo11n_p2_VisDrone_1024_again',
    save_json=True
    # name='test'
    # workspace=4
    # weights=r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/yolov13l.pt'
    )


if __name__ == '__main__':
    model()


# python z-train/detect_train.py | tee train_log.txt