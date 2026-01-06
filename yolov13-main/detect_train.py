from ultralytics import YOLO
 
# model = YOLO(r'ultralytics/cfg/models/v13/yolov13.yaml') 
model = YOLO(r'/home/jack/11/1027/YOLO13/yolov13-main/yolo11n.pt')
# model = YOLO(r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/models/11/yolo11n.yaml')

# model.load('yolo13l.pt') # 权重文件名，官网下载
# model.load('/home/jack/11/1027/YOLO13/yolov13-main/yolov13l.pt')
results = model.train(
    # data=r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VISO_Detection_coco.yaml', # 数据yaml文件
    data=r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/cfg/datasets/VisDrone.yaml', # 数据yaml文件

    epochs=300,
    batch=8,
    device=0,
    lr0=0.01,  # 设置较小的学习率
    workers=4,

    project='/home/jack/11/1027/YOLO13/yolov13-main/runs',
    name='yolo11n.yaml_VisDrone-300'
    # name='test'
    # workspace=4
    # weights=r'/home/jack/11/1027/YOLO13/yolov13-main/ultralytics/yolov13l.pt'
    )

